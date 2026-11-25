/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/fp8_linear.hpp"

#include "kernel_image.hpp"
#include "wmma_tile.hpp"

#include "eclair/formats/fp8.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace eclair::cuda {
namespace {

using detail::kPerThread;
using detail::kStageLd;
using detail::kTileThreads;

constexpr std::size_t kMaxBlocks = 65535U;

// Batches up to this many rows use CUDA cores: one warp per output channel
// decodes each weight once and applies it to every row. Larger batches use
// the shared tensor-core tile loop in wmma_tile.hpp.
constexpr std::uint32_t kSkinnyMaxRows = 4U;
constexpr unsigned int kSkinnyThreads = 128U;

struct KernelShape {
    std::uint32_t m;
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t block_n;
    std::uint32_t block_k;
    std::uint32_t scale_columns;
    // Set by the launcher when K and the base pointers allow the selected
    // kernel's vector widths for every row. Wide weights are the CUDA-core
    // path's 16-byte loads, on top of its 4-byte vector loads.
    bool vector_activations;
    bool vector_weights;
    bool wide_weights;
};

[[nodiscard]] __device__ __half2 as_half2(std::uint32_t bits) {
    return __halves2half2(__ushort_as_half(static_cast<unsigned short>(bits & 0xFFFFU)),
                          __ushort_as_half(static_cast<unsigned short>(bits >> 16U)));
}

[[nodiscard]] __device__ std::uint32_t as_bits(__half2 value) {
    return static_cast<std::uint32_t>(__half_as_ushort(__low2half(value))) |
           (static_cast<std::uint32_t>(__half_as_ushort(__high2half(value))) << 16U);
}

[[nodiscard]] __device__ float half_bits_to_float(std::uint32_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xFFFFU)));
}

// Exact binary16 bits for FP8 bytes 0 and 1 of `pair`, low byte in the low
// lane, decoded together with 32-bit and half2 operations. This must equal
// formats::fp8_to_fp16_bits for all 256 patterns of both formats (NaN payloads
// aside). The GPU tests check every pattern through both kernels.
template <formats::Fp8Format Format>
[[nodiscard]] __device__ std::uint32_t decode_pair(std::uint32_t pair) {
    // Each byte moves to the high byte of its 16-bit lane.
    const std::uint32_t high = __byte_perm(pair, 0U, 0x1404U);
    if constexpr (Format == formats::Fp8Format::E5M2) {
        // E5M2 is the upper byte of binary16.
        return high;
    } else {
        // One right shift aligns E4M3's 4-bit exponent and 3-bit mantissa with
        // binary16's fields. The biases differ by 8, so the shifted value is
        // 2^-8 times too small and the product restores it exactly. E4M3
        // subnormals become binary16 subnormals, which the product also scales
        // exactly.
        const std::uint32_t magnitude = high & 0x7F007F00U;
        const std::uint32_t shifted = (high & 0x80008000U) | (magnitude >> 1U);
        const std::uint32_t value = as_bits(__hmul2(as_half2(shifted), __float2half2_rn(256.0F)));
        // S.1111.111 is E4M3FN's only NaN, which would otherwise decode as 480.
        return value | __vcmpeq2(magnitude, 0x7F007F00U);
    }
}

// The contract's weight values for a decoded pair: value * scale rounded to
// binary16 nearest-even. Callers skip this for a unit scale, which is exact
// because every FP8 value is a binary16 value.
[[nodiscard]] __device__ std::uint32_t scale_pair(std::uint32_t decoded, float low_scale,
                                                  float high_scale) {
    const float2 wide = __half22float2(as_half2(decoded));
    return as_bits(__floats2half2_rn(wide.x * low_scale, wide.y * high_scale));
}

enum class ScaleKind { Tensor, Channel, Block };

template <ScaleKind Kind>
[[nodiscard]] __device__ float weight_scale(const KernelShape& shape,
                                            const ops::Fp8LinearBuffers& buffers, std::uint32_t n,
                                            std::uint32_t k) {
    if constexpr (Kind == ScaleKind::Tensor) {
        return buffers.scales.data[0];
    } else if constexpr (Kind == ScaleKind::Channel) {
        return buffers.scales.data[n];
    } else {
        return buffers.scales
            .data[static_cast<std::size_t>(n / shape.block_n) * shape.scale_columns +
                  k / shape.block_k];
    }
}

// Applies Words groups of four adjacent weights, starting at column k of row
// n, to every activation row. A 4-aligned group shares one block scale when
// block_k is a multiple of four. Activations load as 8-byte groups.
template <formats::Fp8Format Format, ScaleKind Kind, std::uint32_t Words>
__device__ void accumulate_columns(const KernelShape& shape, const ops::Fp8LinearBuffers& buffers,
                                   std::uint32_t n, std::uint32_t k,
                                   const std::uint32_t (&packed)[Words], float row_scale, bool unit,
                                   float (&sums)[kSkinnyMaxRows]) {
    float w[4U * Words];
#pragma unroll
    for (std::uint32_t word = 0; word < Words; ++word) {
        std::uint32_t low = decode_pair<Format>(packed[word]);
        std::uint32_t high = decode_pair<Format>(packed[word] >> 16U);
        if (!unit) {
            const std::uint32_t column = k + 4U * word;
            if (Kind != ScaleKind::Block || shape.block_k % 4U == 0U) {
                const float scale = Kind == ScaleKind::Block
                                        ? weight_scale<Kind>(shape, buffers, n, column)
                                        : row_scale;
                low = scale_pair(low, scale, scale);
                high = scale_pair(high, scale, scale);
            } else {
                low = scale_pair(low, weight_scale<Kind>(shape, buffers, n, column),
                                 weight_scale<Kind>(shape, buffers, n, column + 1U));
                high = scale_pair(high, weight_scale<Kind>(shape, buffers, n, column + 2U),
                                  weight_scale<Kind>(shape, buffers, n, column + 3U));
            }
        }
        w[4U * word] = half_bits_to_float(low);
        w[4U * word + 1U] = half_bits_to_float(low >> 16U);
        w[4U * word + 2U] = half_bits_to_float(high);
        w[4U * word + 3U] = half_bits_to_float(high >> 16U);
    }
#pragma unroll
    for (std::uint32_t m = 0; m < kSkinnyMaxRows; ++m) {
        if (m < shape.m) {
            const auto* row = reinterpret_cast<const uint2*>(
                buffers.activations.data + static_cast<std::size_t>(m) * shape.k + k);
#pragma unroll
            for (std::uint32_t word = 0; word < Words; ++word) {
                const uint2 a = row[word];
                sums[m] = fmaf(half_bits_to_float(a.x), w[4U * word], sums[m]);
                sums[m] = fmaf(half_bits_to_float(a.x >> 16U), w[4U * word + 1U], sums[m]);
                sums[m] = fmaf(half_bits_to_float(a.y), w[4U * word + 2U], sums[m]);
                sums[m] = fmaf(half_bits_to_float(a.y >> 16U), w[4U * word + 3U], sums[m]);
            }
        }
    }
}

// Decode batches without expanding weights into a temporary FP16 matrix or
// computing padded activation rows. Each lane accumulates a strided share of K
// for every row, then the warp reduces each row. Wide 16-byte weight loads
// keep enough bytes in flight to approach DRAM bandwidth.
template <formats::Fp8Format Format, ScaleKind Kind>
__global__ void __launch_bounds__(kSkinnyThreads)
    fp8_linear_skinny_kernel(KernelShape shape, ops::Fp8LinearBuffers buffers) {
    const std::uint32_t lane = threadIdx.x % 32U;
    const std::size_t warps = blockDim.x / 32U;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * warps;
    for (std::size_t channel = blockIdx.x * warps + threadIdx.x / 32U; channel < shape.n;
         channel += stride) {
        const auto n = static_cast<std::uint32_t>(channel);
        const std::uint8_t* weights = buffers.weights.data + channel * shape.k;
        const float row_scale =
            Kind == ScaleKind::Block ? 0.0F : weight_scale<Kind>(shape, buffers, n, 0U);
        const bool unit = Kind != ScaleKind::Block && row_scale == 1.0F;
        float sums[kSkinnyMaxRows] = {};
        if (shape.vector_activations && shape.wide_weights) {
            for (std::uint32_t k = lane * 16U; k < shape.k; k += 512U) {
                const uint4 bytes = *reinterpret_cast<const uint4*>(weights + k);
                const std::uint32_t packed[4] = {bytes.x, bytes.y, bytes.z, bytes.w};
                accumulate_columns<Format, Kind>(shape, buffers, n, k, packed, row_scale, unit,
                                                 sums);
            }
        } else if (shape.vector_activations && shape.vector_weights) {
            for (std::uint32_t k = lane * 4U; k < shape.k; k += 128U) {
                const std::uint32_t packed[1] = {
                    *reinterpret_cast<const std::uint32_t*>(weights + k)};
                accumulate_columns<Format, Kind>(shape, buffers, n, k, packed, row_scale, unit,
                                                 sums);
            }
        } else {
            for (std::uint32_t k = lane; k < shape.k; k += 32U) {
                std::uint32_t bits = decode_pair<Format>(weights[k]);
                if (!unit) {
                    const float scale = Kind == ScaleKind::Block
                                            ? weight_scale<Kind>(shape, buffers, n, k)
                                            : row_scale;
                    bits = scale_pair(bits, scale, 1.0F);
                }
                const float w = half_bits_to_float(bits);
#pragma unroll
                for (std::uint32_t m = 0; m < kSkinnyMaxRows; ++m) {
                    if (m < shape.m) {
                        sums[m] = fmaf(half_bits_to_float(
                                           buffers.activations
                                               .data[static_cast<std::size_t>(m) * shape.k + k]),
                                       w, sums[m]);
                    }
                }
            }
        }
        const float bias = buffers.bias.size == 0U ? 0.0F : buffers.bias.data[n];
#pragma unroll
        for (std::uint32_t m = 0; m < kSkinnyMaxRows; ++m) {
            float sum = sums[m];
            for (int offset = 16; offset != 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
            }
            if (lane == 0U && m < shape.m) {
                buffers.output.data[static_cast<std::size_t>(m) * shape.n + n] = sum + bias;
            }
        }
    }
}

// Scales for one thread's kPerThread contiguous columns starting at k, all in
// row n. Block indices advance at boundaries instead of dividing per element.
// Columns at or past `valid` get 1, since padding never reads a scale.
__device__ void block_scales(const KernelShape& shape, const ops::Fp8LinearBuffers& buffers,
                             std::uint32_t n, std::uint32_t k, std::uint32_t valid,
                             float (&scales)[kPerThread]) {
    if (valid == 0U) {
        for (float& scale : scales) {
            scale = 1.0F;
        }
        return;
    }
    const float* row =
        buffers.scales.data + static_cast<std::size_t>(n / shape.block_n) * shape.scale_columns;
    std::uint32_t block = k / shape.block_k;
    // At most K + block_k, which fits because both are below 2^31.
    std::uint32_t next = (block + 1U) * shape.block_k;
#pragma unroll
    for (std::uint32_t i = 0; i < kPerThread; ++i) {
        if (k + i == next) {
            ++block;
            next += shape.block_k;
        }
        scales[i] = i < valid ? row[block] : 1.0F;
    }
}

// FP8 weights for the shared tile loop: 16 raw bytes per thread and stage,
// decoded and scaled into the transposed B tile when stored. Row n of W holds
// K-contiguous values, which WMMA reads as column-major B[K,N]. Padding is
// zeroed after scaling, so a NaN or infinite scale cannot reach a valid
// output through a zero-padded product.
template <formats::Fp8Format Format, ScaleKind Kind>
struct Fp8TileWeights {
    KernelShape shape;
    ops::Fp8LinearBuffers buffers;

    using Registers = uint4;

    // Bytes outside W hold zero, which decodes to +0.
    __device__ void load(std::uint32_t n_begin, std::uint32_t k_begin, uint4& registers) const {
        const std::uint32_t n = n_begin + detail::stage_row();
        const std::uint32_t k = k_begin + detail::stage_column();
        if (shape.vector_weights && k + kPerThread <= shape.k && n < shape.n) {
            registers = *reinterpret_cast<const uint4*>(buffers.weights.data +
                                                        static_cast<std::size_t>(n) * shape.k + k);
            return;
        }
        std::uint32_t words[4] = {};
        for (std::uint32_t i = 0; i < kPerThread && n < shape.n && k + i < shape.k; ++i) {
            words[i / 4U] |=
                static_cast<std::uint32_t>(
                    buffers.weights.data[static_cast<std::size_t>(n) * shape.k + k + i])
                << (8U * (i % 4U));
        }
        registers = uint4{words[0], words[1], words[2], words[3]};
    }

    __device__ void store(std::uint32_t n_begin, std::uint32_t k_begin, const uint4& registers,
                          __half* b_tile) const {
        const std::uint32_t row = detail::stage_row();
        const std::uint32_t column = detail::stage_column();
        const std::uint32_t n = n_begin + row;
        const std::uint32_t k = k_begin + column;
        std::uint32_t valid = 0U;
        if (n < shape.n && k < shape.k) {
            valid = shape.k - k < kPerThread ? shape.k - k : kPerThread;
        }
        float scales[kPerThread];
        bool unit = false;
        if constexpr (Kind == ScaleKind::Block) {
            block_scales(shape, buffers, n, k, valid, scales);
        } else {
            const float scale = valid == 0U ? 1.0F : weight_scale<Kind>(shape, buffers, n, 0U);
            unit = scale == 1.0F;
            for (float& value : scales) {
                value = scale;
            }
        }
        const std::uint32_t bytes[4] = {registers.x, registers.y, registers.z, registers.w};
        std::uint32_t halves[kPerThread / 2U];
#pragma unroll
        for (std::uint32_t pair = 0; pair < kPerThread / 2U; ++pair) {
            std::uint32_t bits = decode_pair<Format>(bytes[pair / 2U] >> (16U * (pair % 2U)));
            if (!unit) {
                bits = scale_pair(bits, scales[2U * pair], scales[2U * pair + 1U]);
            }
            if (2U * pair + 1U >= valid) {
                bits = 2U * pair < valid ? bits & 0xFFFFU : 0U;
            }
            halves[pair] = bits;
        }
        auto* target = reinterpret_cast<uint4*>(b_tile + row * kStageLd + column);
        target[0] = uint4{halves[0], halves[1], halves[2], halves[3]};
        target[1] = uint4{halves[4], halves[5], halves[6], halves[7]};
    }
};

[[nodiscard]] bool aligned(const void* pointer, std::uintptr_t bytes) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0U;
}

template <formats::Fp8Format Format, ScaleKind Kind>
[[nodiscard]] cudaError_t launch(KernelShape shape, const ops::Fp8LinearBuffers& buffers,
                                 cudaStream_t stream) noexcept {
    if (shape.m <= kSkinnyMaxRows) {
        shape.vector_activations = shape.k % 4U == 0U && aligned(buffers.activations.data, 8U);
        shape.vector_weights = shape.k % 4U == 0U && aligned(buffers.weights.data, 4U);
        shape.wide_weights = shape.k % 16U == 0U && aligned(buffers.weights.data, 16U);
        const std::size_t warps = kSkinnyThreads / 32U;
        const std::size_t blocks = std::min(1U + (shape.n - 1U) / warps, kMaxBlocks);
        fp8_linear_skinny_kernel<Format, Kind>
            <<<static_cast<unsigned int>(blocks), kSkinnyThreads, 0, stream>>>(shape, buffers);
    } else {
        shape.vector_activations = shape.k % 8U == 0U && aligned(buffers.activations.data, 16U);
        shape.vector_weights = shape.k % 16U == 0U && aligned(buffers.weights.data, 16U);
        const detail::TileShape tile{shape.m, shape.n, shape.k, shape.vector_activations};
        const detail::Bias bias{buffers.bias.size == 0U ? nullptr : buffers.bias.data};
        detail::linear_tile_kernel<Fp8TileWeights<Format, Kind>, detail::BiasEpilogue>
            <<<detail::tile_blocks(tile), kTileThreads, 0, stream>>>(
                tile, buffers.activations.data, Fp8TileWeights<Format, Kind>{shape, buffers},
                detail::BiasEpilogue{bias}, buffers.output.data);
    }
    return cudaGetLastError();
}

template <formats::Fp8Format Format>
[[nodiscard]] cudaError_t select_scale(const KernelShape& shape,
                                       const ops::Fp8LinearBuffers& buffers,
                                       cudaStream_t stream) noexcept {
    if (shape.block_n == shape.n && shape.block_k == shape.k) {
        return launch<Format, ScaleKind::Tensor>(shape, buffers, stream);
    }
    if (shape.block_n == 1U && shape.block_k == shape.k) {
        return launch<Format, ScaleKind::Channel>(shape, buffers, stream);
    }
    return launch<Format, ScaleKind::Block>(shape, buffers, stream);
}

template <formats::Fp8Format Format, ScaleKind Kind>
constexpr auto tile_kernel =
    detail::linear_tile_kernel<Fp8TileWeights<Format, Kind>, detail::BiasEpilogue>;

}  // namespace

cudaError_t launch_fp8_linear(const ops::Fp8LinearShape& shape,
                              const ops::Fp8LinearBuffers& buffers, cudaStream_t stream) noexcept {
    if (ops::validate_fp8_linear(shape, buffers) != ops::Status::Success) {
        return cudaErrorInvalidValue;
    }
    if (shape.m == 0U || shape.n == 0U) {
        return cudaSuccess;
    }
    constexpr auto kMaxDimension = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (shape.m > kMaxDimension || shape.n > kMaxDimension || shape.k > kMaxDimension) {
        return cudaErrorInvalidValue;
    }
    const KernelShape kernel_shape{static_cast<std::uint32_t>(shape.m),
                                   static_cast<std::uint32_t>(shape.n),
                                   static_cast<std::uint32_t>(shape.k),
                                   static_cast<std::uint32_t>(std::min(shape.block_n, shape.n)),
                                   static_cast<std::uint32_t>(std::min(shape.block_k, shape.k)),
                                   static_cast<std::uint32_t>(1U + (shape.k - 1U) / shape.block_k),
                                   false,
                                   false,
                                   false};
    return shape.format == formats::Fp8Format::E4M3FN
               ? select_scale<formats::Fp8Format::E4M3FN>(kernel_shape, buffers, stream)
               : select_scale<formats::Fp8Format::E5M2>(kernel_shape, buffers, stream);
}

// Both kernel families for every format and scale layout, so a partial build
// cannot report the operation as available.
cudaError_t check_fp8_linear_image(int device_index) noexcept {
    using formats::Fp8Format;
    return detail::check_kernel_images(
        device_index, fp8_linear_skinny_kernel<Fp8Format::E4M3FN, ScaleKind::Tensor>,
        fp8_linear_skinny_kernel<Fp8Format::E4M3FN, ScaleKind::Channel>,
        fp8_linear_skinny_kernel<Fp8Format::E4M3FN, ScaleKind::Block>,
        fp8_linear_skinny_kernel<Fp8Format::E5M2, ScaleKind::Tensor>,
        fp8_linear_skinny_kernel<Fp8Format::E5M2, ScaleKind::Channel>,
        fp8_linear_skinny_kernel<Fp8Format::E5M2, ScaleKind::Block>,
        tile_kernel<Fp8Format::E4M3FN, ScaleKind::Tensor>,
        tile_kernel<Fp8Format::E4M3FN, ScaleKind::Channel>,
        tile_kernel<Fp8Format::E4M3FN, ScaleKind::Block>,
        tile_kernel<Fp8Format::E5M2, ScaleKind::Tensor>,
        tile_kernel<Fp8Format::E5M2, ScaleKind::Channel>,
        tile_kernel<Fp8Format::E5M2, ScaleKind::Block>);
}

}  // namespace eclair::cuda
