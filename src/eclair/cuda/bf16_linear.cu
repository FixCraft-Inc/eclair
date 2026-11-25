/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/bf16_linear.hpp"

#include "kernel_image.hpp"
#include "wmma_tile.hpp"

#include "eclair/formats/bf16.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace eclair::cuda {
namespace {

constexpr unsigned int kRowThreads = 256U;
constexpr std::size_t kRowWarps = kRowThreads / 32U;
// Enough resident warps to keep any current GPU's memory system busy. A
// grid-stride loop covers more rows.
constexpr std::size_t kMaxRowBlocks = 4096U;
// Values per 16-byte access.
constexpr std::size_t kVector = 8U;

[[nodiscard]] __device__ std::uint32_t larger(std::uint32_t a, std::uint32_t b) {
    return a > b ? a : b;
}

// One warp per row, grid-striding over rows. Lanes reduce the row's largest
// finite magnitude, every lane derives the same shift, then lanes convert. A
// short row is read twice from cache. With `vector`, each access moves eight
// values; the launcher sets it when the column count is a multiple of eight
// and both matrices are 16-byte aligned, so every row is too.
__global__ void __launch_bounds__(kRowThreads)
    bf16_row_scaling_kernel(const std::uint16_t* input, std::uint16_t* output, std::int32_t* shifts,
                            std::size_t rows, std::size_t columns, bool vector) {
    const std::uint32_t lane = threadIdx.x % 32U;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * kRowWarps;
    for (std::size_t row = static_cast<std::size_t>(blockIdx.x) * kRowWarps + threadIdx.x / 32U;
         row < rows; row += stride) {
        const std::uint16_t* source = input + row * columns;
        std::uint16_t* target = output + row * columns;
        std::uint32_t largest = 0U;
        if (vector) {
            for (std::size_t column = lane * kVector; column < columns; column += 32U * kVector) {
                const uint4 values = *reinterpret_cast<const uint4*>(source + column);
                for (const std::uint32_t pair : {values.x, values.y, values.z, values.w}) {
                    largest = larger(largest, formats::bf16_finite_magnitude(
                                                  static_cast<std::uint16_t>(pair & 0xFFFFU)));
                    largest = larger(largest, formats::bf16_finite_magnitude(
                                                  static_cast<std::uint16_t>(pair >> 16U)));
                }
            }
        } else {
            for (std::size_t column = lane; column < columns; column += 32U) {
                largest = larger(largest, formats::bf16_finite_magnitude(source[column]));
            }
        }
        for (int offset = 16; offset != 0; offset /= 2) {
            largest = larger(largest, __shfl_xor_sync(0xFFFFFFFFU, largest, offset));
        }
        const std::int32_t shift = formats::bf16_fp16_shift(static_cast<std::uint16_t>(largest));
        if (lane == 0U) {
            shifts[row] = shift;
        }
        if (vector) {
            for (std::size_t column = lane * kVector; column < columns; column += 32U * kVector) {
                const uint4 values = *reinterpret_cast<const uint4*>(source + column);
                std::uint32_t words[4] = {values.x, values.y, values.z, values.w};
                for (std::uint32_t& word : words) {
                    word = formats::bf16_to_fp16_scaled_bits(
                               static_cast<std::uint16_t>(word & 0xFFFFU), shift) |
                           (static_cast<std::uint32_t>(formats::bf16_to_fp16_scaled_bits(
                                static_cast<std::uint16_t>(word >> 16U), shift))
                            << 16U);
                }
                *reinterpret_cast<uint4*>(target + column) =
                    uint4{words[0], words[1], words[2], words[3]};
            }
        } else {
            for (std::size_t column = lane; column < columns; column += 32U) {
                target[column] = formats::bf16_to_fp16_scaled_bits(source[column], shift);
            }
        }
    }
}

// Undoes both rows' powers of two, then adds the bias. scalbnf rounds once,
// so the power is exact unless the result leaves FP32's normal range.
struct ShiftEpilogue {
    detail::Bias bias;
    const std::int32_t* activation_shifts;
    const std::int32_t* weight_shifts;

    [[nodiscard]] __device__ float operator()(float sum, std::uint32_t m, std::uint32_t n) const {
        const std::int32_t power = -(formats::clamp_bf16_fp16_shift(activation_shifts[m]) +
                                     formats::clamp_bf16_fp16_shift(weight_shifts[n]));
        return scalbnf(sum, power) + bias[n];
    }
};

constexpr auto tile_kernel = detail::linear_tile_kernel<detail::Fp16Weights, ShiftEpilogue>;

// Decode batches already hold binary16 operands after row scaling. A warp
// streams one output channel, reusing its weights across the few real rows
// instead of computing a padded 64-row tensor-core tile.
template <unsigned int Rows, bool Vector>
__global__ void __launch_bounds__(128)
    bf16_linear_skinny_kernel(std::uint32_t n, std::uint32_t k, const std::uint16_t* activations,
                              const std::uint16_t* weights, ShiftEpilogue epilogue, float* output) {
    const unsigned int lane = threadIdx.x % 32U;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * 4U;
    for (std::size_t channel = static_cast<std::size_t>(blockIdx.x) * 4U + threadIdx.x / 32U;
         channel < n; channel += stride) {
        float sums[Rows] = {};
        const auto* w = weights + channel * k;
        if constexpr (Vector) {
            for (std::uint32_t column = lane * 8U; column < k; column += 256U) {
                const uint4 word = *reinterpret_cast<const uint4*>(w + column);
                const std::uint32_t packed[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
                for (unsigned int row = 0; row < Rows; ++row) {
                    const uint4 a = *reinterpret_cast<const uint4*>(
                        activations + static_cast<std::size_t>(row) * k + column);
                    const std::uint32_t ap[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                    for (unsigned int pair = 0; pair < 4U; ++pair) {
                        const float2 wf = __half22float2(__halves2half2(
                            __ushort_as_half(static_cast<unsigned short>(packed[pair])),
                            __ushort_as_half(static_cast<unsigned short>(packed[pair] >> 16U))));
                        const float2 af = __half22float2(__halves2half2(
                            __ushort_as_half(static_cast<unsigned short>(ap[pair])),
                            __ushort_as_half(static_cast<unsigned short>(ap[pair] >> 16U))));
                        sums[row] = fmaf(af.x, wf.x, sums[row]);
                        sums[row] = fmaf(af.y, wf.y, sums[row]);
                    }
                }
            }
        } else {
            for (std::uint32_t column = lane; column < k; column += 32U) {
                const float value = __half2float(__ushort_as_half(w[column]));
#pragma unroll
                for (unsigned int row = 0; row < Rows; ++row) {
                    sums[row] = fmaf(__half2float(__ushort_as_half(
                                         activations[static_cast<std::size_t>(row) * k + column])),
                                     value, sums[row]);
                }
            }
        }
#pragma unroll
        for (unsigned int row = 0; row < Rows; ++row) {
            float sum = sums[row];
            for (int delta = 16; delta != 0; delta /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFFU, sum, delta);
            }
            if (lane == 0U) {
                output[static_cast<std::size_t>(row) * n + channel] =
                    epilogue(sum, row, static_cast<std::uint32_t>(channel));
            }
        }
    }
}

template <unsigned int Rows>
void launch_skinny(std::uint32_t n, std::uint32_t k, const ops::Bf16LinearBuffers& b,
                   ShiftEpilogue epilogue, bool vector, cudaStream_t stream) {
    const auto blocks = static_cast<unsigned int>(std::min<std::size_t>((n + 3U) / 4U, 4096U));
    if (vector) {
        bf16_linear_skinny_kernel<Rows, true><<<blocks, 128, 0, stream>>>(
            n, k, b.activations.data, b.weights.data, epilogue, b.output.data);
    } else {
        bf16_linear_skinny_kernel<Rows, false><<<blocks, 128, 0, stream>>>(
            n, k, b.activations.data, b.weights.data, epilogue, b.output.data);
    }
}

[[nodiscard]] bool aligned(const void* pointer, std::uintptr_t bytes) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0U;
}

}  // namespace

cudaError_t launch_bf16_row_scaling(const ops::Bf16RowScalingShape& shape,
                                    const ops::Bf16RowScalingBuffers& buffers,
                                    cudaStream_t stream) noexcept {
    if (ops::validate_bf16_row_scaling(shape, buffers) != ops::Status::Success) {
        return cudaErrorInvalidValue;
    }
    if (shape.rows == 0U) {
        return cudaSuccess;
    }
    const bool vector = shape.columns % kVector == 0U && aligned(buffers.input.data, 16U) &&
                        aligned(buffers.output.data, 16U);
    const std::size_t blocks = std::min(1U + (shape.rows - 1U) / kRowWarps, kMaxRowBlocks);
    bf16_row_scaling_kernel<<<static_cast<unsigned int>(blocks), kRowThreads, 0, stream>>>(
        buffers.input.data, buffers.output.data, buffers.shifts.data, shape.rows, shape.columns,
        vector);
    return cudaGetLastError();
}

cudaError_t launch_bf16_linear(const ops::Bf16LinearShape& shape,
                               const ops::Bf16LinearBuffers& buffers,
                               cudaStream_t stream) noexcept {
    if (ops::validate_bf16_linear(shape, buffers) != ops::Status::Success) {
        return cudaErrorInvalidValue;
    }
    if (shape.m == 0U || shape.n == 0U) {
        return cudaSuccess;
    }
    constexpr auto kMaxDimension = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (shape.m > kMaxDimension || shape.n > kMaxDimension || shape.k > kMaxDimension) {
        return cudaErrorInvalidValue;
    }
    const auto m = static_cast<std::uint32_t>(shape.m);
    const auto n = static_cast<std::uint32_t>(shape.n);
    const auto k = static_cast<std::uint32_t>(shape.k);
    const detail::TileShape tile{m, n, k,
                                 k % kVector == 0U && aligned(buffers.activations.data, 16U)};
    const detail::Fp16Weights weights{buffers.weights.data, n, k,
                                      k % kVector == 0U && aligned(buffers.weights.data, 16U)};
    const ShiftEpilogue epilogue{
        detail::Bias{buffers.bias.size == 0U ? nullptr : buffers.bias.data},
        buffers.activation_shifts.data, buffers.weight_shifts.data};
    if (m <= 4U) {
        const bool vector = tile.vector_activations && weights.vector;
        switch (m) {
            case 1U:
                launch_skinny<1>(n, k, buffers, epilogue, vector, stream);
                break;
            case 2U:
                launch_skinny<2>(n, k, buffers, epilogue, vector, stream);
                break;
            case 3U:
                launch_skinny<3>(n, k, buffers, epilogue, vector, stream);
                break;
            case 4U:
                launch_skinny<4>(n, k, buffers, epilogue, vector, stream);
                break;
        }
        return cudaGetLastError();
    }
    tile_kernel<<<detail::tile_blocks(tile), detail::kTileThreads, 0, stream>>>(
        tile, buffers.activations.data, weights, epilogue, buffers.output.data);
    return cudaGetLastError();
}

cudaError_t check_bf16_linear_image(int device_index) noexcept {
    return detail::check_kernel_images(
        device_index, bf16_row_scaling_kernel, tile_kernel, bf16_linear_skinny_kernel<1, true>,
        bf16_linear_skinny_kernel<1, false>, bf16_linear_skinny_kernel<2, true>,
        bf16_linear_skinny_kernel<2, false>, bf16_linear_skinny_kernel<3, true>,
        bf16_linear_skinny_kernel<3, false>, bf16_linear_skinny_kernel<4, true>,
        bf16_linear_skinny_kernel<4, false>);
}

}  // namespace eclair::cuda
