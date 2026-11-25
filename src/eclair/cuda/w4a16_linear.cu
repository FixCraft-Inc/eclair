/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/w4a16_linear.hpp"

#include "kernel_image.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace eclair::cuda {
namespace {

// One staged binary16 activation row per batch row and scale group. Eight
// halves of padding shift each row by four banks, so the rows' 16-byte
// fragment reads within a warp fall in distinct banks.
constexpr unsigned int kStageStride = 136U;

__device__ __forceinline__ std::uint32_t half2_bits(__half2 value) {
    const __half2_raw raw = value;
    std::uint32_t bits;
    static_assert(sizeof(raw) == sizeof(bits));
    std::memcpy(&bits, &raw, sizeof(bits));
    return bits;
}

__device__ __forceinline__ __half2 bits_half2(std::uint32_t bits) {
    __half2_raw raw;
    static_assert(sizeof(raw) == sizeof(bits));
    std::memcpy(&raw, &bits, sizeof(raw));
    return __half2(raw);
}

// (value & mask) | binary16 1024 in both halves, as one LOP3 (0xEA).
// Without it the tested CUDA 12.9 build emits separate AND and OR.
__device__ __forceinline__ __half2 biased_pair(std::uint32_t value, std::uint32_t mask) {
    std::uint32_t bits;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(bits) : "r"(value), "r"(mask), "r"(0x64006400U));
    return bits_half2(bits);
}

// Dequantizes one packed word into four binary16 pairs for logical offsets
// {0,1}, {2,3}, {4,5}, {6,7}. Masked nibbles placed in binary16 1024's
// mantissa give 1024 + q (low nibble) or 1024 + 16q (next nibble). Both
// recover q - 8 exactly; the FP16 scale multiplication is the only rounding.
__device__ __forceinline__ void dequantize(std::uint32_t packed, __half2 scale,
                                           std::uint32_t (&pairs)[4]) {
    const __half2 low_offset = __float2half2_rn(1032.0F);
    const __half2 sixteenth = __float2half2_rn(0.0625F);
    const __half2 high_offset = __float2half2_rn(-72.0F);
    const std::uint32_t upper = packed >> 8U;
    const std::uint32_t source[2] = {packed, upper};
#pragma unroll
    for (unsigned int i = 0; i < 2U; ++i) {
        const __half2 low = biased_pair(source[i], 0x000F000FU);
        const __half2 high = biased_pair(source[i], 0x00F000F0U);
        pairs[2U * i] = half2_bits(__hmul2(__hsub2(low, low_offset), scale));
        pairs[2U * i + 1U] = half2_bits(__hmul2(__hfma2(high, sixteenth, high_offset), scale));
    }
}

// Volta HMMA.884 through PTX mma.m8n8k4: each quad pair (lanes 4q..4q+3 and
// 16+4q..16+4q+3) multiplies its own row-major 8x4 binary16 A by a
// column-major 4x8 B with FP32 accumulation. A row and B column are both
// lane%4 + 4*(lane/16). Accumulator i is at row (lane&1) + (i&2) + 4*(lane/16),
// column (i&4) + (lane&2) + (i&1). Products are exact; see W4A16_LINEAR.md for
// the tensor cores' accumulation alignment.
__device__ __forceinline__ void mma_m8n8k4(float (&sums)[8], std::uint32_t a01, std::uint32_t a23,
                                           std::uint32_t b01, std::uint32_t b23) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(sums[0]), "+f"(sums[1]), "+f"(sums[2]), "+f"(sums[3]), "+f"(sums[4]), "+f"(sums[5]),
          "+f"(sums[6]), "+f"(sums[7])
        : "r"(a01), "r"(a23), "r"(b01), "r"(b23));
}

// Each warp covers 32 adjacent output channels as four 8-channel quad pairs,
// so the [K/8,N] pack stays coalesced. Dequantized weights are tensor-core A
// rows; B columns 0..Rows-1 are activation rows and the remaining columns
// duplicate row 0 into results that are never stored. Splits own whole scale
// groups, and a separate FP32 reduction avoids both binary16 atomics and
// cross-block ordering dependence. Every lane executes every mma.
template <unsigned int Rows>
__global__ void w4a16_partial(std::uint32_t n, std::uint32_t k, const std::uint16_t* a,
                              const std::uint32_t* weights, const std::uint16_t* scales,
                              float* workspace) {
    constexpr unsigned int kWords = Rows * 64U;               // binary16 pairs per staged group
    constexpr unsigned int kStaged = (kWords + 127U) / 128U;  // pairs per thread
    __shared__ __align__(16) std::uint16_t stage[2][Rows][kStageStride];
    const unsigned int lane = threadIdx.x % 32U;
    const unsigned int upper = lane / 16U;
    const std::size_t pair_base =
        static_cast<std::size_t>(blockIdx.x) * 128U + threadIdx.x / 32U * 32U + lane / 4U % 4U * 8U;
    const std::size_t channel = pair_base + lane % 4U + upper * 4U;
    const unsigned int row = lane % 4U < Rows ? lane % 4U : 0U;
    const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(a);
    const std::uint32_t groups = k / 128U;
    // The next group's activations are read before this group's products and
    // stored after them, so their L2 latency overlaps the products instead of
    // stalling the shared store before the barrier.
    std::uint32_t staged[kStaged];
    const auto load_activations = [&](std::uint32_t group) {
#pragma unroll
        for (unsigned int j = 0; j < kStaged; ++j) {
            const unsigned int i = threadIdx.x + j * 128U;
            if (i < kWords) {
                staged[j] = activation_pairs[static_cast<std::size_t>(i / 64U) * (k / 2U) +
                                             group * 64U + i % 64U];
            }
        }
    };
    const auto store_activations = [&](unsigned int target) {
#pragma unroll
        for (unsigned int j = 0; j < kStaged; ++j) {
            const unsigned int i = threadIdx.x + j * 128U;
            if (i < kWords) {
                std::memcpy(&stage[target][i / 64U][i % 64U * 2U], &staged[j], sizeof(staged[j]));
            }
        }
    };
    if (blockIdx.y < groups) {
        load_activations(blockIdx.y);
        store_activations(0U);
    }
    // Two accumulator chains halve the dependent mma sequence per group.
    float even[8] = {};
    float odd[8] = {};
    unsigned int buffer = 0;
    for (std::uint32_t group = blockIdx.y; group < groups; group += gridDim.y) {
        std::uint32_t words[16];
        __half scale = __float2half_rn(0.0F);
#pragma unroll
        for (unsigned int word = 0; word < 16U; ++word) {
            words[word] = 0x88888888U;
        }
        if (channel < n) {
            scale = __ushort_as_half(scales[static_cast<std::size_t>(group) * n + channel]);
#pragma unroll
            for (unsigned int word = 0; word < 16U; ++word) {
                words[word] = weights[(static_cast<std::size_t>(group) * 16U + word) * n + channel];
            }
        }
        const bool more = group + gridDim.y < groups;
        if (more) {
            load_activations(group + gridDim.y);
        }
        // Alternating buffers need one barrier per group: a thread stores
        // the next group only after every thread passed this barrier, which
        // follows the last reads of that buffer.
        __syncthreads();
        const __half2 scale2 = __half2half2(scale);
#pragma unroll
        for (unsigned int word = 0; word < 16U; ++word) {
            std::uint32_t pairs[4];
            dequantize(words[word], scale2, pairs);
            uint4 b;
            std::memcpy(&b, &stage[buffer][row][word * 8U], sizeof(b));
            mma_m8n8k4(even, pairs[0], pairs[1], b.x, b.y);
            mma_m8n8k4(odd, pairs[2], pairs[3], b.z, b.w);
        }
        buffer ^= 1U;
        if (more) {
            store_activations(buffer);
        }
    }
#pragma unroll
    for (unsigned int i = 0; i < 4U; ++i) {
        const unsigned int column = (lane & 2U) + (i & 1U);
        const std::size_t output_channel = pair_base + (lane & 1U) + (i & 2U) + upper * 4U;
        if (column < Rows && output_channel < n) {
            workspace[(static_cast<std::size_t>(blockIdx.y) * Rows + column) * n + output_channel] =
                even[i] + odd[i];
        }
    }
}

__global__ void w4a16_reduce(std::size_t count, unsigned int splits, const float* workspace,
                             std::uint16_t* output) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    float sum = 0;
    for (unsigned int split = 0; split < splits; ++split) {
        sum += workspace[static_cast<std::size_t>(split) * count + i];
    }
    output[i] = __half_as_ushort(__float2half_rn(sum));
}

// One thread per packed word row and Columns adjacent channels. Logical
// offset j of a word is half j % 2 of pair j / 2. With four channels each
// row store is 8 bytes, so a warp writes 256 contiguous bytes per K row.
template <unsigned int Columns>
__global__ void w4a16_dequantize(std::uint32_t n, std::uint32_t words, const std::uint32_t* weights,
                                 const std::uint16_t* scales, std::uint16_t* dequantized) {
    const std::uint32_t channel = (blockIdx.x * blockDim.x + threadIdx.x) * Columns;
    if (channel >= n) {
        return;
    }
    for (std::uint32_t word = blockIdx.y; word < words; word += gridDim.y) {
        const std::size_t source = static_cast<std::size_t>(word) * n + channel;
        const std::size_t scale_index = static_cast<std::size_t>(word / 16U) * n + channel;
        std::uint32_t packed[Columns];
        std::uint16_t scale[Columns];
        if constexpr (Columns == 4U) {
            const uint4 four = *reinterpret_cast<const uint4*>(weights + source);
            const uint2 group = *reinterpret_cast<const uint2*>(scales + scale_index);
            packed[0] = four.x;
            packed[1] = four.y;
            packed[2] = four.z;
            packed[3] = four.w;
            std::memcpy(scale, &group, sizeof(group));
        } else {
            packed[0] = weights[source];
            scale[0] = scales[scale_index];
        }
        std::uint32_t pairs[Columns][4];
#pragma unroll
        for (unsigned int c = 0; c < Columns; ++c) {
            dequantize(packed[c], __half2half2(__ushort_as_half(scale[c])), pairs[c]);
        }
#pragma unroll
        for (unsigned int j = 0; j < 8U; ++j) {
            std::uint16_t row[Columns];
#pragma unroll
            for (unsigned int c = 0; c < Columns; ++c) {
                row[c] = static_cast<std::uint16_t>(pairs[c][j / 2U] >> (16U * (j % 2U)));
            }
            std::uint16_t* target =
                dequantized + (static_cast<std::size_t>(word) * 8U + j) * n + channel;
            if constexpr (Columns == 4U) {
                uint2 bits;
                std::memcpy(&bits, row, sizeof(bits));
                *reinterpret_cast<uint2*>(target) = bits;
            } else {
                *target = row[0];
            }
        }
    }
}

template <unsigned int Rows>
void launch(const ops::W4A16LinearShape& s, const ops::W4A16LinearBuffers& b, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>((s.n + 127U) / 128U),
                    static_cast<unsigned int>(s.splits));
    w4a16_partial<Rows><<<grid, 128, 0, stream>>>(
        static_cast<std::uint32_t>(s.n), static_cast<std::uint32_t>(s.k), b.activations.data,
        b.weights.data, b.scales.data, b.workspace.data);
}

}  // namespace

cudaError_t launch_w4a16_linear(const ops::W4A16LinearShape& s, const ops::W4A16LinearBuffers& b,
                                cudaStream_t stream) noexcept {
    if (ops::validate_w4a16_linear(s, b) != ops::Status::Success) {
        return cudaErrorInvalidValue;
    }
    if (s.m == 0U || s.n == 0U) {
        return cudaSuccess;
    }
    constexpr auto max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (s.m > 4U || s.splits > 128U || s.n > max || s.k > max ||
        reinterpret_cast<std::uintptr_t>(b.activations.data) % 4U != 0U) {
        return cudaErrorInvalidValue;
    }
    switch (s.m) {
        case 1:
            launch<1>(s, b, stream);
            break;
        case 2:
            launch<2>(s, b, stream);
            break;
        case 3:
            launch<3>(s, b, stream);
            break;
        case 4:
            launch<4>(s, b, stream);
            break;
    }
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    const std::size_t count = s.m * s.n;
    w4a16_reduce<<<static_cast<unsigned int>((count + 255U) / 256U), 256, 0, stream>>>(
        count, static_cast<unsigned int>(s.splits), b.workspace.data, b.output.data);
    return cudaGetLastError();
}

cudaError_t check_w4a16_linear_image(int device_index) noexcept {
    return detail::check_kernel_images(device_index, w4a16_partial<1>, w4a16_partial<2>,
                                       w4a16_partial<3>, w4a16_partial<4>, w4a16_reduce);
}

cudaError_t launch_w4a16_dequantize(std::size_t n, std::size_t k,
                                    const ops::W4A16DequantizeBuffers& b,
                                    cudaStream_t stream) noexcept {
    if (ops::validate_w4a16_dequantize(n, k, b) != ops::Status::Success) {
        return cudaErrorInvalidValue;
    }
    if (n == 0U) {
        return cudaSuccess;
    }
    constexpr auto max = static_cast<std::size_t>(std::numeric_limits<unsigned int>::max());
    constexpr unsigned int kThreads = 256U;
    constexpr std::size_t kMaxRows = 65535U;  // grid.y limit, later rows loop
    if (n > max - 1023U || k / 8U > max - kMaxRows) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid4(static_cast<unsigned int>((n / 4U + kThreads - 1U) / kThreads),
                     static_cast<unsigned int>(std::min(k / 8U, kMaxRows)));
    const auto aligned = [](const void* pointer, std::uintptr_t bytes) {
        return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0U;
    };
    if (n % 4U == 0U && aligned(b.weights.data, 16U) && aligned(b.scales.data, 8U) &&
        aligned(b.dequantized.data, 8U)) {
        w4a16_dequantize<4><<<grid4, kThreads, 0, stream>>>(
            static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(k / 8U), b.weights.data,
            b.scales.data, b.dequantized.data);
    } else {
        const dim3 grid1(static_cast<unsigned int>((n + kThreads - 1U) / kThreads), grid4.y);
        w4a16_dequantize<1><<<grid1, kThreads, 0, stream>>>(
            static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(k / 8U), b.weights.data,
            b.scales.data, b.dequantized.data);
    }
    return cudaGetLastError();
}

cudaError_t check_w4a16_dequantize_image(int device_index) noexcept {
    return detail::check_kernel_images(device_index, w4a16_dequantize<1>, w4a16_dequantize<4>);
}

}  // namespace eclair::cuda
