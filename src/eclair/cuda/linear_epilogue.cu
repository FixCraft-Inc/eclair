/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/linear_epilogue.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace eclair::cuda {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kMaxRowBlocks = 65535U;

// The __fmul_rn and __fadd_rn intrinsics are never contracted into an FMA,
// which keeps the two FP32 roundings the contract requires.
[[nodiscard]] __device__ std::uint16_t finish(float accumulator, float scale, const float* bias,
                                              std::size_t column) {
    float value = __fmul_rn(accumulator, scale);
    if (bias != nullptr) {
        value = __fadd_rn(value, bias[column]);
    }
    return __half_as_ushort(__float2half_rn(value));
}

// Grid y strides over rows, so each column index is a bias index without a
// division. The vector path needs N % 4 == 0, 16-byte accumulators and 8-byte
// output, which then holds for every row.
__global__ void linear_epilogue_kernel(std::size_t rows, std::size_t columns, bool vector,
                                       ops::LinearEpilogueBuffers buffers) {
    const float scale = buffers.scale.data[0];
    const float* bias = buffers.bias.size == 0U ? nullptr : buffers.bias.data;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const float* input = buffers.accumulators.data + row * columns;
        std::uint16_t* output = buffers.output.data + row * columns;
        if (vector) {
            for (std::size_t column = first * 4U; column < columns; column += stride * 4U) {
                const float4 values = *reinterpret_cast<const float4*>(input + column);
                const uint2 packed{
                    static_cast<std::uint32_t>(finish(values.x, scale, bias, column)) |
                        static_cast<std::uint32_t>(finish(values.y, scale, bias, column + 1U))
                            << 16U,
                    static_cast<std::uint32_t>(finish(values.z, scale, bias, column + 2U)) |
                        static_cast<std::uint32_t>(finish(values.w, scale, bias, column + 3U))
                            << 16U};
                *reinterpret_cast<uint2*>(output + column) = packed;
            }
        } else {
            for (std::size_t column = first; column < columns; column += stride) {
                output[column] = finish(input[column], scale, bias, column);
            }
        }
    }
}

}  // namespace

cudaError_t launch_linear_epilogue(const ops::LinearEpilogueShape& shape,
                                   const ops::LinearEpilogueBuffers& buffers,
                                   cudaStream_t stream) noexcept {
    if (ops::validate_linear_epilogue(shape, buffers) != ops::Status::Success) {
        return cudaErrorInvalidValue;
    }
    if (shape.m == 0U || shape.n == 0U) {
        return cudaSuccess;
    }
    const bool vector = shape.n % 4U == 0U &&
                        reinterpret_cast<std::uintptr_t>(buffers.accumulators.data) % 16U == 0U &&
                        reinterpret_cast<std::uintptr_t>(buffers.output.data) % 8U == 0U;
    const std::size_t per_block = kThreads * (vector ? 4U : 1U);
    // At most 65535 column blocks keeps the x dimension well inside its limit.
    const std::size_t column_blocks = std::min(1U + (shape.n - 1U) / per_block, kMaxRowBlocks);
    const dim3 grid(static_cast<unsigned int>(column_blocks),
                    static_cast<unsigned int>(std::min(shape.m, kMaxRowBlocks)));
    linear_epilogue_kernel<<<grid, kThreads, 0, stream>>>(shape.m, shape.n, vector, buffers);
    return cudaGetLastError();
}

}  // namespace eclair::cuda
