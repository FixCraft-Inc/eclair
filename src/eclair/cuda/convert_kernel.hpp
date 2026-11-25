/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Internal to eclair_cuda: the elementwise kernel and launcher that every
// format conversion uses. Include only from CUDA units. Not a public header.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace eclair::cuda::detail {

// Four consecutive elements, aligned so one access moves all of them.
template <typename T>
struct alignas(4 * sizeof(T)) Quad {
    T value[4];
};

// output[i] = convert(input[i]) for every i < count. With `vector` set, each
// thread moves groups of four and the first threads finish the up to three
// elements after the last whole group. The launcher sets `vector` only when
// both base pointers are aligned for a Quad.
template <typename In, typename Out, typename Convert>
__global__ void convert_kernel(const In* input, Out* output, std::size_t count, bool vector,
                               Convert convert) {
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    if (!vector) {
        for (std::size_t index = first; index < count; index += stride) {
            output[index] = convert(input[index]);
        }
        return;
    }
    const std::size_t groups = count / 4U;
    const auto* source = reinterpret_cast<const Quad<In>*>(input);
    auto* target = reinterpret_cast<Quad<Out>*>(output);
    for (std::size_t group = first; group < groups; group += stride) {
        const Quad<In> values = source[group];
        Quad<Out> converted;
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            converted.value[lane] = convert(values.value[lane]);
        }
        target[group] = converted;
    }
    const std::size_t tail = groups * 4U + first;
    if (tail < count) {
        output[tail] = convert(input[tail]);
    }
}

constexpr unsigned int kConvertThreads = 256U;
// Enough resident threads to keep any current GPU's memory system busy. A
// grid-stride loop covers larger counts.
constexpr std::size_t kMaxConvertBlocks = 4096U;

[[nodiscard]] inline bool aligned(const void* pointer, std::size_t bytes) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0U;
}

// The checks every conversion launcher makes, then the launch. Rejects null
// pointers, byte counts that overflow or wrap the address space, and any
// overlap between input and output. Allocates and synchronizes nothing.
template <typename In, typename Out, typename Convert>
[[nodiscard]] cudaError_t launch_conversion(const In* input, Out* output, std::size_t count,
                                            Convert convert, cudaStream_t stream) noexcept {
    if (count == 0U) {
        return cudaSuccess;
    }
    constexpr std::size_t kLargest = sizeof(In) > sizeof(Out) ? sizeof(In) : sizeof(Out);
    if (input == nullptr || output == nullptr ||
        count > std::numeric_limits<std::size_t>::max() / kLargest) {
        return cudaErrorInvalidValue;
    }
    const auto input_begin = reinterpret_cast<std::uintptr_t>(input);
    const auto output_begin = reinterpret_cast<std::uintptr_t>(output);
    const std::size_t input_bytes = count * sizeof(In);
    const std::size_t output_bytes = count * sizeof(Out);
    constexpr std::uintptr_t kTop = std::numeric_limits<std::uintptr_t>::max();
    if (input_begin > kTop - input_bytes || output_begin > kTop - output_bytes ||
        (input_begin < output_begin + output_bytes && output_begin < input_begin + input_bytes)) {
        return cudaErrorInvalidValue;
    }
    const bool vector = aligned(input, alignof(Quad<In>)) && aligned(output, alignof(Quad<Out>));
    const std::size_t items = vector ? std::max<std::size_t>(count / 4U, 1U) : count;
    const std::size_t blocks =
        std::min((items + kConvertThreads - 1U) / kConvertThreads, kMaxConvertBlocks);
    convert_kernel<In, Out, Convert>
        <<<static_cast<unsigned int>(blocks), kConvertThreads, 0, stream>>>(input, output, count,
                                                                            vector, convert);
    return cudaGetLastError();
}

}  // namespace eclair::cuda::detail
