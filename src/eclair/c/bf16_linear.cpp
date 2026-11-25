/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/c/bf16_linear.h"

#include "eclair/cuda/bf16_linear.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

extern "C" __attribute__((visibility("default"))) int eclair_bf16_linear(
    const EclairBf16LinearArgs* args, void* stream) {
    if (args == nullptr || args->abi != ECLAIR_BF16_LINEAR_ABI) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto& a = *args;
    for (const std::uint64_t value :
         {a.activation_elements, a.activation_shift_elements, a.weight_elements,
          a.weight_shift_elements, a.bias_elements, a.output_elements, a.m, a.n, a.k}) {
        if (value > std::numeric_limits<std::size_t>::max()) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    const auto size = [](std::uint64_t value) { return static_cast<std::size_t>(value); };
    const eclair::ops::Bf16LinearBuffers b{
        {static_cast<const std::uint16_t*>(a.activations), size(a.activation_elements)},
        {a.activation_shifts, size(a.activation_shift_elements)},
        {static_cast<const std::uint16_t*>(a.weights), size(a.weight_elements)},
        {a.weight_shifts, size(a.weight_shift_elements)},
        {a.bias, size(a.bias_elements)},
        {a.output, size(a.output_elements)}};
    return static_cast<int>(eclair::cuda::launch_bf16_linear({size(a.m), size(a.n), size(a.k)}, b,
                                                             static_cast<cudaStream_t>(stream)));
}

extern "C" __attribute__((visibility("default"))) int eclair_bf16_linear_image(int device) {
    return static_cast<int>(eclair::cuda::check_bf16_linear_image(device));
}
