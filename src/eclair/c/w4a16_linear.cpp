/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/c/w4a16_linear.h"

#include "eclair/cuda/w4a16_linear.hpp"

#include <initializer_list>
#include <limits>

extern "C" __attribute__((visibility("default"))) int eclair_w4a16_linear(
    const EclairW4A16LinearArgs* args, void* stream) {
    if (args == nullptr || args->abi != ECLAIR_W4A16_LINEAR_ABI) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto& a = *args;
    for (const std::uint64_t value :
         {a.activation_elements, a.weight_elements, a.scale_elements, a.output_elements,
          a.workspace_elements, a.m, a.n, a.k, a.splits}) {
        if (value > std::numeric_limits<std::size_t>::max()) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    const auto size = [](std::uint64_t v) { return static_cast<std::size_t>(v); };
    const eclair::ops::W4A16LinearBuffers b{
        {static_cast<const std::uint16_t*>(a.activations), size(a.activation_elements)},
        {a.weights, size(a.weight_elements)},
        {static_cast<const std::uint16_t*>(a.scales), size(a.scale_elements)},
        {static_cast<std::uint16_t*>(a.output), size(a.output_elements)},
        {a.workspace, size(a.workspace_elements)}};
    return static_cast<int>(eclair::cuda::launch_w4a16_linear(
        {size(a.m), size(a.n), size(a.k), size(a.splits)}, b, static_cast<cudaStream_t>(stream)));
}

extern "C" __attribute__((visibility("default"))) int eclair_w4a16_linear_image(int device_index) {
    return static_cast<int>(eclair::cuda::check_w4a16_linear_image(device_index));
}
