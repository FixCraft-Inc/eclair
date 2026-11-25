/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/c/gated_delta_output.h"

#include "eclair/cuda/gated_delta_output.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

extern "C" __attribute__((visibility("default"))) int eclair_gated_delta_output(
    const EclairGatedDeltaOutputArgs* args, void* stream) {
    if (args == nullptr || args->abi != ECLAIR_GATED_DELTA_OUTPUT_ABI) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto& a = *args;
    for (const std::uint64_t value :
         {a.query_elements, a.keys_elements, a.values_elements, a.states_elements, a.gates_elements,
          a.sequence_starts_elements, a.chunk_indices_elements, a.output_elements, a.tokens,
          a.sequences, a.chunks, a.heads, a.key_heads, a.key_size, a.value_size}) {
        if (value > std::numeric_limits<std::size_t>::max()) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    const auto size = [](std::uint64_t value) { return static_cast<std::size_t>(value); };
    const eclair::ops::GatedDeltaOutputShape s{
        size(a.tokens),    size(a.sequences), size(a.chunks),     size(a.heads),
        size(a.key_heads), size(a.key_size),  size(a.value_size), a.scale};
    const eclair::ops::GatedDeltaOutputBuffers b{
        {static_cast<const std::uint16_t*>(a.query), size(a.query_elements)},
        {static_cast<const std::uint16_t*>(a.keys), size(a.keys_elements)},
        {static_cast<const std::uint16_t*>(a.values), size(a.values_elements)},
        {static_cast<const std::uint16_t*>(a.states), size(a.states_elements)},
        {a.gates, size(a.gates_elements)},
        {a.sequence_starts, size(a.sequence_starts_elements)},
        {a.chunk_indices, size(a.chunk_indices_elements)},
        {static_cast<std::uint16_t*>(a.output), size(a.output_elements)}};
    return static_cast<int>(
        eclair::cuda::launch_gated_delta_output(s, b, static_cast<cudaStream_t>(stream)));
}

extern "C" __attribute__((visibility("default"))) int eclair_gated_delta_output_image(int device) {
    return static_cast<int>(eclair::cuda::check_gated_delta_output_image(device));
}
