/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/c/gated_delta.h"

#include "eclair/cuda/gated_delta.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace {

[[nodiscard]] bool fits(std::uint64_t value) noexcept {
    return value <= std::numeric_limits<std::size_t>::max();
}

[[nodiscard]] std::size_t size(std::uint64_t value) noexcept {
    return static_cast<std::size_t>(value);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int eclair_gated_delta_states(
    const EclairGatedDeltaStatesArgs* args, void* stream) {
    if (args == nullptr || args->abi != ECLAIR_GATED_DELTA_ABI || args->initial_format > 2U) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const EclairGatedDeltaStatesArgs& a = *args;
    for (const std::uint64_t value :
         {a.keys_elements, a.w_elements, a.u_elements, a.gates_elements, a.initial_elements,
          a.states_elements, a.new_values_elements, a.final_states_elements, a.tokens, a.sequences,
          a.chunks, a.heads, a.key_heads, a.key_size, a.value_size}) {
        if (!fits(value)) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    const eclair::ops::GatedDeltaStatesShape shape{
        size(a.tokens),    size(a.sequences), size(a.chunks),    size(a.heads),
        size(a.key_heads), size(a.key_size),  size(a.value_size)};
    eclair::ops::GatedDeltaStatesBuffers buffers{
        {static_cast<const std::uint16_t*>(a.keys), size(a.keys_elements)},
        {static_cast<const std::uint16_t*>(a.w), size(a.w_elements)},
        {static_cast<const std::uint16_t*>(a.u), size(a.u_elements)},
        {a.gates, size(a.gates_elements)},
        {},
        {},
        {a.sequence_starts, size(a.sequences) + 1U},
        {a.chunk_starts, size(a.sequences)},
        {static_cast<std::uint16_t*>(a.states), size(a.states_elements)},
        {static_cast<std::uint16_t*>(a.new_values), size(a.new_values_elements)},
        {a.final_states, size(a.final_states_elements)}};
    if (a.initial_format == 1U) {
        buffers.initial_binary16 = {static_cast<const std::uint16_t*>(a.initial),
                                    size(a.initial_elements)};
    } else if (a.initial_format == 2U) {
        buffers.initial_float32 = {static_cast<const float*>(a.initial), size(a.initial_elements)};
    }
    return static_cast<int>(
        eclair::cuda::launch_gated_delta_states(shape, buffers, static_cast<cudaStream_t>(stream)));
}

extern "C" __attribute__((visibility("default"))) int eclair_gated_delta_image(int device_index) {
    return static_cast<int>(eclair::cuda::check_gated_delta_image(device_index));
}
