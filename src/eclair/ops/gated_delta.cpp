/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/ops/gated_delta.hpp"

#include "buffer_checks.hpp"

namespace eclair::ops {
namespace {

using detail::AddressRange;
using detail::multiply;
using detail::overlaps;
using detail::range;

// Product of three sizes, or false on overflow.
[[nodiscard]] bool volume(std::size_t a, std::size_t b, std::size_t c,
                          std::size_t& result) noexcept {
    std::size_t partial = 0;
    return multiply(a, b, partial) && multiply(partial, c, result);
}

}  // namespace

Status validate_gated_delta_states(const GatedDeltaStatesShape& shape,
                                   const GatedDeltaStatesBuffers& buffers) noexcept {
    if (shape.tokens == 0U) {
        return Status::Success;
    }
    if (shape.sequences == 0U || shape.chunks == 0U || shape.heads == 0U || shape.key_heads == 0U ||
        shape.heads % shape.key_heads != 0U || shape.key_size == 0U || shape.value_size == 0U ||
        (buffers.initial_binary16.data != nullptr && buffers.initial_float32.data != nullptr)) {
        return Status::InvalidArgument;
    }
    std::size_t keys = 0;
    std::size_t w = 0;
    std::size_t values = 0;
    std::size_t gates = 0;
    std::size_t matrix = 0;
    std::size_t states = 0;
    std::size_t sequence_states = 0;
    if (!volume(shape.tokens, shape.key_heads, shape.key_size, keys) ||
        !volume(shape.tokens, shape.heads, shape.key_size, w) ||
        !volume(shape.tokens, shape.heads, shape.value_size, values) ||
        !multiply(shape.tokens, shape.heads, gates) ||
        !multiply(shape.value_size, shape.key_size, matrix) ||
        !volume(shape.chunks, shape.heads, matrix, states) ||
        !volume(shape.sequences, shape.heads, matrix, sequence_states) ||
        shape.sequences + 1U == 0U) {
        return Status::SizeOverflow;
    }
    AddressRange inputs[8];
    AddressRange outputs[3];
    const Status checks[] = {
        range(buffers.keys, keys, inputs[0]),
        range(buffers.w, w, inputs[1]),
        range(buffers.u, values, inputs[2]),
        range(buffers.gates, gates, inputs[3]),
        range(buffers.initial_binary16,
              buffers.initial_binary16.data != nullptr ? sequence_states : 0U, inputs[4]),
        range(buffers.initial_float32,
              buffers.initial_float32.data != nullptr ? sequence_states : 0U, inputs[5]),
        range(buffers.sequence_starts, shape.sequences + 1U, inputs[6]),
        range(buffers.chunk_starts, shape.sequences, inputs[7]),
        range(buffers.states, states, outputs[0]),
        range(buffers.new_values, values, outputs[1]),
        range(buffers.final_states, sequence_states, outputs[2]),
    };
    for (const Status check : checks) {
        if (check != Status::Success) {
            return check;
        }
    }
    for (std::size_t o = 0; o < 3U; ++o) {
        for (const AddressRange& input : inputs) {
            if (overlaps(outputs[o], input)) {
                return Status::InvalidArgument;
            }
        }
        for (std::size_t other = o + 1U; other < 3U; ++other) {
            if (overlaps(outputs[o], outputs[other])) {
                return Status::InvalidArgument;
            }
        }
    }
    return Status::Success;
}

}  // namespace eclair::ops
