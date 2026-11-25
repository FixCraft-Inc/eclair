/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/ops/gated_delta_output.hpp"

#include "buffer_checks.hpp"

#include <cmath>

namespace eclair::ops {

Status validate_gated_delta_output(const GatedDeltaOutputShape& s,
                                   const GatedDeltaOutputBuffers& b) noexcept {
    if (s.tokens == 0U) {
        return Status::Success;
    }
    if (s.sequences == 0U || s.chunks == 0U || s.heads == 0U || s.key_heads == 0U ||
        s.heads % s.key_heads != 0U || s.key_size == 0U || s.value_size == 0U ||
        !std::isfinite(s.scale)) {
        return Status::InvalidArgument;
    }
    std::size_t query = 0, values = 0, gates = 0, states = 0, indices = 0;
    using detail::multiply;
    if (!multiply(s.tokens, s.key_heads, query) || !multiply(query, s.key_size, query) ||
        !multiply(s.tokens, s.heads, gates) || !multiply(gates, s.value_size, values) ||
        !multiply(s.chunks, s.heads, states) || !multiply(states, s.value_size, states) ||
        !multiply(states, s.key_size, states) || !multiply(s.chunks, 2U, indices) ||
        s.sequences + 1U == 0U) {
        return Status::SizeOverflow;
    }
    detail::AddressRange inputs[7], output;
    const Status checks[] = {
        detail::range(b.query, query, inputs[0]),
        detail::range(b.keys, query, inputs[1]),
        detail::range(b.values, values, inputs[2]),
        detail::range(b.states, states, inputs[3]),
        detail::range(b.gates, gates, inputs[4]),
        detail::range(b.sequence_starts, s.sequences + 1U, inputs[5]),
        detail::range(b.chunk_indices, indices, inputs[6]),
        detail::range(b.output, values, output),
    };
    for (const Status status : checks) {
        if (status != Status::Success) {
            return status;
        }
    }
    for (const auto& input : inputs) {
        if (detail::overlaps(output, input)) {
            return Status::InvalidArgument;
        }
    }
    return Status::Success;
}

}  // namespace eclair::ops
