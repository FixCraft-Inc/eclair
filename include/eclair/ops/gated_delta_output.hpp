/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/buffer.hpp"

#include <cstddef>
#include <cstdint>

namespace eclair::ops {

// Output stage of the scalar-gated, 64-token chunk rule. Sequences are packed
// along the token axis. A chunk's index pair is (sequence, local chunk index),
// in sequence order, including every nonempty chunk exactly once. Its state
// is the binary16 V x K snapshot made by the state pass. For local token i,
// value head h and key head h / (heads / key_heads):
//
//   a_ij = binary16(sum_d q_i[d] * k_j[d] * exp(g_i - g_j)), j <= i
//   o_i  = binary16(scale * (exp(g_i) * q_i * state^T + sum_j a_ij * v_j))
//
// All sums accumulate in FP32 or better, in unspecified order; gates are
// cumulative FP32 log decays within each chunk. Scale applies after the state
// and value contributions are summed. Queries/keys, new values and
// states are binary16. A backend may multiply a masked value inside a chunk
// by zero, so a NaN/infinity there may also reach earlier rows. Slots beyond
// the sequence's end are never read. See docs/GATED_DELTA.md.
struct GatedDeltaOutputShape {
    std::size_t tokens = 0;
    std::size_t sequences = 0;
    std::size_t chunks = 0;
    std::size_t heads = 0;
    std::size_t key_heads = 0;
    std::size_t key_size = 0;
    std::size_t value_size = 0;
    float scale = 1.0F;
};

// Contiguous arrays, with the same state layout as GatedDeltaStatesBuffers.
// Device backends skip an index pair with an out-of-range sequence/chunk or
// invalid sequence bounds. Valid metadata must follow the ordering above;
// bounds checks alone do not establish uniqueness or complete coverage.
struct GatedDeltaOutputBuffers {
    BufferView<const std::uint16_t> query;           // [tokens, key_heads, K]
    BufferView<const std::uint16_t> keys;            // [tokens, key_heads, K]
    BufferView<const std::uint16_t> values;          // [tokens, heads, V], new_values
    BufferView<const std::uint16_t> states;          // [chunks, heads, V, K]
    BufferView<const float> gates;                   // [tokens, heads]
    BufferView<const std::int32_t> sequence_starts;  // [sequences + 1]
    BufferView<const std::int32_t> chunk_indices;    // [chunks, 2]
    BufferView<std::uint16_t> output;                // [tokens, heads, V]
};

// Checks dimensions, finite scale, buffer sizes/alignment, output/input
// overlap and size overflow. Reads no values. Zero tokens needs no buffers.
[[nodiscard]] Status validate_gated_delta_output(const GatedDeltaOutputShape& shape,
                                                 const GatedDeltaOutputBuffers& buffers) noexcept;

}  // namespace eclair::ops
