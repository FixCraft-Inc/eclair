/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/buffer.hpp"

#include <cstddef>
#include <cstdint>

// Experimental state pass of the chunked gated delta rule with one scalar gate
// per head and token, the step of Gated DeltaNet prefill that carries the
// recurrent state from chunk to chunk. Its inputs come from the WY form of
// the preceding steps. See docs/GATED_DELTA.md.
namespace eclair::ops {

// Tokens per chunk.
inline constexpr std::size_t kGatedDeltaChunk = 64U;

// Sequences are packed along the token axis: sequence n holds tokens
// sequence_starts[n] .. sequence_starts[n + 1] - 1 and its chunks have
// global indices chunk_starts[n] onward. Value head h reads key head
// h / (heads / key_heads). For a chunk of tokens a .. b - 1 of sequence n and
// head h, with the FP32 state S of V x K values (the initial state, or zero):
//
//   states[chunk, h]  = binary16(S)
//   d_t               = u_t - w_t * binary16(S)^T      (FP32 sums)
//   new_values[t, h]  = binary16(d_t)
//   e_t               = binary16(d_t * exp(g_{b-1} - g_t))
//   S                 = S * exp(g_{b-1}) + sum_t e_t^T k_t   (FP32 sums)
//
// and final_states[n, h] = S after the sequence's last chunk. Gates g are FP32
// and cumulative within each chunk. k, w and u are binary16. All arrays are
// contiguous: keys [tokens, key_heads, K], w [tokens, heads, K], u and
// new_values [tokens, heads, V], gates [tokens, heads], states [chunks, heads,
// V, K], initial and final states [sequences, heads, V, K].
struct GatedDeltaStatesShape {
    std::size_t tokens = 0;
    std::size_t sequences = 0;
    std::size_t chunks = 0;  // Sum over sequences of ceil(length / kGatedDeltaChunk).
    std::size_t heads = 0;
    std::size_t key_heads = 0;
    std::size_t key_size = 0;
    std::size_t value_size = 0;
};

struct GatedDeltaStatesBuffers {
    BufferView<const std::uint16_t> keys;
    BufferView<const std::uint16_t> w;
    BufferView<const std::uint16_t> u;
    BufferView<const float> gates;
    // At most one initial state, binary16 or FP32. Neither means zero.
    BufferView<const std::uint16_t> initial_binary16;
    BufferView<const float> initial_float32;
    // Device backends cannot read these during validation. They skip a
    // sequence whose bounds or chunks fall outside the shape.
    BufferView<const std::int32_t> sequence_starts;  // sequences + 1 entries.
    BufferView<const std::int32_t> chunk_starts;     // sequences entries.
    BufferView<std::uint16_t> states;
    BufferView<std::uint16_t> new_values;
    BufferView<float> final_states;
};

// Rejects zero sizes, heads that are not a multiple of key heads, two initial
// states, short or misaligned buffers, outputs that overlap each other or an
// input, and size overflow. Zero tokens is a no-op that needs no buffers.
// Reads no values.
[[nodiscard]] Status validate_gated_delta_states(const GatedDeltaStatesShape& shape,
                                                 const GatedDeltaStatesBuffers& buffers) noexcept;

}  // namespace eclair::ops
