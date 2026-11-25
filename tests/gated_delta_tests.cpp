/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/formats/fp16.hpp"
#include "eclair/host/gated_delta.hpp"
#include "eclair/ops/gated_delta.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using eclair::formats::decode_fp16;
using eclair::formats::encode_fp16;
using eclair::host::gated_delta_states_reference;
using eclair::ops::GatedDeltaStatesBuffers;
using eclair::ops::GatedDeltaStatesShape;
using eclair::ops::kGatedDeltaChunk;
using eclair::ops::Status;
using eclair::ops::validate_gated_delta_states;

// Host storage for one problem, zero-filled, with sequences of the given lengths.
struct Problem {
    GatedDeltaStatesShape shape;
    std::vector<std::uint16_t> keys, w, u, states, new_values, initial;
    std::vector<float> gates, final_states;
    std::vector<std::int32_t> starts, chunk_starts;

    Problem(std::vector<std::size_t> lengths, std::size_t heads, std::size_t key_heads,
            std::size_t key_size, std::size_t value_size) {
        std::size_t tokens = 0;
        std::size_t chunks = 0;
        starts.push_back(0);
        for (const std::size_t length : lengths) {
            chunk_starts.push_back(static_cast<std::int32_t>(chunks));
            tokens += length;
            chunks += (length + kGatedDeltaChunk - 1U) / kGatedDeltaChunk;
            starts.push_back(static_cast<std::int32_t>(tokens));
        }
        shape = {tokens, lengths.size(), chunks, heads, key_heads, key_size, value_size};
        keys.assign(tokens * key_heads * key_size, 0U);
        w.assign(tokens * heads * key_size, 0U);
        u.assign(tokens * heads * value_size, 0U);
        gates.assign(tokens * heads, 0.0F);
        states.assign(chunks * heads * value_size * key_size, 0U);
        new_values.assign(tokens * heads * value_size, 0U);
        final_states.assign(lengths.size() * heads * value_size * key_size, 0.0F);
    }

    [[nodiscard]] GatedDeltaStatesBuffers buffers() {
        GatedDeltaStatesBuffers result{{keys.data(), keys.size()},
                                       {w.data(), w.size()},
                                       {u.data(), u.size()},
                                       {gates.data(), gates.size()},
                                       {},
                                       {},
                                       {starts.data(), starts.size()},
                                       {chunk_starts.data(), chunk_starts.size()},
                                       {states.data(), states.size()},
                                       {new_values.data(), new_values.size()},
                                       {final_states.data(), final_states.size()}};
        if (!initial.empty()) {
            result.initial_binary16 = {initial.data(), initial.size()};
        }
        return result;
    }

    [[nodiscard]] Status run() { return gated_delta_states_reference(shape, buffers()); }

    [[nodiscard]] float final_at(std::size_t n, std::size_t h, std::size_t v, std::size_t k) const {
        return final_states[((n * shape.heads + h) * shape.value_size + v) * shape.key_size + k];
    }
};

[[nodiscard]] std::uint16_t f16(float value) {
    return encode_fp16(value);
}

void test_validation() {
    Problem p({3}, 2, 1, 4, 4);
    ECLAIR_CHECK(validate_gated_delta_states(p.shape, p.buffers()) == Status::Success);

    auto zero = p.shape;
    zero.tokens = 0;
    ECLAIR_CHECK(validate_gated_delta_states(zero, {}) == Status::Success);

    auto heads = p.shape;
    heads.key_heads = 3;
    ECLAIR_CHECK(validate_gated_delta_states(heads, p.buffers()) == Status::InvalidArgument);

    GatedDeltaStatesBuffers both = p.buffers();
    std::vector<std::uint16_t> half_state(2 * 4 * 4);
    std::vector<float> float_state(2 * 4 * 4);
    both.initial_binary16 = {half_state.data(), half_state.size()};
    both.initial_float32 = {float_state.data(), float_state.size()};
    ECLAIR_CHECK(validate_gated_delta_states(p.shape, both) == Status::InvalidArgument);

    GatedDeltaStatesBuffers short_u = p.buffers();
    short_u.u.size -= 1U;
    ECLAIR_CHECK(validate_gated_delta_states(p.shape, short_u) == Status::BufferTooSmall);

    GatedDeltaStatesBuffers aliased = p.buffers();
    aliased.new_values = {const_cast<std::uint16_t*>(p.u.data()), p.u.size()};
    ECLAIR_CHECK(validate_gated_delta_states(p.shape, aliased) == Status::InvalidArgument);

    auto huge = p.shape;
    huge.chunks = std::numeric_limits<std::size_t>::max();
    ECLAIR_CHECK(validate_gated_delta_states(huge, p.buffers()) == Status::SizeOverflow);
}

// One token from a zero state: d = u, e = u, and the final state is u^T k.
void test_single_token_outer_product() {
    Problem p({1}, 1, 1, 2, 2);
    p.keys = {f16(1.0F), f16(-2.0F)};
    p.u = {f16(3.0F), f16(0.5F)};
    p.gates = {-0.7F};
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(decode_fp16(p.new_values[0]) == 3.0F && decode_fp16(p.new_values[1]) == 0.5F);
    ECLAIR_CHECK(decode_fp16(p.states[0]) == 0.0F);  // The state before the chunk.
    ECLAIR_CHECK(p.final_at(0, 0, 0, 0) == 3.0F && p.final_at(0, 0, 0, 1) == -6.0F);
    ECLAIR_CHECK(p.final_at(0, 0, 1, 0) == 0.5F && p.final_at(0, 0, 1, 1) == -1.0F);
}

// Without new input the gate decays the state by exp(g_last), and d removes
// the state's projection onto w: d = u - w S^T.
void test_decay_and_projection() {
    Problem p({1}, 1, 1, 2, 2);
    p.initial = {f16(1.0F), f16(2.0F), f16(3.0F), f16(4.0F)};
    p.w = {f16(1.0F), f16(0.0F)};
    p.u = {f16(10.0F), f16(10.0F)};
    p.gates = {std::log(0.5F)};
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(decode_fp16(p.new_values[0]) == 9.0F);  // 10 - 1 * S[0][0].
    ECLAIR_CHECK(decode_fp16(p.new_values[1]) == 7.0F);  // 10 - 1 * S[1][0].
    // Keys are zero, so only the decay acts on the state.
    ECLAIR_CHECK(std::fabs(p.final_at(0, 0, 0, 1) - 1.0F) < 1e-6F);
    ECLAIR_CHECK(std::fabs(p.final_at(0, 0, 1, 1) - 2.0F) < 1e-6F);
}

// A second chunk starts from the first chunk's state, and the second sequence
// starts fresh. Value heads 0 and 1 share key head 0.
void test_chunks_and_sequences() {
    Problem p({kGatedDeltaChunk + 1U, 2}, 2, 1, 2, 2);
    p.keys[0] = f16(1.0F);  // Token 0, key element 0.
    for (std::size_t h = 0; h < 2; ++h) {
        p.u[h * 2] = f16(h == 0 ? 2.0F : 4.0F);  // Token 0, value element 0.
    }
    ECLAIR_CHECK(p.run() == Status::Success);
    // Chunk 0 sets S[0][0] = u for each head, and chunk 1 begins with it.
    const std::size_t matrix = 4;
    ECLAIR_CHECK(decode_fp16(p.states[(1U * 2U + 0U) * matrix]) == 2.0F);
    ECLAIR_CHECK(decode_fp16(p.states[(1U * 2U + 1U) * matrix]) == 4.0F);
    ECLAIR_CHECK(p.final_at(0, 1, 0, 0) == 4.0F);
    ECLAIR_CHECK(p.final_at(1, 0, 0, 0) == 0.0F);  // The second sequence had no input.
}

void test_rejected_bounds() {
    Problem p({4, 4}, 1, 1, 2, 2);
    p.starts[1] = 9;  // Past the 8 tokens.
    ECLAIR_CHECK(p.run() == Status::InvalidArgument);
    Problem q({4, 4}, 1, 1, 2, 2);
    q.chunk_starts[1] = 2;  // Needs a third chunk.
    ECLAIR_CHECK(q.run() == Status::InvalidArgument);
}

}  // namespace

int main() {
    test_validation();
    test_single_token_outer_product();
    test_decay_and_projection();
    test_chunks_and_sequences();
    test_rejected_bounds();
    return eclair::test::finish();
}
