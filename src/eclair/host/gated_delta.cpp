/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/gated_delta.hpp"

#include "eclair/formats/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace eclair::host {
namespace {

using formats::decode_fp16;
using formats::encode_fp16;

constexpr std::size_t kChunk = ops::kGatedDeltaChunk;

[[nodiscard]] double half(std::uint16_t bits) noexcept {
    return static_cast<double>(decode_fp16(bits));
}

}  // namespace

ops::Status gated_delta_states_reference(const ops::GatedDeltaStatesShape& shape,
                                         const ops::GatedDeltaStatesBuffers& buffers) noexcept {
    const ops::Status status = ops::validate_gated_delta_states(shape, buffers);
    if (status != ops::Status::Success || shape.tokens == 0U) {
        return status;
    }
    const std::int32_t* starts = buffers.sequence_starts.data;
    for (std::size_t n = 0; n < shape.sequences; ++n) {
        if (starts[n] < 0 || starts[n + 1U] < starts[n] ||
            static_cast<std::size_t>(starts[n + 1U]) > shape.tokens ||
            buffers.chunk_starts.data[n] < 0) {
            return ops::Status::InvalidArgument;
        }
        const auto length = static_cast<std::size_t>(starts[n + 1U] - starts[n]);
        const std::size_t chunks = (length + kChunk - 1U) / kChunk;
        if (static_cast<std::size_t>(buffers.chunk_starts.data[n]) + chunks > shape.chunks) {
            return ops::Status::InvalidArgument;
        }
    }
    const std::size_t heads = shape.heads;
    const std::size_t key_size = shape.key_size;
    const std::size_t value_size = shape.value_size;
    const std::size_t group = heads / shape.key_heads;
    const std::size_t matrix = value_size * key_size;
    for (std::size_t n = 0; n < shape.sequences; ++n) {
        const auto first = static_cast<std::size_t>(starts[n]);
        const auto last = static_cast<std::size_t>(starts[n + 1U]);
        for (std::size_t h = 0; h < heads; ++h) {
            float* state = buffers.final_states.data + (n * heads + h) * matrix;
            for (std::size_t e = 0; e < matrix; ++e) {
                const std::size_t index = (n * heads + h) * matrix + e;
                state[e] = buffers.initial_binary16.data != nullptr
                               ? decode_fp16(buffers.initial_binary16.data[index])
                           : buffers.initial_float32.data != nullptr
                               ? buffers.initial_float32.data[index]
                               : 0.0F;
            }
            std::size_t chunk = static_cast<std::size_t>(buffers.chunk_starts.data[n]);
            for (std::size_t a = first; a < last; a += kChunk, ++chunk) {
                const std::size_t b = std::min(last, a + kChunk);
                const double g_last = static_cast<double>(buffers.gates.data[(b - 1U) * heads + h]);
                std::uint16_t* saved = buffers.states.data + (chunk * heads + h) * matrix;
                for (std::size_t e = 0; e < matrix; ++e) {
                    saved[e] = encode_fp16(state[e]);
                }
                for (std::size_t v = 0; v < value_size; ++v) {
                    double weighted[kChunk];
                    for (std::size_t t = a; t < b; ++t) {
                        const std::uint16_t* w = buffers.w.data + (t * heads + h) * key_size;
                        double dot = 0.0;
                        for (std::size_t k = 0; k < key_size; ++k) {
                            dot += half(w[k]) * half(saved[v * key_size + k]);
                        }
                        const double d =
                            half(buffers.u.data[(t * heads + h) * value_size + v]) - dot;
                        buffers.new_values.data[(t * heads + h) * value_size + v] =
                            encode_fp16(static_cast<float>(d));
                        const double gate = static_cast<double>(buffers.gates.data[t * heads + h]);
                        weighted[t - a] =
                            half(encode_fp16(static_cast<float>(d * std::exp(g_last - gate))));
                    }
                    const double decay = std::exp(g_last);
                    for (std::size_t k = 0; k < key_size; ++k) {
                        double sum = static_cast<double>(state[v * key_size + k]) * decay;
                        for (std::size_t t = a; t < b; ++t) {
                            const std::size_t key_head = h / group;
                            sum += weighted[t - a] *
                                   half(buffers.keys
                                            .data[(t * shape.key_heads + key_head) * key_size + k]);
                        }
                        state[v * key_size + k] = static_cast<float>(sum);
                    }
                }
            }
        }
    }
    return ops::Status::Success;
}

}  // namespace eclair::host
