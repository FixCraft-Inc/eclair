/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/gated_delta_output.hpp"

#include "eclair/formats/fp16.hpp"
#include "eclair/ops/gated_delta.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace eclair::host {
namespace {

constexpr std::size_t kChunk = ops::kGatedDeltaChunk;

[[nodiscard]] double half(std::uint16_t bits) noexcept {
    return static_cast<double>(formats::decode_fp16(bits));
}

[[nodiscard]] bool metadata_valid(const ops::GatedDeltaOutputShape& s,
                                  const ops::GatedDeltaOutputBuffers& b) noexcept {
    if (b.sequence_starts.data[0] != 0) {
        return false;
    }
    std::size_t chunk = 0;
    for (std::size_t n = 0; n < s.sequences; ++n) {
        const auto first = b.sequence_starts.data[n];
        const auto last = b.sequence_starts.data[n + 1U];
        if (first < 0 || last < first || static_cast<std::size_t>(last) > s.tokens) {
            return false;
        }
        const auto count = (static_cast<std::size_t>(last - first) + kChunk - 1U) / kChunk;
        if (count > s.chunks - chunk) {
            return false;
        }
        for (std::size_t i = 0; i < count; ++i, ++chunk) {
            if (b.chunk_indices.data[2U * chunk] < 0 ||
                static_cast<std::size_t>(b.chunk_indices.data[2U * chunk]) != n ||
                b.chunk_indices.data[2U * chunk + 1U] < 0 ||
                static_cast<std::size_t>(b.chunk_indices.data[2U * chunk + 1U]) != i) {
                return false;
            }
        }
    }
    return chunk == s.chunks &&
           static_cast<std::size_t>(b.sequence_starts.data[s.sequences]) == s.tokens;
}

}  // namespace

ops::Status gated_delta_output_reference(const ops::GatedDeltaOutputShape& s,
                                         const ops::GatedDeltaOutputBuffers& b) noexcept {
    const auto status = ops::validate_gated_delta_output(s, b);
    if (status != ops::Status::Success || s.tokens == 0U) {
        return status;
    }
    if (!metadata_valid(s, b)) {
        return ops::Status::InvalidArgument;
    }
    const std::size_t group = s.heads / s.key_heads;
    for (std::size_t chunk = 0; chunk < s.chunks; ++chunk) {
        const auto sequence = static_cast<std::size_t>(b.chunk_indices.data[2U * chunk]);
        const auto local = static_cast<std::size_t>(b.chunk_indices.data[2U * chunk + 1U]);
        const auto first =
            static_cast<std::size_t>(b.sequence_starts.data[sequence]) + local * kChunk;
        const auto last = std::min(first + kChunk,
                                   static_cast<std::size_t>(b.sequence_starts.data[sequence + 1U]));
        for (std::size_t head = 0; head < s.heads; ++head) {
            for (std::size_t row = first; row < last; ++row) {
                const auto* q = b.query.data + (row * s.key_heads + head / group) * s.key_size;
                const double gate = b.gates.data[row * s.heads + head];
                std::array<double, kChunk> weights{};
                for (std::size_t key = first; key <= row; ++key) {
                    const auto* k = b.keys.data + (key * s.key_heads + head / group) * s.key_size;
                    double score = 0.0;
                    for (std::size_t d = 0; d < s.key_size; ++d) {
                        score += half(q[d]) * half(k[d]);
                    }
                    weights[key - first] = half(formats::encode_fp16(static_cast<float>(
                        score * std::exp(gate - b.gates.data[key * s.heads + head]))));
                }
                for (std::size_t value = 0; value < s.value_size; ++value) {
                    const auto* state =
                        b.states.data +
                        ((chunk * s.heads + head) * s.value_size + value) * s.key_size;
                    double prior = 0.0;
                    for (std::size_t d = 0; d < s.key_size; ++d) {
                        prior += half(q[d]) * half(state[d]);
                    }
                    double sum = prior * std::exp(gate);
                    for (std::size_t key = first; key <= row; ++key) {
                        sum += weights[key - first] *
                               half(b.values.data[(key * s.heads + head) * s.value_size + value]);
                    }
                    b.output.data[(row * s.heads + head) * s.value_size + value] =
                        formats::encode_fp16(static_cast<float>(sum * s.scale));
                }
            }
        }
    }
    return ops::Status::Success;
}

}  // namespace eclair::host
