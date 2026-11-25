/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

// The gated delta rule state pass on the GPU against the FP64 host reference,
// rejection of unsupported arguments, and timings at Qwen3.8 27B's Gated
// DeltaNet shapes (48 value heads, 16 key heads, key and value size 128).
// Inputs follow the scales of a real layer: unit-norm keys, w of the same
// size, standard-normal u, and gates that fall by up to 0.08 per token within
// each chunk.

#include "cuda_support.hpp"

#include "eclair/cuda/gated_delta.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/host/gated_delta.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using eclair::cuda::launch_gated_delta_states;
using eclair::formats::decode_fp16;
using eclair::formats::encode_fp16;
using eclair::ops::GatedDeltaStatesBuffers;
using eclair::ops::GatedDeltaStatesShape;
using eclair::ops::kGatedDeltaChunk;
using eclair::ops::Status;
using eclair::test::checked;
using eclair::test::DeviceBuffer;
using eclair::test::Stream;

enum class Initial : std::uint8_t { None, Binary16, Float32 };

struct Problem {
    GatedDeltaStatesShape shape;
    std::vector<std::uint16_t> keys, w, u, initial16;
    std::vector<float> gates, initial32;
    std::vector<std::int32_t> starts, chunk_starts;
};

[[nodiscard]] Problem make_problem(const std::vector<std::size_t>& lengths, std::size_t heads,
                                   std::size_t key_heads, std::size_t key_size,
                                   std::size_t value_size, Initial initial, std::uint32_t seed) {
    Problem p;
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> step(-0.08F, -0.005F);
    std::size_t tokens = 0;
    std::size_t chunks = 0;
    p.starts.push_back(0);
    for (const std::size_t length : lengths) {
        p.chunk_starts.push_back(static_cast<std::int32_t>(chunks));
        tokens += length;
        chunks += (length + kGatedDeltaChunk - 1U) / kGatedDeltaChunk;
        p.starts.push_back(static_cast<std::int32_t>(tokens));
    }
    p.shape = {tokens, lengths.size(), chunks, heads, key_heads, key_size, value_size};
    const auto unit_rows = [&](std::vector<std::uint16_t>& out, std::size_t rows) {
        out.resize(rows * key_size);
        for (std::size_t r = 0; r < rows; ++r) {
            std::vector<float> row(key_size);
            float norm = 0.0F;
            for (float& value : row) {
                value = normal(generator);
                norm += value * value;
            }
            for (std::size_t k = 0; k < key_size; ++k) {
                out[r * key_size + k] = encode_fp16(row[k] / std::sqrt(norm));
            }
        }
    };
    unit_rows(p.keys, tokens * key_heads);
    unit_rows(p.w, tokens * heads);
    p.u.resize(tokens * heads * value_size);
    for (std::uint16_t& value : p.u) {
        value = encode_fp16(normal(generator));
    }
    p.gates.assign(tokens * heads, 0.0F);
    for (std::size_t n = 0; n < lengths.size(); ++n) {
        for (std::size_t h = 0; h < heads; ++h) {
            float sum = 0.0F;
            for (auto t = static_cast<std::size_t>(p.starts[n]);
                 t < static_cast<std::size_t>(p.starts[n + 1U]); ++t) {
                if ((t - static_cast<std::size_t>(p.starts[n])) % kGatedDeltaChunk == 0U) {
                    sum = 0.0F;
                }
                sum += step(generator);
                p.gates[t * heads + h] = sum;
            }
        }
    }
    const std::size_t states = lengths.size() * heads * value_size * key_size;
    if (initial == Initial::Binary16) {
        p.initial16.resize(states);
        for (std::uint16_t& value : p.initial16) {
            value = encode_fp16(0.3F * normal(generator));
        }
    } else if (initial == Initial::Float32) {
        p.initial32.resize(states);
        for (float& value : p.initial32) {
            value = 0.3F * normal(generator);
        }
    }
    return p;
}

struct Outputs {
    std::vector<std::uint16_t> states, new_values;
    std::vector<float> final_states;
};

[[nodiscard]] Outputs allocate(const GatedDeltaStatesShape& s) {
    return {std::vector<std::uint16_t>(s.chunks * s.heads * s.value_size * s.key_size),
            std::vector<std::uint16_t>(s.tokens * s.heads * s.value_size),
            std::vector<float>(s.sequences * s.heads * s.value_size * s.key_size)};
}

[[nodiscard]] Outputs reference(const Problem& p) {
    Outputs out = allocate(p.shape);
    const GatedDeltaStatesBuffers host{
        {p.keys.data(), p.keys.size()},
        {p.w.data(), p.w.size()},
        {p.u.data(), p.u.size()},
        {p.gates.data(), p.gates.size()},
        {p.initial16.empty() ? nullptr : p.initial16.data(), p.initial16.size()},
        {p.initial32.empty() ? nullptr : p.initial32.data(), p.initial32.size()},
        {p.starts.data(), p.starts.size()},
        {p.chunk_starts.data(), p.chunk_starts.size()},
        {out.states.data(), out.states.size()},
        {out.new_values.data(), out.new_values.size()},
        {out.final_states.data(), out.final_states.size()}};
    if (eclair::host::gated_delta_states_reference(p.shape, host) != Status::Success) {
        throw std::runtime_error("reference rejected a test case");
    }
    return out;
}

struct Device {
    DeviceBuffer<std::uint16_t> keys, w, u, initial16;
    DeviceBuffer<float> gates, initial32;
    DeviceBuffer<std::int32_t> starts, chunk_starts;
    DeviceBuffer<std::uint16_t> states, new_values;
    DeviceBuffer<float> final_states;

    Device(const Problem& p, cudaStream_t stream)
        : keys(p.keys, stream),
          w(p.w, stream),
          u(p.u, stream),
          initial16(p.initial16, stream),
          gates(p.gates, stream),
          initial32(p.initial32, stream),
          starts(p.starts, stream),
          chunk_starts(p.chunk_starts, stream),
          states(p.shape.chunks * p.shape.heads * p.shape.value_size * p.shape.key_size),
          new_values(p.shape.tokens * p.shape.heads * p.shape.value_size),
          final_states(p.shape.sequences * p.shape.heads * p.shape.value_size * p.shape.key_size) {}

    [[nodiscard]] GatedDeltaStatesBuffers buffers() const {
        return {{keys.data(), keys.size()},
                {w.data(), w.size()},
                {u.data(), u.size()},
                {gates.data(), gates.size()},
                {initial16.size() != 0U ? initial16.data() : nullptr, initial16.size()},
                {initial32.size() != 0U ? initial32.data() : nullptr, initial32.size()},
                {starts.data(), starts.size()},
                {chunk_starts.data(), chunk_starts.size()},
                {states.data(), states.size()},
                {new_values.data(), new_values.size()},
                {final_states.data(), final_states.size()}};
    }
};

// Largest difference over the largest reference magnitude, at least 1.
template <typename T, typename Decode>
[[nodiscard]] float relative_difference(const std::vector<T>& actual,
                                        const std::vector<T>& expected, Decode decode,
                                        bool& finite) {
    float difference = 0.0F;
    float magnitude = 1.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float a = decode(actual[i]);
        const float e = decode(expected[i]);
        finite = finite && std::isfinite(a);
        difference = std::fmax(difference, std::fabs(a - e));
        magnitude = std::fmax(magnitude, std::fabs(e));
    }
    return difference / magnitude;
}

void compare(const std::vector<std::size_t>& lengths, std::size_t key_size, Initial initial,
             std::uint32_t seed) {
    const Problem p = make_problem(lengths, 6, 2, key_size, 128, initial, seed);
    Stream stream;
    Device device(p, stream.get());
    checked(launch_gated_delta_states(p.shape, device.buffers(), stream.get()), "state pass");
    const Outputs expected = reference(p);
    const Outputs actual{device.states.download(stream.get()),
                         device.new_values.download(stream.get()),
                         device.final_states.download(stream.get())};
    bool finite = true;
    const auto half = [](std::uint16_t bits) { return decode_fp16(bits); };
    const auto single = [](float value) { return value; };
    const float states = relative_difference(actual.states, expected.states, half, finite);
    const float values = relative_difference(actual.new_values, expected.new_values, half, finite);
    const float finals =
        relative_difference(actual.final_states, expected.final_states, single, finite);
    std::printf(
        "compare: sequences=%zu tokens=%zu key_size=%zu initial=%d states=%.2g "
        "new_values=%.2g final=%.2g finite=%d\n",
        lengths.size(), p.shape.tokens, key_size, static_cast<int>(initial), states, values, finals,
        finite ? 1 : 0);
    // Binary16 roundings of the state can differ by one unit where FP32 and
    // FP64 sums straddle a rounding boundary, so results agree to about 1e-3.
    ECLAIR_CHECK(finite && states <= 1e-2F && values <= 1e-2F && finals <= 1e-2F);
}

void test_rejections() {
    Stream stream;
    const Problem p = make_problem({70}, 6, 2, 128, 128, Initial::None, 3);
    Device device(p, stream.get());
    auto bad_value = p.shape;
    bad_value.value_size = 96;  // Not a multiple of the 64-row state block.
    ECLAIR_CHECK(launch_gated_delta_states(bad_value, device.buffers(), stream.get()) ==
                 cudaErrorInvalidValue);
    auto bad_key = p.shape;
    bad_key.key_size = 96;
    ECLAIR_CHECK(launch_gated_delta_states(bad_key, device.buffers(), stream.get()) ==
                 cudaErrorInvalidValue);
    GatedDeltaStatesBuffers misaligned = device.buffers();
    misaligned.u.data += 1;
    misaligned.u.size -= 1U;
    ECLAIR_CHECK(launch_gated_delta_states(p.shape, misaligned, stream.get()) ==
                 cudaErrorInvalidValue);
    ECLAIR_CHECK(cudaGetLastError() == cudaSuccess);
    ECLAIR_CHECK(eclair::cuda::check_gated_delta_image(0) == cudaSuccess);
}

// Median CUDA-event time of one prefill chunk's state pass at Qwen3.8 27B's
// shapes, and of a long single sequence.
void benchmark() {
    for (const std::size_t tokens : {std::size_t{3072}, std::size_t{30720}}) {
        const Problem p = make_problem({tokens}, 48, 16, 128, 128, Initial::Binary16, 11);
        Stream stream;
        Device device(p, stream.get());
        const auto samples = eclair::test::timings(stream.get(), [&] {
            checked(launch_gated_delta_states(p.shape, device.buffers(), stream.get()), "bench");
        });
        bool finite = true;
        for (const float value : device.final_states.download(stream.get())) {
            finite = finite && std::isfinite(value);
        }
        ECLAIR_CHECK(finite);
        std::printf(
            "benchmark: tokens=%zu heads=48 key_heads=16 size=128 median_ms=%.3f "
            "max_ms=%.3f\n",
            tokens, samples[4], samples[8]);
    }
}

}  // namespace

int main() {
    return eclair::test::contain([] {
        eclair::test::print_device_header();
        compare({70}, 128, Initial::None, 1);
        compare({64, 200, 1, 129}, 128, Initial::Binary16, 2);
        compare({300}, 128, Initial::Float32, 4);
        compare({65, 3}, 64, Initial::Binary16, 5);
        test_rejections();
        benchmark();
    });
}
