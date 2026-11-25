/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "cuda_support.hpp"

#include "eclair/cuda/gated_delta_output.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/host/gated_delta_output.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using eclair::formats::decode_fp16;
using eclair::formats::encode_fp16;
using eclair::ops::GatedDeltaOutputBuffers;
using eclair::ops::GatedDeltaOutputShape;
using eclair::ops::Status;
using eclair::test::checked;
using eclair::test::DeviceBuffer;
using eclair::test::Stream;

struct Problem {
    GatedDeltaOutputShape shape;
    std::vector<std::uint16_t> q, k, v, states, out;
    std::vector<float> gates;
    std::vector<std::int32_t> starts, indices;

    Problem(const std::vector<int>& lengths, std::size_t heads, std::size_t key_heads,
            std::size_t key_size, std::size_t value_size) {
        std::size_t tokens = 0, chunks = 0;
        starts.push_back(0);
        for (std::size_t n = 0; n < lengths.size(); ++n) {
            for (int i = 0; i < (lengths[n] + 63) / 64; ++i) {
                indices.push_back(static_cast<std::int32_t>(n));
                indices.push_back(i);
                ++chunks;
            }
            tokens += static_cast<std::size_t>(lengths[n]);
            starts.push_back(static_cast<std::int32_t>(tokens));
        }
        shape = {
            tokens,    lengths.size(), chunks,     heads,
            key_heads, key_size,       value_size, 1.0F / std::sqrt(static_cast<float>(key_size))};
        q.resize(tokens * key_heads * key_size);
        k.resize(q.size());
        v.resize(tokens * heads * value_size);
        states.resize(chunks * heads * value_size * key_size);
        out.resize(v.size());
        gates.resize(tokens * heads);
        std::mt19937 rng(71);
        std::uniform_real_distribution<float> unit(-0.5F, 0.5F);
        for (auto* data : {&q, &k, &v, &states}) {
            for (auto& value : *data) {
                value = encode_fp16(unit(rng));
            }
        }
        for (std::size_t n = 0; n < lengths.size(); ++n) {
            for (int i = starts[n]; i < starts[n + 1U]; ++i) {
                for (std::size_t h = 0; h < heads; ++h) {
                    gates[static_cast<std::size_t>(i) * heads + h] =
                        -0.01F * static_cast<float>((i - starts[n]) % 64 + 1);
                }
            }
        }
    }

    GatedDeltaOutputBuffers buffers() {
        return {{q.data(), q.size()},
                {k.data(), k.size()},
                {v.data(), v.size()},
                {states.data(), states.size()},
                {gates.data(), gates.size()},
                {starts.data(), starts.size()},
                {indices.data(), indices.size()},
                {out.data(), out.size()}};
    }
};

struct DeviceCase {
    DeviceBuffer<std::uint16_t> q, k, v, states, out;
    DeviceBuffer<float> gates;
    DeviceBuffer<std::int32_t> starts, indices;
    DeviceCase(const Problem& p, cudaStream_t stream)
        : q(p.q, stream),
          k(p.k, stream),
          v(p.v, stream),
          states(p.states, stream),
          out(p.out.size()),
          gates(p.gates, stream),
          starts(p.starts, stream),
          indices(p.indices, stream) {}
    GatedDeltaOutputBuffers buffers() const {
        return {{q.data(), q.size()},
                {k.data(), k.size()},
                {v.data(), v.size()},
                {states.data(), states.size()},
                {gates.data(), gates.size()},
                {starts.data(), starts.size()},
                {indices.data(), indices.size()},
                {out.data(), out.size()}};
    }
};

void compare(Problem& p) {
    ECLAIR_CHECK(eclair::host::gated_delta_output_reference(p.shape, p.buffers()) ==
                 Status::Success);
    Stream stream;
    DeviceCase d(p, stream.get());
    checked(eclair::cuda::launch_gated_delta_output(p.shape, d.buffers(), stream.get()), "output");
    const auto actual = d.out.download(stream.get());
    float largest = 0;
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = decode_fp16(actual[i]), e = decode_fp16(p.out[i]);
        if (std::isnan(e)) {
            wrong += std::isnan(a) ? 0U : 1U;
        } else if (!std::isfinite(a) || !std::isfinite(e)) {
            wrong += a == e ? 0U : 1U;
        } else {
            largest = std::fmax(largest, std::fabs(a - e));
            wrong += std::fabs(a - e) > 0.004F ? 1U : 0U;
        }
    }
    std::printf(
        "output: tokens=%zu sequences=%zu heads=%zu key_heads=%zu K=%zu V=%zu max_abs=%g "
        "wrong=%zu\n",
        p.shape.tokens, p.shape.sequences, p.shape.heads, p.shape.key_heads, p.shape.key_size,
        p.shape.value_size, largest, wrong);
    ECLAIR_CHECK(wrong == 0U);
}

void test_reference_and_nonfinite() {
    for (const std::size_t key_size : {64U, 128U}) {
        Problem p({1, 63, 0, 65, 130}, 6, 2, key_size, 128);
        compare(p);
    }
    Problem p({17, 3}, 2, 1, 64, 64);
    std::fill(p.q.begin(), p.q.end(), encode_fp16(1.0F / 8));
    std::fill(p.k.begin(), p.k.end(), encode_fp16(1.0F / 8));
    std::fill(p.gates.begin(), p.gates.end(), 0);
    p.v[0] = 0x7E00;  // Attended NaN in one value head/column.
    p.v[1] = 0x7C00;  // Attended infinity, with nonzero correlations.
    p.v[2] = 0x0001;  // Subnormal.
    p.v[3] = 0x8000;  // Negative zero.
    compare(p);

    // A rounding boundary with an independently known result: the prior is
    // -(1 + 2^-10)^2; the binary16 correlation is 1 + 2^-9. Their sum is -2^-20.
    Problem rounding({1}, 1, 1, 64, 64);
    std::fill(rounding.q.begin(), rounding.q.end(), 0);
    std::fill(rounding.k.begin(), rounding.k.end(), 0);
    std::fill(rounding.states.begin(), rounding.states.end(), 0);
    std::fill(rounding.v.begin(), rounding.v.end(), 0x3C00);
    std::fill(rounding.gates.begin(), rounding.gates.end(), 0);
    rounding.q[0] = 0x3C01;
    rounding.k[0] = 0x3C01;
    for (std::size_t value = 0; value < 64; ++value) {
        rounding.states[value * 64] = 0xBC01;
    }
    rounding.shape.scale = 1;
    Stream stream;
    DeviceCase d(rounding, stream.get());
    checked(eclair::cuda::launch_gated_delta_output(rounding.shape, d.buffers(), stream.get()),
            "correlation rounding");
    for (const auto bits : d.out.download(stream.get())) {
        ECLAIR_CHECK(bits == 0x8010);
    }
    rounding.q[0] = 0x3C00;
    rounding.k[0] = 0x3C00;
    std::fill(rounding.v.begin(), rounding.v.end(), 0xC000);
    for (std::size_t value = 0; value < 64; ++value) {
        rounding.states[value * 64] = 0x4000;
    }
    rounding.shape.scale = std::numeric_limits<float>::max();
    d.q.upload(rounding.q, stream.get());
    d.k.upload(rounding.k, stream.get());
    d.v.upload(rounding.v, stream.get());
    d.states.upload(rounding.states, stream.get());
    checked(eclair::cuda::launch_gated_delta_output(rounding.shape, d.buffers(), stream.get()),
            "scale after cancellation");
    for (const auto bits : d.out.download(stream.get())) {
        ECLAIR_CHECK(bits == 0);
    }
}

void test_rejections_and_metadata() {
    Problem p({17}, 2, 1, 64, 64);
    Stream stream;
    DeviceCase d(p, stream.get());
    checked(cudaMemsetAsync(d.out.data(), 0x7B, d.out.size() * 2U, stream.get()), "sentinel");
    auto b = d.buffers();
    b.values.size -= 1U;
    ECLAIR_CHECK(eclair::cuda::launch_gated_delta_output(p.shape, b, stream.get()) ==
                 cudaErrorInvalidValue);
    b = d.buffers();
    b.query.data += 1;
    b.query.size -= 1;
    ECLAIR_CHECK(eclair::cuda::launch_gated_delta_output(p.shape, b, stream.get()) ==
                 cudaErrorInvalidValue);
    for (const bool narrow_key : {false, true}) {
        auto unsupported = p.shape;
        if (narrow_key) {
            unsupported.key_size = 32;
        } else {
            unsupported.value_size = 32;
        }
        // Both extents fit in the existing allocations, so only the unsupported
        // tile dimension prevents the launch.
        ECLAIR_CHECK(eclair::ops::validate_gated_delta_output(unsupported, d.buffers()) ==
                     Status::Success);
        ECLAIR_CHECK(eclair::cuda::launch_gated_delta_output(
                         unsupported, d.buffers(), stream.get()) == cudaErrorInvalidValue);
    }
    p.indices[0] = 1;  // Device bounds skip, no writes.
    d.indices.upload(p.indices, stream.get());
    checked(eclair::cuda::launch_gated_delta_output(p.shape, d.buffers(), stream.get()),
            "invalid metadata");
    for (const auto bits : d.out.download(stream.get())) {
        ECLAIR_CHECK(bits == 0x7B7B);
    }
}

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable != nullptr) {
            ECLAIR_CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
        }
        if (graph != nullptr) {
            ECLAIR_CHECK(cudaGraphDestroy(graph) == cudaSuccess);
        }
    }
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
};

void test_graph_replay() {
    Problem p({17}, 2, 1, 64, 64);
    Stream stream;
    DeviceCase d(p, stream.get());
    checked(eclair::cuda::launch_gated_delta_output(p.shape, d.buffers(), stream.get()), "warmup");
    checked(cudaStreamSynchronize(stream.get()), "warmup sync");
    Graph graph;
    checked(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal), "begin capture");
    checked(eclair::cuda::launch_gated_delta_output(p.shape, d.buffers(), stream.get()), "capture");
    checked(cudaStreamEndCapture(stream.get(), &graph.graph), "end capture");
    checked(cudaGraphInstantiate(&graph.executable, graph.graph, 0ULL), "instantiate");
    for (const int rows : {7, 17}) {
        p.starts[1] = rows;
        p.shape.tokens = static_cast<std::size_t>(rows);
        d.starts.upload(p.starts, stream.get());
        std::fill(p.out.begin(), p.out.end(), 0x7B7B);
        checked(cudaMemsetAsync(d.out.data(), 0x7B, d.out.size() * 2U, stream.get()), "sentinel");
        ECLAIR_CHECK(eclair::host::gated_delta_output_reference(p.shape, p.buffers()) ==
                     Status::Success);
        checked(cudaGraphLaunch(graph.executable, stream.get()), "replay");
        const auto actual = d.out.download(stream.get());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            ECLAIR_CHECK(std::fabs(decode_fp16(actual[i]) - decode_fp16(p.out[i])) <= 0.004F);
        }
    }
}

void benchmark() {
    Problem p({3072}, 48, 16, 128, 128);
    Stream stream;
    DeviceCase d(p, stream.get());
    const auto samples = eclair::test::timings(stream.get(), [&] {
        checked(eclair::cuda::launch_gated_delta_output(p.shape, d.buffers(), stream.get()),
                "benchmark");
    });
    for (const auto bits : d.out.download(stream.get())) {
        ECLAIR_CHECK(std::isfinite(decode_fp16(bits)));
    }
    std::printf("benchmark: tokens=3072 H=48 Hg=16 K=128 V=128 median_ms=%.3f max_ms=%.3f\n",
                samples[4], samples[8]);
}

}  // namespace

int main() {
    return eclair::test::contain([] {
        eclair::test::print_device_header();
        test_reference_and_nonfinite();
        test_rejections_and_metadata();
        test_graph_replay();
        benchmark();
    });
}
