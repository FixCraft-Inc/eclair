/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "cuda_support.hpp"

#include "eclair/cuda/w4a16_linear.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/host/w4a16_linear.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using eclair::formats::decode_fp16;
using eclair::formats::encode_fp16;
using eclair::test::checked;
using eclair::test::DeviceBuffer;

struct GraphOwner {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    GraphOwner() = default;
    GraphOwner(const GraphOwner&) = delete;
    GraphOwner& operator=(const GraphOwner&) = delete;
    ~GraphOwner() {
        if (executable != nullptr && cudaGraphExecDestroy(executable) != cudaSuccess) {
            ECLAIR_CHECK(false);
        }
        if (graph != nullptr && cudaGraphDestroy(graph) != cudaSuccess) {
            ECLAIR_CHECK(false);
        }
    }
};

bool same_half(std::uint16_t a, std::uint16_t b) {
    return (std::isnan(decode_fp16(a)) && std::isnan(decode_fp16(b))) ||
           decode_fp16(a) == decode_fp16(b);
}

void test_shape(std::size_t m, std::size_t n, std::size_t k, std::size_t splits) {
    eclair::test::Stream stream;
    const auto st = stream.get();
    std::vector<std::uint16_t> a(m * k), scales(k / 128U * n), out(m * n, 0x7BFF);
    std::vector<std::uint32_t> weights(k / 8U * n, 0U);
    std::vector<float> scratch(splits * m * n);
    // Independent packing from logical indices and exact dyadic operands.
    constexpr unsigned int positions[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = encode_fp16(static_cast<float>(static_cast<int>(i % 17U) - 8) / 16.0F);
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = encode_fp16(static_cast<float>(i % 7U + 1U) / 32.0F);
    }
    for (std::size_t column = 0; column < k; ++column) {
        for (std::size_t channel = 0; channel < n; ++channel) {
            const auto q = static_cast<std::uint32_t>((column * 7U + channel * 3U) % 16U);
            weights[column / 8U * n + channel] |= q << (4U * positions[column % 8U]);
        }
    }
    const eclair::ops::W4A16LinearShape s{m, n, k, splits};
    const eclair::ops::W4A16LinearBuffers host{{a.data(), a.size()},
                                               {weights.data(), weights.size()},
                                               {scales.data(), scales.size()},
                                               {out.data(), out.size()},
                                               {scratch.data(), scratch.size()}};
    ECLAIR_CHECK(eclair::host::w4a16_linear_reference(s, host) == eclair::ops::Status::Success);
    DeviceBuffer<std::uint16_t> da(a, st), ds(scales, st), dout(m * n + 2U);
    DeviceBuffer<std::uint32_t> dw(weights, st);
    DeviceBuffer<float> workspace(scratch.size() + 2U);
    dout.upload(std::vector<std::uint16_t>(dout.size(), 0x7BFF), st);
    workspace.upload(std::vector<float>(workspace.size(), 123), st);
    eclair::ops::W4A16LinearBuffers b{{da.data(), da.size()},
                                      {dw.data(), dw.size()},
                                      {ds.data(), ds.size()},
                                      {dout.data() + 1U, m * n},
                                      {workspace.data() + 1U, scratch.size()}};
    checked(eclair::cuda::launch_w4a16_linear(s, b, st), "W4A16 launch");
    auto actual = dout.download(st);
    ECLAIR_CHECK(actual.front() == 0x7BFF && actual.back() == 0x7BFF);
    for (std::size_t i = 0; i < out.size(); ++i) {
        ECLAIR_CHECK(same_half(actual[i + 1U], out[i]));
    }
    const auto ws = workspace.download(st);
    ECLAIR_CHECK(ws.front() == 123 && ws.back() == 123);
    auto invalid = b;
    invalid.weights.size = 0;
    ECLAIR_CHECK(eclair::cuda::launch_w4a16_linear(s, invalid, st) == cudaErrorInvalidValue);
    invalid = b;
    invalid.workspace.data = reinterpret_cast<float*>(dw.data());
    ECLAIR_CHECK(eclair::cuda::launch_w4a16_linear(s, invalid, st) == cudaErrorInvalidValue);
    const auto unchanged = dout.download(st);
    ECLAIR_CHECK(actual == unchanged);

    checked(cudaStreamSynchronize(st), "before capture");
    GraphOwner owner;
    checked(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), "capture begin");
    checked(eclair::cuda::launch_w4a16_linear(s, b, st), "captured launch");
    checked(cudaStreamEndCapture(st, &owner.graph), "capture end");
    checked(cudaGraphInstantiate(&owner.executable, owner.graph, nullptr, nullptr, 0),
            "instantiate");
    const auto replay = cudaGraphLaunch(owner.executable, st);
    const auto sync = cudaStreamSynchronize(st);
    if (replay == cudaSuccess && sync == cudaSuccess) {
        for (auto& value : a) {
            value = encode_fp16(decode_fp16(value) * 0.5F);
        }
        da.upload(a, st);
        ECLAIR_CHECK(eclair::host::w4a16_linear_reference(s, host) == eclair::ops::Status::Success);
        const auto changed_replay = cudaGraphLaunch(owner.executable, st);
        const auto changed_sync = cudaStreamSynchronize(st);
        ECLAIR_CHECK(changed_replay == cudaSuccess && changed_sync == cudaSuccess);
    }
    checked(replay, "graph replay");
    checked(sync, "graph synchronize");
    actual = dout.download(st);
    for (std::size_t i = 0; i < out.size(); ++i) {
        ECLAIR_CHECK(same_half(actual[i + 1U], out[i]));
    }
}

void test_special_activations() {
    eclair::test::Stream stream;
    const auto st = stream.get();
    std::vector<std::uint16_t> a(4U * 256U, 0), scales{0x3C01, 0x3C00}, expected(4);
    std::vector<std::uint32_t> weights(32U, 0x88888888U);
    weights[0] = 0x8888888BU;  // 3 * (1 + 2^-10) rounds upward as a weight.
    weights[16] = 0x88888889U;
    std::vector<float> scratch(16U * 4U);
    DeviceBuffer<std::uint16_t> da(a.size()), ds(scales, st), out(4);
    DeviceBuffer<std::uint32_t> dw(weights, st);
    DeviceBuffer<float> ws(scratch.size());
    const eclair::ops::W4A16LinearShape s{4, 1, 256, 16};
    const eclair::ops::W4A16LinearBuffers host{{a.data(), a.size()},
                                               {weights.data(), weights.size()},
                                               {scales.data(), scales.size()},
                                               {expected.data(), expected.size()},
                                               {scratch.data(), scratch.size()}};
    const eclair::ops::W4A16LinearBuffers device{{da.data(), da.size()},
                                                 {dw.data(), dw.size()},
                                                 {ds.data(), ds.size()},
                                                 {out.data(), out.size()},
                                                 {ws.data(), ws.size()}};
    for (const auto bits : {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0x7BFF, 0x7C00,
                            0xFC00, 0x7E00, 0xFE00}) {
        for (std::size_t row = 0; row < 4U; ++row) {
            a[row * 256U] = static_cast<std::uint16_t>(bits);
            a[row * 256U + 128U] = bits == 0x3C00 ? 0xC202 : 0;
        }
        ECLAIR_CHECK(eclair::host::w4a16_linear_reference(s, host) == eclair::ops::Status::Success);
        da.upload(a, st);
        checked(eclair::cuda::launch_w4a16_linear(s, device, st), "special activation launch");
        const auto actual = out.download(st);
        for (std::size_t row = 0; row < 4U; ++row) {
            ECLAIR_CHECK(same_half(actual[row], expected[row]));
        }
        if (bits == 0x3C00) {
            ECLAIR_CHECK(decode_fp16(actual[0]) == 0);
        }
    }
}

void test_all_scales() {
    constexpr std::size_t n = 65536;
    eclair::test::Stream stream;
    const auto st = stream.get();
    std::vector<std::uint16_t> a(4U * 128U, 0), scales(n);
    for (std::size_t row = 0; row < 4U; ++row) {
        a[row * 128U] = 0x3C00;
    }
    for (std::size_t i = 0; i < n; ++i) {
        scales[i] = static_cast<std::uint16_t>(i);
    }
    DeviceBuffer<std::uint16_t> da(a, st), ds(scales, st), out(4U * n);
    DeviceBuffer<std::uint32_t> w(16U * n);
    DeviceBuffer<float> scratch(4U * 4U * n);
    for (unsigned int q = 0; q < 16U; ++q) {
        std::vector<std::uint32_t> weights(w.size(), 0x88888888U);
        std::fill(weights.begin(), weights.begin() + n, 0x88888880U | q);
        w.upload(weights, st);
        for (std::size_t m = 1; m <= 4; ++m) {
            const eclair::ops::W4A16LinearShape s{m, n, 128, 4};
            const eclair::ops::W4A16LinearBuffers b{{da.data(), da.size()},
                                                    {w.data(), w.size()},
                                                    {ds.data(), ds.size()},
                                                    {out.data(), out.size()},
                                                    {scratch.data(), scratch.size()}};
            checked(eclair::cuda::launch_w4a16_linear(s, b, st), "all-scale launch");
            const auto actual = out.download(st);
            for (std::size_t row = 0; row < m; ++row) {
                for (std::size_t i = 0; i < n; ++i) {
                    const float scale = decode_fp16(scales[i]);
                    // Nonfinite scales also multiply the zero weights in
                    // the remaining K positions, so their dot is NaN.
                    const auto expected = std::isfinite(scale)
                                              ? encode_fp16((static_cast<float>(q) - 8.0F) * scale)
                                              : 0x7E00;
                    ECLAIR_CHECK(
                        same_half(actual[row * n + i], static_cast<std::uint16_t>(expected)));
                }
            }
        }
    }
}

// Logical K offset to nibble position within its packed word.
constexpr unsigned int kNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};

// One output of sum_k a[k] * w[k] for K=128, N=1, scale 1 and integer
// weights in [-8,7]; every activation row holds the same values. Terms are
// (logical K offset, activation, weight).
struct Term {
    unsigned int offset;
    float activation;
    int weight;
};

float device_terms(const std::vector<Term>& terms) {
    eclair::test::Stream stream;
    const auto st = stream.get();
    std::vector<std::uint16_t> a(4U * 128U, 0);
    std::vector<std::uint32_t> weights(16U, 0x88888888U);
    for (const auto& term : terms) {
        for (std::size_t row = 0; row < 4U; ++row) {
            a[row * 128U + term.offset] = encode_fp16(term.activation);
        }
        const unsigned int shift = 4U * kNibble[term.offset % 8U];
        weights[term.offset / 8U] = (weights[term.offset / 8U] & ~(15U << shift)) |
                                    static_cast<std::uint32_t>(term.weight + 8) << shift;
    }
    DeviceBuffer<std::uint16_t> da(a, st), ds(std::vector<std::uint16_t>{0x3C00}, st), out(4);
    DeviceBuffer<std::uint32_t> dw(weights, st);
    DeviceBuffer<float> ws(4);
    const eclair::ops::W4A16LinearBuffers b{{da.data(), da.size()},
                                            {dw.data(), dw.size()},
                                            {ds.data(), ds.size()},
                                            {out.data(), out.size()},
                                            {ws.data(), ws.size()}};
    checked(eclair::cuda::launch_w4a16_linear({4, 1, 128, 1}, b, st), "terms launch");
    const auto actual = out.download(st);
    for (const auto value : actual) {
        ECLAIR_CHECK(value == actual[0]);
    }
    return decode_fp16(actual[0]);
}

// Tensor cores add exact products after aligning them to the largest term
// and truncating what falls below roughly 23 bits (BF16_LINEAR.md). In
// 1 + x - 1 with x = 1.5 * 2^-j, FP32 summation in either order recovers x
// while j <= 22, and the binary16 output shows whether x survived. Measures
// the window within one mma step (offsets 0..2) and through the FP32
// accumulator across steps (offsets 0, 8, 16).
void check_alignment() {
    const auto window = [](unsigned int second, unsigned int third) {
        int exact_to = 7;
        for (int j = 8; j <= 23; ++j) {
            const float x = std::ldexp(1.5F, -j);
            if (device_terms({{0, 1.0F, 1}, {second, x, 1}, {third, -1.0F, 1}}) != x) {
                break;
            }
            exact_to = j;
        }
        return exact_to;
    };
    const int same_step = window(1, 2);
    const int across_steps = window(8, 16);
    std::printf("ALIGNMENT same_step_exact_to_j=%d across_steps_exact_to_j=%d\n", same_step,
                across_steps);
    ECLAIR_CHECK(same_step >= 22 && across_steps >= 22);
}

// A nonzero binary16 subnormal operand aligns as if its exponent were -14,
// so it weighs 2^-14 in the error bound.
[[nodiscard]] double alignment_magnitude(float value) {
    const double magnitude = std::fabs(static_cast<double>(value));
    return magnitude != 0.0 && magnitude < 0x1p-14 ? 0x1p-14 : magnitude;
}

// Random activations, nibbles and scales against an independent FP64 dot of
// FP16-rounded weights. Summation with truncation and the FP32 split
// reduction errs by at most about (K + splits + 4) * 2^-22 times the aligned
// product magnitudes; binary16 output rounding adds half an ulp.
void test_random_bound() {
    constexpr std::size_t n = 131;
    constexpr std::size_t k = 6144;
    std::uint32_t state = 20261027U;
    const auto next = [&state] {
        state = state * 1664525U + 1013904223U;
        return state >> 8U;  // 24 random bits
    };
    const auto unit = [&next] { return static_cast<float>(next()) * 0x1p-24F; };
    std::vector<std::uint16_t> a(4U * k), scales(k / 128U * n);
    std::vector<std::uint32_t> weights(k / 8U * n, 0U);
    std::vector<int> q(k * n);
    for (auto& value : a) {
        value = encode_fp16((unit() * 2.0F - 1.0F) * 4.0F);
    }
    for (auto& value : scales) {
        value = encode_fp16(std::ldexp(1.0F + unit(), -9 + static_cast<int>(next() % 5U)));
    }
    for (std::size_t column = 0; column < k; ++column) {
        for (std::size_t channel = 0; channel < n; ++channel) {
            const auto nibble = next() % 16U;
            q[column * n + channel] = static_cast<int>(nibble) - 8;
            weights[column / 8U * n + channel] |= nibble << (4U * kNibble[column % 8U]);
        }
    }
    eclair::test::Stream stream;
    const auto st = stream.get();
    DeviceBuffer<std::uint16_t> da(a, st), ds(scales, st), out(4U * n);
    DeviceBuffer<std::uint32_t> dw(weights, st);
    DeviceBuffer<float> ws(16U * 4U * n);
    double worst_ratio = 0.0;
    double worst_relative = 0.0;
    for (std::size_t m = 1; m <= 4; ++m) {
        for (const std::size_t splits : {1U, 16U}) {
            const eclair::ops::W4A16LinearBuffers b{{da.data(), m * k},
                                                    {dw.data(), dw.size()},
                                                    {ds.data(), ds.size()},
                                                    {out.data(), m * n},
                                                    {ws.data(), splits * m * n}};
            checked(eclair::cuda::launch_w4a16_linear({m, n, k, splits}, b, st), "random launch");
            const auto actual = out.download(st);
            for (std::size_t row = 0; row < m; ++row) {
                for (std::size_t channel = 0; channel < n; ++channel) {
                    double exact = 0.0;
                    double magnitude = 0.0;
                    for (std::size_t column = 0; column < k; ++column) {
                        const float scale = decode_fp16(scales[column / 128U * n + channel]);
                        const float weight = decode_fp16(
                            encode_fp16(static_cast<float>(q[column * n + channel]) * scale));
                        const float activation = decode_fp16(a[row * k + column]);
                        exact += static_cast<double>(activation) * weight;
                        magnitude += alignment_magnitude(activation) * alignment_magnitude(weight);
                    }
                    const double sum_error =
                        static_cast<double>(k + splits + 4U) * std::ldexp(magnitude, -22);
                    const double bound =
                        sum_error + std::ldexp(std::fabs(exact) + sum_error, -11) + 0x1p-25;
                    const float value = decode_fp16(actual[row * n + channel]);
                    const double error = std::fabs(static_cast<double>(value) - exact);
                    ECLAIR_CHECK(std::isfinite(value) && error <= bound);
                    worst_ratio = std::max(worst_ratio, error / bound);
                    worst_relative = std::max(worst_relative, error / std::fabs(exact));
                }
            }
        }
    }
    std::printf("RANDOM max_error_to_bound=%.3g max_relative_error=%.3g\n", worst_ratio,
                worst_relative);
}

}  // namespace

int main() {
    return eclair::test::contain([] {
        eclair::test::print_device_header();
        std::puts(
            "W4A16 group128 shuffled K/8,N; HMMA.884 FP32 accumulation, FP32 split "
            "reduction; binary16 output; nondefault stream synchronized; exact dyadic FP64, "
            "all-scale, alignment-window and random FP64-bound checks");
        checked(eclair::cuda::check_w4a16_linear_image(0), "image probe");
        for (std::size_t m = 1; m <= 4; ++m) {
            for (const std::size_t splits : {1U, 4U, 16U, 128U}) {
                test_shape(m, 33, 384, splits);
            }
            test_shape(m, 131, 6144, 16);
        }
        test_all_scales();
        test_special_activations();
        check_alignment();
        test_random_bound();
    });
}
