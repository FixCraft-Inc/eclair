/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "cuda_support.hpp"

#include "eclair/cuda/w4a16_gemm.hpp"
#include "eclair/cuda/w4a16_linear.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/host/w4a16_linear.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using eclair::formats::decode_fp16;
using eclair::formats::encode_fp16;
using eclair::test::checked;
using eclair::test::DeviceBuffer;

// Storage nibble of each logical K offset within a packed word
// (docs/W4A16_LINEAR.md), so test packing is independent of the kernels.
constexpr unsigned int kNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};
constexpr std::size_t kWorkspaceBytes = std::size_t{32} << 20U;

bool same_half(std::uint16_t a, std::uint16_t b) {
    return (std::isnan(decode_fp16(a)) && std::isnan(decode_fp16(b))) ||
           decode_fp16(a) == decode_fp16(b);
}

// Packs logical nibbles q[k * n + channel] into [K/8,N] words.
std::vector<std::uint32_t> pack(const std::vector<unsigned int>& q, std::size_t n, std::size_t k) {
    std::vector<std::uint32_t> words(k / 8U * n, 0U);
    for (std::size_t column = 0; column < k; ++column) {
        for (std::size_t channel = 0; channel < n; ++channel) {
            words[column / 8U * n + channel] |= q[column * n + channel]
                                                << (4U * kNibble[column % 8U]);
        }
    }
    return words;
}

// The contract's weight from its specification: an exact float product of a
// small integer and a binary16 scale, rounded once to binary16.
std::uint16_t spec_weight(unsigned int q, std::uint16_t scale) {
    return encode_fp16((static_cast<float>(q) - 8.0F) * decode_fp16(scale));
}

// Every binary16 scale bit pattern (a superset of BF16-derived scales) times
// every nibble, through the vector path and, with an output that misses its
// 8-byte alignment, the scalar path. Results equal the specification bit for
// bit, including signed zeros. NaN only has to stay NaN.
void test_dequantize_all_scales() {
    constexpr std::size_t n = 65536;
    constexpr std::size_t k = 128;
    std::vector<unsigned int> q(k * n);
    for (std::size_t column = 0; column < k; ++column) {
        for (std::size_t channel = 0; channel < n; ++channel) {
            q[column * n + channel] = static_cast<unsigned int>((column + channel / 4096U) % 16U);
        }
    }
    std::vector<std::uint16_t> scales(n);
    for (std::size_t i = 0; i < n; ++i) {
        scales[i] = static_cast<std::uint16_t>(i);
    }
    eclair::test::Stream stream;
    const auto st = stream.get();
    DeviceBuffer<std::uint32_t> dw(pack(q, n, k), st);
    DeviceBuffer<std::uint16_t> ds(scales, st), out(k * n + 8U);
    std::size_t mismatches = 0;
    for (const std::size_t offset : {std::size_t{0}, std::size_t{1}}) {
        out.upload(std::vector<std::uint16_t>(out.size(), 0x7BFF), st);
        checked(
            eclair::cuda::launch_w4a16_dequantize(
                n, k,
                {{dw.data(), dw.size()}, {ds.data(), ds.size()}, {out.data() + offset, k * n}}, st),
            "all-scale dequantize");
        const auto actual = out.download(st);
        ECLAIR_CHECK(offset == 0U || actual[0] == 0x7BFF);
        ECLAIR_CHECK(actual[offset + k * n] == 0x7BFF);
        for (std::size_t i = 0; i < k * n; ++i) {
            const auto expected = spec_weight(q[i], scales[i % n]);
            const auto value = actual[offset + i];
            const bool ok = std::isnan(decode_fp16(expected)) ? std::isnan(decode_fp16(value))
                                                              : value == expected;
            mismatches += ok ? 0U : 1U;
        }
    }
    std::printf("DEQUANTIZE all_scales outputs=%zu mismatches=%zu\n", 2U * k * n, mismatches);
    ECLAIR_CHECK(mismatches == 0U);
}

// Odd N takes the scalar path; rows beyond the launch's 65535 grid rows loop.
void test_dequantize_shapes() {
    eclair::test::Stream stream;
    const auto st = stream.get();
    for (const auto& [n, k] :
         {std::pair<std::size_t, std::size_t>{33, 384}, {64, 128}, {3, 524416}}) {
        std::vector<unsigned int> q(k * n);
        std::vector<std::uint16_t> scales(k / 128U * n);
        for (std::size_t i = 0; i < q.size(); ++i) {
            q[i] = static_cast<unsigned int>((i * 7U + i / 13U) % 16U);
        }
        for (std::size_t i = 0; i < scales.size(); ++i) {
            scales[i] = encode_fp16(static_cast<float>(i % 9U + 1U) / 64.0F);
        }
        DeviceBuffer<std::uint32_t> dw(pack(q, n, k), st);
        DeviceBuffer<std::uint16_t> ds(scales, st), out(k * n + 2U);
        out.upload(std::vector<std::uint16_t>(out.size(), 0x7BFF), st);
        const eclair::ops::W4A16DequantizeBuffers b{
            {dw.data(), dw.size()}, {ds.data(), ds.size()}, {out.data() + 1U, k * n}};
        checked(eclair::cuda::launch_w4a16_dequantize(n, k, b, st), "dequantize shape");
        const auto actual = out.download(st);
        ECLAIR_CHECK(actual.front() == 0x7BFF && actual.back() == 0x7BFF);
        for (std::size_t i = 0; i < k * n; ++i) {
            ECLAIR_CHECK(actual[i + 1U] == spec_weight(q[i], scales[i / n / 128U * n + i % n]));
        }
        auto invalid = b;
        invalid.scales.size = 0;
        ECLAIR_CHECK(eclair::cuda::launch_w4a16_dequantize(n, k, invalid, st) ==
                     cudaErrorInvalidValue);
        invalid = b;
        invalid.dequantized.data = reinterpret_cast<std::uint16_t*>(dw.data());
        ECLAIR_CHECK(eclair::cuda::launch_w4a16_dequantize(n, k, invalid, st) ==
                     cudaErrorInvalidValue);
        ECLAIR_CHECK(eclair::cuda::launch_w4a16_dequantize(n, k - 8U, b, st) ==
                     cudaErrorInvalidValue);
        ECLAIR_CHECK(out.download(st) == actual);
    }
}

struct GemmProblem {
    std::size_t m, n, k;
    std::vector<std::uint16_t> a, scales;
    std::vector<std::uint32_t> weights;

    // The host FP64 reference of the shared contract.
    [[nodiscard]] std::vector<std::uint16_t> reference() const {
        std::vector<std::uint16_t> out(m * n);
        std::vector<float> unused(m * n);
        const eclair::ops::W4A16LinearBuffers host{{a.data(), a.size()},
                                                   {weights.data(), weights.size()},
                                                   {scales.data(), scales.size()},
                                                   {out.data(), out.size()},
                                                   {unused.data(), unused.size()}};
        ECLAIR_CHECK(eclair::host::w4a16_linear_reference({m, n, k, 1}, host) ==
                     eclair::ops::Status::Success);
        return out;
    }
};

// Uploads the problem, runs the GEMM path with output and scratch guards and
// returns the M*N outputs. Checks the guards and that the reconstruction left
// the specified weights in the scratch.
class GemmRun {
public:
    GemmRun(eclair::cuda::BlasHandle& handle, const GemmProblem& p, cudaStream_t stream)
        : handle_(handle),
          p_(p),
          stream_(stream),
          a_(p.a, stream),
          scales_(p.scales, stream),
          weights_(p.weights, stream),
          dequantized_(p.k * p.n + 2U),
          out_(p.m * p.n + 2U),
          workspace_(kWorkspaceBytes + 256U) {}

    [[nodiscard]] eclair::ops::W4A16GemmBuffers buffers(std::size_t workspace_bytes) const {
        return {{a_.data(), a_.size()},           {weights_.data(), weights_.size()},
                {scales_.data(), scales_.size()}, {dequantized_.data() + 1U, p_.k * p_.n},
                {out_.data() + 1U, p_.m * p_.n},  {workspace_.data(), workspace_bytes}};
    }

    std::vector<std::uint16_t> run(std::size_t workspace_bytes) {
        out_.upload(std::vector<std::uint16_t>(out_.size(), 0x7BFF), stream_);
        dequantized_.upload(std::vector<std::uint16_t>(dequantized_.size(), 0x7BFF), stream_);
        checked(eclair::cuda::launch_w4a16_gemm(handle_, {p_.m, p_.n, p_.k, 1},
                                                buffers(workspace_bytes), stream_),
                "W4A16 GEMM launch");
        const auto out = out_.download(stream_);
        ECLAIR_CHECK(out.front() == 0x7BFF && out.back() == 0x7BFF);
        const auto weights = dequantized_.download(stream_);
        ECLAIR_CHECK(weights.front() == 0x7BFF && weights.back() == 0x7BFF);
        for (std::size_t column = 0; column < p_.k; column += 37U) {
            for (std::size_t channel = 0; channel < p_.n; ++channel) {
                const unsigned int q =
                    (p_.weights[column / 8U * p_.n + channel] >> (4U * kNibble[column % 8U])) & 15U;
                ECLAIR_CHECK(weights[1U + column * p_.n + channel] ==
                             spec_weight(q, p_.scales[column / 128U * p_.n + channel]));
            }
        }
        return {out.begin() + 1, out.end() - 1};
    }

    // Rejected launches must leave every output untouched.
    void check_rejected(const eclair::ops::W4A16LinearShape& shape,
                        const eclair::ops::W4A16GemmBuffers& b) {
        const auto before = out_.download(stream_);
        ECLAIR_CHECK(eclair::cuda::launch_w4a16_gemm(handle_, shape, b, stream_) ==
                     cudaErrorInvalidValue);
        ECLAIR_CHECK(out_.download(stream_) == before);
    }

    [[nodiscard]] std::uint16_t* dequantized() const { return dequantized_.data() + 1U; }
    [[nodiscard]] std::byte* workspace() const { return workspace_.data(); }

private:
    eclair::cuda::BlasHandle& handle_;
    const GemmProblem& p_;
    cudaStream_t stream_;
    DeviceBuffer<std::uint16_t> a_, scales_;
    DeviceBuffer<std::uint32_t> weights_;
    DeviceBuffer<std::uint16_t> dequantized_, out_;
    DeviceBuffer<std::byte> workspace_;
};

// Exact dyadic operands: every partial sum is exact in FP32, so the output
// must equal the FP64 reference bit for bit, with or without a workspace,
// and repeat bit for bit.
void test_gemm_exact(eclair::cuda::BlasHandle& handle, std::size_t m, std::size_t n,
                     std::size_t k) {
    GemmProblem p{
        m, n, k, std::vector<std::uint16_t>(m * k), std::vector<std::uint16_t>(k / 128U * n), {}};
    for (std::size_t i = 0; i < p.a.size(); ++i) {
        p.a[i] = encode_fp16(static_cast<float>(static_cast<int>(i % 17U) - 8) / 16.0F);
    }
    for (std::size_t i = 0; i < p.scales.size(); ++i) {
        p.scales[i] = encode_fp16(static_cast<float>(i % 7U + 1U) / 32.0F);
    }
    std::vector<unsigned int> q(k * n);
    for (std::size_t column = 0; column < k; ++column) {
        for (std::size_t channel = 0; channel < n; ++channel) {
            q[column * n + channel] = static_cast<unsigned int>((column * 7U + channel * 3U) % 16U);
        }
    }
    p.weights = pack(q, n, k);
    const auto expected = p.reference();
    eclair::test::Stream stream;
    GemmRun run(handle, p, stream.get());
    for (const std::size_t bytes : {std::size_t{0}, kWorkspaceBytes}) {
        const auto first = run.run(bytes);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            ECLAIR_CHECK(first[i] == expected[i]);
        }
        ECLAIR_CHECK(run.run(bytes) == first);
    }
    const auto b = run.buffers(kWorkspaceBytes);
    auto invalid = b;
    invalid.weights.size = 0;
    run.check_rejected({m, n, k, 1}, invalid);
    invalid = b;
    invalid.output.data = invalid.dequantized.data + 1;
    run.check_rejected({m, n, k, 1}, invalid);
    invalid = b;
    invalid.workspace.data = run.workspace() + 128;
    run.check_rejected({m, n, k, 1}, invalid);
    invalid = b;
    invalid.workspace.data = reinterpret_cast<std::byte*>(run.dequantized());
    invalid.workspace.size = 256U;
    run.check_rejected({m, n, k, 1}, invalid);
    run.check_rejected({m, n, k, 2}, b);
    run.check_rejected({m, n, k + 8U, 1}, b);
    eclair::cuda::BlasHandle empty;
    ECLAIR_CHECK(eclair::cuda::launch_w4a16_gemm(empty, {m, n, k, 1}, b, stream.get()) ==
                 cudaErrorInvalidValue);
}

// 3 * (1 + 2^-10) rounds upward as a weight. Its exact negative in a second
// scale group cancels it to zero only if the weight was rounded before the
// product. Without that rounding the sum is -2^-10.
void test_cancellation(eclair::cuda::BlasHandle& handle) {
    constexpr std::size_t m = 51;
    GemmProblem p{m,
                  1,
                  256,
                  std::vector<std::uint16_t>(m * 256U, 0),
                  {0x3C01, 0x3C00},
                  std::vector<std::uint32_t>(32, 0x88888888U)};
    for (std::size_t row = 0; row < m; ++row) {
        p.a[row * 256U] = 0x3C00;
        p.a[row * 256U + 128U] = 0xC202;
    }
    p.weights[0] = 0x8888888BU;
    p.weights[16] = 0x88888889U;
    eclair::test::Stream stream;
    GemmRun run(handle, p, stream.get());
    for (const auto value : run.run(kWorkspaceBytes)) {
        ECLAIR_CHECK(value == 0x0000);
    }
}

// Every 32nd activation is +-2048 and the rest are 1/4, with unit weights.
// FP32 sums every partial exactly (at most 14 significant bits) in any order,
// giving 248. Binary16 accumulation loses the quarters added next to 2048,
// whose ulp is 2, in every K order, so this fails for that reason.
void test_fp32_accumulation(eclair::cuda::BlasHandle& handle) {
    constexpr std::size_t m = 51;
    constexpr std::size_t n = 8;
    constexpr std::size_t k = 1024;
    GemmProblem p{m,
                  n,
                  k,
                  std::vector<std::uint16_t>(m * k),
                  std::vector<std::uint16_t>(k / 128U * n, 0x3C00),
                  std::vector<std::uint32_t>(k / 8U * n, 0x99999999U)};
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t column = 0; column < k; ++column) {
            p.a[row * k + column] = column % 32U != 0U   ? 0x3400
                                    : column % 64U == 0U ? 0x6800
                                                         : 0xE800;
        }
    }
    eclair::test::Stream stream;
    GemmRun run(handle, p, stream.get());
    for (const std::size_t bytes : {std::size_t{0}, kWorkspaceBytes}) {
        for (const auto value : run.run(bytes)) {
            ECLAIR_CHECK(decode_fp16(value) == 248.0F);
        }
    }
}

// Zero, subnormal, largest, infinite and NaN activations in M=64 rows
// against the host reference.
void test_special_activations(eclair::cuda::BlasHandle& handle) {
    constexpr std::size_t m = 64;
    GemmProblem p{m,
                  1,
                  256,
                  std::vector<std::uint16_t>(m * 256U, 0),
                  {0x3C01, 0x3C00},
                  std::vector<std::uint32_t>(32, 0x88888888U)};
    p.weights[0] = 0x8888888BU;
    p.weights[16] = 0x88888889U;
    const std::uint16_t specials[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400,
                                      0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00};
    for (std::size_t row = 0; row < m; ++row) {
        const auto bits = specials[row % 12U];
        p.a[row * 256U] = bits;
        p.a[row * 256U + 128U] = bits == 0x3C00 ? 0xC202 : 0;
    }
    const auto expected = p.reference();
    eclair::test::Stream stream;
    GemmRun run(handle, p, stream.get());
    const auto actual = run.run(kWorkspaceBytes);
    for (std::size_t row = 0; row < m; ++row) {
        ECLAIR_CHECK(same_half(actual[row], expected[row]));
    }
}

// A nonzero binary16 subnormal operand aligns as if its exponent were -14.
[[nodiscard]] double alignment_magnitude(float value) {
    const double magnitude = std::fabs(static_cast<double>(value));
    return magnitude != 0.0 && magnitude < 0x1p-14 ? 0x1p-14 : magnitude;
}

// Random operands against an independent FP64 dot of the specified weights.
// The bound is the native kernel's form with up to 128 FP32 split partials,
// since cuBLAS chooses its own split: (K + 132) * 2^-22 times the aligned
// product magnitudes, plus binary16 output rounding.
void test_random_bound(eclair::cuda::BlasHandle& handle) {
    constexpr std::size_t n = 131;
    constexpr std::size_t k = 6144;
    std::uint32_t state = 20261028U;
    const auto next = [&state] {
        state = state * 1664525U + 1013904223U;
        return state >> 8U;  // 24 random bits
    };
    const auto unit = [&next] { return static_cast<float>(next()) * 0x1p-24F; };
    double worst_ratio = 0.0;
    double worst_relative = 0.0;
    for (const std::size_t m : {std::size_t{51}, std::size_t{130}}) {
        GemmProblem p{
            m, n, k, std::vector<std::uint16_t>(m * k), std::vector<std::uint16_t>(k / 128U * n),
            {}};
        for (auto& value : p.a) {
            value = encode_fp16((unit() * 2.0F - 1.0F) * 4.0F);
        }
        for (auto& value : p.scales) {
            value = encode_fp16(std::ldexp(1.0F + unit(), -9 + static_cast<int>(next() % 5U)));
        }
        std::vector<unsigned int> q(k * n);
        for (auto& value : q) {
            value = next() % 16U;
        }
        p.weights = pack(q, n, k);
        std::vector<float> weights(k * n);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] = decode_fp16(spec_weight(q[i], p.scales[i / n / 128U * n + i % n]));
        }
        eclair::test::Stream stream;
        GemmRun run(handle, p, stream.get());
        for (const std::size_t bytes : {std::size_t{0}, kWorkspaceBytes}) {
            const auto actual = run.run(bytes);
            for (std::size_t row = 0; row < m; ++row) {
                for (std::size_t channel = 0; channel < n; ++channel) {
                    double exact = 0.0;
                    double magnitude = 0.0;
                    for (std::size_t column = 0; column < k; ++column) {
                        const float weight = weights[column * n + channel];
                        const float activation = decode_fp16(p.a[row * k + column]);
                        exact += static_cast<double>(activation) * weight;
                        magnitude += alignment_magnitude(activation) * alignment_magnitude(weight);
                    }
                    const double sum_error =
                        static_cast<double>(k + 132U) * std::ldexp(magnitude, -22);
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
            "W4A16 large-M: Eclair binary16 reconstruction of shuffled K/8,N words into K,N, "
            "then cublasGemmEx FP16 operands, COMPUTE_32F, reduced-precision reduction "
            "disallowed, no atomics; workspace 0 and 32 MiB; nondefault stream synchronized; "
            "all-scale bitwise, exact dyadic FP64, cancellation, special and random FP64-bound "
            "checks");
        checked(eclair::cuda::check_w4a16_dequantize_image(0), "image probe");
        eclair::cuda::BlasHandle handle;
        checked(eclair::cuda::BlasHandle::create(0, handle), "cuBLAS handle");
        ECLAIR_CHECK(handle.valid() && handle.device() == 0);
        eclair::cuda::BlasHandle invalid;
        ECLAIR_CHECK(eclair::cuda::BlasHandle::create(-1, invalid) == cudaErrorInvalidValue);
        ECLAIR_CHECK(!invalid.valid());
        test_dequantize_all_scales();
        test_dequantize_shapes();
        for (const std::size_t m : {std::size_t{51}, std::size_t{64}, std::size_t{130}}) {
            test_gemm_exact(handle, m, 33, 384);
            test_gemm_exact(handle, m, 64, 128);
        }
        test_gemm_exact(handle, 51, 131, 6144);
        test_gemm_exact(handle, 130, 131, 6144);
        test_cancellation(handle);
        test_fp32_accumulation(handle);
        test_special_activations(handle);
        test_random_bound(handle);
    });
}
