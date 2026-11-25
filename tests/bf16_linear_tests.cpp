/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/formats/bf16.hpp"
#include "eclair/host/bf16_linear.hpp"
#include "eclair/ops/bf16_linear.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {

using eclair::formats::decode_bf16;
using eclair::formats::encode_bf16;
using eclair::host::bf16_linear_reference;
using eclair::host::bf16_row_scaling_reference;
using eclair::ops::Bf16LinearBuffers;
using eclair::ops::Bf16LinearShape;
using eclair::ops::Bf16RowScalingBuffers;
using eclair::ops::Bf16RowScalingShape;
using eclair::ops::Status;

constexpr float kInfinity = std::numeric_limits<float>::infinity();

[[nodiscard]] std::uint16_t bf16(float value) {
    return encode_bf16(value);
}

// A finite BF16 value with the given sign, unbiased exponent (-126 to 127)
// and 7-bit fraction.
[[nodiscard]] std::uint16_t bf16_bits(bool negative, int exponent, std::uint32_t fraction) {
    return static_cast<std::uint16_t>((negative ? 0x8000U : 0U) |
                                      (static_cast<std::uint32_t>(exponent + 127) << 7U) |
                                      (fraction & 0x7FU));
}

// Row-major storage of a matrix written one row per line.
template <std::size_t Rows, std::size_t Columns>
[[nodiscard]] std::vector<std::uint16_t> flatten(
    const std::array<std::array<std::uint16_t, Columns>, Rows>& rows) {
    std::vector<std::uint16_t> values;
    for (const auto& row : rows) {
        values.insert(values.end(), row.begin(), row.end());
    }
    return values;
}

struct ScaledMatrix {
    std::vector<std::uint16_t> values;
    std::vector<std::int32_t> shifts;
};

[[nodiscard]] ScaledMatrix scale_rows(const std::vector<std::uint16_t>& matrix, std::size_t rows,
                                      std::size_t columns) {
    ScaledMatrix scaled{std::vector<std::uint16_t>(matrix.size()), std::vector<std::int32_t>(rows)};
    const Bf16RowScalingBuffers buffers{{matrix.data(), matrix.size()},
                                        {scaled.values.data(), scaled.values.size()},
                                        {scaled.shifts.data(), scaled.shifts.size()}};
    ECLAIR_CHECK(bf16_row_scaling_reference({rows, columns}, buffers) == Status::Success);
    return scaled;
}

// Row scaling of both operands followed by the linear reference.
[[nodiscard]] std::vector<float> pipeline(const Bf16LinearShape& shape,
                                          const std::vector<std::uint16_t>& a,
                                          const std::vector<std::uint16_t>& w,
                                          const std::vector<float>& bias) {
    const ScaledMatrix scaled_a = scale_rows(a, shape.m, shape.k);
    const ScaledMatrix scaled_w = scale_rows(w, shape.n, shape.k);
    std::vector<float> output(shape.m * shape.n);
    const Bf16LinearBuffers buffers{{scaled_a.values.data(), scaled_a.values.size()},
                                    {scaled_a.shifts.data(), scaled_a.shifts.size()},
                                    {scaled_w.values.data(), scaled_w.values.size()},
                                    {scaled_w.shifts.data(), scaled_w.shifts.size()},
                                    {bias.data(), bias.size()},
                                    {output.data(), output.size()}};
    ECLAIR_CHECK(bf16_linear_reference(shape, buffers) == Status::Success);
    return output;
}

// Nearest float with IEEE overflow, restated independently of the library.
[[nodiscard]] float round_to_float(double value) {
    if (std::fabs(value) >= 0x1.ffffffp127) {
        return value < 0.0 ? -kInfinity : kInfinity;
    }
    return static_cast<float>(value);
}

// The BF16 definition with no scaling: FP64 sums of the products of the BF16
// values themselves, rounded once to FP32, plus the bias.
[[nodiscard]] std::vector<float> direct(const Bf16LinearShape& shape,
                                        const std::vector<std::uint16_t>& a,
                                        const std::vector<std::uint16_t>& w,
                                        const std::vector<float>& bias) {
    std::vector<float> output(shape.m * shape.n);
    for (std::size_t m = 0; m < shape.m; ++m) {
        for (std::size_t n = 0; n < shape.n; ++n) {
            double sum = 0.0;
            for (std::size_t k = 0; k < shape.k; ++k) {
                sum += static_cast<double>(decode_bf16(a[m * shape.k + k])) *
                       static_cast<double>(decode_bf16(w[n * shape.k + k]));
            }
            output[m * shape.n + n] = round_to_float(sum) + (bias.empty() ? 0.0F : bias[n]);
        }
    }
    return output;
}

[[nodiscard]] bool same(float a, float b) {
    return std::isnan(a) ? std::isnan(b) : a == b;
}

void check_row_scaling_known_rows() {
    // Row 0: largest 1.0, so shift 15. 2^-40 is 40 binades below and ties to
    // zero, 2^-32 is the smallest exact distance, and infinity, NaN and -0
    // pass through.
    // Row 1: no finite nonzero value, so shift 0.
    // Row 2: the largest BF16 value lands on 65280, and 1.0 vanishes.
    // Row 3: the smallest BF16 subnormal alone.
    using Rows = std::array<std::array<std::uint16_t, 6>, 4>;
    const Rows input{{
        {bf16(1.0F), bf16(0x1p-40F), bf16(0x1p-32F), 0x7F80U, 0x7FC0U, 0x8000U},
        {0x0000U, 0x7F80U, 0xFF80U, 0xFFC1U, 0x8000U, 0x0000U},
        {0x7F7FU, bf16(1.0F), 0x0000U, 0x0000U, 0x0000U, 0x0000U},
        {0x0001U, 0x0000U, 0x8000U, 0x0000U, 0x0000U, 0x0000U},
    }};
    const Rows expected{{
        {0x7800U, 0x0000U, 0x0080U, 0x7C00U, 0x7E00U, 0x8000U},
        {0x0000U, 0x7C00U, 0xFC00U, 0xFE00U, 0x8000U, 0x0000U},
        {0x7BF8U, 0x0000U, 0x0000U, 0x0000U, 0x0000U, 0x0000U},
        {0x7800U, 0x0000U, 0x8000U, 0x0000U, 0x0000U, 0x0000U},
    }};
    const ScaledMatrix scaled = scale_rows(flatten(input), 4, 6);
    const std::array<std::int32_t, 4> shifts{15, 0, -112, 148};
    for (std::size_t row = 0; row < shifts.size(); ++row) {
        ECLAIR_CHECK(scaled.shifts[row] == shifts[row]);
    }
    ECLAIR_CHECK(scaled.values == flatten(expected));
}

void check_row_scaling_boundaries() {
    const std::array<std::uint16_t, 4> input{bf16(1.0F), bf16(2.0F), bf16(3.0F), bf16(4.0F)};
    // Aligned for int32, so aliasing the shifts onto it fails for overlap.
    alignas(4) std::array<std::uint16_t, 4> output{0x1234U, 0x1234U, 0x1234U, 0x1234U};
    std::array<std::int32_t, 3> shifts{-7, -7, -7};
    Bf16RowScalingShape shape{2, 2};
    Bf16RowScalingBuffers buffers{{input.data(), 4}, {output.data(), 4}, {shifts.data(), 2}};
    const auto untouched = [&] { return output[0] == 0x1234U && shifts[0] == -7; };

    ECLAIR_CHECK(bf16_row_scaling_reference({0, 5}, {}) == Status::Success);
    ECLAIR_CHECK(bf16_row_scaling_reference({2, 0}, buffers) == Status::InvalidArgument);
    buffers.output.size = 3;
    ECLAIR_CHECK(bf16_row_scaling_reference(shape, buffers) == Status::BufferTooSmall);
    buffers.output.size = 4;
    buffers.shifts.size = 1;
    ECLAIR_CHECK(bf16_row_scaling_reference(shape, buffers) == Status::BufferTooSmall);
    buffers.shifts.size = 2;
    buffers.input.data = nullptr;
    ECLAIR_CHECK(bf16_row_scaling_reference(shape, buffers) == Status::InvalidArgument);
    buffers.input.data = input.data();
    buffers.shifts.data =
        reinterpret_cast<std::int32_t*>(reinterpret_cast<std::uintptr_t>(shifts.data()) + 2U);
    ECLAIR_CHECK(bf16_row_scaling_reference(shape, buffers) == Status::InvalidArgument);
    buffers.shifts.data = shifts.data();
    // Output overlapping the input, and shifts overlapping the output.
    buffers.output.data = const_cast<std::uint16_t*>(input.data()) + 1;
    buffers.output.size = 3;
    ECLAIR_CHECK(bf16_row_scaling_reference({1, 3}, buffers) == Status::InvalidArgument);
    buffers.output = {output.data(), 4};
    buffers.shifts.data = reinterpret_cast<std::int32_t*>(output.data());
    ECLAIR_CHECK(bf16_row_scaling_reference(shape, buffers) == Status::InvalidArgument);
    buffers.shifts.data = shifts.data();
    ECLAIR_CHECK(bf16_row_scaling_reference({std::numeric_limits<std::size_t>::max(), 2},
                                            buffers) == Status::SizeOverflow);
    ECLAIR_CHECK(untouched());
    ECLAIR_CHECK(bf16_row_scaling_reference(shape, buffers) == Status::Success);
    ECLAIR_CHECK(shifts[0] == 14 && shifts[1] == 13 && shifts[2] == -7);
}

// When every element is within 32 binades of its row's largest, the scaled
// operands are exact, so the pipeline must equal the unscaled BF16 definition
// bit for bit: the FP64 products differ only by powers of two.
void check_matches_bf16_definition() {
    std::mt19937 random(20261024U);
    const auto next = [&](std::uint32_t bound) {
        return static_cast<std::uint32_t>(random() % bound);
    };
    for (const Bf16LinearShape shape :
         {Bf16LinearShape{3, 5, 7}, Bf16LinearShape{8, 9, 64}, Bf16LinearShape{1, 33, 129}}) {
        std::vector<std::uint16_t> a(shape.m * shape.k);
        std::vector<std::uint16_t> w(shape.n * shape.k);
        for (auto* matrix : {&a, &w}) {
            const std::size_t rows = matrix == &a ? shape.m : shape.n;
            for (std::size_t row = 0; row < rows; ++row) {
                // Rows centered anywhere from 2^-45 to 2^45, spanning 30
                // binades, far outside binary16's range at both ends while
                // most sums stay within FP32's.
                const int center = static_cast<int>(next(91U)) - 45;
                for (std::size_t column = 0; column < shape.k; ++column) {
                    (*matrix)[row * shape.k + column] = bf16_bits(
                        next(2U) == 1U, center + static_cast<int>(next(31U)) - 15, next(128U));
                }
            }
        }
        std::vector<float> bias(shape.n);
        for (float& value : bias) {
            value = static_cast<float>(static_cast<int>(next(2001U)) - 1000) / 8.0F;
        }
        for (const bool with_bias : {true, false}) {
            const std::vector<float> chosen = with_bias ? bias : std::vector<float>{};
            const auto actual = pipeline(shape, a, w, chosen);
            const auto expected = direct(shape, a, w, chosen);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (!same(actual[i], expected[i])) {
                    std::fprintf(stderr, "M=%zu N=%zu K=%zu index=%zu actual=%a expected=%a\n",
                                 shape.m, shape.n, shape.k, i, static_cast<double>(actual[i]),
                                 static_cast<double>(expected[i]));
                    ECLAIR_CHECK(false && "row-scaled pipeline differs from BF16 definition");
                    return;
                }
            }
        }
    }
}

// The documented cost of row scaling: an element more than 32 binades below
// its row's largest rounds away before any cancellation can expose it.
void check_flush_below_window() {
    const std::vector<std::uint16_t> wide{bf16(0x1p40F), bf16(-0x1p40F), bf16(1.0F)};
    const std::vector<std::uint16_t> narrow{bf16(0x1p20F), bf16(-0x1p20F), bf16(1.0F)};
    const std::vector<std::uint16_t> ones{bf16(1.0F), bf16(1.0F), bf16(1.0F)};
    ECLAIR_CHECK(pipeline({1, 1, 3}, wide, ones, {})[0] == 0.0F);
    ECLAIR_CHECK(direct({1, 1, 3}, wide, ones, {})[0] == 1.0F);
    ECLAIR_CHECK(pipeline({1, 1, 3}, narrow, ones, {})[0] == 1.0F);
}

// Results far outside binary16's range, at FP32's limits, and nonfinite
// inputs follow the BF16 definition.
void check_range_and_nonfinite() {
    struct Case {
        std::vector<std::uint16_t> a;
        std::vector<std::uint16_t> w;
        float expected;
    };
    const Case cases[] = {
        {{bf16(0x1p60F)}, {bf16(0x1p60F)}, 0x1p120F},
        {{bf16(0x1p100F)}, {bf16(-0x1p100F)}, -kInfinity},
        {{bf16(0x1p-70F)}, {bf16(0x1p-70F)}, 0x1p-140F},
        {{bf16(0x1p-100F)}, {bf16(0x1p-100F)}, 0.0F},
        {{bf16(3.0F), bf16(0x1p-120F)}, {bf16(0x1p100F), bf16(0x1p-10F)}, 3.0F * 0x1p100F},
        {{0x7F80U, bf16(1.0F)}, {bf16(2.0F), bf16(1.0F)}, kInfinity},
        {{0x7F80U}, {0x0000U}, std::numeric_limits<float>::quiet_NaN()},
        {{0x7F80U, 0xFF80U}, {bf16(1.0F), bf16(1.0F)}, std::numeric_limits<float>::quiet_NaN()},
        {{0x7FC0U, bf16(1.0F)}, {bf16(1.0F), bf16(1.0F)}, std::numeric_limits<float>::quiet_NaN()},
    };
    for (const Case& test : cases) {
        const Bf16LinearShape shape{1, 1, test.a.size()};
        ECLAIR_CHECK(same(pipeline(shape, test.a, test.w, {})[0], test.expected));
        ECLAIR_CHECK(same(direct(shape, test.a, test.w, {})[0], test.expected));
    }
}

// Any int32 shift is defined: clamping to the documented bound changes no
// result, and nothing overflows.
void check_extreme_shifts() {
    const std::uint16_t one = 0x3C00U;  // binary16 1.0.
    const std::uint16_t zero = 0x0000U;
    float output = -1.0F;
    for (const std::int32_t shift :
         {std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::min()}) {
        const std::int32_t zero_shift = 0;
        Bf16LinearBuffers buffers{{&one, 1},        {&shift, 1}, {&one, 1},
                                  {&zero_shift, 1}, {},          {&output, 1}};
        ECLAIR_CHECK(bf16_linear_reference({1, 1, 1}, buffers) == Status::Success);
        ECLAIR_CHECK(output == (shift > 0 ? 0.0F : kInfinity));
        buffers.activations.data = &zero;
        ECLAIR_CHECK(bf16_linear_reference({1, 1, 1}, buffers) == Status::Success);
        ECLAIR_CHECK(output == 0.0F);
    }
}

void check_linear_boundaries() {
    const std::array<std::uint16_t, 4> values{0x3C00U, 0x4000U, 0x4200U, 0x4400U};
    const std::array<std::int32_t, 2> shifts{0, 0};
    std::array<float, 4> output{-5.0F, -5.0F, -5.0F, -5.0F};
    Bf16LinearShape shape{2, 2, 2};
    Bf16LinearBuffers buffers{
        {values.data(), 4}, {shifts.data(), 2}, {values.data(), 4}, {shifts.data(), 2}, {},
        {output.data(), 4}};
    ECLAIR_CHECK(bf16_linear_reference({0, 2, 2}, {}) == Status::Success);
    ECLAIR_CHECK(bf16_linear_reference({2, 0, 0}, {}) == Status::Success);
    ECLAIR_CHECK(bf16_linear_reference({2, 2, 0}, buffers) == Status::InvalidArgument);
    buffers.activation_shifts.size = 1;
    ECLAIR_CHECK(bf16_linear_reference(shape, buffers) == Status::BufferTooSmall);
    buffers.activation_shifts.size = 2;
    buffers.weight_shifts = {nullptr, 2};
    ECLAIR_CHECK(bf16_linear_reference(shape, buffers) == Status::InvalidArgument);
    buffers.weight_shifts = {shifts.data(), 2};
    buffers.bias = {output.data(), 2};
    ECLAIR_CHECK(bf16_linear_reference(shape, buffers) == Status::InvalidArgument);
    buffers.bias = {};
    buffers.output.data =
        reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(output.data()) + 1U);
    ECLAIR_CHECK(bf16_linear_reference(shape, buffers) == Status::InvalidArgument);
    buffers.output.data = reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max() - 7U);
    ECLAIR_CHECK(bf16_linear_reference(shape, buffers) == Status::SizeOverflow);
    buffers.output.data = output.data();
    ECLAIR_CHECK(bf16_linear_reference({std::numeric_limits<std::size_t>::max(), 2, 2}, buffers) ==
                 Status::SizeOverflow);
    ECLAIR_CHECK(output[0] == -5.0F);
    // Inputs may alias each other: A and W share storage here.
    ECLAIR_CHECK(bf16_linear_reference(shape, buffers) == Status::Success);
    const std::array<float, 4> expected{5.0F, 11.0F, 11.0F, 25.0F};
    ECLAIR_CHECK(output == expected);
}

}  // namespace

int main() {
    check_row_scaling_known_rows();
    check_row_scaling_boundaries();
    check_matches_bf16_definition();
    check_flush_below_window();
    check_range_and_nonfinite();
    check_extreme_shifts();
    check_linear_boundaries();
    return eclair::test::finish();
}
