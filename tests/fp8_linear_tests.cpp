/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/formats/fp16.hpp"
#include "eclair/host/fp8_linear.hpp"
#include "eclair/ops/fp8_linear.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using eclair::formats::Fp8Format;
using eclair::host::fp8_linear_reference;
using eclair::ops::Fp8LinearBuffers;
using eclair::ops::Fp8LinearShape;
using eclair::ops::Status;

void check_known_matrix() {
    const Fp8LinearShape shape{2, 3, 3, 2, 2, Fp8Format::E4M3FN};
    const std::array<std::uint16_t, 6> a{0x3C00U, 0x4000U, 0xBC00U, 0U, 0x3C00U, 0x3C00U};
    const std::array<std::uint8_t, 9> w{0x38U, 0x40U, 0x44U, 0xB8U, 0U, 0x38U, 0x30U, 0x38U, 0x40U};
    const std::array<float, 4> scales{2.0F, 0.5F, 4.0F, 2.0F};
    const std::array<float, 3> bias{1.0F, 2.0F, -3.0F};
    std::array<float, 6> output{};
    const Fp8LinearBuffers buffers{{a.data(), a.size()},
                                   {w.data(), w.size()},
                                   {scales.data(), scales.size()},
                                   {bias.data(), bias.size()},
                                   {output.data(), output.size()}};
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::Success);
    const std::array<float, 6> expected{9.5F, -0.5F, 3.0F, 6.5F, 2.5F, 5.0F};
    ECLAIR_CHECK(output == expected);
}

void check_representations() {
    const std::uint16_t one = 0x3C00U;
    const float scale = 1.0F;
    std::uint8_t weight = 0;
    float output = 0;
    Fp8LinearShape shape{1, 1, 1, 1, 1, Fp8Format::E4M3FN};
    const Fp8LinearBuffers buffers{{&one, 1}, {&weight, 1}, {&scale, 1}, {}, {&output, 1}};
    for (const auto format : {Fp8Format::E4M3FN, Fp8Format::E5M2}) {
        shape.format = format;
        for (unsigned int bits = 0; bits < 256U; ++bits) {
            weight = static_cast<std::uint8_t>(bits);
            // Independent specification formula, including exceptional patterns.
            const bool e4 = format == Fp8Format::E4M3FN;
            const unsigned int fraction_bits = e4 ? 3U : 2U;
            const unsigned int fraction = bits & ((1U << fraction_bits) - 1U);
            const unsigned int exponent = (bits & 127U) >> fraction_bits;
            const int bias = e4 ? 7 : 15;
            float expected = exponent == 0U
                                 ? std::ldexp(static_cast<float>(fraction),
                                              1 - bias - static_cast<int>(fraction_bits))
                                 : std::ldexp(1.0F + static_cast<float>(fraction) /
                                                         static_cast<float>(1U << fraction_bits),
                                              static_cast<int>(exponent) - bias);
            const bool nan = e4 ? (bits & 127U) == 127U : exponent == 31U && fraction != 0U;
            if (!e4 && exponent == 31U && fraction == 0U) {
                expected = std::numeric_limits<float>::infinity();
            }
            if ((bits & 128U) != 0U) {
                expected = -expected;
            }
            ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::Success);
            ECLAIR_CHECK(nan ? std::isnan(output) : output == expected);
        }
    }
}

void check_rounding() {
    const std::uint16_t one = 0x3C00U;
    std::uint8_t weight = 0x38U;
    float scale = 1.0F + std::ldexp(1.0F, -11);
    float output = 0;
    const Fp8LinearShape shape{1, 1, 1, 1, 1, Fp8Format::E4M3FN};
    const Fp8LinearBuffers buffers{{&one, 1}, {&weight, 1}, {&scale, 1}, {}, {&output, 1}};
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::Success);
    ECLAIR_CHECK(output == 1.0F);
    scale = std::ldexp(1.0F, -25);
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::Success);
    ECLAIR_CHECK(output == 0.0F);
    scale = std::ldexp(1.0F, -24);
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::Success);
    ECLAIR_CHECK(output == scale);
    weight = 0x7EU;
    scale = 256.0F;
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::Success);
    ECLAIR_CHECK(std::isinf(output));
    scale = std::numeric_limits<float>::quiet_NaN();
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::Success);
    ECLAIR_CHECK(std::isnan(output));
}

void check_boundaries() {
    const std::uint16_t a = 0x3C00U;
    const std::uint8_t w = 0x38U;
    const float scale = 2.0F;
    float output = -123.0F;
    Fp8LinearShape shape{1, 1, 1, 1, 1, Fp8Format::E4M3FN};
    Fp8LinearBuffers buffers{{&a, 1}, {&w, 1}, {&scale, 1}, {}, {&output, 1}};
    buffers.weights.size = 0;
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::BufferTooSmall);
    ECLAIR_CHECK(output == -123.0F);
    buffers.weights = {nullptr, 1};
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::InvalidArgument);
    buffers.weights = {&w, 1};
    buffers.scales = {&output, 1};
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::InvalidArgument);
    buffers.scales = {&scale, 1};
    buffers.output.data = reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max() - 3U);
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::SizeOverflow);
    buffers.output.data = reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(&output) + 1U);
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::InvalidArgument);
    buffers.output.data = &output;
    shape.k = 0;
    ECLAIR_CHECK(fp8_linear_reference(shape, buffers) == Status::InvalidArgument);
    shape.m = 0;
    ECLAIR_CHECK(fp8_linear_reference(shape, {}) == Status::Success);
    shape.format = static_cast<Fp8Format>(255U);
    ECLAIR_CHECK(fp8_linear_reference(shape, {}) == Status::InvalidArgument);
    shape.format = Fp8Format::E4M3FN;
    shape.block_k = 0;
    ECLAIR_CHECK(fp8_linear_reference(shape, {}) == Status::InvalidArgument);
    shape = {std::numeric_limits<std::size_t>::max(), 2, 2, 1, 1, Fp8Format::E4M3FN};
    ECLAIR_CHECK(fp8_linear_reference(shape, {}) == Status::SizeOverflow);
    ECLAIR_CHECK(output == -123.0F);
}

}  // namespace

int main() {
    check_known_matrix();
    check_representations();
    check_rounding();
    check_boundaries();
    return eclair::test::finish();
}
