/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/formats/fp8.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <vector>

namespace {

using eclair::formats::decode_fp8;
using eclair::formats::encode_fp8;
using eclair::formats::Fp8Format;

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr std::initializer_list<Fp8Format> kFormats = {Fp8Format::E4M3FN, Fp8Format::E5M2};

// Values fixed by the OCP 8-bit floating-point specification.
void check_decode_known_values() {
    ECLAIR_CHECK(decode_fp8(0x00U, Fp8Format::E4M3FN) == 0.0F);
    ECLAIR_CHECK(decode_fp8(0x80U, Fp8Format::E4M3FN) == 0.0F);
    ECLAIR_CHECK(std::signbit(decode_fp8(0x80U, Fp8Format::E4M3FN)));
    ECLAIR_CHECK(decode_fp8(0x01U, Fp8Format::E4M3FN) == std::ldexp(1.0F, -9));
    ECLAIR_CHECK(decode_fp8(0x08U, Fp8Format::E4M3FN) == std::ldexp(1.0F, -6));
    ECLAIR_CHECK(decode_fp8(0x38U, Fp8Format::E4M3FN) == 1.0F);
    ECLAIR_CHECK(decode_fp8(0x3CU, Fp8Format::E4M3FN) == 1.5F);
    ECLAIR_CHECK(decode_fp8(0x7EU, Fp8Format::E4M3FN) == 448.0F);
    ECLAIR_CHECK(decode_fp8(0xFEU, Fp8Format::E4M3FN) == -448.0F);
    ECLAIR_CHECK(std::isnan(decode_fp8(0x7FU, Fp8Format::E4M3FN)));
    ECLAIR_CHECK(std::isnan(decode_fp8(0xFFU, Fp8Format::E4M3FN)));

    ECLAIR_CHECK(decode_fp8(0x01U, Fp8Format::E5M2) == std::ldexp(1.0F, -16));
    ECLAIR_CHECK(decode_fp8(0x04U, Fp8Format::E5M2) == std::ldexp(1.0F, -14));
    ECLAIR_CHECK(decode_fp8(0x3CU, Fp8Format::E5M2) == 1.0F);
    ECLAIR_CHECK(decode_fp8(0x7BU, Fp8Format::E5M2) == 57344.0F);
    ECLAIR_CHECK(decode_fp8(0x7CU, Fp8Format::E5M2) == kInfinity);
    ECLAIR_CHECK(decode_fp8(0xFCU, Fp8Format::E5M2) == -kInfinity);
    ECLAIR_CHECK(std::isnan(decode_fp8(0x7DU, Fp8Format::E5M2)));
    ECLAIR_CHECK(std::isnan(decode_fp8(0xFFU, Fp8Format::E5M2)));
}

void check_encode_known_values() {
    ECLAIR_CHECK(encode_fp8(0.0F, Fp8Format::E4M3FN) == 0x00U);
    ECLAIR_CHECK(encode_fp8(-0.0F, Fp8Format::E4M3FN) == 0x80U);
    ECLAIR_CHECK(encode_fp8(1.0F, Fp8Format::E4M3FN) == 0x38U);
    ECLAIR_CHECK(encode_fp8(1.0F, Fp8Format::E5M2) == 0x3CU);
    ECLAIR_CHECK(encode_fp8(kInfinity, Fp8Format::E4M3FN) == 0x7EU);
    ECLAIR_CHECK(encode_fp8(-kInfinity, Fp8Format::E4M3FN) == 0xFEU);
    ECLAIR_CHECK(encode_fp8(kInfinity, Fp8Format::E5M2) == 0x7CU);
    ECLAIR_CHECK(encode_fp8(std::numeric_limits<float>::quiet_NaN(), Fp8Format::E4M3FN) == 0x7FU);
    ECLAIR_CHECK(encode_fp8(-std::numeric_limits<float>::quiet_NaN(), Fp8Format::E5M2) == 0xFFU);

    // Ties go to the even code: halfway between 1 and 1.125 stays 1, halfway
    // between 1.125 and 1.25 becomes 1.25.
    ECLAIR_CHECK(encode_fp8(1.0625F, Fp8Format::E4M3FN) == 0x38U);
    ECLAIR_CHECK(encode_fp8(1.1875F, Fp8Format::E4M3FN) == 0x3AU);
    ECLAIR_CHECK(encode_fp8(1.125F, Fp8Format::E5M2) == 0x3CU);
    ECLAIR_CHECK(encode_fp8(1.375F, Fp8Format::E5M2) == 0x3EU);
    // Halfway to the smallest subnormal rounds to zero, anything above to it.
    ECLAIR_CHECK(encode_fp8(std::ldexp(1.0F, -10), Fp8Format::E4M3FN) == 0x00U);
    ECLAIR_CHECK(encode_fp8(std::nextafter(std::ldexp(1.0F, -10), 1.0F), Fp8Format::E4M3FN) ==
                 0x01U);
    ECLAIR_CHECK(encode_fp8(-std::ldexp(1.0F, -17), Fp8Format::E5M2) == 0x80U);
    ECLAIR_CHECK(encode_fp8(std::nextafter(std::ldexp(1.0F, -17), 1.0F), Fp8Format::E5M2) == 0x01U);
    // Halfway between the largest subnormal and the smallest normal goes to
    // the even code, which is the normal one.
    ECLAIR_CHECK(encode_fp8(7.5F * std::ldexp(1.0F, -9), Fp8Format::E4M3FN) == 0x08U);
    ECLAIR_CHECK(encode_fp8(std::numeric_limits<float>::denorm_min(), Fp8Format::E5M2) == 0x00U);

    // Finite overflow saturates in both formats at every magnitude. Before the
    // reference decided saturation explicitly, E5M2 returned 49152 (0x7A) from
    // about 2^40 up, where float distances to the top candidates merge.
    for (const float value : {449.0F, 464.0F, 480.0F, 1.0e6F, std::ldexp(1.0F, 40), 1.0e30F,
                              std::numeric_limits<float>::max()}) {
        ECLAIR_CHECK(encode_fp8(value, Fp8Format::E4M3FN) == 0x7EU);
        ECLAIR_CHECK(encode_fp8(-value, Fp8Format::E4M3FN) == 0xFEU);
    }
    for (const float value : {57345.0F, 61440.0F, 1.0e6F, std::ldexp(1.0F, 40), 1.0e30F,
                              std::numeric_limits<float>::max()}) {
        ECLAIR_CHECK(encode_fp8(value, Fp8Format::E5M2) == 0x7BU);
        ECLAIR_CHECK(encode_fp8(-value, Fp8Format::E5M2) == 0xFBU);
    }
}

void check_round_trip() {
    for (const Fp8Format format : kFormats) {
        for (unsigned int bits = 0; bits < 256U; ++bits) {
            const auto input = static_cast<std::uint8_t>(bits);
            const float decoded = decode_fp8(input, format);
            if (!std::isnan(decoded)) {
                ECLAIR_CHECK(encode_fp8(decoded, format) == input);
            }
        }
    }
}

// Every input that can change the bit-level result: each FP8 value, the
// midpoints between neighbors (and one float either side), the midpoint past
// the largest value, special values, and a spread over all binary32 patterns.
[[nodiscard]] std::vector<float> encoder_inputs(Fp8Format format) {
    std::vector<float> inputs = {0.0F,
                                 kInfinity,
                                 std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::denorm_min(),
                                 std::numeric_limits<float>::min(),
                                 std::numeric_limits<float>::max()};
    const std::uint8_t max_finite = format == Fp8Format::E4M3FN ? 0x7EU : 0x7BU;
    for (unsigned int bits = 0; bits <= max_finite; ++bits) {
        const float value = decode_fp8(static_cast<std::uint8_t>(bits), format);
        const float next =
            bits == max_finite
                ? value + (value - decode_fp8(static_cast<std::uint8_t>(bits - 1U), format))
                : decode_fp8(static_cast<std::uint8_t>(bits + 1U), format);
        const float midpoint = value + (next - value) / 2.0F;
        for (const float input : {value, midpoint, std::nextafter(midpoint, 0.0F),
                                  std::nextafter(midpoint, kInfinity)}) {
            inputs.push_back(input);
        }
    }
    for (std::uint64_t bits = 0; bits <= 0xFFFFFFFFU; bits += 65521U) {
        inputs.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(bits)));
    }
    std::mt19937 random(8U);
    for (int index = 0; index < 100000; ++index) {
        inputs.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(random())));
    }
    const std::size_t positive = inputs.size();
    for (std::size_t index = 0; index < positive; ++index) {
        inputs.push_back(-inputs[index]);
    }
    return inputs;
}

void check_bit_encoder_matches_reference() {
    for (const Fp8Format format : kFormats) {
        for (const float input : encoder_inputs(format)) {
            const std::uint8_t expected = encode_fp8(input, format);
            const std::uint8_t actual =
                eclair::formats::fp32_bits_to_fp8(std::bit_cast<std::uint32_t>(input), format);
            ECLAIR_CHECK(actual == expected);
        }
    }
    // The bit encoder is constexpr, so a wrong rounding rule cannot compile.
    static_assert(eclair::formats::fp32_bits_to_fp8(0x3F800000U, Fp8Format::E4M3FN) == 0x38U);
    static_assert(eclair::formats::fp32_bits_to_fp8(0xFF800000U, Fp8Format::E5M2) == 0xFCU);
}

}  // namespace

int main() {
    check_decode_known_values();
    check_encode_known_values();
    check_round_trip();
    check_bit_encoder_matches_reference();
    return eclair::test::finish();
}
