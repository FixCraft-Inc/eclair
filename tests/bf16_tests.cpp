/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/formats/bf16.hpp"
#include "eclair/formats/fp16.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>

namespace {

using eclair::formats::bf16_finite_magnitude;
using eclair::formats::bf16_fp16_shift;
using eclair::formats::bf16_to_fp16_scaled_bits;
using eclair::formats::decode_bf16;
using eclair::formats::encode_bf16;

constexpr float kInfinity = std::numeric_limits<float>::infinity();

[[nodiscard]] float from_bits(std::uint32_t bits) {
    return std::bit_cast<float>(bits);
}

// Values from the format definition: bfloat16 is binary32's upper half.
void check_known_values() {
    ECLAIR_CHECK(decode_bf16(0x3F80U) == 1.0F);
    ECLAIR_CHECK(decode_bf16(0xC000U) == -2.0F);
    ECLAIR_CHECK(decode_bf16(0x7F7FU) == std::ldexp(255.0F, 120));
    ECLAIR_CHECK(decode_bf16(0x0080U) == std::ldexp(1.0F, -126));
    ECLAIR_CHECK(decode_bf16(0x0001U) == std::ldexp(1.0F, -133));
    ECLAIR_CHECK(decode_bf16(0x7F80U) == kInfinity);
    ECLAIR_CHECK(decode_bf16(0xFF80U) == -kInfinity);
    ECLAIR_CHECK(std::isnan(decode_bf16(0x7FC0U)));
    ECLAIR_CHECK(std::isnan(decode_bf16(0x7F81U)));
    ECLAIR_CHECK(decode_bf16(0x8000U) == 0.0F && std::signbit(decode_bf16(0x8000U)));

    ECLAIR_CHECK(encode_bf16(1.0F) == 0x3F80U);
    ECLAIR_CHECK(encode_bf16(-0.0F) == 0x8000U);
    // Ties go to the even code. 1 + 2^-8 is halfway between 1 and 1 + 2^-7.
    ECLAIR_CHECK(encode_bf16(1.0F + std::ldexp(1.0F, -8)) == 0x3F80U);
    ECLAIR_CHECK(encode_bf16(1.0F + 3.0F * std::ldexp(1.0F, -8)) == 0x3F82U);
    ECLAIR_CHECK(encode_bf16(1.0F + std::ldexp(1.0F, -8) + std::ldexp(1.0F, -23)) == 0x3F81U);
    // Overflow starts at the midpoint between the largest value and 2^128.
    ECLAIR_CHECK(encode_bf16(from_bits(0x7F7F7FFFU)) == 0x7F7FU);
    ECLAIR_CHECK(encode_bf16(from_bits(0x7F7F8000U)) == 0x7F80U);
    ECLAIR_CHECK(encode_bf16(std::numeric_limits<float>::max()) == 0x7F80U);
    ECLAIR_CHECK(encode_bf16(-std::numeric_limits<float>::max()) == 0xFF80U);
    ECLAIR_CHECK(encode_bf16(kInfinity) == 0x7F80U);
    // Gradual underflow: half the smallest subnormal ties to zero, and 1.5
    // units ties to the even code of two units.
    ECLAIR_CHECK(encode_bf16(from_bits(0x00008000U)) == 0x0000U);
    ECLAIR_CHECK(encode_bf16(from_bits(0x00008001U)) == 0x0001U);
    ECLAIR_CHECK(encode_bf16(from_bits(0x00018000U)) == 0x0002U);
    ECLAIR_CHECK(encode_bf16(from_bits(0x807FFFFFU)) == 0x8080U);
    // NaN stays NaN with its sign, including a signaling NaN whose payload is
    // entirely in the dropped half, which truncation would turn into infinity.
    ECLAIR_CHECK(encode_bf16(std::numeric_limits<float>::quiet_NaN()) == 0x7FC0U);
    ECLAIR_CHECK(encode_bf16(-std::numeric_limits<float>::quiet_NaN()) == 0xFFC0U);
    ECLAIR_CHECK(encode_bf16(from_bits(0x7F800001U)) == 0x7FC0U);
    ECLAIR_CHECK(encode_bf16(from_bits(0xFFFFFFFFU)) == 0xFFC0U);
}

void check_every_pattern() {
    for (std::uint32_t bits = 0; bits < 65536U; ++bits) {
        const auto pattern = static_cast<std::uint16_t>(bits);
        const float value = decode_bf16(pattern);
        const bool nan_pattern = (bits & 0x7F80U) == 0x7F80U && (bits & 0x007FU) != 0U;
        ECLAIR_CHECK(std::isnan(value) == nan_pattern);
        if (!nan_pattern) {
            ECLAIR_CHECK(encode_bf16(value) == pattern);
        }
    }
}

// Nearest bfloat16 by exact distance, independently of the carry trick. Both
// neighbors and the input are exact in double, and so are their differences.
// Past the largest finite value, the upper neighbor stands for 2^128, which
// is where IEEE rounding places the overflow threshold.
[[nodiscard]] std::uint16_t nearest_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const auto lower = static_cast<std::uint16_t>((bits >> 16U) & 0x7FFFU);
    const double magnitude = std::fabs(static_cast<double>(value));
    const double lower_value = std::fabs(static_cast<double>(decode_bf16(lower)));
    const double upper_value =
        lower == 0x7F7FU ? std::ldexp(1.0, 128)
                         : static_cast<double>(decode_bf16(static_cast<std::uint16_t>(lower + 1U)));
    const double below = magnitude - lower_value;
    const double above = upper_value - magnitude;
    const bool round_up = above < below || (above == below && (lower & 1U) != 0U);
    return static_cast<std::uint16_t>(sign | (round_up ? lower + 1U : lower));
}

// Every exponent and kept fraction, each with the dropped halves that sit at
// or next to the rounding decisions.
void check_rounding_against_oracle() {
    for (std::uint32_t high = 0; high < 65536U; ++high) {
        if ((high & 0x7F80U) == 0x7F80U) {
            continue;
        }
        for (const std::uint32_t low : {0x0000U, 0x0001U, 0x7FFFU, 0x8000U, 0x8001U, 0xFFFFU}) {
            const float value = from_bits((high << 16U) | low);
            ECLAIR_CHECK(encode_bf16(value) == nearest_bf16(value));
        }
    }
}

// The BF16 value times 2^shift rounded to binary16, from exact double
// arithmetic and encode_fp16 rather than the integer path under test. A BF16
// value has 8 significant bits, so the scaled double is exact, and between
// 2^-26 and 65520 it is also an exact float in encode_fp16's input range.
// Below 2^-26 every value rounds to zero, and from 65520, the midpoint between
// 65504 and 2^16, to infinity.
[[nodiscard]] std::uint16_t scaled_oracle(std::uint16_t bits, int shift) {
    const auto sign = static_cast<std::uint16_t>(bits & 0x8000U);
    const float value = decode_bf16(bits);
    if (std::isnan(value)) {
        return static_cast<std::uint16_t>(sign | 0x7E00U);
    }
    const double scaled = std::ldexp(static_cast<double>(value), shift);
    if (std::fabs(scaled) >= 65520.0) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    if (std::fabs(scaled) < std::ldexp(1.0, -26)) {
        return sign;
    }
    return eclair::formats::encode_fp16(static_cast<float>(scaled));
}

void check_scaled_known_values() {
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x3F80U, 0) == 0x3C00U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x3F80U, 15) == 0x7800U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x3F80U, 16) == 0x7C00U);
    // The largest BF16 value lands on 65280 in binary16's top binade.
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x7F7FU, -112) == 0x7BF8U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0xFF7FU, -111) == 0xFC00U);
    // The smallest BF16 subnormal, 2^-133, reaches the binary16 subnormals.
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x0001U, 109) == 0x0001U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x0001U, 148) == 0x7800U);
    // Rounding to units of 2^-24: half a unit ties to zero, 0.75 rounds up,
    // just above half rounds up, 1.5 units ties to two.
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x3F80U, -24) == 0x0001U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x3F80U, -25) == 0x0000U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x3FC0U, -25) == 0x0001U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x3F81U, -25) == 0x0001U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x4040U, -25) == 0x0002U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0xBF80U, -26) == 0x8000U);
    // 255 * 2^-25 is 127.5 units, a tie that rounds to the even 128.
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x3FFFU, -18) == 0x0080U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x8000U, 40) == 0x8000U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x7F80U, -1000) == 0x7C00U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0xFF80U, 1000) == 0xFC00U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0x7FC1U, 0) == 0x7E00U);
    ECLAIR_CHECK(bf16_to_fp16_scaled_bits(0xFF81U, 0) == 0xFE00U);
    // Extreme shifts are clamped without changing a result.
    for (const std::uint16_t bits : {std::uint16_t{0x0001U}, std::uint16_t{0x7F7FU},
                                     std::uint16_t{0x8080U}, std::uint16_t{0x3F80U}}) {
        ECLAIR_CHECK(bf16_to_fp16_scaled_bits(bits, std::numeric_limits<std::int32_t>::max()) ==
                     scaled_oracle(bits, 4096));
        ECLAIR_CHECK(bf16_to_fp16_scaled_bits(bits, std::numeric_limits<std::int32_t>::min()) ==
                     scaled_oracle(bits, -4096));
    }
}

// Every BF16 pattern at every shift that can matter, and some past the clamp.
void check_scaled_every_pattern() {
    for (std::uint32_t bits = 0; bits < 65536U; ++bits) {
        const auto pattern = static_cast<std::uint16_t>(bits);
        for (int shift = -170; shift <= 170; ++shift) {
            if (bf16_to_fp16_scaled_bits(pattern, shift) != scaled_oracle(pattern, shift)) {
                std::fprintf(stderr, "bits=0x%04X shift=%d\n", static_cast<unsigned int>(bits),
                             shift);
                ECLAIR_CHECK(false && "scaled conversion differs from the oracle");
                return;
            }
        }
    }
}

// The row shift puts every finite nonzero magnitude in [2^15, 2^16), where
// the conversion is exact, and ignores infinity and NaN.
void check_row_shift() {
    ECLAIR_CHECK(bf16_fp16_shift(0U) == 0);
    ECLAIR_CHECK(bf16_finite_magnitude(0xFF7FU) == 0x7F7FU);
    ECLAIR_CHECK(bf16_finite_magnitude(0x8001U) == 0x0001U);
    ECLAIR_CHECK(bf16_finite_magnitude(0x7F80U) == 0U);
    ECLAIR_CHECK(bf16_finite_magnitude(0xFFC0U) == 0U);
    for (std::uint32_t magnitude = 1; magnitude < 0x7F80U; ++magnitude) {
        const auto bits = static_cast<std::uint16_t>(magnitude);
        const std::int32_t shift = bf16_fp16_shift(bits);
        const double scaled = std::ldexp(static_cast<double>(decode_bf16(bits)), shift);
        const std::uint16_t converted = bf16_to_fp16_scaled_bits(bits, shift);
        if (scaled < 32768.0 || scaled >= 65536.0 ||
            static_cast<double>(eclair::formats::decode_fp16(converted)) != scaled) {
            std::fprintf(stderr, "magnitude=0x%04X shift=%d\n",
                         static_cast<unsigned int>(magnitude), static_cast<int>(shift));
            ECLAIR_CHECK(false && "row shift misses binary16's top binade");
            return;
        }
    }
}

}  // namespace

int main() {
    check_known_values();
    check_every_pattern();
    check_rounding_against_oracle();
    check_scaled_known_values();
    check_scaled_every_pattern();
    check_row_shift();
    return eclair::test::finish();
}
