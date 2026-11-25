/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/formats/fp16.hpp"

#include <bit>
#include <cmath>
#include <limits>

namespace eclair::formats {

float decode_fp16(std::uint16_t bits) noexcept {
    const auto exponent = static_cast<unsigned int>((bits >> 10U) & 31U);
    const auto fraction = static_cast<unsigned int>(bits & 1023U);
    float magnitude = 0.0F;
    if (exponent == 0U) {
        magnitude = std::ldexp(static_cast<float>(fraction), -24);
    } else if (exponent == 31U) {
        magnitude = fraction == 0U ? std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0F + static_cast<float>(fraction) / 1024.0F,
                               static_cast<int>(exponent) - 15);
    }
    return (bits & 0x8000U) == 0U ? magnitude : -magnitude;
}

std::uint16_t encode_fp16(float value) noexcept {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 255U;
    const std::uint32_t fraction = bits & 0x7FFFFFU;
    if (exponent == 255U) {
        return static_cast<std::uint16_t>(sign | (fraction == 0U ? 0x7C00U : 0x7E00U));
    }
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    if (half_exponent < -10) {
        return static_cast<std::uint16_t>(sign);
    }
    const unsigned int shift =
        half_exponent <= 0 ? static_cast<unsigned int>(14 - half_exponent) : 13U;
    const std::uint32_t significand = half_exponent <= 0 ? fraction | 0x800000U : fraction;
    std::uint32_t rounded = significand >> shift;
    const std::uint32_t remainder = significand & ((1U << shift) - 1U);
    const std::uint32_t midpoint = 1U << (shift - 1U);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1U) != 0U)) {
        ++rounded;
    }
    // Carry from the fraction naturally increments the exponent, including
    // subnormal-to-normal and finite-to-infinity transitions.
    const std::uint32_t encoded_exponent =
        half_exponent <= 0 ? 0U : static_cast<std::uint32_t>(half_exponent) << 10U;
    return static_cast<std::uint16_t>(sign | (encoded_exponent + rounded));
}

}  // namespace eclair::formats
