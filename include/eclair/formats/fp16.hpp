/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/formats/fp8.hpp"
#include "eclair/formats/host_device.hpp"

#include <cstdint>

namespace eclair::formats {

// IEEE binary16 interchange bits. Conversion uses nearest-even rounding,
// gradual underflow and infinity on overflow. NaN payloads are not preserved.
[[nodiscard]] float decode_fp16(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t encode_fp16(float value) noexcept;

// Exact FP8 widening. NaN payload equality is not required. The caller
// validates the format. Tests compare it with decode_fp8's float oracle.
[[nodiscard]] ECLAIR_HOST_DEVICE constexpr std::uint16_t fp8_to_fp16_bits(
    std::uint8_t bits, Fp8Format format) noexcept {
    const auto sign = static_cast<std::uint16_t>((bits & 0x80U) << 8U);
    if (format == Fp8Format::E5M2) {
        return static_cast<std::uint16_t>(static_cast<std::uint16_t>(bits) << 8U);
    }
    const unsigned int exponent = (bits >> 3U) & 15U;
    const unsigned int fraction = bits & 7U;
    if (exponent == 15U && fraction == 7U) {
        return static_cast<std::uint16_t>(sign | 0x7E00U);
    }
    if (exponent != 0U) {
        return static_cast<std::uint16_t>(sign | ((exponent + 8U) << 10U) | (fraction << 7U));
    }
    if (fraction == 0U) {
        return sign;
    }
    const unsigned int shift = fraction >= 4U ? 2U : (fraction >= 2U ? 1U : 0U);
    return static_cast<std::uint16_t>(sign | ((6U + shift) << 10U) |
                                      ((fraction - (1U << shift)) << (10U - shift)));
}

}  // namespace eclair::formats
