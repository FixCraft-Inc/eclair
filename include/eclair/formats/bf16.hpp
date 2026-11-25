/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/formats/host_device.hpp"

#include <cstdint>

// bfloat16 is the upper half of an IEEE binary32 value: 1 sign, 8 exponent and
// 7 fraction bits. It has binary32's range, subnormals included, with 8
// significant bits.
namespace eclair::formats {

// Exact widening to binary32 bits.
[[nodiscard]] ECLAIR_HOST_DEVICE constexpr std::uint32_t bf16_to_fp32_bits(
    std::uint16_t bits) noexcept {
    return static_cast<std::uint32_t>(bits) << 16U;
}

// Round binary32 bits to bfloat16, nearest even. Overflow rounds to infinity
// and binary32 subnormals round to bfloat16 subnormals. NaN keeps its sign and
// becomes the quiet NaN S.11111111.1000000. Payloads are not preserved.
[[nodiscard]] ECLAIR_HOST_DEVICE constexpr std::uint16_t fp32_bits_to_bf16(
    std::uint32_t bits) noexcept {
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) {
        return static_cast<std::uint16_t>(((bits >> 16U) & 0x8000U) | 0x7FC0U);
    }
    // Adding just under half a unit, plus the kept low bit, carries into the
    // kept bits exactly when the dropped half is above a half, or exactly a
    // half with an odd kept value. A carry out of the fraction increments the
    // exponent, up to infinity. A finite magnitude cannot carry into the sign.
    const std::uint32_t kept_low_bit = (bits >> 16U) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7FFFU + kept_low_bit) >> 16U);
}

// Host conveniences over the bit functions above.
[[nodiscard]] float decode_bf16(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t encode_bf16(float value) noexcept;

// Scaling BF16 onto binary16. A power-of-two scale changes only the exponent,
// and a BF16 value has 8 significant bits to binary16's 11, so any BF16 value
// scaled into [2^-17, 2^16) converts exactly. Scaling a row so that its
// largest magnitude lands in [2^15, 2^16) therefore keeps exact every element
// whose exponent is at most 32 below the largest one's. Smaller elements
// round to binary16 subnormals or zero.

// Shifts beyond this bound map every finite nonzero BF16 value to zero or
// infinity in binary16 (BF16 magnitudes lie in [2^-133, 2^128)), so clamping
// a shift to it never changes a result.
inline constexpr std::int32_t kBf16Fp16ShiftLimit = 160;

[[nodiscard]] ECLAIR_HOST_DEVICE constexpr std::int32_t clamp_bf16_fp16_shift(
    std::int32_t shift) noexcept {
    return shift < -kBf16Fp16ShiftLimit
               ? -kBf16Fp16ShiftLimit
               : (shift > kBf16Fp16ShiftLimit ? kBf16Fp16ShiftLimit : shift);
}

// The magnitude bits of a finite value, or 0 for infinity and NaN. Finite
// magnitude bits order like the values, so their maximum finds a row's
// largest finite magnitude.
[[nodiscard]] ECLAIR_HOST_DEVICE constexpr std::uint16_t bf16_finite_magnitude(
    std::uint16_t bits) noexcept {
    const auto magnitude = static_cast<std::uint16_t>(bits & 0x7FFFU);
    return magnitude < 0x7F80U ? magnitude : std::uint16_t{0};
}

// The shift that puts `largest`, a row's largest finite magnitude from
// bf16_finite_magnitude, in [2^15, 2^16). The largest BF16 significand there
// is 65280, below binary16's 65504. A row without a finite nonzero value
// (largest == 0) gets shift 0.
[[nodiscard]] ECLAIR_HOST_DEVICE constexpr std::int32_t bf16_fp16_shift(
    std::uint16_t largest) noexcept {
    if (largest == 0U) {
        return 0;
    }
    const std::uint32_t field = largest >> 7U;
    if (field != 0U) {
        // Exponent field - 127 is the leading bit's power, which moves to 15.
        return 142 - static_cast<std::int32_t>(field);
    }
    // A subnormal is fraction * 2^-133, with its leading bit at position 0-6.
    std::int32_t leading = 6;
    while ((largest >> static_cast<std::uint32_t>(leading)) == 0U) {
        --leading;
    }
    return 148 - leading;
}

// binary16 bits of the BF16 value times 2^shift, rounded to nearest even with
// gradual underflow and overflow to infinity. Infinity stays infinite, zero
// keeps its sign, and NaN becomes the quiet NaN S.11111.1000000000.
[[nodiscard]] ECLAIR_HOST_DEVICE constexpr std::uint16_t bf16_to_fp16_scaled_bits(
    std::uint16_t bits, std::int32_t shift) noexcept {
    const std::uint32_t sign = bits & 0x8000U;
    const std::uint32_t magnitude = bits & 0x7FFFU;
    if (magnitude >= 0x7F80U) {
        return static_cast<std::uint16_t>(sign | (magnitude == 0x7F80U ? 0x7C00U : 0x7E00U));
    }
    if (magnitude == 0U) {
        return static_cast<std::uint16_t>(sign);
    }
    // The value is significand * 2^power, with the significand's leading bit
    // moved to position 10 where binary16 keeps its implicit bit.
    const std::uint32_t field = magnitude >> 7U;
    std::uint32_t significand = 0;
    std::int32_t power = clamp_bf16_fp16_shift(shift);
    if (field != 0U) {
        significand = (0x80U | (magnitude & 0x7FU)) << 3U;
        power += static_cast<std::int32_t>(field) - 137;
    } else {
        significand = magnitude;
        power -= 133;
        while (significand < 0x400U) {
            significand <<= 1U;
            --power;
        }
    }
    // significand * 2^power = 1.f * 2^(power + 10), so the biased exponent is
    // power + 25. A normal result needs no rounding.
    const std::int32_t exponent = power + 25;
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    if (exponent >= 1) {
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10U) |
                                          (significand - 0x400U));
    }
    // Subnormal results count units of 2^-24, dropping 1 - exponent bits.
    // Below half a unit (more than 11 dropped bits) the result is zero.
    const auto dropped = static_cast<std::uint32_t>(1 - exponent);
    if (dropped > 11U) {
        return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t kept = significand >> dropped;
    const std::uint32_t remainder = significand & ((1U << dropped) - 1U);
    const std::uint32_t half = 1U << (dropped - 1U);
    const bool round_up = remainder > half || (remainder == half && (kept & 1U) != 0U);
    return static_cast<std::uint16_t>(sign | (kept + (round_up ? 1U : 0U)));
}

}  // namespace eclair::formats
