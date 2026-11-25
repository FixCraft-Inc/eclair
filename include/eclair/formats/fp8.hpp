/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/formats/host_device.hpp"

#include <cstdint>

namespace eclair::formats {

// OCP 8-bit floating-point interchange formats.
enum class Fp8Format : std::uint8_t {
    E4M3FN,  // Bias 7, finite only: S.1111.111 is NaN, the largest value is 448.
    E5M2,    // Bias 15, IEEE-style infinities and NaNs, the largest finite is 57344.
};

namespace detail {

struct Fp8Layout {
    int mantissa_bits;
    int exponent_bits;
    int bias;
    std::uint8_t max_finite;  // Bits of the largest finite magnitude.
};

[[nodiscard]] ECLAIR_HOST_DEVICE constexpr Fp8Layout layout_for(Fp8Format format) noexcept {
    if (format == Fp8Format::E4M3FN) {
        return {.mantissa_bits = 3, .exponent_bits = 4, .bias = 7, .max_finite = 0x7EU};
    }
    return {.mantissa_bits = 2, .exponent_bits = 5, .bias = 15, .max_finite = 0x7BU};
}

}  // namespace detail

// Portable reference decode, evaluated from the format definition.
[[nodiscard]] float decode_fp8(std::uint8_t bits, Fp8Format format) noexcept;

// Portable reference encode: the nearest finite value, ties to even. Finite
// magnitudes past the largest finite value saturate to it in both formats.
// Infinity saturates in E4M3FN and stays infinite in E5M2. NaN keeps its sign
// and encodes as S.1111111. This search is the oracle for fp32_bits_to_fp8.
[[nodiscard]] std::uint8_t encode_fp8(float value, Fp8Format format) noexcept;

// IEEE binary32 bits to FP8 bits with exactly encode_fp8's results. Integer
// arithmetic only, so host code and CUDA kernels share this definition.
[[nodiscard]] ECLAIR_HOST_DEVICE constexpr std::uint8_t fp32_bits_to_fp8(
    std::uint32_t bits, Fp8Format format) noexcept {
    const auto sign = static_cast<std::uint8_t>((bits >> 24U) & 0x80U);
    const std::uint32_t magnitude = bits & 0x7FFFFFFFU;
    if (magnitude > 0x7F800000U) {
        return static_cast<std::uint8_t>(sign | 0x7FU);
    }
    if (magnitude == 0x7F800000U && format == Fp8Format::E5M2) {
        return static_cast<std::uint8_t>(sign | 0x7CU);
    }
    const detail::Fp8Layout layout = detail::layout_for(format);
    const int exponent = static_cast<int>(magnitude >> 23U) - 127;
    const int min_normal_exponent = 1 - layout.bias;
    // Past the binade of the largest finite value everything saturates,
    // including E4M3FN's infinity.
    if (exponent > (layout.max_finite >> layout.mantissa_bits) - layout.bias) {
        return static_cast<std::uint8_t>(sign | layout.max_finite);
    }
    // Below half the smallest subnormal, which covers binary32 zeros and
    // subnormals, the nearest value is zero.
    if (exponent < min_normal_exponent - layout.mantissa_bits - 1) {
        return sign;
    }
    // Round the significand to this binade's FP8 spacing. A normal value drops
    // the implicit bit, so a carry out of the fraction increments the exponent
    // field, and a subnormal that rounds up to 2^(1-bias) becomes the smallest
    // normal. The code's low bit is the fraction's, which decides ties.
    const bool normal = exponent >= min_normal_exponent;
    const std::uint32_t fraction = magnitude & 0x7FFFFFU;
    const std::uint32_t significand = normal ? fraction : (fraction | 0x800000U);
    const auto shift = static_cast<std::uint32_t>(23 - layout.mantissa_bits +
                                                  (normal ? 0 : min_normal_exponent - exponent));
    std::uint32_t rounded = significand >> shift;
    const std::uint32_t remainder = significand & ((1U << shift) - 1U);
    const std::uint32_t midpoint = 1U << (shift - 1U);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1U) != 0U)) {
        ++rounded;
    }
    const std::uint32_t exponent_field =
        normal ? static_cast<std::uint32_t>(exponent + layout.bias) << layout.mantissa_bits : 0U;
    const std::uint32_t code = exponent_field + rounded;
    return static_cast<std::uint8_t>(sign | (code > layout.max_finite ? layout.max_finite : code));
}

}  // namespace eclair::formats
