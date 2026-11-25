/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/formats/fp8.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace eclair::formats {
namespace {

using detail::Fp8Layout;
using detail::layout_for;

[[nodiscard]] bool is_nan_encoding(std::uint8_t bits, Fp8Format format) noexcept {
    if (format == Fp8Format::E4M3FN) {
        return (bits & 0x7FU) == 0x7FU;
    }
    return (bits & 0x7CU) == 0x7CU && (bits & 0x03U) != 0U;
}

[[nodiscard]] bool is_even_candidate(std::uint8_t bits) noexcept {
    return (bits & 1U) == 0U;
}

}  // namespace

// Subnormals are 0.m * 2^(1 - bias) and normals 1.m * 2^(e - bias). The
// all-ones exponent holds E5M2's infinities and NaNs, and E4M3FN's NaN only
// when the mantissa is all ones too. NaN is returned without a sign.
float decode_fp8(std::uint8_t bits, Fp8Format format) noexcept {
    const Fp8Layout layout = layout_for(format);
    const unsigned int magnitude_bits = bits & 0x7FU;
    const unsigned int exponent = magnitude_bits >> layout.mantissa_bits;
    const unsigned int mantissa = magnitude_bits & ((1U << layout.mantissa_bits) - 1U);
    const unsigned int max_exponent = (1U << layout.exponent_bits) - 1U;
    if (is_nan_encoding(bits, format)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float magnitude = 0.0F;
    if (format == Fp8Format::E5M2 && exponent == max_exponent) {
        magnitude = std::numeric_limits<float>::infinity();
    } else if (exponent == 0U) {
        magnitude =
            std::ldexp(static_cast<float>(mantissa), 1 - layout.bias - layout.mantissa_bits);
    } else {
        const float significand =
            1.0F + static_cast<float>(mantissa) / static_cast<float>(1U << layout.mantissa_bits);
        magnitude = std::ldexp(significand, static_cast<int>(exponent) - layout.bias);
    }
    return (bits & 0x80U) == 0U ? magnitude : -magnitude;
}

std::uint8_t encode_fp8(float value, Fp8Format format) noexcept {
    const std::uint8_t sign = std::signbit(value) ? 0x80U : 0U;
    if (std::isnan(value)) {
        return static_cast<std::uint8_t>(sign | 0x7FU);
    }
    if (std::isinf(value) && format == Fp8Format::E5M2) {
        return static_cast<std::uint8_t>(sign | 0x7CU);
    }
    if (value == 0.0F) {
        return sign;
    }
    const Fp8Layout layout = layout_for(format);
    const float magnitude = std::fabs(value);
    // Decide saturation before searching. Far above the largest value, float
    // distances to the top candidates round to one number, and the tie rule
    // would then choose a smaller even candidate.
    if (magnitude >= decode_fp8(layout.max_finite, format)) {
        return static_cast<std::uint8_t>(sign | layout.max_finite);
    }

    // Below the largest value, the distances to the two values bracketing the
    // magnitude are exact in float, so this exhaustive search is exactly round
    // to nearest even. It stays deliberately simple: it is the oracle.
    std::uint8_t best = 0U;
    float best_distance = std::numeric_limits<float>::infinity();
    for (std::uint16_t candidate = 0; candidate < 128U; ++candidate) {
        const auto bits = static_cast<std::uint8_t>(candidate);
        if (is_nan_encoding(bits, format)) {
            continue;
        }
        const float representable = decode_fp8(bits, format);
        if (!std::isfinite(representable)) {
            continue;
        }
        const float distance = std::fabs(magnitude - representable);
        if (distance < best_distance || (distance == best_distance && is_even_candidate(bits))) {
            best = bits;
            best_distance = distance;
        }
    }
    return static_cast<std::uint8_t>(sign | best);
}

}  // namespace eclair::formats
