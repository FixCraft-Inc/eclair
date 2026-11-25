/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/formats/bf16.hpp"

#include <bit>
#include <cstdint>
#include <limits>

namespace eclair::formats {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == sizeof(std::uint32_t));

float decode_bf16(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(bf16_to_fp32_bits(bits));
}

std::uint16_t encode_bf16(float value) noexcept {
    return fp32_bits_to_bf16(std::bit_cast<std::uint32_t>(value));
}

}  // namespace eclair::formats
