/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/formats/fp16.hpp"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

int main() {
    using eclair::formats::decode_fp16;
    using eclair::formats::encode_fp16;
    ECLAIR_CHECK(decode_fp16(0x3C00U) == 1.0F);
    ECLAIR_CHECK(decode_fp16(0x0001U) == std::ldexp(1.0F, -24));
    ECLAIR_CHECK(decode_fp16(0x03FFU) == std::ldexp(1023.0F, -24));
    ECLAIR_CHECK(decode_fp16(0x0400U) == std::ldexp(1.0F, -14));
    ECLAIR_CHECK(decode_fp16(0x7BFFU) == 65504.0F);
    ECLAIR_CHECK(std::isinf(decode_fp16(0x7C00U)));
    ECLAIR_CHECK(std::isnan(decode_fp16(0x7C01U)));
    ECLAIR_CHECK(std::signbit(decode_fp16(0x8000U)));
    ECLAIR_CHECK(encode_fp16(0.0F) == 0x0000U);
    ECLAIR_CHECK(encode_fp16(-0.0F) == 0x8000U);
    ECLAIR_CHECK(encode_fp16(1.0F + std::ldexp(1.0F, -11)) == 0x3C00U);
    ECLAIR_CHECK(encode_fp16(1.0F + 3.0F * std::ldexp(1.0F, -11)) == 0x3C02U);
    ECLAIR_CHECK(encode_fp16(std::ldexp(1.0F, -25)) == 0x0000U);
    ECLAIR_CHECK(encode_fp16(-std::ldexp(1.0F, -25)) == 0x8000U);
    ECLAIR_CHECK(encode_fp16(std::nextafter(std::ldexp(1.0F, -25), 1.0F)) == 0x0001U);
    ECLAIR_CHECK(encode_fp16(std::ldexp(1023.5F, -24)) == 0x0400U);
    ECLAIR_CHECK(encode_fp16(65519.0F) == 0x7BFFU);
    ECLAIR_CHECK(encode_fp16(65520.0F) == 0x7C00U);
    ECLAIR_CHECK(encode_fp16(-65520.0F) == 0xFC00U);
    ECLAIR_CHECK(encode_fp16(std::numeric_limits<float>::denorm_min()) == 0U);
    ECLAIR_CHECK(encode_fp16(std::numeric_limits<float>::infinity()) == 0x7C00U);
    ECLAIR_CHECK((encode_fp16(std::numeric_limits<float>::quiet_NaN()) & 0x7FFFU) == 0x7E00U);
    for (std::uint32_t bits = 0; bits < 65536U; ++bits) {
        const auto half = static_cast<std::uint16_t>(bits);
        const float decoded = decode_fp16(half);
        if (!std::isnan(decoded)) {
            ECLAIR_CHECK(encode_fp16(decoded) == half);
        }
    }
    for (const auto format :
         {eclair::formats::Fp8Format::E4M3FN, eclair::formats::Fp8Format::E5M2}) {
        for (unsigned int bits = 0; bits < 256U; ++bits) {
            const auto fp8 = static_cast<std::uint8_t>(bits);
            const float oracle = eclair::formats::decode_fp8(fp8, format);
            const auto half = eclair::formats::fp8_to_fp16_bits(fp8, format);
            ECLAIR_CHECK(std::isnan(oracle) ? std::isnan(decode_fp16(half))
                                            : half == encode_fp16(oracle));
        }
    }
    return eclair::test::finish();
}
