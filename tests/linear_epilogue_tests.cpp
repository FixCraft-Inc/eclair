/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/host/linear_epilogue.hpp"
#include "eclair/ops/linear_epilogue.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using eclair::host::linear_epilogue_reference;
using eclair::ops::LinearEpilogueBuffers;
using eclair::ops::LinearEpilogueShape;
using eclair::ops::Status;
using eclair::ops::validate_linear_epilogue;

constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
constexpr float kInfinity = std::numeric_limits<float>::infinity();

[[nodiscard]] bool is_half_nan(std::uint16_t bits) {
    return (bits & 0x7C00U) == 0x7C00U && (bits & 0x03FFU) != 0U;
}

// Expected bits come from the binary16 specification, not from encode_fp16.
struct Case {
    float accumulator;
    float scale;
    bool has_bias;
    float bias;
    std::uint16_t expected;
};

void check_single_values() {
    const std::array<Case, 19> cases{{
        {1.0F, 0.5F, true, 0.25F, 0x3A00U},
        {-2.0F, 0.5F, false, 0.0F, 0xBC00U},
        // Signed zero survives only without an addition.
        {-0.0F, 1.0F, false, 0.0F, 0x8000U},
        {-0.0F, 1.0F, true, 0.0F, 0x0000U},
        {0.0F, -1.0F, false, 0.0F, 0x8000U},
        // Overflow: 65520 is the tie between 65504 and 65536 and rounds to even.
        {65520.0F, 1.0F, false, 0.0F, 0x7C00U},
        {65519.0F, 1.0F, false, 0.0F, 0x7BFFU},
        {-65520.0F, 1.0F, false, 0.0F, 0xFC00U},
        {1.0e30F, 1.0e30F, false, 0.0F, 0x7C00U},
        // Gradual underflow and ties to even in the subnormal range.
        {std::ldexp(1.0F, -24), 1.0F, false, 0.0F, 0x0001U},
        {std::ldexp(1.0F, -25), 1.0F, false, 0.0F, 0x0000U},
        {std::ldexp(3.0F, -25), 1.0F, false, 0.0F, 0x0002U},
        // Normal ties to even.
        {1.0F + std::ldexp(1.0F, -11), 1.0F, false, 0.0F, 0x3C00U},
        {1.0F + std::ldexp(3.0F, -11), 1.0F, false, 0.0F, 0x3C02U},
        // The product rounds to FP32 before the bias: the exact sum 2^-25 is a
        // binary16 tie that rounds to zero. A fused multiply-add would keep
        // 2^-25 + 2^-48 and round to the smallest subnormal, 0x0001.
        {0x1.000002p-3F, 0x1.000004p+0F, true, -0x1.000002p-3F, 0x0000U},
        {kInfinity, 1.0F, false, 0.0F, 0x7C00U},
        {kInfinity, 1.0F, true, -kInfinity, 0x7E00U},
        {kInfinity, 0.0F, false, 0.0F, 0x7E00U},
        {kNan, 1.0F, true, 1.0F, 0x7E00U},
    }};
    for (const Case& value : cases) {
        std::uint16_t output = 0x5555U;
        const LinearEpilogueBuffers buffers{{&value.accumulator, 1U},
                                            {&value.scale, 1U},
                                            {&value.bias, value.has_bias ? 1U : 0U},
                                            {&output, 1U}};
        ECLAIR_CHECK(linear_epilogue_reference({1U, 1U}, buffers) == Status::Success);
        if (is_half_nan(value.expected)) {
            ECLAIR_CHECK(is_half_nan(output));
        } else {
            ECLAIR_CHECK(output == value.expected);
        }
    }
}

void check_matrix_and_bias_broadcast() {
    const std::array<float, 6> accumulators{1.0F, 2.0F, 3.0F, -1.0F, -2.0F, -3.0F};
    const std::array<float, 3> bias{0.5F, -1.0F, 2.0F};
    const float scale = 2.0F;
    std::array<std::uint16_t, 6> output{};
    const LinearEpilogueBuffers buffers{{accumulators.data(), accumulators.size()},
                                        {&scale, 1U},
                                        {bias.data(), bias.size()},
                                        {output.data(), output.size()}};
    ECLAIR_CHECK(linear_epilogue_reference({2U, 3U}, buffers) == Status::Success);
    // 2.5, 3, 8, -1.5, -5, -4
    const std::array<std::uint16_t, 6> expected{0x4100U, 0x4200U, 0x4800U,
                                                0xBE00U, 0xC500U, 0xC400U};
    ECLAIR_CHECK(output == expected);
}

void check_nan_scale() {
    const std::array<float, 2> accumulators{0.0F, 1.0F};
    const float scale = kNan;
    std::array<std::uint16_t, 2> output{};
    const LinearEpilogueBuffers buffers{
        {accumulators.data(), 2U}, {&scale, 1U}, {}, {output.data(), 2U}};
    ECLAIR_CHECK(linear_epilogue_reference({1U, 2U}, buffers) == Status::Success);
    ECLAIR_CHECK(is_half_nan(output[0]) && is_half_nan(output[1]));
}

void check_validation() {
    std::vector<float> accumulators(6, 1.0F);
    const float scale = 1.0F;
    std::vector<float> bias(3, 0.0F);
    std::vector<std::uint16_t> output(6, 0U);
    const LinearEpilogueBuffers valid{
        {accumulators.data(), 6U}, {&scale, 1U}, {bias.data(), 3U}, {output.data(), 6U}};
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, valid) == Status::Success);
    ECLAIR_CHECK(validate_linear_epilogue({0U, 3U}, {}) == Status::Success);
    ECLAIR_CHECK(validate_linear_epilogue({2U, 0U}, {}) == Status::Success);

    auto invalid = valid;
    invalid.accumulators.size = 5U;
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::BufferTooSmall);
    invalid = valid;
    invalid.scale.size = 0U;
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::BufferTooSmall);
    invalid = valid;
    invalid.bias.size = 2U;
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::BufferTooSmall);
    invalid = valid;
    invalid.output.size = 5U;
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::BufferTooSmall);
    invalid = valid;
    invalid.output.data = nullptr;
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::InvalidArgument);
    invalid = valid;
    invalid.scale.data = nullptr;
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::InvalidArgument);

    // Output may not overlap any input it reads.
    invalid = valid;
    invalid.output = {reinterpret_cast<std::uint16_t*>(accumulators.data()), 6U};
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::InvalidArgument);
    invalid = valid;
    invalid.output = {reinterpret_cast<std::uint16_t*>(bias.data()), 6U};
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::InvalidArgument);

    // Misaligned binary16 output.
    std::array<std::uint32_t, 8> raw{};
    invalid = valid;
    invalid.output = {
        reinterpret_cast<std::uint16_t*>(reinterpret_cast<unsigned char*>(raw.data()) + 1), 6U};
    ECLAIR_CHECK(validate_linear_epilogue({2U, 3U}, invalid) == Status::InvalidArgument);

    const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2U;
    ECLAIR_CHECK(validate_linear_epilogue({huge, 3U}, valid) == Status::SizeOverflow);

    // Rejection leaves output unchanged.
    std::vector<std::uint16_t> guard(6, 0x1234U);
    invalid = valid;
    invalid.output = {guard.data(), 6U};
    invalid.bias.size = 1U;
    ECLAIR_CHECK(linear_epilogue_reference({2U, 3U}, invalid) == Status::BufferTooSmall);
    for (const std::uint16_t value : guard) {
        ECLAIR_CHECK(value == 0x1234U);
    }
}

}  // namespace

int main() {
    check_single_values();
    check_matrix_and_bias_broadcast();
    check_nan_scale();
    check_validation();
    return eclair::test::finish();
}
