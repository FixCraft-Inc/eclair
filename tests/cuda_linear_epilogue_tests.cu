/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"
#include "cuda_support.hpp"

#include "eclair/cuda/linear_epilogue.hpp"
#include "eclair/host/linear_epilogue.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using eclair::ops::LinearEpilogueBuffers;
using eclair::ops::LinearEpilogueShape;
using eclair::test::checked;
using eclair::test::DeviceBuffer;
using eclair::test::Stream;

constexpr float kFmaScale = 0x1.000004p+0F;

[[nodiscard]] bool is_half_nan(std::uint16_t bits) {
    return (bits & 0x7C00U) == 0x7C00U && (bits & 0x03FFU) != 0U;
}

// Values spread over exponents that cover binary16 subnormals, normals and
// overflow, with exceptional values and rounding ties mixed in.
[[nodiscard]] float sample(std::mt19937& random) {
    const std::uint32_t choice = random() % 64U;
    switch (choice) {
        case 0U:
            return std::numeric_limits<float>::quiet_NaN();
        case 1U:
            return std::numeric_limits<float>::infinity();
        case 2U:
            return -std::numeric_limits<float>::infinity();
        case 3U:
            return -0.0F;
        case 4U:
            return 65520.0F;
        case 5U:
            return std::ldexp(3.0F, -25);
        case 6U:
            return 1.0F + std::ldexp(1.0F, -11);
        default:
            break;
    }
    const std::uint32_t exponent = 127U - 40U + random() % 60U;
    const std::uint32_t bits =
        (random() & 0x80000000U) | (exponent << 23U) | (random() & 0x007FFFFFU);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Runs the GPU epilogue on views starting `offset` elements into each device
// vector and requires bit equality with the host oracle (NaNs by class).
void check_case(std::size_t m, std::size_t n, float scale, bool with_bias, std::size_t offset,
                std::uint32_t seed) {
    std::mt19937 random(seed);
    std::vector<float> accumulators(offset + m * n);
    for (float& value : accumulators) {
        value = sample(random);
    }
    // With this scale and bias, element 0 is the host tests' FMA detector:
    // unfused rounding gives 0x0000 and a fused multiply-add gives 0x0001.
    const bool fma_probe = scale == kFmaScale && with_bias && m * n != 0U;
    if (fma_probe) {
        accumulators[offset] = 0x1.000002p-3F;
    }
    std::vector<float> bias(with_bias ? n : 0U);
    for (float& value : bias) {
        value = (random() % 8U == 0U) ? sample(random)
                                      : static_cast<float>(static_cast<int>(random() % 17U) - 8);
    }
    if (fma_probe) {
        bias[0] = -0x1.000002p-3F;
    }
    std::vector<std::uint16_t> expected(m * n);
    const LinearEpilogueBuffers host{{accumulators.data() + offset, m * n},
                                     {&scale, 1U},
                                     {bias.data(), bias.size()},
                                     {expected.data(), expected.size()}};
    ECLAIR_CHECK(eclair::host::linear_epilogue_reference({m, n}, host) ==
                 eclair::ops::Status::Success);

    Stream stream;
    const DeviceBuffer<float> device_accumulators(accumulators, stream.get());
    const DeviceBuffer<float> device_scale(std::vector<float>{scale}, stream.get());
    const DeviceBuffer<float> device_bias(bias, stream.get());
    // Sentinels on both sides of the written range detect stray stores.
    const DeviceBuffer<std::uint16_t> device_output(
        std::vector<std::uint16_t>(offset + m * n + 1U, 0x5A5AU), stream.get());
    const LinearEpilogueBuffers device{{device_accumulators.data() + offset, m * n},
                                       {device_scale.data(), 1U},
                                       {device_bias.data(), bias.size()},
                                       {device_output.data() + offset, m * n}};
    checked(eclair::cuda::launch_linear_epilogue({m, n}, device, stream.get()), "epilogue launch");
    const auto actual = device_output.download(stream.get());
    for (std::size_t i = 0; i < offset; ++i) {
        ECLAIR_CHECK(actual[i] == 0x5A5AU);
    }
    ECLAIR_CHECK(actual.back() == 0x5A5AU);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const std::uint16_t value = actual[offset + i];
        const bool same = is_half_nan(expected[i]) ? is_half_nan(value) : value == expected[i];
        if (!same && mismatches++ < 4U) {
            std::fprintf(stderr, "M=%zu N=%zu offset=%zu index=%zu actual=0x%04x expected=0x%04x\n",
                         m, n, offset, i, value, expected[i]);
        }
    }
    ECLAIR_CHECK(mismatches == 0U);
    if (fma_probe) {
        ECLAIR_CHECK(expected[0] == 0x0000U && actual[offset] == 0x0000U);
    }
    if (m * n == 0U) {
        return;
    }
    // Rejection enqueues nothing: the output keeps the successful result.
    auto invalid = device;
    invalid.output = {reinterpret_cast<std::uint16_t*>(device_accumulators.data() + offset), m * n};
    ECLAIR_CHECK(eclair::cuda::launch_linear_epilogue({m, n}, invalid, stream.get()) ==
                 cudaErrorInvalidValue);
    invalid = device;
    invalid.scale.size = 0U;
    ECLAIR_CHECK(eclair::cuda::launch_linear_epilogue({m, n}, invalid, stream.get()) ==
                 cudaErrorInvalidValue);
    ECLAIR_CHECK(device_output.download(stream.get()) == actual);
}

void run() {
    eclair::test::print_device_header();
    std::puts(
        "FP32 accumulators * scale + bias -> binary16 RNE; bit-exact against host oracle; "
        "nonblocking stream; stream synchronize; NaNs compared by class");

    ECLAIR_CHECK(eclair::cuda::launch_linear_epilogue({}, {}, nullptr) == cudaSuccess);
    ECLAIR_CHECK(eclair::cuda::launch_linear_epilogue({0U, 7U}, {}, nullptr) == cudaSuccess);
    std::uint32_t seed = 101U;
    for (const bool with_bias : {false, true}) {
        // Offset 0 with N % 4 == 0 takes the vector path. Odd N or a
        // one-element offset takes the scalar path.
        check_case(1U, 1U, 1.0F, with_bias, 0U, seed++);
        check_case(3U, 5U, 1.0F, with_bias, 0U, seed++);
        check_case(7U, 64U, 0.007F, with_bias, 0U, seed++);
        check_case(7U, 64U, 0.007F, with_bias, 1U, seed++);
        check_case(33U, 1028U, 1.0F, with_bias, 0U, seed++);
        check_case(33U, 1027U, 3.5e-4F, with_bias, 0U, seed++);
        check_case(70000U, 4U, 1.0F, with_bias, 0U, seed++);
        check_case(70001U, 3U, 2.0F, with_bias, 1U, seed++);
        check_case(5U, 8U, 1.0e5F, with_bias, 0U, seed++);
        check_case(4U, 12U, kFmaScale, with_bias, 0U, seed++);
        check_case(4U, 11U, kFmaScale, with_bias, 1U, seed++);
        check_case(5U, 8U, std::numeric_limits<float>::quiet_NaN(), with_bias, 0U, seed++);
    }
    std::puts(
        "checked: vector and scalar paths, >65535 rows, misaligned views, NaN/inf/signed zero, "
        "subnormal, overflow, ties, unfused rounding, rejection without writes");
}

}  // namespace

int main() {
    return eclair::test::contain(run);
}
