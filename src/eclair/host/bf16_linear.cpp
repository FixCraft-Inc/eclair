/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/bf16_linear.hpp"

#include "eclair/formats/bf16.hpp"
#include "eclair/formats/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace eclair::host {
namespace {

// Nearest-even FP32 with IEEE overflow to infinity. Converting a double beyond
// FP32's range is undefined behavior in C++, and rescaled sums can get there.
[[nodiscard]] float to_float(double value) noexcept {
    // The midpoint between the largest float and 2^128. It ties to the even
    // neighbor, 2^128, so it overflows too.
    constexpr double kOverflow = 0x1.ffffffp127;
    if (std::fabs(value) >= kOverflow) {
        return value < 0.0 ? -std::numeric_limits<float>::infinity()
                           : std::numeric_limits<float>::infinity();
    }
    return static_cast<float>(value);
}

}  // namespace

ops::Status bf16_row_scaling_reference(const ops::Bf16RowScalingShape& shape,
                                       const ops::Bf16RowScalingBuffers& buffers) noexcept {
    const ops::Status status = ops::validate_bf16_row_scaling(shape, buffers);
    if (status != ops::Status::Success || shape.rows == 0U) {
        return status;
    }
    for (std::size_t row = 0; row < shape.rows; ++row) {
        const std::uint16_t* input = buffers.input.data + row * shape.columns;
        std::uint16_t largest = 0;
        for (std::size_t column = 0; column < shape.columns; ++column) {
            largest = std::max(largest, formats::bf16_finite_magnitude(input[column]));
        }
        const std::int32_t shift = formats::bf16_fp16_shift(largest);
        buffers.shifts.data[row] = shift;
        std::uint16_t* output = buffers.output.data + row * shape.columns;
        for (std::size_t column = 0; column < shape.columns; ++column) {
            output[column] = formats::bf16_to_fp16_scaled_bits(input[column], shift);
        }
    }
    return ops::Status::Success;
}

ops::Status bf16_linear_reference(const ops::Bf16LinearShape& shape,
                                  const ops::Bf16LinearBuffers& buffers) noexcept {
    const ops::Status status = ops::validate_bf16_linear(shape, buffers);
    if (status != ops::Status::Success || shape.m == 0U || shape.n == 0U) {
        return status;
    }
    for (std::size_t m = 0; m < shape.m; ++m) {
        const std::int32_t a_shift =
            formats::clamp_bf16_fp16_shift(buffers.activation_shifts.data[m]);
        for (std::size_t n = 0; n < shape.n; ++n) {
            double sum = 0.0;
            for (std::size_t k = 0; k < shape.k; ++k) {
                sum += static_cast<double>(
                           formats::decode_fp16(buffers.activations.data[m * shape.k + k])) *
                       static_cast<double>(
                           formats::decode_fp16(buffers.weights.data[n * shape.k + k]));
            }
            // Both shifts are clamped to 160, so the power stays far inside
            // double's range and ldexp is exact.
            const std::int32_t power =
                -(a_shift + formats::clamp_bf16_fp16_shift(buffers.weight_shifts.data[n]));
            const float bias = buffers.bias.size == 0U ? 0.0F : buffers.bias.data[n];
            buffers.output.data[m * shape.n + n] = to_float(std::ldexp(sum, power)) + bias;
        }
    }
    return ops::Status::Success;
}

}  // namespace eclair::host
