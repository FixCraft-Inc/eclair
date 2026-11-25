/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/ops/fp8_linear.hpp"

#include "buffer_checks.hpp"

namespace eclair::ops {

using detail::AddressRange;
using detail::multiply;
using detail::overlaps;
using detail::range;

Status fp8_linear_sizes(const Fp8LinearShape& shape, Fp8LinearSizes& sizes) noexcept {
    sizes = {};
    if ((shape.format != formats::Fp8Format::E4M3FN && shape.format != formats::Fp8Format::E5M2) ||
        shape.block_n == 0U || shape.block_k == 0U) {
        return Status::InvalidArgument;
    }
    if (shape.m == 0U || shape.n == 0U) {
        return Status::Success;
    }
    if (shape.k == 0U) {
        return Status::InvalidArgument;
    }
    Fp8LinearSizes result;
    const std::size_t scale_rows = 1U + (shape.n - 1U) / shape.block_n;
    const std::size_t scale_columns = 1U + (shape.k - 1U) / shape.block_k;
    std::size_t bytes = 0;
    if (!multiply(shape.m, shape.k, result.activations) ||
        !multiply(result.activations, sizeof(std::uint16_t), bytes) ||
        !multiply(shape.n, shape.k, result.weights) ||
        !multiply(scale_rows, scale_columns, result.scales) ||
        !multiply(result.scales, sizeof(float), bytes) ||
        !multiply(shape.m, shape.n, result.output) ||
        !multiply(result.output, sizeof(float), bytes)) {
        return Status::SizeOverflow;
    }
    sizes = result;
    return Status::Success;
}

Status validate_fp8_linear(const Fp8LinearShape& shape, const Fp8LinearBuffers& buffers) noexcept {
    Fp8LinearSizes sizes;
    const Status status = fp8_linear_sizes(shape, sizes);
    if (status != Status::Success || sizes.output == 0U) {
        return status;
    }
    AddressRange inputs[4];
    AddressRange output;
    const Status checks[] = {
        range(buffers.activations, sizes.activations, inputs[0]),
        range(buffers.weights, sizes.weights, inputs[1]),
        range(buffers.scales, sizes.scales, inputs[2]),
        range(buffers.bias, buffers.bias.size == 0U ? 0U : shape.n, inputs[3]),
        range(buffers.output, sizes.output, output),
    };
    for (const Status check : checks) {
        if (check != Status::Success) {
            return check;
        }
    }
    for (const AddressRange input : inputs) {
        if (overlaps(input, output)) {
            return Status::InvalidArgument;
        }
    }
    return Status::Success;
}

}  // namespace eclair::ops
