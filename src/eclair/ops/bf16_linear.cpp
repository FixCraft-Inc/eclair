/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/ops/bf16_linear.hpp"

#include "buffer_checks.hpp"

namespace eclair::ops {

using detail::AddressRange;
using detail::multiply;
using detail::overlaps;
using detail::range;

Status validate_bf16_row_scaling(const Bf16RowScalingShape& shape,
                                 const Bf16RowScalingBuffers& buffers) noexcept {
    if (shape.rows == 0U) {
        return Status::Success;
    }
    if (shape.columns == 0U) {
        return Status::InvalidArgument;
    }
    std::size_t elements = 0;
    std::size_t bytes = 0;
    if (!multiply(shape.rows, shape.columns, elements) ||
        !multiply(elements, sizeof(std::uint16_t), bytes)) {
        return Status::SizeOverflow;
    }
    AddressRange input;
    AddressRange output;
    AddressRange shifts;
    const Status checks[] = {
        range(buffers.input, elements, input),
        range(buffers.output, elements, output),
        range(buffers.shifts, shape.rows, shifts),
    };
    for (const Status check : checks) {
        if (check != Status::Success) {
            return check;
        }
    }
    if (overlaps(input, output) || overlaps(input, shifts) || overlaps(output, shifts)) {
        return Status::InvalidArgument;
    }
    return Status::Success;
}

Status bf16_linear_sizes(const Bf16LinearShape& shape, Bf16LinearSizes& sizes) noexcept {
    sizes = {};
    if (shape.m == 0U || shape.n == 0U) {
        return Status::Success;
    }
    if (shape.k == 0U) {
        return Status::InvalidArgument;
    }
    Bf16LinearSizes result;
    std::size_t bytes = 0;
    if (!multiply(shape.m, shape.k, result.activations) ||
        !multiply(result.activations, sizeof(std::uint16_t), bytes) ||
        !multiply(shape.n, shape.k, result.weights) ||
        !multiply(result.weights, sizeof(std::uint16_t), bytes) ||
        !multiply(shape.m, shape.n, result.output) ||
        !multiply(result.output, sizeof(float), bytes)) {
        return Status::SizeOverflow;
    }
    sizes = result;
    return Status::Success;
}

Status validate_bf16_linear(const Bf16LinearShape& shape,
                            const Bf16LinearBuffers& buffers) noexcept {
    Bf16LinearSizes sizes;
    const Status status = bf16_linear_sizes(shape, sizes);
    if (status != Status::Success || sizes.output == 0U) {
        return status;
    }
    AddressRange inputs[5];
    AddressRange output;
    const Status checks[] = {
        range(buffers.activations, sizes.activations, inputs[0]),
        range(buffers.activation_shifts, shape.m, inputs[1]),
        range(buffers.weights, sizes.weights, inputs[2]),
        range(buffers.weight_shifts, shape.n, inputs[3]),
        range(buffers.bias, buffers.bias.size == 0U ? 0U : shape.n, inputs[4]),
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
