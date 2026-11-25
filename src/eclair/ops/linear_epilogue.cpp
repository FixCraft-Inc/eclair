/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/ops/linear_epilogue.hpp"

#include "buffer_checks.hpp"

namespace eclair::ops {

Status validate_linear_epilogue(const LinearEpilogueShape& shape,
                                const LinearEpilogueBuffers& buffers) noexcept {
    if (shape.m == 0U || shape.n == 0U) {
        return Status::Success;
    }
    std::size_t elements = 0;
    std::size_t bytes = 0;
    if (!detail::multiply(shape.m, shape.n, elements) ||
        !detail::multiply(elements, sizeof(float), bytes)) {
        return Status::SizeOverflow;
    }
    detail::AddressRange inputs[3];
    detail::AddressRange output;
    const Status checks[] = {
        detail::range(buffers.accumulators, elements, inputs[0]),
        detail::range(buffers.scale, 1U, inputs[1]),
        detail::range(buffers.bias, buffers.bias.size == 0U ? 0U : shape.n, inputs[2]),
        detail::range(buffers.output, elements, output),
    };
    for (const Status check : checks) {
        if (check != Status::Success) {
            return check;
        }
    }
    for (const detail::AddressRange input : inputs) {
        if (detail::overlaps(input, output)) {
            return Status::InvalidArgument;
        }
    }
    return Status::Success;
}

}  // namespace eclair::ops
