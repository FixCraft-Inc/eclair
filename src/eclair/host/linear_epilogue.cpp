/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/linear_epilogue.hpp"

#include "eclair/formats/fp16.hpp"

namespace eclair::host {

ops::Status linear_epilogue_reference(const ops::LinearEpilogueShape& shape,
                                      const ops::LinearEpilogueBuffers& buffers) noexcept {
    const ops::Status status = ops::validate_linear_epilogue(shape, buffers);
    if (status != ops::Status::Success || shape.m == 0U || shape.n == 0U) {
        return status;
    }
    // A product or sum of two floats computed in double and then rounded to
    // float is the correctly rounded FP32 result: double carries more than
    // 2 * 24 + 2 significand bits. This also prevents FMA contraction.
    const double scale = buffers.scale.data[0];
    for (std::size_t m = 0; m < shape.m; ++m) {
        for (std::size_t n = 0; n < shape.n; ++n) {
            const std::size_t index = m * shape.n + n;
            float value = static_cast<float>(buffers.accumulators.data[index] * scale);
            if (buffers.bias.size != 0U) {
                value = static_cast<float>(static_cast<double>(value) + buffers.bias.data[n]);
            }
            buffers.output.data[index] = formats::encode_fp16(value);
        }
    }
    return ops::Status::Success;
}

}  // namespace eclair::host
