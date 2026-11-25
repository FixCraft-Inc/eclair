/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/fp8_linear.hpp"

#include "eclair/formats/fp16.hpp"
#include "eclair/formats/fp8.hpp"

namespace eclair::host {

ops::Status fp8_linear_reference(const ops::Fp8LinearShape& shape,
                                 const ops::Fp8LinearBuffers& buffers) noexcept {
    const ops::Status status = ops::validate_fp8_linear(shape, buffers);
    if (status != ops::Status::Success || shape.m == 0U || shape.n == 0U) {
        return status;
    }
    const std::size_t scale_columns = 1U + (shape.k - 1U) / shape.block_k;
    for (std::size_t m = 0; m < shape.m; ++m) {
        for (std::size_t n = 0; n < shape.n; ++n) {
            double sum = 0.0;
            for (std::size_t k = 0; k < shape.k; ++k) {
                const float a = formats::decode_fp16(buffers.activations.data[m * shape.k + k]);
                const float w =
                    formats::decode_fp8(buffers.weights.data[n * shape.k + k], shape.format);
                const float scale =
                    buffers.scales.data[(n / shape.block_n) * scale_columns + k / shape.block_k];
                const float rounded = formats::decode_fp16(formats::encode_fp16(w * scale));
                sum += static_cast<double>(a) * static_cast<double>(rounded);
            }
            const float bias = buffers.bias.size == 0U ? 0.0F : buffers.bias.data[n];
            buffers.output.data[m * shape.n + n] = static_cast<float>(sum) + bias;
        }
    }
    return ops::Status::Success;
}

}  // namespace eclair::host
