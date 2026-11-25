/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/w4a16_linear.hpp"

#include "eclair/formats/fp16.hpp"

namespace eclair::host {

ops::Status w4a16_linear_reference(const ops::W4A16LinearShape& s,
                                   const ops::W4A16LinearBuffers& b) noexcept {
    const auto status = ops::validate_w4a16_linear(s, b);
    if (status != ops::Status::Success || s.m == 0U || s.n == 0U) {
        return status;
    }
    using formats::decode_fp16;
    using formats::encode_fp16;
    for (std::size_t m = 0; m < s.m; ++m) {
        for (std::size_t n = 0; n < s.n; ++n) {
            double sum = 0;
            for (std::size_t k = 0; k < s.k; ++k) {
                const unsigned int i = static_cast<unsigned int>(k % 8U);
                const unsigned int shift = 4U * (i / 2U + 4U * (i % 2U));
                const auto q =
                    static_cast<int>((b.weights.data[(k / 8U) * s.n + n] >> shift) & 15U);
                const float scale = decode_fp16(b.scales.data[(k / 128U) * s.n + n]);
                const float weight = decode_fp16(encode_fp16(static_cast<float>(q - 8) * scale));
                sum += static_cast<double>(decode_fp16(b.activations.data[m * s.k + k])) * weight;
            }
            b.output.data[m * s.n + n] = encode_fp16(static_cast<float>(sum));
        }
    }
    return ops::Status::Success;
}

}  // namespace eclair::host
