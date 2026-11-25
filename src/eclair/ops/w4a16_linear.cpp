/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/ops/w4a16_linear.hpp"

#include "buffer_checks.hpp"

namespace eclair::ops {

Status validate_w4a16_linear(const W4A16LinearShape& s, const W4A16LinearBuffers& b) noexcept {
    if (s.m == 0U || s.n == 0U) {
        return Status::Success;
    }
    if (s.k == 0U || s.k % 128U != 0U || s.splits == 0U) {
        return Status::InvalidArgument;
    }
    std::size_t a = 0, w = 0, scales = 0, out = 0, scratch = 0;
    using detail::multiply;
    if (!multiply(s.m, s.k, a) || !multiply(s.k / 8U, s.n, w) ||
        !multiply(s.k / 128U, s.n, scales) || !multiply(s.m, s.n, out) ||
        !multiply(out, s.splits, scratch)) {
        return Status::SizeOverflow;
    }
    detail::AddressRange inputs[3], output, workspace;
    const Status checks[] = {
        detail::range(b.activations, a, inputs[0]),     detail::range(b.weights, w, inputs[1]),
        detail::range(b.scales, scales, inputs[2]),     detail::range(b.output, out, output),
        detail::range(b.workspace, scratch, workspace),
    };
    for (const auto status : checks) {
        if (status != Status::Success) {
            return status;
        }
    }
    for (const auto input : inputs) {
        if (detail::overlaps(output, input) || detail::overlaps(workspace, input)) {
            return Status::InvalidArgument;
        }
    }
    return detail::overlaps(output, workspace) ? Status::InvalidArgument : Status::Success;
}

Status validate_w4a16_dequantize(std::size_t n, std::size_t k,
                                 const W4A16DequantizeBuffers& b) noexcept {
    if (n == 0U) {
        return Status::Success;
    }
    if (k == 0U || k % 128U != 0U) {
        return Status::InvalidArgument;
    }
    std::size_t words = 0, scales = 0, weights = 0;
    using detail::multiply;
    if (!multiply(k / 8U, n, words) || !multiply(k / 128U, n, scales) || !multiply(k, n, weights)) {
        return Status::SizeOverflow;
    }
    detail::AddressRange inputs[2], output;
    const Status checks[] = {
        detail::range(b.weights, words, inputs[0]),
        detail::range(b.scales, scales, inputs[1]),
        detail::range(b.dequantized, weights, output),
    };
    for (const auto status : checks) {
        if (status != Status::Success) {
            return status;
        }
    }
    for (const auto input : inputs) {
        if (detail::overlaps(output, input)) {
            return Status::InvalidArgument;
        }
    }
    return Status::Success;
}

Status validate_w4a16_gemm(const W4A16LinearShape& s, const W4A16GemmBuffers& b) noexcept {
    if (s.m == 0U || s.n == 0U) {
        return Status::Success;
    }
    if (s.k == 0U || s.k % 128U != 0U || s.splits != 1U) {
        return Status::InvalidArgument;
    }
    std::size_t a = 0, words = 0, scales = 0, weights = 0, out = 0;
    using detail::multiply;
    if (!multiply(s.m, s.k, a) || !multiply(s.k / 8U, s.n, words) ||
        !multiply(s.k / 128U, s.n, scales) || !multiply(s.k, s.n, weights) ||
        !multiply(s.m, s.n, out)) {
        return Status::SizeOverflow;
    }
    detail::AddressRange inputs[3], written[3];
    const Status checks[] = {
        detail::range(b.activations, a, inputs[0]),
        detail::range(b.weights, words, inputs[1]),
        detail::range(b.scales, scales, inputs[2]),
        detail::range(b.dequantized, weights, written[0]),
        detail::range(b.output, out, written[1]),
        detail::range(b.workspace, b.workspace.size, written[2]),
    };
    for (const auto status : checks) {
        if (status != Status::Success) {
            return status;
        }
    }
    for (std::size_t i = 0; i < 3U; ++i) {
        for (const auto input : inputs) {
            if (detail::overlaps(written[i], input)) {
                return Status::InvalidArgument;
            }
        }
        for (std::size_t j = i + 1U; j < 3U; ++j) {
            if (detail::overlaps(written[i], written[j])) {
                return Status::InvalidArgument;
            }
        }
    }
    return Status::Success;
}

}  // namespace eclair::ops
