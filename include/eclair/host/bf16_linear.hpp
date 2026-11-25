/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/bf16_linear.hpp"

namespace eclair::host {

// Host pointers only, and the caller owns all storage. No allocation. Applies
// the row-scaling definition through the same formats functions as the GPU
// kernel, so results must match it bit for bit.
[[nodiscard]] ops::Status bf16_row_scaling_reference(
    const ops::Bf16RowScalingShape& shape, const ops::Bf16RowScalingBuffers& buffers) noexcept;

// Host pointers only, and the caller owns all storage. No allocation. Sums the
// exact binary16 products in FP64, applies the power of two in FP64, rounds
// once to FP32 (overflowing to infinity), then adds the FP32 bias. GPU
// accumulation order and FP32 rounding are compared with a stated tolerance.
[[nodiscard]] ops::Status bf16_linear_reference(const ops::Bf16LinearShape& shape,
                                                const ops::Bf16LinearBuffers& buffers) noexcept;

}  // namespace eclair::host
