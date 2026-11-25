/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/fp8_linear.hpp"

namespace eclair::host {

// Host pointers only; caller owns all storage. No allocation. Uses FP64 sums
// of the specified FP16 operands as the accuracy oracle, then FP32 bias and
// output rounding. GPU accumulation order is compared with a stated tolerance.
[[nodiscard]] ops::Status fp8_linear_reference(const ops::Fp8LinearShape& shape,
                                               const ops::Fp8LinearBuffers& buffers) noexcept;

}  // namespace eclair::host
