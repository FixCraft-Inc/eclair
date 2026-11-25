/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/w4a16_linear.hpp"

namespace eclair::host {

// Host storage only, FP64 products/sums with the contract's binary16 weight
// rounding. No allocation. Scratch is validated but left untouched.
[[nodiscard]] ops::Status w4a16_linear_reference(const ops::W4A16LinearShape& shape,
                                                 const ops::W4A16LinearBuffers& buffers) noexcept;

}  // namespace eclair::host
