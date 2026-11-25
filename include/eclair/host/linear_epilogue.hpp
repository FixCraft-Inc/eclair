/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/linear_epilogue.hpp"

namespace eclair::host {

// Host pointers only, and the caller owns all storage. No allocation.
// Bit-exact oracle for the epilogue contract, independent of the platform's
// FP32 contraction and rounding-mode choices.
[[nodiscard]] ops::Status linear_epilogue_reference(
    const ops::LinearEpilogueShape& shape, const ops::LinearEpilogueBuffers& buffers) noexcept;

}  // namespace eclair::host
