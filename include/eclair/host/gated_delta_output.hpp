/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/gated_delta_output.hpp"

namespace eclair::host {

// Host buffers, no allocation. Checks all metadata before writing: sequences
// cover [0, tokens] in order, and chunk_indices names every chunk exactly
// once in sequence order. FP64 sums with the contract's binary16 roundings.
[[nodiscard]] ops::Status gated_delta_output_reference(
    const ops::GatedDeltaOutputShape& shape, const ops::GatedDeltaOutputBuffers& buffers) noexcept;

}  // namespace eclair::host
