/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/gated_delta.hpp"

namespace eclair::host {

// Host pointers only, and the caller owns all storage. No allocation.
// Validates, then reads sequence_starts and chunk_starts and returns
// InvalidArgument, before writing, when a sequence's bounds are decreasing or
// outside the tokens, or its chunks fall outside the shape. The state between
// chunks is the FP32 value of the contract, kept in final_states. Within a
// chunk, products are summed in FP64 and rounded once where the contract
// rounds.
[[nodiscard]] ops::Status gated_delta_states_reference(
    const ops::GatedDeltaStatesShape& shape, const ops::GatedDeltaStatesBuffers& buffers) noexcept;

}  // namespace eclair::host
