/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/gated_delta_output.hpp"

#include <cuda_runtime_api.h>

namespace eclair::cuda {

// Key size 64/128, value size a multiple of 64. Binary16 arrays must be
// 16-byte aligned and counts fit int. Device pointers only. Caller owns all
// storage and the stream until completion. No allocation or synchronization.
[[nodiscard]] bool gated_delta_output_supported(
    const ops::GatedDeltaOutputShape& shape, const ops::GatedDeltaOutputBuffers& buffers) noexcept;

[[nodiscard]] cudaError_t launch_gated_delta_output(const ops::GatedDeltaOutputShape& shape,
                                                    const ops::GatedDeltaOutputBuffers& buffers,
                                                    cudaStream_t stream) noexcept;

[[nodiscard]] cudaError_t check_gated_delta_output_image(int device_index) noexcept;

}  // namespace eclair::cuda
