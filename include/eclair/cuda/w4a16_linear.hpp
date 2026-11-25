/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/w4a16_linear.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace eclair::cuda {

// Experimental skinny implementation: M=1..4, splits<=128,
// N/K within signed-int range. Other layouts/shapes are refused before any
// work is enqueued. Caller owns device buffers and stream until completion;
// no allocation, synchronization or device change. Graph capture is allowed.
// Each weight is dequantized once and multiplied for all rows on tensor
// cores (PTX mma.m8n8k4, sm_70+). Padding columns are computed, never stored.
[[nodiscard]] cudaError_t launch_w4a16_linear(const ops::W4A16LinearShape& shape,
                                              const ops::W4A16LinearBuffers& buffers,
                                              cudaStream_t stream) noexcept;
[[nodiscard]] cudaError_t check_w4a16_linear_image(int device_index) noexcept;

// Writes the contract's binary16 weights into [K,N] (ops::validate_w4a16_dequantize)
// with the linear kernel's exact dequantization and single rounding. N and
// K/8 must leave launch headroom below 2^32. Same ownership rules as above,
// and graph capture is allowed.
[[nodiscard]] cudaError_t launch_w4a16_dequantize(std::size_t n, std::size_t k,
                                                  const ops::W4A16DequantizeBuffers& buffers,
                                                  cudaStream_t stream) noexcept;
[[nodiscard]] cudaError_t check_w4a16_dequantize_image(int device_index) noexcept;

}  // namespace eclair::cuda
