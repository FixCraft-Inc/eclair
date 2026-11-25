/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once
#include "eclair/ops/ray_trace.hpp"
#include <cuda_runtime_api.h>

namespace eclair::cuda {
// Caller owns the device RGBA allocation and stream. No allocation or synchronization.
[[nodiscard]] cudaError_t launch_ray_trace(const ops::RayScene& scene, const ops::RayCamera& camera,
                                           const ops::RayTraceShape& shape,
                                           ops::BufferView<std::uint8_t> rgba,
                                           cudaStream_t stream) noexcept;
[[nodiscard]] cudaError_t check_ray_trace_image(int device_index) noexcept;
}  // namespace eclair::cuda
