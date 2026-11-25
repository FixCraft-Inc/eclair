/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/ray_trace.hpp"

namespace eclair::host {
// Synchronous portable oracle, caller-owned output. Does not spawn threads.
[[nodiscard]] ops::Status ray_trace_reference(const ops::RayScene& scene,
                                              const ops::RayCamera& camera,
                                              const ops::RayTraceShape& shape,
                                              ops::BufferView<std::uint8_t> rgba) noexcept;
}  // namespace eclair::host
