/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once
#include "eclair/ops/ray_trace.hpp"

#include <cstdint>
#include <span>

namespace eclair::demo {
struct FrameScene {
    ops::RayScene scene;
    ops::RayCamera camera;
};
enum class SceneKind : std::uint32_t { Cornell, Materials };
[[nodiscard]] FrameScene cornell_scene(float seconds);
[[nodiscard]] FrameScene materials_scene(float seconds);
[[nodiscard]] FrameScene make_scene(SceneKind kind, float seconds);
void cpu_ray_frame(const FrameScene& frame, const ops::RayTraceShape& shape,
                   std::span<std::uint8_t> rgba, unsigned threads);
void raster_frame(const FrameScene& frame, const ops::RayTraceShape& shape,
                  std::span<std::uint8_t> rgba);
}  // namespace eclair::demo
