/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/buffer.hpp"
#include "eclair/render/geometry.hpp"

#include <cstddef>
#include <cstdint>

namespace eclair::ops {

inline constexpr std::uint32_t kMaxRayPrimitives = 16;
struct RayScene {
    render::Primitive primitives[kMaxRayPrimitives]{};
    std::uint32_t count = 0;
    render::Vec3 light_center{0, 1.98F, 0};
    render::Vec3 light_u{0.3F, 0, 0};
    render::Vec3 light_v{0, 0, 0.3F};
    render::Vec3 light_emission{12, 12, 12};
    render::Vec3 environment{};  // Constant linear radiance on a ray miss.
};
struct RayCamera {
    render::Vec3 origin{0, 1, 3.5F};
    render::Vec3 forward{0, 0, -1};
    render::Vec3 right{1, 0, 0};
    render::Vec3 up{0, 1, 0};
    float vertical_tangent = 0.36F;
};
struct RayTraceShape {
    std::uint32_t width = 0, height = 0;
    std::uint32_t samples = 1, bounces = 2, seed = 1;
};
// RGBA8 bytes in increasing R,G,B,A address order, tightly packed rows.
// Maximum 4096x4096, eight samples/pixel, eight scattering vertices.
[[nodiscard]] Status validate_ray_trace(const RayScene& scene, const RayCamera& camera,
                                        const RayTraceShape& shape) noexcept;
[[nodiscard]] Status ray_frame_bytes(const RayTraceShape& shape, std::size_t* bytes) noexcept;

}  // namespace eclair::ops
