/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/ops/ray_trace.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>

namespace eclair::ops {
Status ray_frame_bytes(const RayTraceShape& s, std::size_t* bytes) noexcept {
    if (bytes == nullptr || s.width == 0 || s.height == 0 || s.width > 4096 || s.height > 4096)
        return Status::InvalidArgument;
    const std::size_t width = s.width, height = s.height;
    if (width > std::numeric_limits<std::size_t>::max() / height / 4) return Status::SizeOverflow;
    *bytes = width * height * 4;
    return Status::Success;
}
Status validate_ray_trace(const RayScene& scene, const RayCamera& camera,
                          const RayTraceShape& shape) noexcept {
    using namespace render;
    std::size_t bytes = 0;
    if (ray_frame_bytes(shape, &bytes) != Status::Success || shape.samples == 0 ||
        shape.samples > 8 || shape.bounces == 0 || shape.bounces > 8 || scene.count == 0 ||
        scene.count > kMaxRayPrimitives)
        return Status::InvalidArgument;
    for (std::uint32_t i = 0; i < scene.count; ++i)
        if (!valid_primitive(scene.primitives[i])) return Status::InvalidArgument;
    // A single air/solid boundary has no medium stack. Reject potentially
    // nested/overlapping glass solids using conservative world-space bounds.
    const auto glass_extent = [](const Primitive& p) -> Vec3 {
        if (p.kind == PrimitiveKind::Sphere) return {p.radius, p.radius, p.radius};
        return {std::fabs(p.rotation_cos) * p.u.x + std::fabs(p.rotation_sin) * p.u.z, p.u.y,
                std::fabs(p.rotation_sin) * p.u.x + std::fabs(p.rotation_cos) * p.u.z};
    };
    for (std::uint32_t i = 0; i < scene.count; ++i) {
        const auto& a = scene.primitives[i];
        if (a.material.kind != MaterialKind::Dielectric) continue;
        for (std::uint32_t j = 0; j < i; ++j) {
            const auto& b = scene.primitives[j];
            if (b.material.kind != MaterialKind::Dielectric) continue;
            const Vec3 extent = glass_extent(a) + glass_extent(b), delta = a.center - b.center;
            if (std::fabs(delta.x) < extent.x && std::fabs(delta.y) < extent.y &&
                std::fabs(delta.z) < extent.z)
                return Status::InvalidArgument;
        }
    }
    const Primitive light{PrimitiveKind::Quad, scene.light_center, scene.light_u, scene.light_v};
    if (!finite(scene.environment) || scene.environment.x < 0 || scene.environment.y < 0 ||
        scene.environment.z < 0 || scene.environment.x > 10000 || scene.environment.y > 10000 ||
        scene.environment.z > 10000)
        return Status::InvalidArgument;
    if (!valid_primitive(light) || !finite(scene.light_emission) || scene.light_emission.x < 0 ||
        scene.light_emission.y < 0 || scene.light_emission.z < 0 ||
        scene.light_emission.x > 10000 || scene.light_emission.y > 10000 ||
        scene.light_emission.z > 10000 || !finite(camera.origin) ||
        std::fabs(camera.origin.x) > 10000 || std::fabs(camera.origin.y) > 10000 ||
        std::fabs(camera.origin.z) > 10000 || !finite(camera.vertical_tangent) ||
        camera.vertical_tangent < 0.05F || camera.vertical_tangent > 2)
        return Status::InvalidArgument;
    for (const Vec3 axis : {camera.forward, camera.right, camera.up})
        if (!finite(axis) || std::fabs(dot(axis, axis) - 1) > 1.0e-4F)
            return Status::InvalidArgument;
    if (std::fabs(dot(camera.forward, camera.right)) > 1.0e-4F ||
        std::fabs(dot(camera.forward, camera.up)) > 1.0e-4F ||
        std::fabs(dot(camera.right, camera.up)) > 1.0e-4F)
        return Status::InvalidArgument;
    return Status::Success;
}
}  // namespace eclair::ops
