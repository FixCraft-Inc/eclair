/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/render/shading.hpp"

namespace eclair::render {

// Preview-only outgoing diffuse radiance, including the material reflectance.
// Tiles live in object coordinates, so camera motion never moves their samples.
inline constexpr std::uint32_t kLightingResolution = 64;
inline constexpr std::uint32_t kLightingTilePixels = kLightingResolution * kLightingResolution;
struct SurfaceLighting {
    Vec3 color{};
    float history = 1;
    Vec3 direct{};
};
inline constexpr std::size_t kLightingBytesPerTexel =
    2 * sizeof(SurfaceLighting) + 2 * sizeof(Vec3);
ECLAIR_HOST_DEVICE inline bool lighting_emitter(const ops::RayScene& scene, const Primitive& p) {
    const Vec3 a = p.center - scene.light_center, b = p.u - scene.light_u, c = p.v - scene.light_v;
    return p.kind == PrimitiveKind::Quad && dot(a, a) < 1.0e-10F && dot(b, b) < 1.0e-10F &&
           dot(c, c) < 1.0e-10F;
}
// Fixed area quadrature removes direct-light Monte Carlo flicker. Visibility
// still traces shadow rays; the grid is a bounded preview approximation.
ECLAIR_HOST_DEVICE inline Vec3 lighting_direct(const ops::RayScene& scene, const Material& m,
                                               Vec3 point, Vec3 normal, Vec3 view,
                                               bool glossy = false) {
    const unsigned grid = glossy ? 2U : 4U;
    const Vec3 light_normal = normalized(cross(scene.light_u, scene.light_v));
    const float area =
        4 * ::sqrtf(dot(scene.light_u, scene.light_u) * dot(scene.light_v, scene.light_v));
    Vec3 sum{};
    for (unsigned sample = 0; sample < grid * grid; ++sample) {
        const float u =
            2 * (static_cast<float>(sample % grid) + 0.5F) / static_cast<float>(grid) - 1;
        const float v =
            2 * (static_cast<float>(sample / grid) + 0.5F) / static_cast<float>(grid) - 1;
        const Vec3 delta = scene.light_center + scene.light_u * u + scene.light_v * v - point;
        const float distance2 = dot(delta, delta), distance = ::sqrtf(distance2);
        if (distance <= 2 * kRayOffset) continue;
        const Vec3 direction = delta / distance;
        const float cosine = ::fmaxf(0, dot(normal, direction));
        const float light_cosine = ::fmaxf(0, -dot(light_normal, direction));
        if (cosine <= 0 || light_cosine <= 0 ||
            occluded(scene, {point + normal * kRayOffset, direction}, distance - 2 * kRayOffset))
            continue;
        sum = sum + evaluate_bsdf(m, normal, view, direction).value * scene.light_emission *
                        (cosine * light_cosine * area / distance2);
    }
    return sum / static_cast<float>(grid * grid);
}
struct LightingLayout {
    std::uint32_t offset[ops::kMaxRayPrimitives]{};
    std::uint32_t tiles = 0;
};
inline LightingLayout lighting_layout(const ops::RayScene& scene) {
    LightingLayout layout;
    for (std::uint32_t i = 0; i < scene.count && i < ops::kMaxRayPrimitives; ++i) {
        layout.offset[i] = layout.tiles;
        const auto& p = scene.primitives[i];
        if (p.material.kind == MaterialKind::Diffuse && p.material.emission == 0)
            layout.tiles += p.kind == PrimitiveKind::Box ? 12U : 2U;
    }
    return layout;
}
ECLAIR_HOST_DEVICE inline std::uint32_t lighting_primitive(const LightingLayout& layout,
                                                           std::uint32_t count,
                                                           std::uint32_t tile) {
    std::uint32_t primitive = 0;
    while (primitive + 1 < count && layout.offset[primitive + 1] <= tile) ++primitive;
    return primitive;
}
struct LightingPoint {
    Vec3 position{}, normal{};
};
ECLAIR_HOST_DEVICE inline LightingPoint lighting_point(const Primitive& p, std::uint32_t face,
                                                       float u, float v) {
    if (p.kind == PrimitiveKind::Quad) {
        return {p.center + p.u * (2 * u - 1) + p.v * (2 * v - 1),
                normalized(cross(p.u, p.v)) * (face == 0 ? 1.0F : -1.0F)};
    }
    if (p.kind == PrimitiveKind::Sphere) {
        const float phi = 2 * kPi * u, theta = kPi * v;
        const Vec3 normal{::sinf(theta) * ::cosf(phi), ::cosf(theta), ::sinf(theta) * ::sinf(phi)};
        return {p.center + normal * p.radius, normal * (face == 0 ? 1.0F : -1.0F)};
    }
    const auto side = face % 6;
    const int axis = static_cast<int>(side / 2), a = (axis + 1) % 3, b = (axis + 2) % 3;
    const float sign = side % 2 == 0 ? 1.0F : -1.0F;
    const Vec3 normal = axis_vector(axis, sign);
    const Vec3 point = normal * component(p.u, axis) +
                       axis_vector(a, (2 * u - 1) * component(p.u, a)) +
                       axis_vector(b, (2 * v - 1) * component(p.u, b));
    return {p.center + box_world(p, point), box_world(p, normal) * (face < 6 ? 1.0F : -1.0F)};
}
ECLAIR_HOST_DEVICE inline void lighting_coordinates(const Primitive& p, Vec3 position,
                                                    const Hit& hit, std::uint32_t& face, float& u,
                                                    float& v) {
    const Vec3 local = position - p.center;
    face = hit.front_face ? 0U : 1U;
    if (p.kind == PrimitiveKind::Quad) {
        u = 0.5F + 0.5F * dot(local, p.u) / dot(p.u, p.u);
        v = 0.5F + 0.5F * dot(local, p.v) / dot(p.v, p.v);
    } else if (p.kind == PrimitiveKind::Sphere) {
        const Vec3 n = normalized(local);
        u = ::atan2f(n.z, n.x) / (2 * kPi);
        if (u < 0) u += 1;
        v = ::acosf(::fmaxf(-1, ::fminf(1, n.y))) / kPi;
    } else {
        const Vec3 normal = box_local(p, hit.front_face ? hit.normal : -hit.normal);
        const Vec3 point = box_local(p, local);
        const int axis = ::fabsf(normal.x) > 0.5F ? 0 : (::fabsf(normal.y) > 0.5F ? 1 : 2);
        face = static_cast<std::uint32_t>(axis * 2) + (component(normal, axis) < 0 ? 1U : 0U) +
               (hit.front_face ? 0U : 6U);
        const int a = (axis + 1) % 3, b = (axis + 2) % 3;
        u = 0.5F + 0.5F * component(point, a) / component(p.u, a);
        v = 0.5F + 0.5F * component(point, b) / component(p.u, b);
    }
}
ECLAIR_HOST_DEVICE inline SurfaceLighting lighting_sample(const ops::RayScene& scene,
                                                          const LightingLayout& layout,
                                                          const ops::RayTraceShape& shape,
                                                          std::uint32_t index) {
    const auto tile = index / kLightingTilePixels;
    const auto primitive = lighting_primitive(layout, scene.count, tile);
    const auto texel = index % kLightingTilePixels;
    const auto point = lighting_point(scene.primitives[primitive], tile - layout.offset[primitive],
                                      (static_cast<float>(texel % kLightingResolution) + 0.5F) /
                                          static_cast<float>(kLightingResolution),
                                      (static_cast<float>(texel / kLightingResolution) + 0.5F) /
                                          static_cast<float>(kLightingResolution));
    const auto& material = scene.primitives[primitive].material;
    const Vec3 direct =
        lighting_direct(scene, material, point.position, point.normal, point.normal);
    std::uint32_t state = (index * 747796405U + shape.seed * 2891336453U) ^ 277803737U;
    if (state == 0) state = 1;
    Vec3 sum{};
    // The cache amortizes these paths across direct views and all specular rays.
    if (shape.bounces > 1) {
        for (unsigned sample = 0; sample < 8; ++sample) {
            const auto bsdf = sample_bsdf(material, point.normal, -point.normal, true, state);
            const Ray ray{point.position + point.normal * kRayOffset, bsdf.direction};
            Hit hit;
            if (!nearest(scene, ray, 1.0e6F, hit))
                sum = sum + bsdf.weight * scene.environment;
            else if (!lighting_emitter(scene, scene.primitives[hit.primitive]))
                sum = sum + bsdf.weight * trace(scene, ray, shape.bounces - 1, state, &hit);
        }
    }
    return {sum / 8, 1, direct};
}
ECLAIR_HOST_DEVICE inline Vec3 lighting_spatial(const SurfaceLighting* current, const Vec3* input,
                                                std::uint32_t index, bool vertical) {
    const auto tile = index / kLightingTilePixels, texel = index % kLightingTilePixels;
    const int x = static_cast<int>(texel % kLightingResolution),
              y = static_cast<int>(texel / kLightingResolution);
    Vec3 sum{};
    float weights = 0;
    for (int d = -2; d <= 2; ++d) {
        const int nx = x + (vertical ? 0 : d), ny = y + (vertical ? d : 0);
        if (nx < 0 || ny < 0 || nx >= static_cast<int>(kLightingResolution) ||
            ny >= static_cast<int>(kLightingResolution))
            continue;
        const auto i = tile * kLightingTilePixels +
                       static_cast<std::uint32_t>(ny) * kLightingResolution +
                       static_cast<std::uint32_t>(nx);
        const float weight = d == 0 ? 6.0F : (::abs(d) == 1 ? 4.0F : 1.0F);
        sum = sum + (input ? input[i] : current[i].color) * weight;
        weights += weight;
    }
    return sum / weights;
}
ECLAIR_HOST_DEVICE inline SurfaceLighting lighting_temporal(const SurfaceLighting& old, Vec3 color,
                                                            bool valid_history, bool stationary,
                                                            Vec3 direct = {}) {
    if (!valid_history || !finite(old.color) || !finite(old.history) || old.history < 1)
        return {color, 1, direct};
    const float length = ::fminf(stationary ? 1024.0F : 32.0F, old.history + 1);
    return {color / length + old.color * (1 - 1 / length), length, direct};
}
ECLAIR_HOST_DEVICE inline Vec3 lighting_lookup(const ops::RayScene& scene,
                                               const LightingLayout& layout,
                                               const SurfaceLighting* lighting, Vec3 position,
                                               const Hit& hit) {
    std::uint32_t face = 0;
    float u = 0, v = 0;
    const auto& p = scene.primitives[hit.primitive];
    lighting_coordinates(p, position, hit, face, u, v);
    const float x = ::fmaxf(0, ::fminf(static_cast<float>(kLightingResolution - 1),
                                       u * static_cast<float>(kLightingResolution) - 0.5F));
    const float y = ::fmaxf(0, ::fminf(static_cast<float>(kLightingResolution - 1),
                                       v * static_cast<float>(kLightingResolution) - 0.5F));
    const auto ix = static_cast<std::uint32_t>(x), iy = static_cast<std::uint32_t>(y);
    const auto nx = ix + 1 < kLightingResolution ? ix + 1 : ix;
    const auto ny = iy + 1 < kLightingResolution ? iy + 1 : iy;
    const auto base = (layout.offset[hit.primitive] + face) * kLightingTilePixels;
    const float fx = x - static_cast<float>(ix), fy = y - static_cast<float>(iy);
    const auto value = [&](std::uint32_t i) { return lighting[i].color + lighting[i].direct; };
    const Vec3 a = value(base + iy * kLightingResolution + ix) * (1 - fx) +
                   value(base + iy * kLightingResolution + nx) * fx;
    const Vec3 b = value(base + ny * kLightingResolution + ix) * (1 - fx) +
                   value(base + ny * kLightingResolution + nx) * fx;
    return a * (1 - fy) + b * fy;
}

}  // namespace eclair::render
