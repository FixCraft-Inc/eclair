/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once
#include "eclair/render/surface_lighting.hpp"

namespace eclair::render {
// Bounded Whitted-style specular preview with GGX samples and shared diffuse
// lighting. Both Fresnel branches are evaluated, eliminating binary glass noise.
// The unfiltered path tracer remains the independent numerical oracle.
ECLAIR_HOST_DEVICE inline Vec3 trace_lit(const ops::RayScene& scene, const LightingLayout& layout,
                                         const SurfaceLighting* lighting, Ray ray,
                                         std::uint32_t bounces, float u, float v,
                                         float primary_light_weight = 1) {
    struct Path {
        Ray ray;
        Vec3 weight;
        std::uint32_t bounce;
        bool light_sampled, rough_path, rough_scattered;
        float u, v;
    };
    Path stack[8];
    unsigned pending = 0;
    Vec3 result{}, weight{1, 1, 1};
    std::uint32_t bounce = 0;
    bool light_sampled = false;
    bool rough_path = false;
    bool rough_scattered = false;
    while (true) {
        for (; bounce < bounces; ++bounce) {
            Hit hit;
            if (!nearest(scene, ray, 1.0e6F, hit)) {
                result = result + weight * scene.environment;
                break;
            }
            const auto& material = scene.primitives[hit.primitive].material;
            const Vec3 point = ray.origin + ray.direction * hit.distance;
            if (material.emission > 0) {
                if ((!light_sampled && !rough_path) ||
                    !lighting_emitter(scene, scene.primitives[hit.primitive]))
                    result = result + weight * material.base_color * material.emission;
                break;
            }
            if (material.kind == MaterialKind::Diffuse) {
                result = result + weight * lighting_lookup(scene, layout, lighting, point, hit);
                break;
            }
            light_sampled = material.kind == MaterialKind::Conductor && material.roughness >= 0.25F;
            // The interactive cache omits emitter caustics through a rough
            // reflection followed by further specular interfaces. Their rare,
            // high-energy paths do not converge in a moving eight-ray budget.
            // Direct quadrature and reflected diffuse lighting remain present;
            // the raw integrator retains the complete path contribution.
            rough_path = rough_path || rough_scattered;
            rough_scattered = rough_scattered ||
                              (material.kind == MaterialKind::Conductor && material.roughness > 0);
            if (light_sampled && (bounce != 0 || primary_light_weight > 0))
                result = result + weight *
                                      lighting_direct(scene, material, point, hit.normal,
                                                      -ray.direction, true) *
                                      (bounce == 0 ? primary_light_weight : 1.0F);
            if (bounce + 1 == bounces) break;
            Vec3 direction;
            if (material.kind == MaterialKind::Dielectric) {
                const float eta = hit.front_face ? 1 / material.ior : material.ior;
                const float fresnel = dielectric_fresnel(-dot(ray.direction, hit.normal), eta);
                Vec3 transmitted;
                if (refract(ray.direction, hit.normal, eta, transmitted)) {
                    if (fresnel > 0 && pending < 8) {
                        stack[pending++] = {
                            {point + hit.normal * kRayOffset, reflect(ray.direction, hit.normal)},
                            weight * fresnel,
                            bounce + 1,
                            false,
                            rough_path,
                            rough_scattered,
                            u,
                            v};
                    }
                    direction = transmitted;
                    weight = weight * material.base_color * ((1 - fresnel) * eta * eta);
                } else {
                    direction = reflect(ray.direction, hit.normal);
                }
            } else if (material.roughness == 0) {
                direction = reflect(ray.direction, hit.normal);
                weight = weight *
                         conductor_fresnel(material.base_color, -dot(ray.direction, hit.normal));
            } else {
                const auto sample = sample_conductor(material, hit.normal, ray.direction, u, v);
                if (sample.pdf <= 0) break;
                direction = sample.direction;
                weight = weight * sample.weight;
                // Decorrelate later interfaces while preserving smooth samples.
                u = u + 0.618033989F;
                u -= ::floorf(u);
                v = v + 0.414213562F;
                v -= ::floorf(v);
            }
            const float side = dot(direction, hit.normal) >= 0 ? 1.0F : -1.0F;
            ray = {point + hit.normal * (side * kRayOffset), direction};
        }
        if (pending == 0) break;
        const auto path = stack[--pending];
        ray = path.ray;
        weight = path.weight;
        bounce = path.bounce;
        light_sampled = path.light_sampled;
        rough_path = path.rough_path;
        rough_scattered = path.rough_scattered;
        u = path.u;
        v = path.v;
    }
    return result;
}
ECLAIR_HOST_DEVICE inline Vec3 lit_pixel(const ops::RayScene& scene, const ops::RayCamera& camera,
                                         const ops::RayTraceShape& shape, std::uint32_t index,
                                         const LightingLayout& layout,
                                         const SurfaceLighting* lighting) {
    const auto ray = camera_ray(camera, shape, static_cast<float>(index % shape.width) + 0.5F,
                                static_cast<float>(index / shape.width) + 0.5F);
    // Stratify lobe samples, with per-pixel rotations to avoid coherent rings.
    // Both dimensions rotate over frames for stationary convergence.
    std::uint32_t state = (index * 747796405U) ^ 277803737U;
    if (state == 0) state = 1;
    const float rotation_x =
        static_cast<float>(shape.seed % 1024) * 0.569840296F + random_unit(state);
    const float rotation_y =
        static_cast<float>(shape.seed % 1024) * 0.754877666F + random_unit(state);
    const float shift_x = rotation_x - ::floorf(rotation_x);
    const float shift_y = rotation_y - ::floorf(rotation_y);
    Vec3 sum{};
    for (std::uint32_t sample = 0; sample < shape.samples; ++sample) {
        float u = (static_cast<float>(sample) + 0.5F) / static_cast<float>(shape.samples);
        u += shift_x;
        u -= ::floorf(u);
        float v = static_cast<float>(sample) * 0.618033989F + shift_y;
        v -= ::floorf(v);
        sum = sum + trace_lit(scene, layout, lighting, ray, shape.bounces, u, v,
                              sample == 0 ? static_cast<float>(shape.samples) : 0.0F);
    }
    return sum / static_cast<float>(shape.samples);
}
}  // namespace eclair::render
