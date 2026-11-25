/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/ray_trace.hpp"
#include "eclair/render/bsdf.hpp"

namespace eclair::render {

inline constexpr float kRayOffset = 0.0002F;
ECLAIR_HOST_DEVICE inline bool nearest(const ops::RayScene& scene, const Ray& ray, float maximum,
                                       Hit& hit) {
    bool found = false;
    for (std::uint32_t i = 0; i < scene.count; ++i) {
        if (intersect_unchecked(scene.primitives[i], ray, kRayOffset, maximum, hit)) {
            maximum = hit.distance;
            hit.primitive = i;
            found = true;
        }
    }
    return found;
}
ECLAIR_HOST_DEVICE inline Ray camera_ray(const ops::RayCamera& camera,
                                         const ops::RayTraceShape& shape, float x, float y) {
    const float fx = (2 * x / static_cast<float>(shape.width) - 1) *
                     (static_cast<float>(shape.width) / static_cast<float>(shape.height)) *
                     camera.vertical_tangent;
    const float fy = (1 - 2 * y / static_cast<float>(shape.height)) * camera.vertical_tangent;
    return {camera.origin, normalized(camera.forward + camera.right * fx + camera.up * fy)};
}
// Shadow visibility needs any blocker, not a closest-hit traversal. Glass is
// opaque to this estimator: refracted paths are traced, focused caustics are not.
ECLAIR_HOST_DEVICE inline bool occluded(const ops::RayScene& scene, const Ray& ray, float maximum) {
    Hit hit;
    for (std::uint32_t i = 0; i < scene.count; ++i)
        if (intersect_unchecked(scene.primitives[i], ray, kRayOffset, maximum, hit)) return true;
    return false;
}
// An optional initial_hit must describe this exact ray. Jittered pixel-area
// samples must query their own visibility.
ECLAIR_HOST_DEVICE inline Vec3 trace(const ops::RayScene& scene, Ray ray, std::uint32_t bounces,
                                     std::uint32_t& state, const Hit* initial_hit = nullptr) {
    Vec3 result{}, throughput{1, 1, 1};
    float previous_pdf = 0;
    bool delta_path = true, previous_light_sampled = false;
    const Vec3 light_normal = normalized(cross(scene.light_u, scene.light_v));
    const float light_area =
        4 * ::sqrtf(dot(scene.light_u, scene.light_u) * dot(scene.light_v, scene.light_v));
    for (std::uint32_t bounce = 0; bounce < bounces; ++bounce) {
        Hit hit;
        bool found = true;
        if (bounce == 0 && initial_hit)
            hit = *initial_hit;
        else
            found = nearest(scene, ray, 1.0e6F, hit);
        if (!found) {
            result = result + throughput * scene.environment;
            break;
        }
        const Primitive& p = scene.primitives[hit.primitive];
        const Material& material = p.material;
        if (material.emission > 0) {
            float weight = 1;
            const Vec3 light_delta = p.center - scene.light_center;
            const Vec3 u_delta = p.u - scene.light_u, v_delta = p.v - scene.light_v;
            const float light_cosine = -dot(light_normal, ray.direction);
            if (!delta_path && previous_light_sampled && p.kind == PrimitiveKind::Quad &&
                dot(light_delta, light_delta) < 1.0e-10F && dot(u_delta, u_delta) < 1.0e-10F &&
                dot(v_delta, v_delta) < 1.0e-10F && light_cosine > 0) {
                const float light_pdf = hit.distance * hit.distance / (light_area * light_cosine);
                weight = power_weight(previous_pdf, light_pdf);
            }
            result = result + throughput * material.base_color * (material.emission * weight);
            break;
        }
        const Vec3 point = ray.origin + ray.direction * hit.distance;
        // Narrow conductor lobes sample the area light more effectively from
        // the BSDF. Omitting light sampling here also removes its MIS weight
        // on the subsequent emitter hit, preserving the expected contribution.
        const bool sample_light = material.kind == MaterialKind::Diffuse ||
                                  (material.kind == MaterialKind::Conductor &&
                                   (material.roughness >= 0.25F || bounce + 1 == bounces));
        if (sample_light) {
            // Sequence RNG updates explicitly: host and CUDA operand evaluation
            // order can differ. Use MIS to avoid counting the emitter twice.
            const float light_u = 2 * random_unit(state) - 1;
            const float light_v = 2 * random_unit(state) - 1;
            const Vec3 light =
                scene.light_center + scene.light_u * light_u + scene.light_v * light_v;
            const Vec3 delta = light - point;
            const float distance2 = dot(delta, delta), distance = ::sqrtf(distance2);
            if (distance > 2 * kRayOffset) {
                const Vec3 direction = delta / distance;
                const float cosine = ::fmaxf(0, dot(hit.normal, direction));
                const float light_cosine = ::fmaxf(0, -dot(light_normal, direction));
                if (cosine > 0 && light_cosine > 0 &&
                    !occluded(scene, {point + hit.normal * kRayOffset, direction},
                              distance - 2 * kRayOffset)) {
                    const auto bsdf =
                        evaluate_bsdf(material, hit.normal, -ray.direction, direction);
                    const float light_pdf = distance2 / (light_cosine * light_area);
                    const float weight =
                        bounce + 1 < bounces ? power_weight(light_pdf, bsdf.pdf) : 1;
                    result = result + throughput * bsdf.value * scene.light_emission *
                                          (cosine * weight / light_pdf);
                }
            }
        }
        if (bounce + 1 == bounces) break;
        const auto sample = sample_bsdf(material, hit.normal, ray.direction, hit.front_face, state);
        if ((!sample.delta && sample.pdf <= 0) || dot(sample.direction, sample.direction) < 0.5F)
            break;
        throughput = throughput * sample.weight;
        if (throughput.x <= 0 && throughput.y <= 0 && throughput.z <= 0) break;
        delta_path = sample.delta;
        previous_light_sampled = sample_light;
        previous_pdf = sample.pdf;
        const float side = dot(sample.direction, hit.normal) >= 0 ? 1.0F : -1.0F;
        ray = {point + hit.normal * (side * kRayOffset), sample.direction};
    }
    return result;
}
ECLAIR_HOST_DEVICE inline std::uint8_t display_channel(float linear) {
    if (!finite(linear)) return linear > 0 ? 255 : 0;
    // Narkowicz's published ACES fit, followed by the IEC sRGB transfer.
    // Bound the input before the rational polynomial, including finite overflow.
    const float x = ::fminf(64, ::fmaxf(0, linear));
    const float mapped = ::fminf(1, (x * (2.51F * x + 0.03F)) / (x * (2.43F * x + 0.59F) + 0.14F));
    const float srgb =
        mapped <= 0.0031308F ? 12.92F * mapped : 1.055F * ::powf(mapped, 1.0F / 2.4F) - 0.055F;
    return static_cast<std::uint8_t>(::fminf(255, srgb * 255 + 0.5F));
}
ECLAIR_HOST_DEVICE inline void store_pixel(Vec3 color, std::uint8_t* rgba) {
    rgba[0] = display_channel(color.x);
    rgba[1] = display_channel(color.y);
    rgba[2] = display_channel(color.z);
    rgba[3] = 255;
}
ECLAIR_HOST_DEVICE inline Vec3 ray_color(const ops::RayScene& scene, const ops::RayCamera& camera,
                                         const ops::RayTraceShape& shape, std::uint32_t index,
                                         float* variance = nullptr, bool pixel_area = true,
                                         const Hit* initial_hit = nullptr) {
    std::uint32_t state = (index * 747796405U + shape.seed * 2891336453U) ^ 277803737U;
    if (state == 0) state = 1;
    Vec3 color{};
    float square_sum = 0;
    const float rotation = pixel_area && shape.samples > 1 ? random_unit(state) : 0;
    for (std::uint32_t sample = 0; sample < shape.samples; ++sample) {
        // Latin hypercube in X and a rotated base-two sequence in Y. A
        // single sample keeps the center-ray oracle; multisampling covers
        // pixel area, including silhouettes and emissive panel boundaries.
        float dx = 0.5F, dy = 0.5F;
        if (pixel_area && shape.samples > 1) {
            dx = (static_cast<float>(sample) + random_unit(state)) /
                 static_cast<float>(shape.samples);
            std::uint32_t bits = sample;
            float inverse = 0, scale = 0.5F;
            while (bits) {
                inverse += static_cast<float>(bits & 1U) * scale;
                bits >>= 1;
                scale *= 0.5F;
            }
            dy = inverse + rotation;
            dy -= ::floorf(dy);
        }
        const float x = static_cast<float>(index % shape.width) + dx;
        const float y = static_cast<float>(index / shape.width) + dy;
        const auto sample_color =
            trace(scene, camera_ray(camera, shape, x, y), shape.bounces, state,
                  pixel_area && shape.samples > 1 ? nullptr : initial_hit);
        color = color + sample_color;
        const float luminance = (sample_color.x + sample_color.y + sample_color.z) / 3;
        square_sum += luminance * luminance;
    }
    const auto mean = color / static_cast<float>(shape.samples);
    const float luminance = (mean.x + mean.y + mean.z) / 3;
    if (variance)
        *variance =
            ::fmaxf(0, square_sum / static_cast<float>(shape.samples) - luminance * luminance) /
            static_cast<float>(shape.samples);
    return mean;
}
ECLAIR_HOST_DEVICE inline void ray_pixel(const ops::RayScene& scene, const ops::RayCamera& camera,
                                         const ops::RayTraceShape& shape, std::uint32_t index,
                                         std::uint8_t* rgba) {
    store_pixel(ray_color(scene, camera, shape, index), rgba);
}
}  // namespace eclair::render
