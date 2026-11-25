/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "scene.hpp"
#include "eclair/render/shading.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

namespace eclair::demo {
namespace {
using namespace render;
void validate(const FrameScene& frame, const ops::RayTraceShape& shape,
              std::span<std::uint8_t> rgba) {
    std::size_t bytes = 0;
    if (ops::validate_ray_trace(frame.scene, frame.camera, shape) != ops::Status::Success ||
        ops::ray_frame_bytes(shape, &bytes) != ops::Status::Success || rgba.size() < bytes)
        throw std::invalid_argument("invalid scene, camera, shape or frame extent");
}
struct ScreenVertex {
    float x, y, reciprocal_z;
};
float edge(const ScreenVertex& a, const ScreenVertex& b, float x, float y) {
    return (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
}
}  // namespace

FrameScene cornell_scene(float seconds) {
    if (!std::isfinite(seconds) || std::fabs(seconds) > 1.0e6F)
        throw std::invalid_argument("invalid scene time");
    FrameScene frame;
    auto& scene = frame.scene;
    const auto quad = [&](Vec3 center, Vec3 u, Vec3 v, Vec3 color, float emission = 0) {
        Primitive p;
        p.kind = PrimitiveKind::Quad;
        p.center = center;
        p.u = u;
        p.v = v;
        p.material.base_color = color;
        p.material.emission = emission;
        scene.primitives[scene.count++] = p;
    };
    const Vec3 white{0.73F, 0.73F, 0.73F};
    quad({-1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0.65F, 0.05F, 0.05F});
    quad({1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0.12F, 0.45F, 0.15F});
    quad({0, 0, 0}, {1, 0, 0}, {0, 0, 1}, white);
    quad({0, 2, 0}, {1, 0, 0}, {0, 0, 1}, white);
    quad({0, 1, -1}, {1, 0, 0}, {0, 1, 0}, white);
    quad(scene.light_center, scene.light_u, scene.light_v, {1, 1, 1}, 12);
    const auto box = [&](Vec3 center, Vec3 extent, float angle) {
        Primitive p;
        p.kind = PrimitiveKind::Box;
        p.center = center;
        p.u = extent;
        p.material.base_color = white;
        p.rotation_cos = std::cos(angle);
        p.rotation_sin = std::sin(angle);
        scene.primitives[scene.count++] = p;
    };
    box({-0.4F, 0.38F, 0.3F}, {0.3F, 0.38F, 0.3F}, -0.25F + 0.22F * std::sin(seconds * 0.8F));
    box({0.38F, 0.68F, -0.35F}, {0.28F, 0.68F, 0.28F}, 0.3F);
    frame.camera.origin = {0.18F * std::sin(seconds * 0.5F), 1.05F, 3.6F};
    frame.camera.forward = normalized(Vec3{0, 0.95F, -0.1F} - frame.camera.origin);
    frame.camera.right = normalized(cross(frame.camera.forward, {0, 1, 0}));
    frame.camera.up = cross(frame.camera.right, frame.camera.forward);
    frame.camera.vertical_tangent = 0.34F;
    return frame;
}

FrameScene materials_scene(float seconds) {
    auto frame = cornell_scene(seconds);
    auto& s = frame.scene;
    s.count = 6;
    s.environment = {0.1F, 0.13F, 0.18F};
    s.primitives[0].material.base_color = {0.72F, 0.018F, 0.035F};
    s.primitives[1].material.base_color = {0.018F, 0.42F, 0.62F};
    auto& floor = s.primitives[2].material;
    floor.kind = MaterialKind::Conductor;
    floor.base_color = {0.55F, 0.59F, 0.65F};
    floor.roughness = 0.16F;
    s.primitives[3].material.base_color = {0.65F, 0.65F, 0.65F};
    s.primitives[4].material.base_color = {0.6F, 0.62F, 0.67F};
    s.light_u = {0.42F, 0, 0};
    s.light_v = {0, 0, 0.35F};
    s.light_emission = {10, 9.6F, 9.0F};
    s.primitives[5].u = s.light_u;
    s.primitives[5].v = s.light_v;
    s.primitives[5].material.base_color = {1, 0.96F, 0.9F};
    s.primitives[5].material.emission = 10;
    Primitive gold;
    gold.center = {-0.46F, 0.35F, 0.3F};
    gold.radius = 0.35F;
    gold.material.kind = MaterialKind::Conductor;
    gold.material.base_color = {0.95F, 0.64F, 0.19F};
    gold.material.roughness = 0.12F;
    s.primitives[s.count++] = gold;
    Primitive glass;
    glass.center = {0.4F, 0.46F, 0.22F};
    glass.radius = 0.46F;
    glass.material.kind = MaterialKind::Dielectric;
    glass.material.base_color = {0.96F, 0.99F, 1};
    glass.material.roughness = 0;
    glass.material.ior = 1.5F;
    s.primitives[s.count++] = glass;
    Primitive copper;
    copper.kind = PrimitiveKind::Box;
    copper.center = {-0.38F, 0.65F, -0.62F};
    copper.u = {0.27F, 0.65F, 0.27F};
    const float angle = 0.25F + 0.2F * std::sin(seconds * 0.7F);
    copper.rotation_cos = std::cos(angle);
    copper.rotation_sin = std::sin(angle);
    copper.material.kind = MaterialKind::Conductor;
    copper.material.base_color = {0.93F, 0.4F, 0.22F};
    copper.material.roughness = 0.42F;
    s.primitives[s.count++] = copper;
    Primitive accent;
    accent.kind = PrimitiveKind::Box;
    accent.center = {0.62F, 0.65F, -0.7F};
    accent.u = {0.14F, 0.65F, 0.14F};
    accent.material.base_color = {0.65F, 0.025F, 0.22F};
    s.primitives[s.count++] = accent;
    return frame;
}

FrameScene make_scene(SceneKind kind, float seconds) {
    switch (kind) {
        case SceneKind::Cornell:
            return cornell_scene(seconds);
        case SceneKind::Materials:
            return materials_scene(seconds);
    }
    throw std::invalid_argument("invalid demo scene");
}

void cpu_ray_frame(const FrameScene& frame, const ops::RayTraceShape& shape,
                   std::span<std::uint8_t> rgba, unsigned threads) {
    validate(frame, shape, rgba);
    threads = std::clamp(threads, 1U, 8U);
    const std::uint32_t pixels = shape.width * shape.height;
    std::vector<std::jthread> workers;
    workers.reserve(threads);
    for (unsigned thread = 0; thread < threads; ++thread) {
        workers.emplace_back([&, thread] {
            for (std::uint32_t y = thread; y < shape.height; y += threads)
                for (std::uint32_t x = 0; x < shape.width; ++x) {
                    const std::uint32_t i = y * shape.width + x;
                    if (i < pixels)
                        ray_pixel(frame.scene, frame.camera, shape, i,
                                  rgba.data() + static_cast<std::size_t>(i) * 4);
                }
        });
    }
}

void raster_frame(const FrameScene& frame, const ops::RayTraceShape& shape,
                  std::span<std::uint8_t> rgba) {
    validate(frame, shape, rgba);
    const std::size_t count = static_cast<std::size_t>(shape.width) * shape.height;
    std::vector<float> depth(count, 0);
    for (std::size_t i = 0; i < count; ++i) store_pixel({}, rgba.data() + i * 4);
    const float width = static_cast<float>(shape.width), height = static_cast<float>(shape.height);
    const auto triangle = [&](const Primitive& p, Vec3 a, Vec3 b, Vec3 c, Vec3 normal) {
        const Vec3 center = (a + b + c) / 3;
        if (dot(normal, frame.camera.origin - center) < 0) normal = -normal;
        const Vec3 delta = frame.scene.light_center - center;
        const float brightness = 0.12F + 1.5F * std::fmax(0.0F, dot(normal, normalized(delta)));
        const Vec3 color = p.material.emission > 0 ? p.material.base_color * p.material.emission
                                                   : p.material.base_color * brightness;
        std::array<ScreenVertex, 3> projected{};
        const std::array<Vec3, 3> vertices{a, b, c};
        for (std::size_t i = 0; i < 3; ++i) {
            const Vec3 v = vertices[i] - frame.camera.origin;
            const float z = dot(v, frame.camera.forward);
            if (z < 0.01F) return;
            const float x =
                dot(v, frame.camera.right) / (z * frame.camera.vertical_tangent * (width / height));
            const float y = dot(v, frame.camera.up) / (z * frame.camera.vertical_tangent);
            projected[i] = {(x + 1) * 0.5F * width, (1 - y) * 0.5F * height, 1 / z};
        }
        const auto& v0 = projected[0];
        const auto& v1 = projected[1];
        const auto& v2 = projected[2];
        const float area = edge(v0, v1, v2.x, v2.y);
        if (std::fabs(area) < 1.0e-6F) return;
        const int x0 =
            static_cast<int>(std::clamp(std::floor(std::min({v0.x, v1.x, v2.x})), 0.0F, width - 1));
        const int x1 =
            static_cast<int>(std::clamp(std::ceil(std::max({v0.x, v1.x, v2.x})), 0.0F, width - 1));
        const int y0 = static_cast<int>(
            std::clamp(std::floor(std::min({v0.y, v1.y, v2.y})), 0.0F, height - 1));
        const int y1 =
            static_cast<int>(std::clamp(std::ceil(std::max({v0.y, v1.y, v2.y})), 0.0F, height - 1));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const float px = static_cast<float>(x) + 0.5F, py = static_cast<float>(y) + 0.5F;
                const float u = edge(v1, v2, px, py) / area, v = edge(v2, v0, px, py) / area,
                            w = 1 - u - v;
                if (u < 0 || v < 0 || w < 0) continue;
                const float z = u * v0.reciprocal_z + v * v1.reciprocal_z + w * v2.reciprocal_z;
                const std::size_t index =
                    static_cast<std::size_t>(y) * shape.width + static_cast<std::size_t>(x);
                if (z > depth[index]) {
                    depth[index] = z;
                    store_pixel(color, rgba.data() + index * 4);
                }
            }
    };
    const auto face = [&](const Primitive& p, Vec3 center, Vec3 u, Vec3 v) {
        const Vec3 normal = normalized(cross(u, v));
        triangle(p, center - u - v, center + u - v, center + u + v, normal);
        triangle(p, center - u - v, center + u + v, center - u + v, normal);
    };
    for (std::uint32_t i = 0; i < frame.scene.count; ++i) {
        const auto& p = frame.scene.primitives[i];
        switch (p.kind) {
            case PrimitiveKind::Quad:
                face(p, p.center, p.u, p.v);
                break;
            case PrimitiveKind::Box:
                for (int axis = 0; axis < 3; ++axis)
                    for (float sign : {-1.0F, 1.0F}) {
                        const Vec3 u = axis_vector((axis + 1) % 3, component(p.u, (axis + 1) % 3));
                        const Vec3 v = axis_vector((axis + 2) % 3, component(p.u, (axis + 2) % 3));
                        const Vec3 center =
                            p.center + box_world(p, axis_vector(axis, sign * component(p.u, axis)));
                        face(p, center, box_world(p, u), box_world(p, v));
                    }
                break;
            case PrimitiveKind::Sphere:
                // Tessellation is only the CPU raster baseline; ray tracing
                // continues to use analytic spheres. No refractive raster claim.
                for (unsigned y = 0; y < 16; ++y) {
                    for (unsigned x = 0; x < 32; ++x) {
                        const auto vertex = [&](unsigned latitude, unsigned longitude) {
                            const float theta = kPi * static_cast<float>(latitude) / 16;
                            const float phi = 2 * kPi * static_cast<float>(longitude) / 32;
                            return p.center + Vec3{std::sin(theta) * std::cos(phi), std::cos(theta),
                                                   std::sin(theta) * std::sin(phi)} *
                                                  p.radius;
                        };
                        const Vec3 a = vertex(y, x), b = vertex(y + 1, x), c = vertex(y + 1, x + 1),
                                   d = vertex(y, x + 1);
                        triangle(p, a, b, c, normalized((a + b + c) / 3 - p.center));
                        triangle(p, a, c, d, normalized((a + c + d) / 3 - p.center));
                    }
                }
                break;
        }
    }
}
}  // namespace eclair::demo
