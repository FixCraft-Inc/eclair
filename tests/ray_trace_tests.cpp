/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/host/ray_trace.hpp"
#include "eclair/render/shading.hpp"

#include <cmath>
#include <limits>
#include <vector>
#include <cstdio>

namespace {
void glossy_light_energy() {
    using namespace eclair::render;
    eclair::ops::RayScene scene;
    scene.count = 2;
    scene.light_center = {0, 0, 2};
    scene.light_u = {0.5F, 0, 0};
    scene.light_v = {0, -0.5F, 0};
    scene.light_emission = {1, 1, 1};
    auto& surface = scene.primitives[0];
    surface.kind = PrimitiveKind::Quad;
    surface.u = {10, 0, 0};
    surface.v = {0, 10, 0};
    surface.material.kind = MaterialKind::Conductor;
    surface.material.base_color = {1, 1, 1};
    auto& light = scene.primitives[1];
    light.kind = PrimitiveKind::Quad;
    light.center = scene.light_center;
    light.u = scene.light_u;
    light.v = scene.light_v;
    light.material.base_color = {1, 1, 1};
    light.material.emission = 1;
    for (const float roughness : {0.2F, 0.4F}) {
        surface.material.roughness = roughness;
        // Independent area quadrature checks both sampling policies against
        // the rendering equation, rather than another stochastic path trace.
        double expected = 0;
        for (unsigned y = 0; y < 200; ++y) {
            for (unsigned x = 0; x < 200; ++x) {
                const Vec3 delta{(static_cast<float>(x) + 0.5F) / 200 - 0.5F,
                                 (static_cast<float>(y) + 0.5F) / 200 - 0.5F, 2};
                const Vec3 direction = normalized(delta);
                const auto bsdf = evaluate_bsdf(surface.material, {0, 0, 1}, {0, 0, 1}, direction);
                expected += static_cast<double>(bsdf.value.x * direction.z * direction.z /
                                                dot(delta, delta)) /
                            40000;
            }
        }
        std::uint32_t state = 123456;
        double sum = 0;
        for (unsigned i = 0; i < 40000; ++i)
            sum += trace(scene, {{0, 0, 1}, {0, 0, -1}}, 2, state).x;
        const double mean = sum / 40000;
        std::printf("roughness=%.2f area_quadrature=%.6f path_mean=%.6f tolerance=0.015\n",
                    roughness, expected, mean);
        ECLAIR_CHECK(std::fabs(mean - expected) < 0.015);
    }
    surface.material.roughness = 0.2F;
    std::uint32_t state = 7;
    ECLAIR_CHECK(trace(scene, {{0, 0, 1}, {0, 0, -1}}, 1, state).x > 0);
}
}  // namespace

int main() {
    using namespace eclair::render;
    using eclair::ops::Status;
    glossy_light_energy();
    Primitive sphere;
    Hit hit;
    ECLAIR_CHECK(intersect(sphere, {{0, 0, 3}, {0, 0, -1}}, 0, 10, hit));
    ECLAIR_CHECK(hit.distance == 2 && hit.normal.z == 1);
    ECLAIR_CHECK(intersect(sphere, {{0, 0, 0}, {1, 0, 0}}, 0.001F, 10, hit));
    ECLAIR_CHECK(hit.distance == 1 && hit.normal.x == -1);
    ECLAIR_CHECK(intersect(sphere, {{1, 0, 3}, {0, 0, -1}}, 0, 10, hit));
    ECLAIR_CHECK(hit.distance == 3);
    ECLAIR_CHECK(!intersect(sphere, {{2, 0, 3}, {0, 0, -1}}, 0, 10, hit));
    ECLAIR_CHECK(!intersect(sphere, {{0, 0, 3}, {0, 0, -1}}, 0, 1.99F, hit));
    ECLAIR_CHECK(!intersect(sphere, {{0, 0, 3}, {0, 0, 0}}, 0, 10, hit));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    ECLAIR_CHECK(!intersect(sphere, {{nan, 0, 3}, {0, 0, -1}}, 0, 10, hit));
    ECLAIR_CHECK(!intersect(sphere, {{0, 0, 3}, {inf, 0, -1}}, 0, 10, hit));
    ECLAIR_CHECK(!intersect(sphere, {{0, 0, 3}, {0, 0, -1}}, nan, 10, hit));
    sphere.radius = 0;
    ECLAIR_CHECK(!valid_primitive(sphere));
    sphere.radius = 1;
    Primitive box;
    box.kind = PrimitiveKind::Box;
    box.u = {1, 1, 1};
    for (const float zero : {0.0F, -0.0F, std::numeric_limits<float>::denorm_min()}) {
        ECLAIR_CHECK(intersect(box, {{0, 0, 3}, {zero, zero, -1}}, 0, 10, hit));
        ECLAIR_CHECK(hit.distance == 2 && hit.normal.z == 1);
        ECLAIR_CHECK(!intersect(box, {{2, 0, 3}, {zero, zero, -1}}, 0, 10, hit));
    }
    ECLAIR_CHECK(intersect(box, {{0, 0, 0}, {1, 0, 0}}, 0.001F, 10, hit));
    ECLAIR_CHECK(hit.distance == 1 && hit.normal.x == -1);
    box.rotation_cos = 0;
    box.rotation_sin = 1;
    box.u = {0.5F, 1, 2};
    ECLAIR_CHECK(intersect(box, {{0, 0, 3}, {0, 0, -1}}, 0, 10, hit));
    ECLAIR_CHECK(hit.distance == 2.5F);
    box.u.x = 0;
    ECLAIR_CHECK(!valid_primitive(box));
    Primitive quad;
    quad.kind = PrimitiveKind::Quad;
    quad.u = {1, 0, 0};
    quad.v = {0, 1, 0};
    ECLAIR_CHECK(intersect(quad, {{0, 0, 3}, {0, 0, -1}}, 0, 10, hit));
    ECLAIR_CHECK(hit.distance == 3 && hit.normal.z == 1);
    ECLAIR_CHECK(!intersect(quad, {{2, 0, 3}, {0, 0, -1}}, 0, 10, hit));
    ECLAIR_CHECK(!intersect(quad, {{0, 0, 3}, {1, 0, 0}}, 0, 10, hit));
    quad.v = quad.u;
    ECLAIR_CHECK(!valid_primitive(quad));
    eclair::ops::RayScene scene;
    scene.count = 1;
    scene.primitives[0] = sphere;
    scene.primitives[0].material.emission = 1;
    scene.primitives[0].material.base_color = {1, 0, 0};
    eclair::ops::RayCamera camera;
    camera.origin = {0, 0, 3};
    eclair::ops::RayTraceShape shape{1, 1, 1, 2, 1};
    std::vector<std::uint8_t> rgba(8, 77);
    ECLAIR_CHECK(eclair::host::ray_trace_reference(scene, camera, shape, {rgba.data(), 4}) ==
                 Status::Success);
    ECLAIR_CHECK(rgba[0] == 232 && rgba[1] == 0 && rgba[2] == 0 && rgba[3] == 255);
    ECLAIR_CHECK(rgba[4] == 77 && rgba[7] == 77);
    ECLAIR_CHECK(eclair::host::ray_trace_reference(scene, camera, shape, {rgba.data(), 3}) ==
                 Status::BufferTooSmall);
    ECLAIR_CHECK(eclair::host::ray_trace_reference(scene, camera, shape, {nullptr, 4}) ==
                 Status::InvalidArgument);
    camera.right = camera.forward;
    ECLAIR_CHECK(eclair::ops::validate_ray_trace(scene, camera, shape) == Status::InvalidArgument);
    camera = {};
    scene.count = 17;
    ECLAIR_CHECK(eclair::ops::validate_ray_trace(scene, camera, shape) == Status::InvalidArgument);
    scene.count = 1;
    shape.width = std::numeric_limits<std::uint32_t>::max();
    std::size_t bytes = 0;
    ECLAIR_CHECK(eclair::ops::ray_frame_bytes(shape, &bytes) == Status::InvalidArgument);
    shape.width = 1;
    shape.samples = 0;
    ECLAIR_CHECK(eclair::ops::validate_ray_trace(scene, camera, shape) == Status::InvalidArgument);
    ECLAIR_CHECK(display_channel(0) == 0 && display_channel(1) == 232);
    ECLAIR_CHECK(display_channel(inf) == 255 && display_channel(-inf) == 0 &&
                 display_channel(nan) == 0);
    ECLAIR_CHECK(display_channel(-0.0F) == 0 &&
                 display_channel(std::numeric_limits<float>::denorm_min()) == 0);
    // Refraction must cross BOTH interfaces, offset into/out of the solid,
    // and cancel the radiance eta^2 factors on returning to air.
    scene = eclair::ops::RayScene{};
    scene.count = 2;
    auto& glass = scene.primitives[0];
    glass.material.kind = MaterialKind::Dielectric;
    glass.material.base_color = {1, 1, 1};
    glass.material.roughness = 0;
    auto& emitter = scene.primitives[1];
    emitter.kind = PrimitiveKind::Quad;
    emitter.center = {0, 0, -3};
    emitter.u = {2, 0, 0};
    emitter.v = {0, 2, 0};
    emitter.material.base_color = {1, 0, 0};
    emitter.material.emission = 1;
    std::uint32_t state = 999;
    const auto through = trace(scene, {{0, 0, 3}, {0, 0, -1}}, 3, state);
    ECLAIR_CHECK(std::fabs(through.x - 1) < 1.0e-5F && through.y == 0 && through.z == 0);
    glass.kind = PrimitiveKind::Box;
    glass.u = {1, 1, 1};
    state = 999;
    const auto through_box = trace(scene, {{0, 0, 3}, {0, 0, -1}}, 3, state);
    ECLAIR_CHECK(std::fabs(through_box.x - 1) < 1.0e-5F);
    shape.samples = 1;
    shape.bounces = 8;
    ECLAIR_CHECK(eclair::ops::validate_ray_trace(scene, camera, shape) == Status::Success);
    emitter = glass;
    ECLAIR_CHECK(eclair::ops::validate_ray_trace(scene, camera, shape) == Status::InvalidArgument);
    emitter.center.x = 3;
    ECLAIR_CHECK(eclair::ops::validate_ray_trace(scene, camera, shape) == Status::Success);
    scene.count = 1;
    scene.primitives[0].center = {8, 0, 0};
    scene.environment = {0.25F, 0.5F, 1};
    state = 99;
    const auto sky = trace(scene, {{0, 0, 3}, {0, 0, -1}}, 3, state);
    ECLAIR_CHECK(sky.x == 0.25F && sky.y == 0.5F && sky.z == 1);
    scene.environment.x = nan;
    ECLAIR_CHECK(eclair::ops::validate_ray_trace(scene, camera, shape) == Status::InvalidArgument);
    return eclair::test::finish();
}
