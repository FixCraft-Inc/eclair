/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026 FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */
#include "check.hpp"
#include "eclair/render/denoise.hpp"
#include <limits>
#include <vector>

int main() {
    using namespace eclair;
    using namespace render;
    ops::RayScene scene;
    scene.count = 1;
    auto& plane = scene.primitives[0];
    plane.kind = PrimitiveKind::Quad;
    plane.u = {10, 0, 0};
    plane.v = {0, 10, 0};
    ops::RayCamera camera;
    camera.origin = {0, 0, 3};
    ops::RayTraceShape shape{8, 8, 8, 2, 1};
    std::vector<PreviewPixel> current(64), previous(64);
    std::vector<Vec3> scratch(64), temporary(64);
    double raw_error = 0, spatial_error = 0;
    for (unsigned i = 0; i < 64; ++i) {
        current[i] = preview_sample(scene, camera, shape, i);
        const float value = (i + i / 8) % 2 == 0 ? 0.5F : 1.5F;
        current[i].color = {value, value, value};
        previous[i] = current[i];
        previous[i].color = {1, 1, 1};
        previous[i].history = 7;
    }
    ECLAIR_CHECK(!preview_boundary(current.data(), shape, 27));
    current[26].primitive = 1;
    ECLAIR_CHECK(preview_boundary(current.data(), shape, 27));
    current[26].primitive = 0;
    const auto saved_normal = current[26].normal;
    current[26].normal = -saved_normal;
    ECLAIR_CHECK(preview_boundary(current.data(), shape, 27));
    current[26].normal = saved_normal;
    for (unsigned i = 0; i < 64; ++i) scratch[i] = current[i].color;
    for (unsigned pass = 0; pass < kPreviewSpatialPasses; ++pass) {
        for (unsigned i = 0; i < 64; ++i)
            temporary[i] = spatial_preview(current.data(), scratch.data(), shape, i,
                                           1 << (pass / 2), pass % 2 != 0);
        scratch.swap(temporary);
    }
    for (unsigned i = 0; i < 64; ++i) {
        raw_error += std::pow(current[i].color.x - 1, 2);
        spatial_error += std::pow(scratch[i].x - 1, 2);
    }
    ECLAIR_CHECK(spatial_error < raw_error / 4);
    auto filtered = temporal_preview(current.data(), previous.data(), scratch.data(), scene, scene,
                                     camera, shape, 27, true);
    ECLAIR_CHECK(filtered.history == 8 && finite(filtered.color));
    ECLAIR_CHECK(std::fabs(filtered.color.x - 1) < std::fabs(scratch[27].x - 1));
    filtered = temporal_preview(current.data(), previous.data(), scratch.data(), scene, scene,
                                camera, shape, 27, false);
    ECLAIR_CHECK(filtered.history == 1 && filtered.color.x == scratch[27].x);
    for (auto& p : previous) p.primitive = 1;
    filtered = temporal_preview(current.data(), previous.data(), scratch.data(), scene, scene,
                                camera, shape, 27, true);
    ECLAIR_CHECK(filtered.history == 1);  // Disocclusion / different object.
    for (auto& p : previous) {
        p.primitive = 0;
        p.position.z = -100;
    }
    filtered = temporal_preview(current.data(), previous.data(), scratch.data(), scene, scene,
                                camera, shape, 27, true);
    ECLAIR_CHECK(filtered.history == 1);  // Reprojected depth mismatch.
    for (auto& p : previous) p.color.x = std::numeric_limits<float>::quiet_NaN();
    filtered = temporal_preview(current.data(), previous.data(), scratch.data(), scene, scene,
                                camera, shape, 27, true);
    ECLAIR_CHECK(filtered.history == 1 && finite(filtered.color));
    // Regression: a tiny range after spatial smoothing must not erase a
    // valid history value that lies inside the Monte Carlo variance margin.
    for (unsigned i = 0; i < 64; ++i) {
        current[i].variance = 0.01F;
        previous[i] = current[i];
        previous[i].color = {1.03F, 1.03F, 1.03F};
        previous[i].history = 31;
        scratch[i] = {1, 1, 1};
    }
    filtered = temporal_preview(current.data(), previous.data(), scratch.data(), scene, scene,
                                camera, shape, 27, true);
    ECLAIR_CHECK(filtered.history == 32 && filtered.color.x > 1.02F && filtered.color.x <= 1.03F);
    float x = 0, y = 0, footprint = 0;
    ECLAIR_CHECK(project_previous({0, 0, 0}, camera, shape, x, y, footprint));
    ECLAIR_CHECK(x == 3.5F && y == 3.5F && footprint > 0);
    ECLAIR_CHECK(!project_previous({0, 0, 4}, camera, shape, x, y, footprint));
    ECLAIR_CHECK(!project_previous({100, 0, 0}, camera, shape, x, y, footprint));
    ECLAIR_CHECK(!project_previous({std::numeric_limits<float>::infinity(), 0, 0}, camera, shape, x,
                                   y, footprint));
    PreviewPixel other = current[0];
    other.normal = -other.normal;
    ECLAIR_CHECK(!same_surface(current[0], other));
    other = current[0];
    other.primitive = kNoSurface;
    ECLAIR_CHECK(!same_surface(current[0], other));
    // Half of this single pixel covers an emissive square. Multisampling
    // must integrate coverage, instead of repeating the same center ray.
    plane.center = {1, 0, 0};
    plane.u = {1, 0, 0};
    plane.v = {0, 2, 0};
    plane.material.base_color = {1, 0, 0};
    plane.material.emission = 1;
    camera.vertical_tangent = 0.5F;
    shape.width = shape.height = 1;
    float variance = 0;
    ECLAIR_CHECK(preview_sample(scene, camera, shape, 0).filterable == 0);
    const auto aa = ray_color(scene, camera, shape, 0, &variance);
    ECLAIR_CHECK(std::fabs(variance - 1.0F / 288) < 1.0e-8F);
    ECLAIR_CHECK(aa.x == 0.5F && aa.y == 0 && aa.z == 0);
    const auto emitter_guide = preview_sample(scene, camera, shape, 0);
    const auto resolved_emitter = coverage_preview(&emitter_guide, scene, camera, shape, 0);
    ECLAIR_CHECK(resolved_emitter.x == 0.5F && resolved_emitter.y == 0 && resolved_emitter.z == 0);

    // Regression: a boundary must use denoised lighting from BOTH surfaces,
    // without inserting random raw paths into the displayed seam or mixing
    // coverage into the per-surface history. This pixel covers each half-plane
    // equally; change the seed to ensure coverage itself cannot flicker.
    scene.count = 2;
    plane.center = {-1, 0, 0};
    plane.u = {1, 0, 0};
    plane.v = {0, 2, 0};
    plane.material.emission = 0;
    scene.primitives[1] = plane;
    scene.primitives[1].center = {1, 0, 0};
    shape = {3, 3, 4, 3, 1};
    std::vector<PreviewPixel> guides(9);
    for (std::uint32_t i = 0; i < 9; ++i) {
        guides[i] = preview_sample(scene, camera, shape, i);
        guides[i].color = guides[i].primitive == 0 ? Vec3{0.2F, 0, 0} : Vec3{0, 0, 0.8F};
        ECLAIR_CHECK(guides[i].filterable == 1);
    }
    ECLAIR_CHECK(preview_boundary(guides.data(), shape, 4));
    const auto history_color = guides[4].color;
    for (std::uint32_t seed = 1; seed <= 100; ++seed) {
        shape.seed = seed;
        const auto color = coverage_preview(guides.data(), scene, camera, shape, 4);
        ECLAIR_CHECK(std::fabs(color.x - 0.1F) < 1.0e-6F && color.y == 0 &&
                     std::fabs(color.z - 0.4F) < 1.0e-6F);
    }
    ECLAIR_CHECK(guides[4].color.x == history_color.x && guides[4].color.z == history_color.z);
    const auto surface_color = spatial_preview(guides.data(), nullptr, shape, 4, 1, false);
    ECLAIR_CHECK(surface_color.x == 0 && std::fabs(surface_color.z - 0.8F) < 1.0e-6F);

    // Low-frequency indirect-light mottling remains after fine-scale passes.
    // Constant illumination is the known answer; the coarsest pair must reduce
    // eight-pixel noise patches without borrowing across geometry boundaries.
    shape = {64, 64, 4, 3, 1};
    guides.assign(4096, guides[4]);
    scratch.resize(4096);
    temporary.resize(4096);
    for (std::uint32_t i = 0; i < 4096; ++i) {
        const float value = ((i % 64) / 8 + (i / 64) / 8) % 2 ? 0.85F : 1.15F;
        guides[i].color = {value, value, value};
        scratch[i] = guides[i].color;
    }
    double fine_error = 0, coarse_error = 0;
    for (unsigned pass = 0; pass < kPreviewSpatialPasses; ++pass) {
        for (std::uint32_t i = 0; i < 4096; ++i)
            temporary[i] = spatial_preview(guides.data(), scratch.data(), shape, i, 1 << (pass / 2),
                                           pass % 2 != 0);
        scratch.swap(temporary);
        if (pass == 5)
            for (const auto color : scratch) fine_error += std::pow(color.x - 1, 2);
    }
    for (const auto color : scratch) coarse_error += std::pow(color.x - 1, 2);
    ECLAIR_CHECK(coarse_error < fine_error / 3);
    // Specular detail is averaged only if the ENTIRE scene and camera are
    // stationary. Motion offscreen can still change a reflection/refraction.
    scene.count = 1;
    plane = {};
    plane.material.kind = MaterialKind::Conductor;
    plane.material.roughness = 0.2F;
    shape = {1, 1, 4, 4, 1};
    auto guide = preview_sample(scene, camera, shape, 0);
    ECLAIR_CHECK(guide.filterable == 2);
    const Vec3 specular_color{2, 1, 0.5F};
    guide.color = specular_color;
    auto old = guide;
    old.color = {0, 0, 0};
    old.history = 7;
    ECLAIR_CHECK(spatial_preview(&guide, nullptr, shape, 0, 1, false).x == 2);
    filtered =
        temporal_preview(&guide, &old, &specular_color, scene, scene, camera, shape, 0, true, true);
    ECLAIR_CHECK(filtered.history == 8 && filtered.color.x == 0.25F);
    filtered = temporal_preview(&guide, &old, &specular_color, scene, scene, camera, shape, 0, true,
                                false);
    ECLAIR_CHECK(filtered.history == 1 && filtered.color.x == 2);
    old.history = 2000;
    filtered =
        temporal_preview(&guide, &old, &specular_color, scene, scene, camera, shape, 0, true, true);
    ECLAIR_CHECK(filtered.history == 1024);
    ECLAIR_CHECK(stationary_preview_scene(scene, camera, scene, camera));
    auto changed = scene;
    changed.primitives[0].material.roughness = 0.3F;
    ECLAIR_CHECK(!stationary_preview_scene(scene, camera, changed, camera));
    changed = scene;
    changed.light_emission.x += 1;
    ECLAIR_CHECK(!stationary_preview_scene(scene, camera, changed, camera));
    auto moved_camera = camera;
    moved_camera.origin.x += 0.01F;
    ECLAIR_CHECK(!stationary_preview_scene(scene, camera, scene, moved_camera));
    // Rough reflection filtering is limited to the fine-scale pair; sharp
    // highlights and refracted detail must never acquire the wide diffuse blur.
    plane.material.roughness = 0.4F;
    ECLAIR_CHECK(preview_sample(scene, camera, shape, 0).filterable == 3);
    shape = {5, 5, 1, 4, 1};
    guides.assign(25, guide);
    for (auto& p : guides) {
        p.filterable = 3;
        p.color = {1, 1, 1};
    }
    guides[12].color = {9, 9, 9};
    ECLAIR_CHECK(spatial_preview(guides.data(), nullptr, shape, 12, 1, false).x < 9);
    ECLAIR_CHECK(spatial_preview(guides.data(), nullptr, shape, 12, 2, false).x == 9);
    guides[12].filterable = 2;
    ECLAIR_CHECK(spatial_preview(guides.data(), nullptr, shape, 12, 1, false).x == 9);

    // A stationary diffuse surface must converge without neighborhood clipping.
    // The old 32-frame/clipped path retained blotches indefinitely at one spp.
    plane.material.kind = MaterialKind::Diffuse;
    guide.filterable = old.filterable = 1;
    old.color = {0.5F, 0.5F, 0.5F};
    old.history = 127;
    filtered = temporal_preview(&guide, &old, &specular_color, scene, scene, camera,
                                {1, 1, 1, 4, 1}, 0, true, true);
    ECLAIR_CHECK(filtered.history == 128);
    ECLAIR_CHECK(std::fabs(filtered.color.x - (0.5F + 1.5F / 128)) < 1.0e-6F);

    // Object-space parameterization must round-trip every face and both sides,
    // including rotated boxes. These are geometric known answers, not images.
    for (auto kind : {PrimitiveKind::Quad, PrimitiveKind::Box, PrimitiveKind::Sphere}) {
        Primitive p;
        p.kind = kind;
        p.center = {1, 2, 3};
        p.u = kind == PrimitiveKind::Quad ? Vec3{2, 0, 0} : Vec3{2, 3, 4};
        p.v = {0, 3, 0};
        p.rotation_cos = std::cos(0.4F);
        p.rotation_sin = std::sin(0.4F);
        const unsigned faces = kind == PrimitiveKind::Box ? 12U : 2U;
        for (unsigned face = 0; face < faces; ++face) {
            const auto point = lighting_point(p, face, 0.23F, 0.71F);
            Hit hit;
            hit.normal = point.normal;
            hit.front_face = face < faces / 2;
            std::uint32_t recovered_face = 99;
            float u = 0, v = 0;
            lighting_coordinates(p, point.position, hit, recovered_face, u, v);
            ECLAIR_CHECK(recovered_face == face);
            ECLAIR_CHECK(std::fabs(u - 0.23F) < 1.0e-5F);
            ECLAIR_CHECK(std::fabs(v - 0.71F) < 1.0e-5F);
        }
    }
    const ops::RayScene empty_scene;
    scene = empty_scene;
    scene.count = 3;
    scene.primitives[0].material.kind = MaterialKind::Conductor;
    scene.primitives[1].kind = PrimitiveKind::Quad;
    scene.primitives[1].u = {1, 0, 0};
    scene.primitives[1].v = {0, 1, 0};
    scene.primitives[2].material.emission = 1;
    const auto layout = lighting_layout(scene);
    ECLAIR_CHECK(layout.tiles == 2 && layout.offset[0] == 0 && layout.offset[1] == 0 &&
                 layout.offset[2] == 2);
    ECLAIR_CHECK(lighting_primitive(layout, scene.count, 0) == 1);
    std::vector<SurfaceLighting> lighting(2 * kLightingTilePixels);
    for (std::uint32_t i = 0; i < kLightingTilePixels; ++i) {
        lighting[i].color = {0.2F, 0.4F, 0.6F};
        lighting[i + kLightingTilePixels].color = {0.8F, 0.1F, 0.3F};
    }
    Hit hit;
    hit.primitive = 1;
    hit.normal = {0, 0, 1};
    const auto front = lighting_lookup(scene, layout, lighting.data(), {-1, 1, 0}, hit);
    ECLAIR_CHECK(std::fabs(front.x - 0.2F) < 1.0e-6F);
    hit.front_face = false;
    hit.normal = -hit.normal;
    const auto back = lighting_lookup(scene, layout, lighting.data(), {1, -1, 0}, hit);
    ECLAIR_CHECK(std::fabs(back.x - 0.8F) < 1.0e-6F);
    ECLAIR_CHECK(std::fabs(lighting_spatial(lighting.data(), nullptr, 0, false).x - 0.2F) <
                 1.0e-6F);
    auto accumulated = lighting_temporal({{0, 0, 0}, 7}, {1, 1, 1}, true, true);
    ECLAIR_CHECK(accumulated.history == 8 && accumulated.color.x == 0.125F);
    accumulated = lighting_temporal({{0, 0, 0}, 1024}, {1, 1, 1}, true, false);
    ECLAIR_CHECK(accumulated.history == 32 && accumulated.color.x == 0.03125F);
    accumulated = lighting_temporal({{0, 0, 0}, 1024}, {1, 1, 1}, false, true);
    ECLAIR_CHECK(accumulated.history == 1 && accumulated.color.x == 1);
    // Clear glass in a constant environment conserves radiance across both
    // interfaces. IOR=1 with tint has the exact two-interface squared tint.
    scene = empty_scene;
    scene.count = 1;
    scene.environment = {1, 1, 1};
    auto& glass = scene.primitives[0];
    glass.material.kind = MaterialKind::Dielectric;
    glass.material.roughness = 0;
    glass.material.base_color = {1, 1, 1};
    const Ray glass_ray{{0, 0, 3}, {0, 0, -1}};
    const auto clear = trace_lit(scene, {}, nullptr, glass_ray, 8, 0.1F, 0.2F);
    ECLAIR_CHECK(std::fabs(clear.x - 1) < 1.0e-5F && finite(clear));
    glass.material.ior = 1;
    glass.material.base_color = {0.5F, 0.8F, 1};
    const auto tinted = trace_lit(scene, {}, nullptr, glass_ray, 8, 0.7F, 0.8F);
    ECLAIR_CHECK(std::fabs(tinted.x - 0.25F) < 1.0e-6F && std::fabs(tinted.y - 0.64F) < 1.0e-6F &&
                 tinted.z == 1);
    // The narrow glass median removes an isolated spark but preserves a
    // monotonic refracted edge and never borrows across a primary silhouette.
    shape = {5, 5, 4, 6, 1};
    guides.assign(25, guide);
    for (auto& p : guides) {
        p.filterable = 5;
        p.color = {1, 1, 1};
    }
    guides[12].color = {9, 9, 9};
    ECLAIR_CHECK(spatial_preview(guides.data(), nullptr, shape, 12, 1, false, true).x == 1);
    guides[13].color = {9, 9, 9};
    ECLAIR_CHECK(spatial_preview(guides.data(), nullptr, shape, 12, 1, false, true).x == 9);
    guides[13].primitive = kNoSurface;
    ECLAIR_CHECK(spatial_preview(guides.data(), nullptr, shape, 12, 1, false, true).x == 9);
    return test::finish();
}
