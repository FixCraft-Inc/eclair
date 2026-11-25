/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "cuda_support.hpp"
#include "eclair/cuda/ray_trace.hpp"
#include "eclair/cuda/ray_preview.hpp"
#include "eclair/host/ray_trace.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    return eclair::test::contain([] {
        using namespace eclair;
        using test::checked;
        test::print_device_header();
        checked(cuda::check_ray_trace_image(0), "ray kernel image");
        ops::RayScene scene;
        const auto quad = [&](render::Vec3 center, render::Vec3 u, render::Vec3 v,
                              render::Vec3 color, float emission = 0) {
            render::Primitive p;
            p.kind = render::PrimitiveKind::Quad;
            p.center = center;
            p.u = u;
            p.v = v;
            p.material.base_color = color;
            p.material.emission = emission;
            scene.primitives[scene.count++] = p;
        };
        quad({-1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0.65F, 0.05F, 0.05F});
        quad({1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0.12F, 0.45F, 0.15F});
        quad({0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0.73F, 0.73F, 0.73F});
        quad({0, 2, 0}, {1, 0, 0}, {0, 0, 1}, {0.73F, 0.73F, 0.73F});
        quad({0, 1, -1}, {1, 0, 0}, {0, 1, 0}, {0.73F, 0.73F, 0.73F});
        quad(scene.light_center, scene.light_u, scene.light_v, {1, 1, 1}, 12);
        render::Primitive box;
        box.kind = render::PrimitiveKind::Box;
        box.center = {-0.4F, 0.4F, 0.3F};
        box.u = {0.3F, 0.4F, 0.3F};
        box.material.kind = render::MaterialKind::Conductor;
        box.material.roughness = 0.5F;
        scene.primitives[scene.count++] = box;
        render::Primitive sphere;
        sphere.center = {0.4F, 0.5F, -0.3F};
        sphere.radius = 0.5F;
        sphere.material.kind = render::MaterialKind::Conductor;
        sphere.material.roughness = 0.12F;
        scene.primitives[scene.count++] = sphere;
        render::Primitive glass;
        glass.center = {0.2F, 0.3F, 0.5F};
        glass.radius = 0.24F;
        glass.material.kind = render::MaterialKind::Dielectric;
        glass.material.base_color = {0.95F, 0.98F, 1};
        glass.material.roughness = 0;
        scene.primitives[scene.count++] = glass;
        ops::RayCamera camera;
        ops::RayTraceShape shape{97, 65, 2, 6, 1};
        std::size_t bytes = 0;
        ECLAIR_CHECK(ops::ray_frame_bytes(shape, &bytes) == ops::Status::Success);
        test::Stream stream;
        std::vector<std::uint8_t> initial(bytes + 16, 77);
        test::DeviceBuffer<std::uint8_t> device(initial, stream.get());
        std::vector<std::uint8_t> reference(bytes);
        for (const auto seed : {1U, 42U, 123456U}) {
            shape.seed = seed;
            ECLAIR_CHECK(host::ray_trace_reference(scene, camera, shape,
                                                   {reference.data(), reference.size()}) ==
                         ops::Status::Success);
            checked(
                cuda::launch_ray_trace(scene, camera, shape, {device.data(), bytes}, stream.get()),
                "trace frame");
            const auto actual = device.download(stream.get());
            std::size_t outliers = 0;
            unsigned maximum = 0;
            std::uint64_t sum = 0;
            for (std::size_t i = 0; i < bytes; ++i) {
                const unsigned difference = static_cast<unsigned>(
                    std::abs(static_cast<int>(reference[i]) - static_cast<int>(actual[i])));
                if (difference > 2) ++outliers;
                maximum = std::max(maximum, difference);
                sum += difference;
            }
            const double mean = static_cast<double>(sum) / static_cast<double>(bytes);
            std::printf(
                "shape=%ux%u samples=2 bounces=6 seed=%u FP32 RGBA8 channel_tolerance=2 "
                "outliers=%zu/%zu max=%u mean=%.6f synchronization=stream\n",
                shape.width, shape.height, seed, outliers, bytes, maximum, mean);
            ECLAIR_CHECK(outliers <= bytes / 1000 && mean <= 0.1);
            for (std::size_t i = bytes; i < actual.size(); ++i) ECLAIR_CHECK(actual[i] == 77);
        }
        ECLAIR_CHECK(cuda::launch_ray_trace(scene, camera, shape, {device.data(), bytes - 1},
                                            stream.get()) == cudaErrorInvalidValue);
        ECLAIR_CHECK(cuda::launch_ray_trace(scene, camera, shape, {nullptr, bytes}, stream.get()) ==
                     cudaErrorInvalidValue);
        // Compare both filtering passes against the scalar reference across
        // history creation and reuse. Device buffers include output sentinels.
        const std::size_t count = bytes / 4;
        std::vector<render::PreviewPixel> empty(count), host_previous(count), host_current(count);
        std::vector<render::Vec3> host_scratch(count), temporary(count), device_scratch(count * 2);
        test::DeviceBuffer<render::PreviewPixel> a(empty, stream.get()), b(empty, stream.get());
        test::DeviceBuffer<render::Vec3> scratch(device_scratch, stream.get());
        for (unsigned frame = 0; frame < 3; ++frame) {
            shape.seed = frame + 44;
            for (std::uint32_t i = 0; i < count; ++i)
                host_current[i] = render::preview_sample(scene, camera, shape, i);
            for (std::uint32_t i = 0; i < count; ++i) host_scratch[i] = host_current[i].color;
            for (unsigned pass = 0; pass < render::kPreviewSpatialPasses; ++pass) {
                for (std::uint32_t i = 0; i < count; ++i)
                    temporary[i] =
                        render::spatial_preview(host_current.data(), host_scratch.data(), shape, i,
                                                1 << (pass / 2), pass % 2 != 0);
                host_scratch.swap(temporary);
            }
            auto next = host_current;
            for (std::uint32_t i = 0; i < count; ++i) {
                next[i] = render::temporal_preview(host_current.data(), host_previous.data(),
                                                   host_scratch.data(), scene, scene, camera, shape,
                                                   i, frame != 0, true);
            }
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto color =
                    render::preview_boundary(next.data(), shape, i)
                        ? render::coverage_preview(next.data(), scene, camera, shape, i)
                        : next[i].color;
                render::store_pixel(color, reference.data() + i * 4);
            }
            auto* current = frame % 2 == 0 ? a.data() : b.data();
            const auto* previous = frame % 2 == 0 ? b.data() : a.data();
            checked(cuda::launch_ray_preview(scene, camera, scene, camera, shape, frame != 0,
                                             {current, count}, {previous, count},
                                             {scratch.data(), count * 2}, {device.data(), bytes},
                                             stream.get()),
                    "filtered preview");
            const auto actual = device.download(stream.get());
            unsigned maximum = 0;
            for (std::size_t i = 0; i < bytes; ++i) {
                const auto difference = static_cast<unsigned>(
                    std::abs(static_cast<int>(actual[i]) - static_cast<int>(reference[i])));
                maximum = std::max(maximum, difference);
                ECLAIR_CHECK(difference <= 2);
            }
            for (std::size_t i = bytes; i < actual.size(); ++i) ECLAIR_CHECK(actual[i] == 77);
            std::printf("filtered_preview frame=%u max_channel_error=%u tolerance=2 sync=stream\n",
                        frame, maximum);
            host_previous = std::move(next);
        }
        // Cached lighting must match the scalar pipeline too, including both
        // Fresnel branches and a full atlas history update. This fixture has
        // reflected/refracted diffuse surfaces as well as direct views.
        const auto layout = render::lighting_layout(scene);
        const std::size_t texels = layout.tiles * render::kLightingTilePixels;
        std::vector<render::SurfaceLighting> atlas(texels), atlas_previous(texels);
        std::vector<render::Vec3> lighting_x(texels), lighting_y(texels);
        test::DeviceBuffer<render::SurfaceLighting> ca(atlas, stream.get()),
            cb(atlas, stream.get());
        std::vector<render::Vec3> cache_scratch(texels * 2);
        test::DeviceBuffer<render::Vec3> cs(cache_scratch, stream.get());
        for (unsigned frame = 0; frame < 2; ++frame) {
            shape.seed = frame + 71;
            for (std::uint32_t i = 0; i < texels; ++i)
                atlas[i] = render::lighting_sample(scene, layout, shape, i);
            for (std::uint32_t i = 0; i < texels; ++i)
                lighting_x[i] = render::lighting_spatial(atlas.data(), nullptr, i, false);
            for (std::uint32_t i = 0; i < texels; ++i)
                lighting_y[i] = render::lighting_spatial(atlas.data(), lighting_x.data(), i, true);
            for (std::uint32_t i = 0; i < texels; ++i)
                atlas[i] = render::lighting_temporal(atlas_previous[i], lighting_y[i], frame != 0,
                                                     true, atlas[i].direct);
            for (std::uint32_t i = 0; i < count; ++i) {
                host_current[i] =
                    render::preview_sample(scene, camera, shape, i, layout, atlas.data());
                host_scratch[i] = host_current[i].color;
            }
            for (unsigned pass = 0; pass < render::kPreviewSpatialPasses; ++pass) {
                for (std::uint32_t i = 0; i < count; ++i)
                    temporary[i] =
                        render::spatial_preview(host_current.data(), host_scratch.data(), shape, i,
                                                1 << (pass / 2), pass % 2 != 0, true);
                host_scratch.swap(temporary);
            }
            auto next = host_current;
            for (std::uint32_t i = 0; i < count; ++i)
                next[i] = render::temporal_preview(
                    host_current.data(), host_previous.data(), host_scratch.data(), scene, scene,
                    camera, shape, i, frame != 0 && host_current[i].filterable != 1, true);
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto color = render::preview_boundary(next.data(), shape, i)
                                       ? render::coverage_preview(next.data(), scene, camera, shape,
                                                                  i, layout, atlas.data())
                                       : next[i].color;
                render::store_pixel(color, reference.data() + i * 4);
            }
            auto* cache_current = frame == 0 ? ca.data() : cb.data();
            const auto* cache_old = frame == 0 ? cb.data() : ca.data();
            auto* current = frame == 0 ? a.data() : b.data();
            const auto* previous = frame == 0 ? b.data() : a.data();
            checked(cuda::launch_ray_preview(
                        scene, camera, scene, camera, shape, frame != 0, {current, count},
                        {previous, count}, {scratch.data(), count * 2}, {device.data(), bytes},
                        stream.get(),
                        {{cache_current, texels}, {cache_old, texels}, {cs.data(), texels * 2}}),
                    "cached preview");
            const auto actual = device.download(stream.get());
            unsigned maximum = 0;
            for (std::size_t i = 0; i < bytes; ++i) {
                const auto difference = static_cast<unsigned>(
                    std::abs(static_cast<int>(actual[i]) - static_cast<int>(reference[i])));
                maximum = std::max(maximum, difference);
                ECLAIR_CHECK(difference <= 2);
            }
            for (std::size_t i = bytes; i < actual.size(); ++i) ECLAIR_CHECK(actual[i] == 77);
            std::printf(
                "cached_preview frame=%u texels=%zu max_channel_error=%u tolerance=2 "
                "sync=stream\n",
                frame, texels, maximum);
            atlas_previous = atlas;
            host_previous = std::move(next);
        }
        ECLAIR_CHECK(cuda::launch_ray_preview(
                         scene, camera, scene, camera, shape, false, {a.data(), count},
                         {b.data(), count}, {scratch.data(), count * 2}, {device.data(), bytes},
                         stream.get(),
                         {{ca.data(), texels - 1}, {cb.data(), texels}, {cs.data(), texels * 2}}) ==
                     cudaErrorInvalidValue);
        ECLAIR_CHECK(cuda::launch_ray_preview(
                         scene, camera, scene, camera, shape, false, {a.data(), count},
                         {b.data(), count}, {scratch.data(), count * 2}, {device.data(), bytes},
                         stream.get(),
                         {{ca.data(), texels}, {ca.data(), texels}, {cs.data(), texels * 2}}) ==
                     cudaErrorInvalidValue);
        ECLAIR_CHECK(cuda::launch_ray_preview(scene, camera, scene, camera, shape, false,
                                              {a.data(), count}, {a.data(), count},
                                              {scratch.data(), count * 2}, {device.data(), bytes},
                                              stream.get()) == cudaErrorInvalidValue);
        scene.primitives[scene.count - 1].material.roughness = 0.1F;
        ECLAIR_CHECK(cuda::launch_ray_trace(scene, camera, shape, {device.data(), bytes},
                                            stream.get()) == cudaErrorInvalidValue);
        scene.primitives[scene.count - 1].material.roughness = 0;
        scene.primitives[0].material.base_color.x = std::numeric_limits<float>::quiet_NaN();
        ECLAIR_CHECK(cuda::launch_ray_trace(scene, camera, shape, {device.data(), bytes},
                                            stream.get()) == cudaErrorInvalidValue);
    });
}
