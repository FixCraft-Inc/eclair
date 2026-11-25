/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once
#include "eclair/render/shading.hpp"
#include "eclair/render/preview_shading.hpp"

namespace eclair::render {
// Experimental preview state, in linear light. Two caller-owned frames plus
// two Vec3 scratch frames. No history or geometry leaves VRAM in the demo.
inline constexpr std::uint32_t kNoSurface = 0xffffffffU;
inline constexpr unsigned kPreviewSpatialPasses = 8;
static_assert(kPreviewSpatialPasses > 0 && kPreviewSpatialPasses % 2 == 0,
              "paired X/Y passes leave the final colors in the second scratch frame");
struct PreviewPixel {
    Vec3 color{}, position{}, normal{};
    // 0: miss/emitter, 1: diffuse, 2: sharp specular, 3: rough conductor,
    // 4: cached glossy conductor (strides one/two), 5: cached glass (median pair).
    std::uint32_t primitive = kNoSurface, filterable = 0;
    float history = 1, variance = 0;
};
inline constexpr std::size_t kPreviewBytesPerPixel = 2 * sizeof(PreviewPixel) + 2 * sizeof(Vec3);
ECLAIR_HOST_DEVICE inline PreviewPixel preview_sample(const ops::RayScene& scene,
                                                      const ops::RayCamera& camera,
                                                      const ops::RayTraceShape& shape,
                                                      std::uint32_t index,
                                                      const LightingLayout& layout = {},
                                                      const SurfaceLighting* lighting = nullptr) {
    PreviewPixel result;
    const auto ray = camera_ray(camera, shape, static_cast<float>(index % shape.width) + 0.5F,
                                static_cast<float>(index / shape.width) + 0.5F);
    Hit hit;
    if (nearest(scene, ray, 1.0e6F, hit)) {
        result.position = ray.origin + ray.direction * hit.distance;
        result.normal = hit.normal;
        result.primitive = hit.primitive;
        const auto& surface = scene.primitives[hit.primitive];
        // Cached broad lobes get multiscale filtering; glossy lobes use a
        // narrower footprint. Glass and delta mirrors keep sharp detail.
        result.filterable =
            surface.material.emission > 0
                ? 0U
                : (surface.material.kind == MaterialKind::Diffuse
                       ? 1U
                       : (surface.material.kind == MaterialKind::Conductor &&
                                  surface.material.roughness >= (lighting ? 0.08F : 0.3F)
                              ? 3U
                              : 2U));
        if (lighting && result.filterable == 3 && surface.material.roughness < 0.3F)
            result.filterable = 4;
        if (lighting && surface.material.kind == MaterialKind::Dielectric) result.filterable = 5;
        if (surface.material.emission > 0) {
            result.color = surface.material.base_color * surface.material.emission;
            return result;
        }
    } else {
        // Boundary coverage is reconstructed after filtering. Interior misses
        // need one visibility query, independent of the lighting sample count.
        result.color = scene.environment;
        return result;
    }
    // Diffuse history describes one surface, never a changing mixture of
    // foreground/background coverage. Resolve pixel coverage after denoising.
    if (lighting) {
        result.color = result.filterable == 1
                           ? lighting_lookup(scene, layout, lighting, result.position, hit)
                           : lit_pixel(scene, camera, shape, index, layout, lighting);
    } else {
        result.color =
            ray_color(scene, camera, shape, index, &result.variance, result.filterable != 1, &hit);
    }
    return result;
}
// Primary guide discontinuities identify silhouette/crease pixels. Extra
// coverage samples are confined to their one-pixel neighborhood; the rest of
// the frame keeps the lower lighting sample budget.
ECLAIR_HOST_DEVICE inline bool preview_boundary(const PreviewPixel* pixels,
                                                const ops::RayTraceShape& shape,
                                                std::uint32_t index) {
    const auto& center = pixels[index];
    const int x = static_cast<int>(index % shape.width), y = static_cast<int>(index / shape.width);
    for (int axis = 0; axis < 2; ++axis) {
        for (int sign = -1; sign <= 1; sign += 2) {
            const int nx = x + (axis == 0 ? sign : 0), ny = y + (axis == 1 ? sign : 0);
            if (nx < 0 || ny < 0 || nx >= static_cast<int>(shape.width) ||
                ny >= static_cast<int>(shape.height))
                continue;
            const auto& other = pixels[static_cast<std::uint32_t>(ny) * shape.width +
                                       static_cast<std::uint32_t>(nx)];
            if (center.primitive != other.primitive ||
                (center.primitive != kNoSurface && dot(center.normal, other.normal) < 0.98F))
                return true;
        }
    }
    return false;
}
// Find already-denoised diffuse lighting on the requested side of an edge.
// Matching identity AND normal keeps the faces of a box separate. A very thin
// surface without a center-ray guide falls back to tracing that coverage ray.
ECLAIR_HOST_DEVICE inline bool coverage_lighting(const PreviewPixel* pixels,
                                                 const ops::RayTraceShape& shape,
                                                 std::uint32_t index, const Hit& hit, Vec3& color) {
    const auto matches = [&](const PreviewPixel& p) {
        return p.filterable == 1 && p.primitive == hit.primitive &&
               dot(p.normal, hit.normal) > 0.98F;
    };
    if (matches(pixels[index])) {
        color = pixels[index].color;
        return true;
    }
    const int x = static_cast<int>(index % shape.width), y = static_cast<int>(index / shape.width);
    int nearest_distance = 9;
    bool found = false;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int nx = x + dx, ny = y + dy, distance = dx * dx + dy * dy;
            if (distance >= nearest_distance || nx < 0 || ny < 0 ||
                nx >= static_cast<int>(shape.width) || ny >= static_cast<int>(shape.height))
                continue;
            const auto& p = pixels[static_cast<std::uint32_t>(ny) * shape.width +
                                   static_cast<std::uint32_t>(nx)];
            if (matches(p)) {
                color = p.color;
                nearest_distance = distance;
                found = true;
            }
        }
    }
    return found;
}
// Coverage and illumination have different estimators: deterministic primary
// rays integrate pixel area, while diffuse lighting comes from spatial/temporal
// history. Random low-sample paths must not bypass denoising along room seams.
// The returned mixture is display-only; never feed it back into surface history.
ECLAIR_HOST_DEVICE inline Vec3 coverage_preview(
    const PreviewPixel* pixels, const ops::RayScene& scene, const ops::RayCamera& camera,
    const ops::RayTraceShape& shape, std::uint32_t index, const LightingLayout& layout = {},
    const SurfaceLighting* lighting = nullptr) {
    Vec3 sum{};
    std::uint32_t state = (index * 747796405U + shape.seed * 2891336453U) ^ 277803737U;
    if (state == 0) state = 1;
    for (std::uint32_t sample = 0; sample < 16; ++sample) {
        const float dx = (static_cast<float>(sample) + 0.5F) / 16;
        const unsigned reversed = ((sample & 1U) << 3) | ((sample & 2U) << 1) |
                                  ((sample & 4U) >> 1) | ((sample & 8U) >> 3);
        const float dy = (static_cast<float>(reversed) + 0.5F) / 16;
        const auto ray = camera_ray(camera, shape, static_cast<float>(index % shape.width) + dx,
                                    static_cast<float>(index / shape.width) + dy);
        Hit hit;
        if (!nearest(scene, ray, 1.0e6F, hit)) {
            sum = sum + scene.environment;
            continue;
        }
        const auto& p = scene.primitives[hit.primitive];
        Vec3 color;
        if (lighting) {
            if (p.material.kind != MaterialKind::Diffuse || p.material.emission > 0 ||
                !coverage_lighting(pixels, shape, index, hit, color))
                color = trace_lit(scene, layout, lighting, ray, shape.bounces, dx, dy);
        } else if (p.material.emission > 0)
            color = p.material.base_color * p.material.emission;
        else if (p.material.kind != MaterialKind::Diffuse ||
                 !coverage_lighting(pixels, shape, index, hit, color))
            color = trace(scene, ray, shape.bounces, state, &hit);
        sum = sum + color;
    }
    return sum / 16;
}
ECLAIR_HOST_DEVICE inline bool same_surface(const PreviewPixel& a, const PreviewPixel& b) {
    return a.filterable != 0 && b.filterable != 0 && a.primitive != kNoSurface &&
           a.primitive == b.primitive && dot(a.normal, b.normal) > 0.98F;
}
// A five-tap separable edge-aware pass. Alternating X/Y at strides 1,2,4,8
// suppresses low-sample noise at several scales without crossing box/wall edges.
ECLAIR_HOST_DEVICE inline Vec3 spatial_preview(const PreviewPixel* pixels, const Vec3* colors,
                                               const ops::RayTraceShape& shape, std::uint32_t index,
                                               int stride, bool vertical,
                                               bool cached_lighting = false) {
    const auto& center = pixels[index];
    const Vec3 center_color = colors ? colors[index] : center.color;
    if (center.filterable == 5) {
        // A one-pixel median rejects isolated secondary-lobe sparks without
        // averaging across the sharp refracted color boundary. Very thin
        // one-pixel highlights can be removed by this preview approximation.
        if (!cached_lighting || stride != 1) return center_color;
        const int x = static_cast<int>(index % shape.width),
                  y = static_cast<int>(index / shape.width);
        const int ax = x - (vertical ? 0 : 1), ay = y - (vertical ? 1 : 0),
                  bx = x + (vertical ? 0 : 1), by = y + (vertical ? 1 : 0);
        if (ax < 0 || ay < 0 || bx >= static_cast<int>(shape.width) ||
            by >= static_cast<int>(shape.height))
            return center_color;
        const auto a = static_cast<std::uint32_t>(ay) * shape.width +
                       static_cast<std::uint32_t>(ax),
                   b = static_cast<std::uint32_t>(by) * shape.width +
                       static_cast<std::uint32_t>(bx);
        if (!same_surface(center, pixels[a]) || !same_surface(center, pixels[b]))
            return center_color;
        const Vec3 ca = colors ? colors[a] : pixels[a].color,
                   cb = colors ? colors[b] : pixels[b].color;
        const float la = ca.x + ca.y + ca.z, lb = cb.x + cb.y + cb.z,
                    lc = center_color.x + center_color.y + center_color.z;
        if ((la <= lc && lc <= lb) || (lb <= lc && lc <= la)) return center_color;
        return ((lc <= la && la <= lb) || (lb <= la && la <= lc)) ? ca : cb;
    }
    if (center.filterable != 1 && !(center.filterable == 3 && (stride == 1 || cached_lighting)) &&
        !(center.filterable == 4 && stride <= 2))
        return center_color;
    const int x = static_cast<int>(index % shape.width), y = static_cast<int>(index / shape.width);
    Vec3 sum{};
    float weight_sum = 0;
    for (int d = -2; d <= 2; ++d) {
        const int nx = x + (vertical ? 0 : d * stride), ny = y + (vertical ? d * stride : 0);
        if (nx < 0 || ny < 0 || nx >= static_cast<int>(shape.width) ||
            ny >= static_cast<int>(shape.height))
            continue;
        const auto i =
            static_cast<std::uint32_t>(ny) * shape.width + static_cast<std::uint32_t>(nx);
        const auto& sample = pixels[i];
        if (!same_surface(center, sample)) continue;
        const Vec3 color = colors ? colors[i] : sample.color;
        const float binomial = d == 0 ? 6.0F : (::abs(d) == 1 ? 4.0F : 1.0F);
        const auto difference = color - center_color;
        const float weight =
            binomial / (1 + 0.25F * ::fabsf(difference.x + difference.y + difference.z));
        sum = sum + color * weight;
        weight_sum += weight;
    }
    return weight_sum > 0 ? sum / weight_sum : center_color;
}
ECLAIR_HOST_DEVICE inline bool project_previous(Vec3 position, const ops::RayCamera& camera,
                                                const ops::RayTraceShape& shape, float& x, float& y,
                                                float& footprint) {
    const Vec3 delta = position - camera.origin;
    const float depth = dot(delta, camera.forward);
    if (!finite(depth) || depth <= kRayOffset) return false;
    const float height = static_cast<float>(shape.height), width = static_cast<float>(shape.width);
    const float scale = depth * camera.vertical_tangent;
    x = (dot(delta, camera.right) / (scale * (width / height)) + 1) * width * 0.5F - 0.5F;
    y = (1 - dot(delta, camera.up) / scale) * height * 0.5F - 0.5F;
    footprint = 2 * scale / height;
    return finite(x) && finite(y) && x >= 0 && y >= 0 && x <= width - 1 && y <= height - 1;
}
ECLAIR_HOST_DEVICE inline Vec3 clamp_color(Vec3 c, Vec3 low, Vec3 high) {
    return {::fmaxf(low.x, ::fminf(high.x, c.x)), ::fmaxf(low.y, ::fminf(high.y, c.y)),
            ::fmaxf(low.z, ::fminf(high.z, c.z))};
}
// Host-side identity test, once per launch. No padding-byte comparisons and
// no per-pixel scene scan. A changed light or any moving object invalidates
// view-dependent history, including motion visible only through a reflection.
inline bool stationary_preview_scene(const ops::RayScene& a, const ops::RayCamera& ac,
                                     const ops::RayScene& b, const ops::RayCamera& bc) {
    const auto same = [](Vec3 x, Vec3 y) { return x.x == y.x && x.y == y.y && x.z == y.z; };
    if (a.count != b.count || !same(ac.origin, bc.origin) || !same(ac.forward, bc.forward) ||
        !same(ac.right, bc.right) || !same(ac.up, bc.up) ||
        ac.vertical_tangent != bc.vertical_tangent || !same(a.light_center, b.light_center) ||
        !same(a.light_u, b.light_u) || !same(a.light_v, b.light_v) ||
        !same(a.light_emission, b.light_emission) || !same(a.environment, b.environment))
        return false;
    for (std::uint32_t i = 0; i < a.count; ++i) {
        const auto& p = a.primitives[i];
        const auto& q = b.primitives[i];
        if (p.kind != q.kind || !same(p.center, q.center) || !same(p.u, q.u) || !same(p.v, q.v) ||
            p.radius != q.radius || p.rotation_cos != q.rotation_cos ||
            p.rotation_sin != q.rotation_sin || p.material.kind != q.material.kind ||
            !same(p.material.base_color, q.material.base_color) ||
            p.material.roughness != q.material.roughness || p.material.ior != q.material.ior ||
            p.material.emission != q.material.emission)
            return false;
    }
    return true;
}
// Only reads current geometry; writes to current colors happen in parallel.
// Previous history and the spatial scratch frame must remain immutable.
ECLAIR_HOST_DEVICE inline PreviewPixel temporal_preview(
    const PreviewPixel* current, const PreviewPixel* previous, const Vec3* spatial,
    const ops::RayScene& scene, const ops::RayScene& previous_scene,
    const ops::RayCamera& previous_camera, const ops::RayTraceShape& shape, std::uint32_t index,
    bool valid_history, bool stationary_scene = false) {
    PreviewPixel result = current[index];
    result.color = spatial[index];
    result.history = 1;
    if (!valid_history || result.primitive >= scene.count ||
        result.primitive >= previous_scene.count)
        return result;
    const auto& primitive = scene.primitives[result.primitive];
    const auto& old_primitive = previous_scene.primitives[result.primitive];
    if (primitive.kind != old_primitive.kind || primitive.material.emission > 0) return result;
    if (stationary_scene || primitive.material.kind != MaterialKind::Diffuse) {
        // No reprojection for reflections/refraction: primary-surface motion is
        // insufficient to track the reflected/refracted hit. Exact stationary
        // scenes can accumulate without blurring highlights or bending detail.
        if (!stationary_scene) return result;
        const auto& old = previous[index];
        if (!same_surface(result, old) || !finite(old.color) || !finite(old.history) ||
            old.history < 1)
            return result;
        result.history = ::fminf(1024, old.history + 1);
        result.color = result.color / result.history + old.color * (1 - 1 / result.history);
        return result;
    }
    Vec3 position = result.position - primitive.center, normal = result.normal;
    if (primitive.kind == PrimitiveKind::Box) {
        position = box_world(old_primitive, box_local(primitive, position));
        normal = box_world(old_primitive, box_local(primitive, normal));
    }
    position = position + old_primitive.center;
    float px = 0, py = 0, footprint = 0;
    if (!project_previous(position, previous_camera, shape, px, py, footprint)) return result;
    const int ox = static_cast<int>(::floorf(px)), oy = static_cast<int>(::floorf(py));
    Vec3 history{};
    float weight_sum = 0, length_sum = 0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int x = ox + dx, y = oy + dy;
            if (x >= static_cast<int>(shape.width) || y >= static_cast<int>(shape.height)) continue;
            const auto& old = previous[static_cast<std::uint32_t>(y) * shape.width +
                                       static_cast<std::uint32_t>(x)];
            const Vec3 delta = old.position - position;
            const float tolerance = 2 * footprint + 0.001F;
            if (old.primitive != result.primitive || dot(old.normal, normal) < 0.98F ||
                dot(delta, delta) > tolerance * tolerance || !finite(old.color) ||
                !finite(old.history) || old.history < 1)
                continue;
            const float weight = (1 - ::fabsf(px - static_cast<float>(x))) *
                                 (1 - ::fabsf(py - static_cast<float>(y)));
            history = history + old.color * weight;
            length_sum += old.history * weight;
            weight_sum += weight;
        }
    }
    if (weight_sum < 0.01F) return result;
    // Clip history to the current neighborhood to limit ghosting when a
    // moving box changes the illumination of an otherwise static wall.
    Vec3 low = result.color, high = result.color;
    const int cx = static_cast<int>(index % shape.width),
              cy = static_cast<int>(index / shape.width);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= static_cast<int>(shape.width) ||
                y >= static_cast<int>(shape.height))
                continue;
            const auto i =
                static_cast<std::uint32_t>(y) * shape.width + static_cast<std::uint32_t>(x);
            if (!same_surface(result, current[i])) continue;
            const auto c = spatial[i];
            low = {::fminf(low.x, c.x), ::fminf(low.y, c.y), ::fminf(low.z, c.z)};
            high = {::fmaxf(high.x, c.x), ::fmaxf(high.y, c.y), ::fmaxf(high.z, c.z)};
        }
    }
    // Clamping only to an already smoothed 3x3 range erases valid history
    // whenever Monte Carlo noise shifts that narrow range. Keep a variance
    // margin while bounding lighting changes from moving occluders.
    const float margin = 2 * ::sqrtf(::fmaxf(0, result.variance) / 32) + 0.005F;
    low = low - Vec3{margin, margin, margin};
    high = high + Vec3{margin, margin, margin};
    result.history = ::fminf(32, length_sum / weight_sum + 1);
    const float alpha = 1 / result.history;
    result.color =
        result.color * alpha + clamp_color(history / weight_sum, low, high) * (1 - alpha);
    return result;
}
}  // namespace eclair::render
