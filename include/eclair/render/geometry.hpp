/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/render/material.hpp"

#include <cmath>
#include <cstdint>

namespace eclair::render {

enum class PrimitiveKind : std::uint32_t { Sphere, Box, Quad };

// Quad u/v are orthogonal half-edge vectors. Box u holds positive half-extents;
// rotation is around Y. Sphere uses radius. Unused fields must still be finite.
struct Primitive {
    PrimitiveKind kind = PrimitiveKind::Sphere;
    Vec3 center{};
    Vec3 u{};
    Vec3 v{};
    Material material{};
    float radius = 1;
    float rotation_cos = 1;
    float rotation_sin = 0;
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};
struct Hit {
    float distance = 1.0e6F;
    Vec3 normal{};
    std::uint32_t primitive = 0;
    bool front_face = true;
};

ECLAIR_HOST_DEVICE inline bool valid_primitive(const Primitive& p) {
    if (!finite(p.center) || !finite(p.u) || !finite(p.v) || !finite(p.radius) ||
        !finite(p.rotation_cos) || !finite(p.rotation_sin) || !valid_material(p.material))
        return false;
    if (::fabsf(p.center.x) > 10000 || ::fabsf(p.center.y) > 10000 || ::fabsf(p.center.z) > 10000 ||
        (p.material.kind == MaterialKind::Dielectric && p.kind == PrimitiveKind::Quad))
        return false;
    switch (p.kind) {
        case PrimitiveKind::Sphere:
            return p.radius >= 1.0e-5F && p.radius <= 10000;
        case PrimitiveKind::Box:
            return p.u.x >= 1.0e-5F && p.u.y >= 1.0e-5F && p.u.z >= 1.0e-5F && p.u.x <= 10000 &&
                   p.u.y <= 10000 && p.u.z <= 10000 &&
                   ::fabsf(p.rotation_cos * p.rotation_cos + p.rotation_sin * p.rotation_sin - 1) <
                       1.0e-4F;
        case PrimitiveKind::Quad: {
            const float a = dot(p.u, p.u), b = dot(p.v, p.v);
            return a >= 1.0e-10F && b >= 1.0e-10F && a <= 1.0e8F && b <= 1.0e8F &&
                   ::fabsf(dot(p.u, p.v)) <= 1.0e-4F * ::sqrtf(a * b);
        }
    }
    return false;
}

ECLAIR_HOST_DEVICE inline Vec3 box_local(const Primitive& p, Vec3 a) {
    return {p.rotation_cos * a.x - p.rotation_sin * a.z, a.y,
            p.rotation_sin * a.x + p.rotation_cos * a.z};
}
ECLAIR_HOST_DEVICE inline Vec3 box_world(const Primitive& p, Vec3 a) {
    return {p.rotation_cos * a.x + p.rotation_sin * a.z, a.y,
            -p.rotation_sin * a.x + p.rotation_cos * a.z};
}

// Hot path: scene validation and a finite, nonzero ray are caller preconditions.
// Distances are inclusive. Directions below 1e-20 on a slab axis are parallel;
// they cannot cross its minimum 1e-5 extent within the bounded 1e6 ray range.
ECLAIR_HOST_DEVICE inline bool intersect_unchecked(const Primitive& p, const Ray& ray,
                                                   float minimum, float maximum, Hit& hit) {
    float distance = 0;
    Vec3 normal{};
    switch (p.kind) {
        case PrimitiveKind::Sphere: {
            const Vec3 offset = ray.origin - p.center;
            const float a = dot(ray.direction, ray.direction), b = dot(offset, ray.direction);
            const float c = dot(offset, offset) - p.radius * p.radius;
            const float discriminant = b * b - a * c;
            if (discriminant < 0) return false;
            const float root = ::sqrtf(discriminant);
            distance = (-b - root) / a;
            if (distance < minimum) distance = (-b + root) / a;
            if (distance < minimum || distance > maximum) return false;
            normal = (ray.origin + ray.direction * distance - p.center) / p.radius;
            break;
        }
        case PrimitiveKind::Quad: {
            normal = normalized(cross(p.u, p.v));
            const float denominator = dot(normal, ray.direction);
            if (::fabsf(denominator) < 1.0e-8F) return false;
            distance = dot(p.center - ray.origin, normal) / denominator;
            if (distance < minimum || distance > maximum) return false;
            const Vec3 local = ray.origin + ray.direction * distance - p.center;
            if (::fabsf(dot(local, p.u)) > dot(p.u, p.u) ||
                ::fabsf(dot(local, p.v)) > dot(p.v, p.v))
                return false;
            break;
        }
        case PrimitiveKind::Box: {
            const Vec3 origin = box_local(p, ray.origin - p.center),
                       direction = box_local(p, ray.direction);
            float near = -1.0e30F, far = 1.0e30F;
            Vec3 near_normal{}, far_normal{};
            for (int axis = 0; axis < 3; ++axis) {
                const float o = component(origin, axis), d = component(direction, axis),
                            e = component(p.u, axis);
                if (::fabsf(d) < 1.0e-20F) {
                    if (o < -e || o > e) return false;
                    continue;
                }
                float first = (-e - o) / d, last = (e - o) / d;
                float sign = -1;
                if (first > last) {
                    const float tmp = first;
                    first = last;
                    last = tmp;
                    sign = 1;
                }
                if (first > near) {
                    near = first;
                    near_normal = axis_vector(axis, sign);
                }
                if (last < far) {
                    far = last;
                    far_normal = axis_vector(axis, -sign);
                }
                if (near > far) return false;
            }
            distance = near >= minimum ? near : far;
            if (distance < minimum || distance > maximum) return false;
            normal = box_world(p, near >= minimum ? near_normal : far_normal);
            break;
        }
    }
    hit.front_face = dot(normal, ray.direction) <= 0;
    if (!hit.front_face) normal = -normal;
    hit.distance = distance;
    hit.normal = normal;
    return true;
}

ECLAIR_HOST_DEVICE inline bool intersect(const Primitive& p, const Ray& ray, float minimum,
                                         float maximum, Hit& hit) {
    const float length2 = dot(ray.direction, ray.direction);
    if (!valid_primitive(p) || !finite(ray.origin) || !finite(ray.direction) || !finite(length2) ||
        ::fabsf(ray.origin.x) > 10000 || ::fabsf(ray.origin.y) > 10000 ||
        ::fabsf(ray.origin.z) > 10000 || length2 > 1.0e8F || length2 < 1.0e-20F ||
        !finite(minimum) || !finite(maximum) || minimum < 0 || maximum > 1.0e6F ||
        maximum < minimum)
        return false;
    return intersect_unchecked(p, ray, minimum, maximum, hit);
}

}  // namespace eclair::render
