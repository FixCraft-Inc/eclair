/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/render/material.hpp"

namespace eclair::render {

inline constexpr float kPi = 3.14159265358979323846F;

ECLAIR_HOST_DEVICE inline float random_unit(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) * (1.0F / 16777216.0F);
}
ECLAIR_HOST_DEVICE inline Vec3 reflect(Vec3 incident, Vec3 normal) {
    return incident - normal * (2 * dot(incident, normal));
}
// normal opposes incident; eta is incident IOR / transmitted IOR. Return false
// on total internal reflection without writing an invalid transmitted ray.
ECLAIR_HOST_DEVICE inline bool refract(Vec3 incident, Vec3 normal, float eta, Vec3& transmitted) {
    if (eta == 1) {
        transmitted = incident;
        return true;
    }
    const float cosine = ::fminf(1, ::fmaxf(0, -dot(incident, normal)));
    const float sine2 = eta * eta * ::fmaxf(0, 1 - cosine * cosine);
    if (sine2 >= 1) return false;
    transmitted = normalized(incident * eta + normal * (eta * cosine - ::sqrtf(1 - sine2)));
    return true;
}
ECLAIR_HOST_DEVICE inline float dielectric_fresnel(float cosine, float eta) {
    cosine = ::fminf(1, ::fmaxf(0, cosine));
    if (eta == 1) return 0;
    const float sine2 = eta * eta * ::fmaxf(0, 1 - cosine * cosine);
    if (sine2 >= 1) return 1;
    const float transmitted = ::sqrtf(1 - sine2);
    const float s = (eta * cosine - transmitted) / (eta * cosine + transmitted);
    const float p = (cosine - eta * transmitted) / (cosine + eta * transmitted);
    return 0.5F * (s * s + p * p);
}
ECLAIR_HOST_DEVICE inline Vec3 conductor_fresnel(Vec3 f0, float cosine) {
    const float x = ::fmaxf(0, 1 - cosine), x2 = x * x;
    return f0 + (Vec3{1, 1, 1} - f0) * (x2 * x2 * x);
}
struct TangentFrame {
    Vec3 tangent, bitangent, normal;
    ECLAIR_HOST_DEVICE explicit TangentFrame(Vec3 n) : normal(n) {
        tangent = normalized(cross(::fabsf(n.y) < 0.9F ? Vec3{0, 1, 0} : Vec3{1, 0, 0}, n));
        bitangent = cross(n, tangent);
    }
    ECLAIR_HOST_DEVICE Vec3 local(Vec3 v) const {
        return {dot(v, tangent), dot(v, bitangent), dot(v, normal)};
    }
    ECLAIR_HOST_DEVICE Vec3 world(Vec3 v) const {
        return tangent * v.x + bitangent * v.y + normal * v.z;
    }
};
ECLAIR_HOST_DEVICE inline float ggx_distribution(float nh, float alpha) {
    // This form avoids cancellation near nh=1 at small roughness.
    const float a2 = alpha * alpha;
    const float d = ::fmaxf(0, 1 - nh * nh) + a2 * nh * nh;
    return a2 / (kPi * d * d);
}
ECLAIR_HOST_DEVICE inline float ggx_masking(float cosine, float alpha) {
    if (cosine <= 0) return 0;
    return 2 * cosine / (cosine + ::sqrtf(alpha * alpha + (1 - alpha * alpha) * cosine * cosine));
}
struct BsdfValue {
    Vec3 value{};
    float pdf = 0;
};
ECLAIR_HOST_DEVICE inline BsdfValue evaluate_bsdf(const Material& m, Vec3 n, Vec3 view,
                                                  Vec3 light) {
    const float nv = dot(n, view), nl = dot(n, light);
    if (nv <= 0 || nl <= 0) return {};
    if (m.kind == MaterialKind::Diffuse) return {m.base_color / kPi, nl / kPi};
    if (m.kind != MaterialKind::Conductor || m.roughness == 0) return {};
    const float alpha = ::fmaxf(0.001F, m.roughness * m.roughness);
    const Vec3 half = normalized(view + light);
    const float distribution = ggx_distribution(::fmaxf(0, dot(n, half)), alpha);
    const float visible = ggx_masking(nv, alpha);
    const Vec3 fresnel = conductor_fresnel(m.base_color, ::fmaxf(0, dot(view, half)));
    return {fresnel * (distribution * visible * ggx_masking(nl, alpha) / (4 * nv * nl)),
            distribution * visible / (4 * nv)};
}
struct BsdfSample {
    Vec3 direction{}, weight{};
    float pdf = 0;
    bool delta = false;
};
ECLAIR_HOST_DEVICE inline BsdfSample sample_conductor(const Material& m, Vec3 n, Vec3 incident,
                                                      float u, float v) {
    const TangentFrame basis(n);
    Vec3 direction;
    // GGX visible-normal sampling: stretch the view, sample its projected disk,
    // then unstretch. Public algorithm: Heitz, JCGT 7(4), 2018; independent code.
    const float alpha = ::fmaxf(0.001F, m.roughness * m.roughness);
    const Vec3 view = basis.local(-incident);
    const Vec3 stretched = normalized({alpha * view.x, alpha * view.y, view.z});
    const float length2 = stretched.x * stretched.x + stretched.y * stretched.y;
    const Vec3 t1 =
        length2 > 0 ? Vec3{-stretched.y, stretched.x, 0} / ::sqrtf(length2) : Vec3{1, 0, 0};
    const Vec3 t2 = cross(stretched, t1);
    const float r = ::sqrtf(u), phi = 2 * kPi * v, x = r * ::cosf(phi);
    const float s = 0.5F * (1 + stretched.z);
    const float y = (1 - s) * ::sqrtf(::fmaxf(0, 1 - x * x)) + s * r * ::sinf(phi);
    const Vec3 h = t1 * x + t2 * y + stretched * ::sqrtf(::fmaxf(0, 1 - x * x - y * y));
    const Vec3 half = basis.world(normalized({alpha * h.x, alpha * h.y, ::fmaxf(0, h.z)}));
    direction = normalized(reflect(incident, half));
    if (dot(n, direction) <= 0) return {};
    const auto value = evaluate_bsdf(m, n, -incident, direction);
    // VNDF cancellation avoids dividing tiny BRDF/PDF values at grazing angles.
    const Vec3 weight = conductor_fresnel(m.base_color, ::fmaxf(0, dot(-incident, half))) *
                        ggx_masking(dot(n, direction), alpha);
    return {direction, weight, value.pdf, false};
}
ECLAIR_HOST_DEVICE inline BsdfSample sample_bsdf(const Material& m, Vec3 n, Vec3 incident,
                                                 bool front_face, std::uint32_t& state) {
    if (m.kind == MaterialKind::Dielectric) {
        const float eta = front_face ? 1 / m.ior : m.ior;
        const float fresnel = dielectric_fresnel(-dot(incident, n), eta);
        Vec3 transmitted;
        if (!refract(incident, n, eta, transmitted) || random_unit(state) < fresnel)
            return {reflect(incident, n), {1, 1, 1}, 0, true};
        return {transmitted, m.base_color * (eta * eta), 0, true};
    }
    if (m.kind == MaterialKind::Conductor && m.roughness == 0)
        return {reflect(incident, n), conductor_fresnel(m.base_color, -dot(incident, n)), 0, true};
    const TangentFrame basis(n);
    const float u = random_unit(state), v = random_unit(state);
    Vec3 direction;
    if (m.kind == MaterialKind::Diffuse) {
        const float r = ::sqrtf(u), phi = 2 * kPi * v;
        direction = basis.world({r * ::cosf(phi), r * ::sinf(phi), ::sqrtf(1 - u)});
        return {direction, m.base_color, ::fmaxf(0, dot(n, direction)) / kPi, false};
    }
    return sample_conductor(m, n, incident, u, v);
}

ECLAIR_HOST_DEVICE inline float power_weight(float a, float b) {
    // Normalize before squaring to avoid overflow for narrow glossy lobes.
    const float scale = ::fmaxf(a, b);
    if (scale <= 0) return 0;
    a /= scale;
    b /= scale;
    return a * a / (a * a + b * b);
}
}  // namespace eclair::render
