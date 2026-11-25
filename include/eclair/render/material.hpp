/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/render/vector.hpp"

namespace eclair::render {

enum class MaterialKind : std::uint32_t { Diffuse, Conductor, Dielectric };

// Linear RGB: diffuse reflectance, conductor normal-incidence Fresnel, or
// dielectric interface transmission tint. Roughness is perceptual (alpha=r*r).
// Dielectrics are smooth, closed, non-nested solids surrounded by air. Emission
// is two-sided for the primary view; explicit area-light sampling is one-sided.
struct Material {
    MaterialKind kind = MaterialKind::Diffuse;
    Vec3 base_color{0.7F, 0.7F, 0.7F};
    float roughness = 0.5F;
    float ior = 1.5F;
    float emission = 0;
};

ECLAIR_HOST_DEVICE inline bool valid_material(const Material& m) {
    if (!finite(m.base_color) || m.base_color.x < 0 || m.base_color.y < 0 || m.base_color.z < 0 ||
        m.base_color.x > 1 || m.base_color.y > 1 || m.base_color.z > 1 || !finite(m.roughness) ||
        m.roughness < 0 || m.roughness > 1 || !finite(m.ior) || m.ior < 1 || m.ior > 3 ||
        !finite(m.emission) || m.emission < 0 || m.emission > 10000)
        return false;
    switch (m.kind) {
        case MaterialKind::Diffuse:
        case MaterialKind::Conductor:
            return true;
        case MaterialKind::Dielectric:
            return m.roughness == 0 && m.emission == 0;
    }
    return false;
}

}  // namespace eclair::render
