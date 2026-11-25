/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"
#include "eclair/render/bsdf.hpp"
#include "eclair/render/geometry.hpp"

#include <limits>
#include <initializer_list>

int main() {
    using namespace eclair::render;
    const auto close = [](float a, float b) { return std::fabs(a - b) < 1.0e-5F; };
    const Vec3 normal{0, 0, 1};
    Vec3 transmitted{9, 9, 9};
    // Air -> glass: sin(theta_t) = sin(30 deg)/1.5 = 1/3.
    ECLAIR_CHECK(refract({0.5F, 0, -std::sqrt(0.75F)}, normal, 1 / 1.5F, transmitted));
    ECLAIR_CHECK(close(transmitted.x, 1.0F / 3) && close(transmitted.z, -std::sqrt(8.0F / 9)));
    ECLAIR_CHECK(refract(transmitted, normal, 1.5F, transmitted));
    ECLAIR_CHECK(close(transmitted.x, 0.5F) && close(transmitted.z, -std::sqrt(0.75F)));
    ECLAIR_CHECK(refract({0, 0, -1}, normal, 1.5F, transmitted));
    ECLAIR_CHECK(transmitted.z == -1);
    transmitted = {9, 9, 9};
    ECLAIR_CHECK(!refract({std::sqrt(0.75F), 0, -0.5F}, normal, 1.5F, transmitted));
    ECLAIR_CHECK(transmitted.x == 9);  // Glass -> air beyond critical angle.
    ECLAIR_CHECK(close(dielectric_fresnel(1, 1 / 1.5F), 0.04F));
    ECLAIR_CHECK(close(dielectric_fresnel(1, 1.5F), 0.04F));
    ECLAIR_CHECK(dielectric_fresnel(0.5F, 1.5F) == 1);
    ECLAIR_CHECK(dielectric_fresnel(0, 1) == 0);
    ECLAIR_CHECK(conductor_fresnel({0.9F, 0.6F, 0.2F}, 1).y == 0.6F);
    ECLAIR_CHECK(conductor_fresnel({0.9F, 0.6F, 0.2F}, 0).y == 1);
    ECLAIR_CHECK(close(conductor_fresnel({0.5F, 0.5F, 0.5F}, 0.5F).x, 0.515625F));
    ECLAIR_CHECK(close(ggx_distribution(1, 0.5F), 4 / kPi));
    ECLAIR_CHECK(ggx_masking(1, 0.5F) == 1);
    ECLAIR_CHECK(close(power_weight(2, 1), 0.8F));
    ECLAIR_CHECK(close(power_weight(1.0e30F, 1.0e30F), 0.5F));
    Material m;
    ECLAIR_CHECK(valid_material(m));
    for (float bad : {-1.0F, 1.1F, std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()}) {
        m.roughness = bad;
        ECLAIR_CHECK(!valid_material(m));
    }
    for (float zero : {0.0F, -0.0F, std::numeric_limits<float>::denorm_min()}) {
        m.roughness = zero;
        ECLAIR_CHECK(valid_material(m));
    }
    m.kind = static_cast<MaterialKind>(77);
    ECLAIR_CHECK(!valid_material(m));
    m.kind = MaterialKind::Dielectric;
    m.roughness = 0.2F;
    ECLAIR_CHECK(!valid_material(m));  // Rough transmission is explicitly unsupported.
    m.roughness = 0;
    for (float bad : {0.9F, 3.1F, std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()}) {
        m.ior = bad;
        ECLAIR_CHECK(!valid_material(m));
    }
    m.ior = 1.5F;
    m.base_color = {1, 1, 1};
    std::uint32_t fresnel_state = 77;
    unsigned reflections = 0;
    for (unsigned i = 0; i < 10000; ++i) {
        const auto s = sample_bsdf(m, normal, {0, 0, -1}, true, fresnel_state);
        if (s.direction.z > 0)
            ++reflections;
        else
            ECLAIR_CHECK(close(s.weight.x, 4.0F / 9));
    }
    ECLAIR_CHECK(reflections > 300 && reflections < 500);  // Fresnel R(0)=4%.
    Primitive sphere;
    sphere.material = m;
    Hit hit;
    ECLAIR_CHECK(intersect(sphere, {{0, 0, 3}, {0, 0, -1}}, 0, 10, hit) && hit.front_face);
    ECLAIR_CHECK(intersect(sphere, {{0, 0, 0}, {0, 0, 1}}, 0, 10, hit) && !hit.front_face);
    Primitive sheet;
    sheet.kind = PrimitiveKind::Quad;
    sheet.u = {1, 0, 0};
    sheet.v = {0, 1, 0};
    sheet.material = m;
    ECLAIR_CHECK(!valid_primitive(sheet));
    std::uint32_t state = 123;
    auto sample = sample_bsdf(m, normal, {std::sqrt(0.75F), 0, -0.5F}, false, state);
    ECLAIR_CHECK(sample.delta && close(sample.direction.z, 0.5F) && sample.weight.x == 1);
    m.kind = MaterialKind::Diffuse;
    m.base_color = {0.8F, 0.3F, 0.1F};
    const auto lambert = evaluate_bsdf(m, normal, normal, normal);
    ECLAIR_CHECK(close(lambert.value.x, 0.8F / kPi) && close(lambert.pdf, 1 / kPi));
    ECLAIR_CHECK(evaluate_bsdf(m, normal, normal, -normal).pdf == 0);
    m.kind = MaterialKind::Conductor;
    m.base_color = {1, 1, 1};
    m.roughness = 0.5F;
    ECLAIR_CHECK(close(evaluate_bsdf(m, normal, normal, normal).value.x, 4 / kPi));
    // A white furnace may lose energy from single-scattering GGX masking, but
    // must never gain it. Check grazing views and every supported roughness.
    for (float roughness : {0.0F, std::numeric_limits<float>::denorm_min(), 0.05F, 0.3F, 1.0F}) {
        m.roughness = roughness;
        for (Vec3 incident : {Vec3{0, 0, -1}, normalized(Vec3{1, 0, -0.01F})}) {
            for (unsigned i = 0; i < 2048; ++i) {
                sample = sample_bsdf(m, normal, incident, true, state);
                ECLAIR_CHECK(finite(sample.weight) && finite(sample.direction) &&
                             finite(sample.pdf));
                ECLAIR_CHECK(sample.weight.x >= 0 && sample.weight.x <= 1.00001F);
                if (sample.pdf > 0) {
                    const auto value = evaluate_bsdf(m, normal, -incident, sample.direction);
                    ECLAIR_CHECK(
                        std::fabs(value.value.x * dot(normal, sample.direction) / value.pdf -
                                  sample.weight.x) < 0.002F);
                }
            }
        }
    }
    m.roughness = 0.15F;
    const auto polished = evaluate_bsdf(m, normal, normal, normal);
    m.roughness = 0.7F;
    const auto rough = evaluate_bsdf(m, normal, normal, normal);
    ECLAIR_CHECK(polished.value.x > rough.value.x * 100);
    return eclair::test::finish();
}
