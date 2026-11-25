/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/host_backend.hpp"
#include "eclair/host/bf16_linear.hpp"
#include "eclair/host/fp8_linear.hpp"
#include "eclair/host/gated_delta.hpp"
#include "eclair/host/gated_delta_output.hpp"
#include "eclair/host/paged_attention.hpp"
#include "eclair/host/ray_trace.hpp"

#include "eclair/capability/feature.hpp"
#include "eclair/formats/bf16.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/formats/fp8.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace eclair::host {
namespace {

using capability::ExecutionPath;
using capability::Feature;
using capability::FeatureState;
using capability::FeatureStatus;
using formats::decode_fp8;
using formats::encode_fp8;
using formats::Fp8Format;

constexpr float kInfinity = std::numeric_limits<float>::infinity();

// Values fixed by the OCP 8-bit floating-point specification, independent of
// the implementation under check.
[[nodiscard]] bool decode_known_answers(Fp8Format format) noexcept {
    if (format == Fp8Format::E4M3FN) {
        return decode_fp8(0x38U, format) == 1.0F && decode_fp8(0x7EU, format) == 448.0F &&
               std::isnan(decode_fp8(0x7FU, format)) && std::signbit(decode_fp8(0x80U, format));
    }
    return decode_fp8(0x3CU, format) == 1.0F && decode_fp8(0x7CU, format) == kInfinity &&
           std::isnan(decode_fp8(0x7DU, format)) && decode_fp8(0x7BU, format) == 57344.0F;
}

[[nodiscard]] bool encode_known_answers(Fp8Format format) noexcept {
    if (format == Fp8Format::E4M3FN) {
        return encode_fp8(1.0F, format) == 0x38U && encode_fp8(kInfinity, format) == 0x7EU &&
               encode_fp8(-0.0F, format) == 0x80U;
    }
    return encode_fp8(1.0F, format) == 0x3CU && encode_fp8(kInfinity, format) == 0x7CU &&
           encode_fp8(1.0e6F, format) == 0x7BU && encode_fp8(-0.0F, format) == 0x80U;
}

// Values fixed by the bfloat16 definition: one, a tie that rounds to even,
// the overflow midpoint and a NaN whose payload lies only in the dropped half.
[[nodiscard]] bool bf16_known_answers() noexcept {
    return formats::decode_bf16(0x3F80U) == 1.0F && formats::decode_bf16(0xFF80U) == -kInfinity &&
           formats::encode_bf16(1.0F + 0x1.0p-8F) == 0x3F80U &&
           formats::encode_bf16(0x1.FFp127F) == 0x7F80U &&
           formats::encode_bf16(std::bit_cast<float>(0x7F800001U)) == 0x7FC0U;
}

[[nodiscard]] FeatureStatus checked(Feature feature, bool passed, std::string detail) {
    if (!passed) {
        return {feature, FeatureState::Unsupported, ExecutionPath::None,
                "reference failed its known-answer check"};
    }
    return {feature, FeatureState::Enabled, ExecutionPath::HostReference, std::move(detail)};
}

[[nodiscard]] bool linear_known_answer() noexcept {
    const std::uint16_t a = 0x4000U;  // 2 in binary16.
    const std::uint8_t w = 0x44U;     // 3 in E4M3FN.
    const float scale = 0.5F;
    const float bias = 1.0F;
    float output = 0;
    const ops::Fp8LinearShape shape{1, 1, 1, 1, 1, Fp8Format::E4M3FN};
    const ops::Fp8LinearBuffers buffers{{&a, 1}, {&w, 1}, {&scale, 1}, {&bias, 1}, {&output, 1}};
    return fp8_linear_reference(shape, buffers) == ops::Status::Success && output == 4.0F;
}

// Operands and result far beyond binary16's range, so the row scaling must
// work: (2^20, 2^10) . (2^20, 2^12) = 2^40 + 2^22, which is exact in FP32.
[[nodiscard]] bool bf16_linear_known_answer() noexcept {
    const std::uint16_t a[2] = {formats::encode_bf16(0x1p20F), formats::encode_bf16(0x1p10F)};
    const std::uint16_t w[2] = {formats::encode_bf16(0x1p20F), formats::encode_bf16(0x1p12F)};
    std::uint16_t scaled_a[2] = {};
    std::uint16_t scaled_w[2] = {};
    std::int32_t a_shift = 0;
    std::int32_t w_shift = 0;
    float output = 0;
    const ops::Bf16RowScalingShape row{1, 2};
    const bool scaled = bf16_row_scaling_reference(row, {{a, 2}, {scaled_a, 2}, {&a_shift, 1}}) ==
                            ops::Status::Success &&
                        bf16_row_scaling_reference(row, {{w, 2}, {scaled_w, 2}, {&w_shift, 1}}) ==
                            ops::Status::Success;
    const ops::Bf16LinearBuffers buffers{
        {scaled_a, 2}, {&a_shift, 1}, {scaled_w, 2}, {&w_shift, 1}, {}, {&output, 1}};
    return scaled && bf16_linear_reference({1, 1, 2}, buffers) == ops::Status::Success &&
           output == 0x1p40F + 0x1p22F;
}

// A two-token paged cache, one head of four elements. Both keys are zero, so
// the scores tie and the softmax weights are equal: a query row at position 1
// reads the mean of the values 2 and 6, which is 4 exactly.
struct TwoTokenCache {
    std::int8_t keys[8] = {};
    std::int8_t values[8] = {2, 2, 2, 2, 6, 6, 6, 6};
    float scales[2] = {1.0F, 1.0F};
    std::int32_t table[1] = {0};
    std::uint16_t query[4] = {};
    std::uint16_t output[4] = {};
    // One block of two slots, four bytes per slot.
    ops::PagedInt8CacheLayout layout{1, 2, 1, 4, 8, 4, 4, 2, 1, 1};

    [[nodiscard]] bool output_is_four() const noexcept {
        for (const std::uint16_t bits : output) {
            if (formats::decode_fp16(bits) != 4.0F) {
                return false;
            }
        }
        return true;
    }
};

[[nodiscard]] bool paged_prefill_known_answer() noexcept {
    TwoTokenCache cache;
    const ops::PagedPrefillShape shape{1, 1, 1, 4, 4, 1.0F};
    const ops::PagedPrefillBuffers buffers{{cache.query, 4},  {cache.output, 4}, {cache.keys, 8},
                                           {cache.values, 8}, {cache.scales, 2}, {cache.scales, 2},
                                           {cache.table, 1}};
    return paged_prefill_reference(shape, cache.layout, buffers) == ops::Status::Success &&
           cache.output_is_four();
}

[[nodiscard]] bool paged_decode_known_answer() noexcept {
    TwoTokenCache cache;
    const std::int32_t starts[2] = {0, 1};
    const std::int32_t lengths[1] = {2};
    float workspace[6] = {};
    const ops::PagedDecodeShape shape{1, 1, 1, 1, 1, 1, 1, 4, 4, 1.0F};
    const ops::PagedDecodeBuffers buffers{
        {cache.query, 4},  {cache.output, 4}, {cache.keys, 8}, {cache.values, 8}, {cache.scales, 2},
        {cache.scales, 2}, {cache.table, 1},  {starts, 2},     {lengths, 1},      {workspace, 6}};
    return paged_decode_reference(shape, cache.layout, buffers) == ops::Status::Success &&
           cache.output_is_four();
}

// One token with an FP32 initial state of 2 and a zero gate. The contract
// gives states = 2, d = u - w * 2 = 3 - 1 = 2 and a final state of
// 2 + d * k = 4, all exact in binary16 and FP32.
[[nodiscard]] bool gated_delta_known_answer() noexcept {
    const std::uint16_t k = formats::encode_fp16(1.0F);
    const std::uint16_t w = formats::encode_fp16(0.5F);
    const std::uint16_t u = formats::encode_fp16(3.0F);
    const float gate = 0.0F;
    const float initial = 2.0F;
    const std::int32_t sequence_starts[2] = {0, 1};
    const std::int32_t chunk_starts[1] = {0};
    std::uint16_t states = 0;
    std::uint16_t new_values = 0;
    float final_state = 0.0F;
    const ops::GatedDeltaStatesShape shape{1, 1, 1, 1, 1, 1, 1};
    const ops::GatedDeltaStatesBuffers buffers{{&k, 1},
                                               {&w, 1},
                                               {&u, 1},
                                               {&gate, 1},
                                               {},
                                               {&initial, 1},
                                               {sequence_starts, 2},
                                               {chunk_starts, 1},
                                               {&states, 1},
                                               {&new_values, 1},
                                               {&final_state, 1}};
    return gated_delta_states_reference(shape, buffers) == ops::Status::Success &&
           formats::decode_fp16(states) == 2.0F && formats::decode_fp16(new_values) == 2.0F &&
           final_state == 4.0F;
}

// q=1, k=2, v=3, state=4, gate=0: output = q*state + (q*k)*v = 10.
[[nodiscard]] bool gated_delta_output_known_answer() noexcept {
    const std::uint16_t q = formats::encode_fp16(1), k = formats::encode_fp16(2),
                        v = formats::encode_fp16(3), state = formats::encode_fp16(4);
    const float gate = 0;
    const std::int32_t starts[] = {0, 1}, indices[] = {0, 0};
    std::uint16_t out = 0;
    const ops::GatedDeltaOutputShape s{1, 1, 1, 1, 1, 1, 1, 1};
    const ops::GatedDeltaOutputBuffers b{{&q, 1},    {&k, 1},     {&v, 1},      {&state, 1},
                                         {&gate, 1}, {starts, 2}, {indices, 2}, {&out, 1}};
    return gated_delta_output_reference(s, b) == ops::Status::Success &&
           formats::decode_fp16(out) == 10;
}

[[nodiscard]] bool ray_known_answer() noexcept {
    ops::RayScene scene;
    scene.count = 1;
    scene.primitives[0].material.base_color = {1, 0, 0};
    scene.primitives[0].material.emission = 1;
    ops::RayCamera camera;
    camera.origin = {0, 0, 3};
    const ops::RayTraceShape shape{1, 1, 1, 2, 1};
    std::uint8_t rgba[4]{};
    return ray_trace_reference(scene, camera, shape, {rgba, 4}) == ops::Status::Success &&
           rgba[0] == 232 && rgba[1] == 0 && rgba[2] == 0 && rgba[3] == 255;
}

[[nodiscard]] FeatureStatus evaluate(Feature feature) {
    switch (feature) {
        case Feature::Fp8E4m3fnDecode:
            return checked(feature, decode_known_answers(Fp8Format::E4M3FN), "decode to float");
        case Feature::Fp8E5m2Decode:
            return checked(feature, decode_known_answers(Fp8Format::E5M2), "decode to float");
        case Feature::Fp8E4m3fnEncode:
            return checked(feature, encode_known_answers(Fp8Format::E4M3FN),
                           "round to nearest even, saturating");
        case Feature::Fp8E5m2Encode:
            return checked(feature, encode_known_answers(Fp8Format::E5M2),
                           "round to nearest even, saturating finite values, keeps infinity");
        case Feature::Fp8Gemm:
            return checked(feature, linear_known_answer(),
                           "scaled FP8 linear; FP16 operands with FP64 reference sums");
        case Feature::Bf16Compat:
            return checked(feature, bf16_known_answers(),
                           "conversion to and from FP32, nearest even");
        case Feature::Bf16Gemm:
            return checked(feature, bf16_linear_known_answer(),
                           "row-scaled binary16 operands with FP64 reference sums");
        case Feature::PagedAttentionPrefill:
            return checked(feature, paged_prefill_known_answer(),
                           "binary16-rounded keys and values, FP64 softmax and sums");
        case Feature::PagedAttentionDecode:
            return checked(feature, paged_decode_known_answer(),
                           "the prefill reference applied per sequence");
        case Feature::GatedDeltaStates:
            return checked(feature, gated_delta_known_answer(),
                           "FP64 sums with the contract's roundings");
        case Feature::GatedDeltaOutput:
            return checked(feature, gated_delta_output_known_answer(),
                           "FP64 projections with binary16 correlations");
        case Feature::RayTracingEmulation:
            return checked(feature, ray_known_answer(), "bounded analytic scenes, FP32 reference");
    }
    return {feature, FeatureState::Unsupported, ExecutionPath::None, "unrecognized feature"};
}

}  // namespace

capability::BackendReport probe_backend() {
    capability::DeviceReport device;
    device.id = std::string(kBackendName);
    device.name = "CPU reference";
    device.architecture = "portable C++20";
    for (const capability::FeatureInfo& info : capability::feature_catalog()) {
        device.features.push_back(evaluate(info.feature));
    }

    capability::BackendReport report;
    report.name = std::string(kBackendName);
    report.label = "CPU reference";
    report.is_reference = true;
    report.state = capability::BackendState::Available;
    report.detail = "portable correctness checks; does not accelerate GPU workloads";
    report.devices.push_back(std::move(device));
    return report;
}

}  // namespace eclair::host
