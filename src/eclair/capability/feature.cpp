/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/capability/feature.hpp"

#include <array>

namespace eclair::capability {
namespace {

constexpr std::array<FeatureInfo, kFeatureCount> kCatalog{{
    {Feature::Fp8E4m3fnDecode, "fp8.e4m3fn.decode", "FP8 E4M3FN decode",
     "Widen OCP E4M3FN bytes (finite range, one NaN pattern) to a wider format."},
    {Feature::Fp8E5m2Decode, "fp8.e5m2.decode", "FP8 E5M2 decode",
     "Widen OCP E5M2 bytes (with infinities and NaNs) to a wider format."},
    {Feature::Fp8E4m3fnEncode, "fp8.e4m3fn.encode", "FP8 E4M3FN encode",
     "Round wider values to E4M3FN, saturating overflow to the largest finite value."},
    {Feature::Fp8E5m2Encode, "fp8.e5m2.encode", "FP8 E5M2 encode",
     "Round wider values to E5M2, saturating finite overflow and keeping infinities."},
    {Feature::Fp8Gemm, "fp8.gemm", "FP8-weight matrix multiplication",
     "Scaled FP8 weights with FP16 activations, optional FP32 bias and FP32 output."},
    // The key predates the narrower title. BF16 arithmetic needs its own key.
    {Feature::Bf16Compat, "bf16.compat", "BF16 conversion",
     "Convert BF16 to and from FP32 with nearest-even rounding, emulated where hardware "
     "lacks BF16."},
    {Feature::Bf16Gemm, "bf16.gemm", "BF16 matrix multiplication",
     "BF16 operands scaled per row by powers of two onto binary16, with CUDA-core or tensor-core "
     "FP32 accumulation, optional FP32 bias and FP32 output."},
    {Feature::PagedAttentionPrefill, "attention.paged_int8.prefill",
     "Paged int8 attention, prefill",
     "Causal attention of a chunk of query rows over a paged int8 key and value cache with "
     "one FP32 scale per token and key-value head."},
    {Feature::PagedAttentionDecode, "attention.paged_int8.decode", "Paged int8 attention, decode",
     "The same attention for the few query rows per sequence that decode and speculative "
     "verification produce, with each context split into slices and merged."},
    {Feature::GatedDeltaStates, "gated_delta.states", "Gated DeltaNet state pass",
     "The chunk-to-chunk recurrent state update of Gated DeltaNet prefill, with one scalar "
     "gate per head and token."},
    {Feature::GatedDeltaOutput, "gated_delta.output", "Gated DeltaNet output pass",
     "Causal within-chunk correlations and the gated state projection of Gated DeltaNet prefill."},
    {Feature::RayTracingEmulation, "rt.emulation", "Ray-tracing emulation",
     "Bounded FP32 software ray tracing of up to 16 spheres, rotated boxes or quads, "
     "with area lighting, diffuse bounces and optional mirror reflection."},
}};

// The catalog is indexed by enumerator value, so both must use one order.
constexpr bool catalog_matches_enum() {
    for (std::size_t index = 0; index < kCatalog.size(); ++index) {
        if (static_cast<std::size_t>(kCatalog[index].feature) != index) {
            return false;
        }
    }
    return true;
}
static_assert(catalog_matches_enum());

}  // namespace

std::span<const FeatureInfo> feature_catalog() noexcept {
    return kCatalog;
}

const FeatureInfo& feature_info(Feature feature) noexcept {
    return kCatalog[static_cast<std::size_t>(feature)];
}

std::optional<Feature> parse_feature_key(std::string_view key) noexcept {
    for (const FeatureInfo& info : kCatalog) {
        if (info.key == key) {
            return info.feature;
        }
    }
    return std::nullopt;
}

}  // namespace eclair::capability
