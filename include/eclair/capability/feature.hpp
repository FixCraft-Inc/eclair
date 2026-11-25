/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace eclair::capability {

// Operations Eclair defines independently of any backend. A backend reports
// whether it can execute each one. The catalog itself claims nothing about
// hardware support.
enum class Feature : std::uint8_t {
    Fp8E4m3fnDecode,
    Fp8E5m2Decode,
    Fp8E4m3fnEncode,
    Fp8E5m2Encode,
    Fp8Gemm,
    Bf16Compat,
    Bf16Gemm,
    PagedAttentionPrefill,
    PagedAttentionDecode,
    GatedDeltaStates,
    GatedDeltaOutput,
    RayTracingEmulation,
};

inline constexpr std::size_t kFeatureCount = 12U;

struct FeatureInfo {
    Feature feature;
    // Stable dotted key used by policy, the CLI, and JSON. Never reuse a key
    // for a different operation.
    std::string_view key;
    std::string_view title;
    std::string_view summary;
};

// Every feature exactly once, in display order.
[[nodiscard]] std::span<const FeatureInfo> feature_catalog() noexcept;

[[nodiscard]] const FeatureInfo& feature_info(Feature feature) noexcept;

[[nodiscard]] std::optional<Feature> parse_feature_key(std::string_view key) noexcept;

}  // namespace eclair::capability
