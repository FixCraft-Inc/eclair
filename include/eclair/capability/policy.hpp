/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/capability/feature.hpp"

#include <bitset>
#include <optional>
#include <string>
#include <string_view>

namespace eclair::capability {

// Environment variable read by Eclair programs for the default policy. Its
// value uses the same syntax as FeaturePolicy::disable_list.
inline constexpr std::string_view kDisableEnvironmentVariable = "ECLAIR_DISABLE";

// The set of features a caller has turned off. Dispatch must treat a disabled
// feature as unavailable and take its documented fallback.
class FeaturePolicy {
public:
    void disable(Feature feature) noexcept;
    [[nodiscard]] bool is_disabled(Feature feature) const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    // Disables a comma-separated list of feature keys. An entry ending in
    // ".*" selects every key under that prefix, so "fp8.*" covers all FP8
    // features. Surrounding spaces and empty entries are ignored. On error
    // the policy is unchanged and the message names the rejected entry.
    [[nodiscard]] std::optional<std::string> disable_list(std::string_view list);

private:
    std::bitset<kFeatureCount> disabled_;
};

}  // namespace eclair::capability
