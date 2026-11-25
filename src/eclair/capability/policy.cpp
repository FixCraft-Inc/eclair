/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/capability/policy.hpp"

#include <cstddef>

namespace eclair::capability {
namespace {

[[nodiscard]] std::string_view trim(std::string_view text) noexcept {
    constexpr std::string_view kSpace = " \t";
    const std::size_t first = text.find_first_not_of(kSpace);
    if (first == std::string_view::npos) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(kSpace);
    return text.substr(first, last - first + 1U);
}

// Returns the features one entry selects, or nothing when the entry is unknown.
[[nodiscard]] std::bitset<kFeatureCount> select(std::string_view entry) noexcept {
    std::bitset<kFeatureCount> selected;
    constexpr std::string_view kWildcard = ".*";
    if (entry.size() > kWildcard.size() && entry.ends_with(kWildcard)) {
        // Keep the dot so "fp8.*" cannot match a hypothetical "fp8x.y".
        const std::string_view prefix = entry.substr(0, entry.size() - 1U);
        for (const FeatureInfo& info : feature_catalog()) {
            if (info.key.starts_with(prefix)) {
                selected.set(static_cast<std::size_t>(info.feature));
            }
        }
        return selected;
    }
    if (const std::optional<Feature> feature = parse_feature_key(entry)) {
        selected.set(static_cast<std::size_t>(*feature));
    }
    return selected;
}

}  // namespace

void FeaturePolicy::disable(Feature feature) noexcept {
    disabled_.set(static_cast<std::size_t>(feature));
}

bool FeaturePolicy::is_disabled(Feature feature) const noexcept {
    return disabled_.test(static_cast<std::size_t>(feature));
}

bool FeaturePolicy::empty() const noexcept {
    return disabled_.none();
}

std::optional<std::string> FeaturePolicy::disable_list(std::string_view list) {
    std::bitset<kFeatureCount> pending;
    while (!list.empty()) {
        const std::size_t comma = list.find(',');
        const std::string_view entry = trim(list.substr(0, comma));
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1U);
        if (entry.empty()) {
            continue;
        }
        const std::bitset<kFeatureCount> selected = select(entry);
        if (selected.none()) {
            return "unknown feature '" + std::string(entry) + "'";
        }
        pending |= selected;
    }
    disabled_ |= pending;
    return std::nullopt;
}

}  // namespace eclair::capability
