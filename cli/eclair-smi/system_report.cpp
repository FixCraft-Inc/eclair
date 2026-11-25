/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "system_report.hpp"

#include "eclair/capability/feature.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace eclair::smi {
namespace {

[[nodiscard]] bool is_index(std::string_view text) noexcept {
    return !text.empty() && std::all_of(text.begin(), text.end(), [](char character) {
        return character >= '0' && character <= '9';
    });
}

[[nodiscard]] bool matches(std::string_view selector, const capability::DeviceReport& device) {
    if (device.id == selector) {
        return true;
    }
    if (!is_index(selector)) {
        return false;
    }
    const std::size_t colon = device.id.rfind(':');
    return colon != std::string::npos && std::string_view(device.id).substr(colon + 1U) == selector;
}

}  // namespace

void apply_policy(const capability::FeaturePolicy& policy, std::string source,
                  SystemReport& report) {
    report.disabled_keys.clear();
    report.policy_source.clear();
    if (policy.empty()) {
        return;
    }
    for (capability::BackendReport& backend : report.backends) {
        capability::apply_policy(policy, backend);
    }
    for (const capability::FeatureInfo& info : capability::feature_catalog()) {
        if (policy.is_disabled(info.feature)) {
            report.disabled_keys.push_back(info.key);
        }
    }
    report.policy_source = std::move(source);
}

std::optional<std::string> select_device(std::string_view selector, SystemReport& report) {
    std::size_t match_count = 0U;
    for (const capability::BackendReport& backend : report.backends) {
        match_count += static_cast<std::size_t>(std::count_if(
            backend.devices.begin(), backend.devices.end(),
            [&](const capability::DeviceReport& device) { return matches(selector, device); }));
    }
    if (match_count == 0U) {
        return "no device matches '" + std::string(selector) + "', see eclair-smi --list";
    }
    if (match_count > 1U) {
        return "'" + std::string(selector) + "' matches devices in several backends, use a full id";
    }
    for (capability::BackendReport& backend : report.backends) {
        std::erase_if(backend.devices, [&](const capability::DeviceReport& device) {
            return !matches(selector, device);
        });
    }
    return std::nullopt;
}

}  // namespace eclair::smi
