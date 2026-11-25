/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/capability/policy.hpp"
#include "eclair/capability/report.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace eclair::smi {

// Everything one invocation prints, assembled before any rendering.
struct SystemReport {
    std::string version;
    std::vector<capability::BackendReport> backends;
    // Disabled feature keys in catalog order. Empty when no policy is active.
    std::vector<std::string_view> disabled_keys;
    // Where the policy came from, such as "ECLAIR_DISABLE". Empty without one.
    std::string policy_source;
};

// Every compiled-in backend in a fixed order: accelerators first, the host
// reference last. The build selects the implementation, so this declaration
// names no vendor.
[[nodiscard]] std::vector<capability::BackendReport> probe_backends();

// Applies the policy to every backend and records the disabled keys.
void apply_policy(const capability::FeaturePolicy& policy, std::string source,
                  SystemReport& report);

// Keeps only the device the selector names and drops every other device.
// Backends stay listed so their state remains visible. A selector is a full
// id such as "cuda:0" or "host", or a bare index that matches "<backend>:N"
// in exactly one backend. Returns an error message when nothing or more than
// one device matches.
[[nodiscard]] std::optional<std::string> select_device(std::string_view selector,
                                                       SystemReport& report);

}  // namespace eclair::smi
