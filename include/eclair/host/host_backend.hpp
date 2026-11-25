/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/capability/report.hpp"

#include <string_view>

namespace eclair::host {

inline constexpr std::string_view kBackendName = "host";

// Reports the portable reference implementations. Each enabled feature runs a
// small known-answer check first, so a broken reference build is reported as
// unsupported instead of enabled.
[[nodiscard]] capability::BackendReport probe_backend();

}  // namespace eclair::host
