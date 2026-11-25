/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/capability/report.hpp"

#include <string_view>

namespace eclair::cuda {

inline constexpr std::string_view kBackendName = "cuda";

// Enumerates CUDA devices through the supported runtime API and reports each
// one's hardware units and Eclair features. Probing initializes each device's
// primary context to check kernel images. Failures become an Unavailable
// report rather than an error, so diagnostics can always print something.
[[nodiscard]] capability::BackendReport probe_backend();

}  // namespace eclair::cuda
