/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

// Backend selection for builds without CUDA. The CUDA backend still appears,
// marked not built, so the report explains why no GPU is listed.

#include "system_report.hpp"

#include "eclair/host/host_backend.hpp"

#include <utility>
#include <vector>

namespace eclair::smi {

std::vector<capability::BackendReport> probe_backends() {
    capability::BackendReport cuda;
    cuda.name = "cuda";
    cuda.label = "NVIDIA GPU";
    cuda.state = capability::BackendState::NotBuilt;
    cuda.detail = "GPU support is not included in this build; use a CUDA-enabled Eclair build";

    std::vector<capability::BackendReport> backends;
    backends.push_back(std::move(cuda));
    backends.push_back(host::probe_backend());
    return backends;
}

}  // namespace eclair::smi
