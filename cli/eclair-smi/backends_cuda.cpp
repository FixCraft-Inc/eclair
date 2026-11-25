/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

// Backend selection for builds with the CUDA backend.

#include "system_report.hpp"

#include "eclair/cuda/cuda_backend.hpp"
#include "eclair/host/host_backend.hpp"

#include <vector>

namespace eclair::smi {

std::vector<capability::BackendReport> probe_backends() {
    std::vector<capability::BackendReport> backends;
    backends.push_back(cuda::probe_backend());
    backends.push_back(host::probe_backend());
    return backends;
}

}  // namespace eclair::smi
