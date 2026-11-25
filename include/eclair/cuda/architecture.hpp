/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/capability/report.hpp"

#include <string>
#include <string_view>
#include <vector>

// Host-only model of NVIDIA compute capabilities. It needs no CUDA toolkit,
// so the rules are tested on any machine. The CUDA backend supplies the live
// probe results.
namespace eclair::cuda {

struct ComputeCapability {
    int major = 0;
    int minor = 0;
};

// Whether the build holds kernel images that this device can load, and if
// not, the CUDA error name that says why.
struct KernelImage {
    bool loadable = false;
    std::string error;
};

// Results of runtime checks the backend performs on the device itself, one
// per kernel family.
struct KernelProbe {
    KernelImage fp8_decode;
    KernelImage fp8_encode;
    KernelImage fp8_linear;
    KernelImage bf16_convert;
    KernelImage bf16_linear;
    KernelImage paged_prefill;
    KernelImage paged_decode;
    KernelImage gated_delta;
    KernelImage gated_delta_output;
    KernelImage ray_trace;
};

// "sm_70" style spelling.
[[nodiscard]] std::string architecture_name(ComputeCapability capability);

// Marketing family such as "Volta", or empty when unknown.
[[nodiscard]] std::string_view architecture_family(ComputeCapability capability) noexcept;

// Dedicated units by compute capability. A unit that differs between products
// of one capability, such as RT cores on sm_75, reports Unknown.
[[nodiscard]] std::vector<capability::HardwareUnitStatus> hardware_units(
    ComputeCapability capability);

// Eclair feature states for one CUDA device, one entry per catalog feature in
// catalog order.
[[nodiscard]] std::vector<capability::FeatureStatus> evaluate_features(ComputeCapability capability,
                                                                       const KernelProbe& probe);

}  // namespace eclair::cuda
