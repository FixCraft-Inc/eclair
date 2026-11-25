/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/capability/report.hpp"

#include "eclair/capability/policy.hpp"

namespace eclair::capability {

std::string_view to_string(FeatureState state) noexcept {
    switch (state) {
        case FeatureState::Enabled:
            return "enabled";
        case FeatureState::Disabled:
            return "disabled";
        case FeatureState::NotImplemented:
            return "not-implemented";
        case FeatureState::Unsupported:
            return "unsupported";
        case FeatureState::NotApplicable:
            return "n/a";
    }
    return "invalid";
}

std::string_view to_string(ExecutionPath path) noexcept {
    switch (path) {
        case ExecutionPath::None:
            return "none";
        case ExecutionPath::Hardware:
            return "hardware";
        case ExecutionPath::Emulated:
            return "emulated";
        case ExecutionPath::HostReference:
            return "host-reference";
    }
    return "invalid";
}

std::string_view to_string(HardwareUnit unit) noexcept {
    switch (unit) {
        case HardwareUnit::Fp16TensorCores:
            return "fp16_tensor_cores";
        case HardwareUnit::Int8TensorCores:
            return "int8_tensor_cores";
        case HardwareUnit::Tf32TensorCores:
            return "tf32_tensor_cores";
        case HardwareUnit::Bf16Arithmetic:
            return "bf16_arithmetic";
        case HardwareUnit::Fp8TensorCores:
            return "fp8_tensor_cores";
        case HardwareUnit::RayTracingCores:
            return "ray_tracing_cores";
    }
    return "invalid";
}

std::string_view to_string(HardwareSupport support) noexcept {
    switch (support) {
        case HardwareSupport::Present:
            return "present";
        case HardwareSupport::Absent:
            return "absent";
        case HardwareSupport::Unknown:
            return "unknown";
    }
    return "invalid";
}

std::string_view to_string(BackendState state) noexcept {
    switch (state) {
        case BackendState::Available:
            return "available";
        case BackendState::Unavailable:
            return "unavailable";
        case BackendState::NotBuilt:
            return "not-built";
    }
    return "invalid";
}

std::string_view display_label(HardwareUnit unit) noexcept {
    switch (unit) {
        case HardwareUnit::Fp16TensorCores:
            return "FP16 tensor cores";
        case HardwareUnit::Int8TensorCores:
            return "INT8 tensor cores";
        case HardwareUnit::Tf32TensorCores:
            return "TF32 tensor cores";
        case HardwareUnit::Bf16Arithmetic:
            return "BF16";
        case HardwareUnit::Fp8TensorCores:
            return "FP8 tensor cores";
        case HardwareUnit::RayTracingCores:
            return "RT cores";
    }
    return "invalid";
}

void apply_policy(const FeaturePolicy& policy, BackendReport& report) {
    for (DeviceReport& device : report.devices) {
        for (FeatureStatus& status : device.features) {
            if (status.state == FeatureState::Enabled && policy.is_disabled(status.feature)) {
                status.state = FeatureState::Disabled;
                status.detail = "disabled by policy, " + status.detail;
            }
        }
    }
}

}  // namespace eclair::capability
