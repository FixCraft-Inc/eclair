/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "linear_selection.hpp"

#include "eclair/capability/policy.hpp"
#include "eclair/capability/report.hpp"

namespace eclair::runtime::detail {

LinearSelection select_linear(capability::Feature feature, const capability::DeviceReport& device,
                              const capability::FeaturePolicy& policy, bool gpu_fits) noexcept {
    if (policy.is_disabled(feature)) {
        return {LinearPlanState::Disabled, LinearImplementation::None};
    }
    const capability::FeatureStatus* found = nullptr;
    for (const auto& status : device.features) {
        if (status.feature == feature) {
            if (found != nullptr) {
                return {};  // Ambiguous reports cannot authorize execution.
            }
            found = &status;
        }
    }
    if (found == nullptr) {
        return {};
    }
    if (found->state == capability::FeatureState::Disabled) {
        return {LinearPlanState::Disabled, LinearImplementation::None};
    }
    if (found->state != capability::FeatureState::Enabled) {
        return {};
    }
    switch (found->path) {
        case capability::ExecutionPath::HostReference:
            return {LinearPlanState::Ready, LinearImplementation::CpuReference};
        case capability::ExecutionPath::Emulated:
            if (!gpu_fits) {
                return {LinearPlanState::InvalidArgument, LinearImplementation::None};
            }
            return {LinearPlanState::Ready, LinearImplementation::GpuEmulation};
        case capability::ExecutionPath::None:
        case capability::ExecutionPath::Hardware:
            break;
    }
    return {};
}

}  // namespace eclair::runtime::detail
