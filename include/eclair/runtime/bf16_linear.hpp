/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/bf16_linear.hpp"
#include "eclair/runtime/linear_plan.hpp"

namespace eclair::capability {
struct DeviceReport;
class FeaturePolicy;
}  // namespace eclair::capability

namespace eclair::runtime {

// A ready plan covers one call: row scaling of A into caller-provided
// workspace (sizes.activations binary16 elements and M shifts), then the
// linear operation on W row-scaled in advance.
struct Bf16LinearPlan {
    LinearPlanState state = LinearPlanState::Unavailable;
    LinearImplementation implementation = LinearImplementation::None;
    ops::Bf16LinearSizes sizes;
};

// Same selection rules as plan_fp8_linear, for the bf16.gemm feature.
[[nodiscard]] Bf16LinearPlan plan_bf16_linear(const ops::Bf16LinearShape& shape,
                                              const capability::DeviceReport& device,
                                              const capability::FeaturePolicy& policy) noexcept;

}  // namespace eclair::runtime
