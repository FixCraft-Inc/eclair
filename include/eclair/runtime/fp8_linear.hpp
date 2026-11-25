/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/fp8_linear.hpp"
#include "eclair/runtime/linear_plan.hpp"

namespace eclair::capability {
struct DeviceReport;
class FeaturePolicy;
}  // namespace eclair::capability

namespace eclair::runtime {

struct Fp8LinearPlan {
    LinearPlanState state = LinearPlanState::Unavailable;
    LinearImplementation implementation = LinearImplementation::None;
    ops::Fp8LinearSizes sizes;
};

// Experimental selection seam shared by applications and adapters. Invalid
// shapes, unknown capabilities and disabled features fail closed. No buffers,
// copies, allocation or synchronization are owned by a plan. A CPU fallback
// requires explicit selection of a CPU report and host-accessible buffers.
// The caller must match the GPU report to the current device before launch.
[[nodiscard]] Fp8LinearPlan plan_fp8_linear(const ops::Fp8LinearShape& shape,
                                            const capability::DeviceReport& device,
                                            const capability::FeaturePolicy& policy) noexcept;

}  // namespace eclair::runtime
