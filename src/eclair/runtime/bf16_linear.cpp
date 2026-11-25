/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/runtime/bf16_linear.hpp"

#include "linear_selection.hpp"

namespace eclair::runtime {

Bf16LinearPlan plan_bf16_linear(const ops::Bf16LinearShape& shape,
                                const capability::DeviceReport& device,
                                const capability::FeaturePolicy& policy) noexcept {
    Bf16LinearPlan plan;
    const ops::Status validation = ops::bf16_linear_sizes(shape, plan.sizes);
    if (validation != ops::Status::Success) {
        plan.state = validation == ops::Status::SizeOverflow ? LinearPlanState::SizeOverflow
                                                             : LinearPlanState::InvalidArgument;
        return plan;
    }
    const detail::LinearSelection selection = detail::select_linear(
        capability::Feature::Bf16Gemm, device, policy,
        detail::gpu_dimensions_fit(plan.sizes.output == 0U, shape.m, shape.n, shape.k));
    plan.state = selection.state;
    plan.implementation = selection.implementation;
    return plan;
}

}  // namespace eclair::runtime
