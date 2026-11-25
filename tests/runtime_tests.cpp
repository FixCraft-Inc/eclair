/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/capability/policy.hpp"
#include "eclair/cuda/architecture.hpp"
#include "eclair/host/host_backend.hpp"
#include "eclair/runtime/bf16_linear.hpp"
#include "eclair/runtime/fp8_linear.hpp"

#include <limits>

namespace {

using eclair::capability::Feature;
using eclair::runtime::LinearImplementation;
using eclair::runtime::LinearPlanState;

// The BF16 plan shares the FP8 plan's selection rules but answers to its own
// feature, so each feature's state must decide only its own plan.
void check_bf16_plan() {
    using eclair::runtime::plan_bf16_linear;
    const eclair::ops::Bf16LinearShape shape{2, 3, 5};
    eclair::capability::FeaturePolicy policy;
    const auto host = eclair::host::probe_backend();
    const auto reference = plan_bf16_linear(shape, host.devices[0], policy);
    ECLAIR_CHECK(reference.state == LinearPlanState::Ready);
    ECLAIR_CHECK(reference.implementation == LinearImplementation::CpuReference);
    ECLAIR_CHECK(reference.sizes.activations == 10U);
    ECLAIR_CHECK(reference.sizes.weights == 15U);
    ECLAIR_CHECK(reference.sizes.output == 6U);

    eclair::capability::DeviceReport gpu;
    eclair::cuda::KernelProbe probe;
    probe.bf16_linear.loadable = true;
    gpu.features = eclair::cuda::evaluate_features({7, 0}, probe);
    ECLAIR_CHECK(plan_bf16_linear(shape, gpu, policy).implementation ==
                 LinearImplementation::GpuEmulation);
    auto too_deep = shape;
    too_deep.k = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1U;
    ECLAIR_CHECK(plan_bf16_linear(too_deep, gpu, policy).state == LinearPlanState::InvalidArgument);
    ECLAIR_CHECK(plan_bf16_linear(too_deep, host.devices[0], policy).state ==
                 LinearPlanState::Ready);

    policy.disable(Feature::Fp8Gemm);
    ECLAIR_CHECK(plan_bf16_linear(shape, gpu, policy).state == LinearPlanState::Ready);
    policy.disable(Feature::Bf16Gemm);
    ECLAIR_CHECK(plan_bf16_linear(shape, gpu, policy).state == LinearPlanState::Disabled);
    policy = {};
    probe.bf16_linear.loadable = false;
    probe.fp8_linear.loadable = true;
    gpu.features = eclair::cuda::evaluate_features({7, 0}, probe);
    ECLAIR_CHECK(plan_bf16_linear(shape, gpu, policy).state == LinearPlanState::Unavailable);
    ECLAIR_CHECK(plan_bf16_linear(shape, host.devices[0], policy).state == LinearPlanState::Ready);

    auto invalid = shape;
    invalid.k = 0;
    ECLAIR_CHECK(plan_bf16_linear(invalid, host.devices[0], policy).state ==
                 LinearPlanState::InvalidArgument);
    invalid = shape;
    invalid.n = std::numeric_limits<std::size_t>::max();
    ECLAIR_CHECK(plan_bf16_linear(invalid, host.devices[0], policy).state ==
                 LinearPlanState::SizeOverflow);
}

}  // namespace

int main() {
    using eclair::runtime::plan_fp8_linear;
    const eclair::ops::Fp8LinearShape shape{2, 3, 5, 1, 5};
    eclair::capability::FeaturePolicy policy;
    const auto host = eclair::host::probe_backend();
    const auto reference = plan_fp8_linear(shape, host.devices[0], policy);
    ECLAIR_CHECK(reference.state == LinearPlanState::Ready);
    ECLAIR_CHECK(reference.implementation == LinearImplementation::CpuReference);
    ECLAIR_CHECK(reference.sizes.output == 6U);
    eclair::capability::DeviceReport gpu;
    eclair::cuda::KernelProbe probe;
    probe.fp8_linear.loadable = true;
    gpu.features = eclair::cuda::evaluate_features({7, 0}, probe);
    ECLAIR_CHECK(plan_fp8_linear(shape, gpu, policy).implementation ==
                 LinearImplementation::GpuEmulation);
    auto too_wide = shape;
    too_wide.n = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1U;
    ECLAIR_CHECK(plan_fp8_linear(too_wide, gpu, policy).state == LinearPlanState::InvalidArgument);
    auto empty = too_wide;
    empty.m = 0;
    ECLAIR_CHECK(plan_fp8_linear(empty, gpu, policy).state == LinearPlanState::Ready);
    // CPU shapes do not share CUDA's signed-int dimension restriction.
    ECLAIR_CHECK(plan_fp8_linear(too_wide, host.devices[0], policy).state ==
                 LinearPlanState::Ready);
    policy.disable(Feature::Fp8Gemm);
    ECLAIR_CHECK(plan_fp8_linear(shape, gpu, policy).state == LinearPlanState::Disabled);
    ECLAIR_CHECK(plan_fp8_linear(shape, gpu, policy).implementation == LinearImplementation::None);
    policy = {};
    probe.fp8_linear.loadable = false;
    gpu.features = eclair::cuda::evaluate_features({7, 0}, probe);
    ECLAIR_CHECK(plan_fp8_linear(shape, gpu, policy).state == LinearPlanState::Unavailable);
    // CPU fallback requires a separate, explicit plan for the CPU device.
    ECLAIR_CHECK(plan_fp8_linear(shape, host.devices[0], policy).state == LinearPlanState::Ready);
    gpu.features.clear();
    ECLAIR_CHECK(plan_fp8_linear(shape, gpu, policy).state == LinearPlanState::Unavailable);
    gpu.features = {{Feature::Fp8Gemm, eclair::capability::FeatureState::Enabled,
                     eclair::capability::ExecutionPath::Hardware, "unsupported implementation"}};
    ECLAIR_CHECK(plan_fp8_linear(shape, gpu, policy).state == LinearPlanState::Unavailable);
    gpu.features = host.devices[0].features;
    gpu.features.push_back(gpu.features[4]);
    ECLAIR_CHECK(plan_fp8_linear(shape, gpu, policy).state == LinearPlanState::Unavailable);
    auto invalid = shape;
    invalid.block_k = 0;
    ECLAIR_CHECK(plan_fp8_linear(invalid, host.devices[0], policy).state ==
                 LinearPlanState::InvalidArgument);
    invalid = shape;
    invalid.m = std::numeric_limits<std::size_t>::max();
    ECLAIR_CHECK(plan_fp8_linear(invalid, host.devices[0], policy).state ==
                 LinearPlanState::SizeOverflow);
    check_bf16_plan();
    return eclair::test::finish();
}
