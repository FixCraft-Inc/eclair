/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Internal to eclair_runtime: the capability and policy decision behind every
// linear plan. Not a public header.

#include "eclair/capability/feature.hpp"
#include "eclair/runtime/linear_plan.hpp"

#include <cstddef>
#include <limits>

namespace eclair::capability {
struct DeviceReport;
class FeaturePolicy;
}  // namespace eclair::capability

namespace eclair::runtime::detail {

struct LinearSelection {
    LinearPlanState state = LinearPlanState::Unavailable;
    LinearImplementation implementation = LinearImplementation::None;
};

// CUDA kernels take nonempty dimensions as int, so a GPU plan needs each to
// fit. An empty operation launches nothing and always fits.
[[nodiscard]] inline bool gpu_dimensions_fit(bool empty, std::size_t m, std::size_t n,
                                             std::size_t k) noexcept {
    constexpr auto kMax = static_cast<std::size_t>(std::numeric_limits<int>::max());
    return empty || (m <= kMax && n <= kMax && k <= kMax);
}

// Selects an implementation of an already validated operation. A policy or
// report that disables the feature yields Disabled. A missing, duplicated or
// unimplemented feature, or an unknown execution path, fails closed as
// Unavailable. A GPU path whose dimensions do not fit is InvalidArgument.
[[nodiscard]] LinearSelection select_linear(capability::Feature feature,
                                            const capability::DeviceReport& device,
                                            const capability::FeaturePolicy& policy,
                                            bool gpu_fits) noexcept;

}  // namespace eclair::runtime::detail
