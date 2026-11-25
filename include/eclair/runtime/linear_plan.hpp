/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include <cstdint>

// Outcome types shared by every linear operation's plan.
namespace eclair::runtime {

enum class LinearPlanState : std::uint8_t {
    Ready,
    InvalidArgument,
    SizeOverflow,
    Disabled,
    Unavailable,
};

enum class LinearImplementation : std::uint8_t {
    None,
    CpuReference,
    GpuEmulation,
};

}  // namespace eclair::runtime
