/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Internal to eclair_cuda: shared-memory limits the kernels size themselves
// against. Not a public header.

#include <cstddef>

namespace eclair::cuda::detail {

// Volta's opt-in limit of dynamic shared memory per block, in bytes. Kernels
// that use more than 48 KiB raise their limit with cudaFuncSetAttribute.
inline constexpr std::size_t kMaxSharedBytes = 98304U;

}  // namespace eclair::cuda::detail
