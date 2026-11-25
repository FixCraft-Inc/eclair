/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Marks a format function that host code and CUDA kernels share, so both
// compile from one definition. Headers using it stay C++17 for CUDA units.
#if defined(__CUDACC__)
#define ECLAIR_HOST_DEVICE __host__ __device__
#else
#define ECLAIR_HOST_DEVICE
#endif
