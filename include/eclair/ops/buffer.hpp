/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include <cstddef>
#include <cstdint>

// Types shared by every operation contract in ops/.
namespace eclair::ops {

enum class Status : std::uint8_t {
    Success,
    InvalidArgument,
    SizeOverflow,
    BufferTooSmall,
};

// A caller-owned span in one execution backend's memory domain: host memory
// for host references, memory accessible on the current device for CUDA
// launchers. `size` is the caller's claim. Validation cannot establish the
// actual allocation size or residency from a raw pointer.
template <typename T>
struct BufferView {
    T* data = nullptr;
    std::size_t size = 0;  // Element count, not bytes. Caller owns storage.
};

}  // namespace eclair::ops
