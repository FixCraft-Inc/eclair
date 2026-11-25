/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Internal to eclair_ops: shared extent, alignment and overlap checks for
// operation validators. Not a public header.

#include "eclair/ops/buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace eclair::ops::detail {

[[nodiscard]] inline bool multiply(std::size_t a, std::size_t b, std::size_t& result) noexcept {
    if (b != 0U && a > std::numeric_limits<std::size_t>::max() / b) {
        return false;
    }
    result = a * b;
    return true;
}

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

// Checks that a view claims at least `needed` elements, is aligned for T and
// does not wrap the address space. A zero-length use leaves `address` empty.
template <typename T>
[[nodiscard]] Status range(BufferView<T> buffer, std::size_t needed,
                           AddressRange& address) noexcept {
    if (buffer.size < needed) {
        return Status::BufferTooSmall;
    }
    if (needed == 0U) {
        return Status::Success;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
    std::size_t bytes = 0;
    if (buffer.data == nullptr || begin % alignof(T) != 0U) {
        return Status::InvalidArgument;
    }
    if (!multiply(needed, sizeof(T), bytes) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return Status::SizeOverflow;
    }
    address = {begin, begin + bytes};
    return Status::Success;
}

[[nodiscard]] inline bool overlaps(AddressRange a, AddressRange b) noexcept {
    return a.begin < b.end && b.begin < a.end;
}

}  // namespace eclair::ops::detail
