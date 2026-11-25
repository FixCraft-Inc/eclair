/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Minimal test assertions that stay active in every build type, unlike
// assert(), which NDEBUG removes from Release builds.

#include <cstdio>
#include <cstdlib>

namespace eclair::test {

inline int& failure_count() noexcept {
    static int count = 0;
    return count;
}

inline void record_failure(const char* expression, const char* file, int line) noexcept {
    std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
    ++failure_count();
}

// Returns the process exit status for a test main().
[[nodiscard]] inline int finish() noexcept {
    if (failure_count() != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failure_count());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

}  // namespace eclair::test

#define ECLAIR_CHECK(expression)                                             \
    do {                                                                     \
        if (!(expression)) {                                                 \
            ::eclair::test::record_failure(#expression, __FILE__, __LINE__); \
        }                                                                    \
    } while (false)
