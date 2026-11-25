/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "presenter.hpp"
#include <stdexcept>

namespace eclair::demo {
struct Presenter::Impl {};
Presenter::Presenter(const ops::RayTraceShape&) {
    throw std::runtime_error(
        "display support not built; use --headless or ECLAIR_BUILD_RENDER_DEMO=ON");
}
Presenter::~Presenter() = default;
std::vector<std::string> Presenter::events() {
    return {};
}
void Presenter::upload(std::span<const std::uint8_t>) {}
void Presenter::present(const std::string&, double) {}
}  // namespace eclair::demo
