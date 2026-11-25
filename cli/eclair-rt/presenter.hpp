/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once
#include "eclair/ops/ray_trace.hpp"
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace eclair::demo {
class Presenter {
public:
    explicit Presenter(const ops::RayTraceShape& shape);
    ~Presenter();
    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;
    [[nodiscard]] std::vector<std::string> events();
    void upload(std::span<const std::uint8_t> rgba);
    void present(const std::string& mode, double fps);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace eclair::demo
