/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace eclair::smi {

enum class Command : std::uint8_t {
    Report,
    List,
    Catalog,
    Help,
    Version,
};

// Plain is words only with no color, like nvidia-smi, and is safe in logs and
// any terminal. Pretty adds color on terminals with the same text.
enum class StyleMode : std::uint8_t {
    Plain,
    Pretty,
};

// An explicit override of the style's color choice.
enum class ColorMode : std::uint8_t {
    Auto,
    Always,
    Never,
};

struct Options {
    Command command = Command::Report;
    // Unset means ECLAIR_SMI_STYLE, then the build default.
    std::optional<StyleMode> style;
    ColorMode color = ColorMode::Auto;
    bool json = false;
    bool details = false;
    std::optional<std::string> device;
    // Raw --disable values in command-line order. The capability policy
    // validates them, so parsing here stays syntactic.
    std::vector<std::string> disable_lists;
};

struct ParseResult {
    Options options;
    std::optional<std::string> error;
};

// Parses arguments after the program name.
[[nodiscard]] ParseResult parse_options(std::span<const std::string_view> arguments);

[[nodiscard]] std::string_view usage_text() noexcept;

// "plain" or "pretty", as accepted by --style and ECLAIR_SMI_STYLE.
[[nodiscard]] std::optional<StyleMode> parse_style_mode(std::string_view value) noexcept;

}  // namespace eclair::smi
