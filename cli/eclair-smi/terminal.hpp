/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "options.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace eclair::smi {

// The parts of the process environment that decide presentation, captured
// once so rendering never reads global state.
struct TerminalEnvironment {
    bool stdout_is_terminal = false;
    std::optional<std::string> no_color;
    std::optional<std::string> term;
    std::optional<std::string> style;  // ECLAIR_SMI_STYLE
};

struct Style {
    bool color = false;
};

struct StyleResolution {
    StyleMode mode = StyleMode::Plain;
    std::optional<std::string> error;
};

enum class Tone : std::uint8_t {
    Plain,
    Bold,
    Dim,
    Good,     // Enabled, present.
    Warning,  // Disabled, unknown.
    Info,     // Informational diagnostics.
    Bad,      // Unsupported, unavailable.
};

[[nodiscard]] TerminalEnvironment capture_environment();

// The flag wins, then a non-empty ECLAIR_SMI_STYLE, then the build default.
[[nodiscard]] StyleResolution resolve_style_mode(std::optional<StyleMode> requested,
                                                 const TerminalEnvironment& environment,
                                                 StyleMode build_default);

// Plain uses color only with --color=always. Pretty colors a terminal,
// honoring NO_COLOR (any non-empty value) and TERM=dumb. Both use words
// and ASCII punctuation for status; presentation does not depend on locale.
[[nodiscard]] Style choose_style(StyleMode mode, ColorMode color,
                                 const TerminalEnvironment& environment);

// Wraps text in an SGR sequence when color is on. Pad before painting, since
// escape codes have no display width.
[[nodiscard]] std::string paint(const Style& style, Tone tone, std::string_view text);

}  // namespace eclair::smi
