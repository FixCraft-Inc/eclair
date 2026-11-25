/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "terminal.hpp"

#include <unistd.h>

#include <cstdlib>

namespace eclair::smi {
namespace {

[[nodiscard]] std::optional<std::string> read_environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string(value);
}

[[nodiscard]] bool is_set(const std::optional<std::string>& value) noexcept {
    return value.has_value() && !value->empty();
}

[[nodiscard]] std::string_view sgr(Tone tone) noexcept {
    switch (tone) {
        case Tone::Plain:
            return {};
        case Tone::Bold:
            return "\x1b[1m";
        case Tone::Dim:
            return "\x1b[2m";
        case Tone::Good:
            return "\x1b[32m";
        case Tone::Warning:
            return "\x1b[33m";
        case Tone::Info:
            return "\x1b[36m";
        case Tone::Bad:
            return "\x1b[31m";
    }
    return {};
}

}  // namespace

TerminalEnvironment capture_environment() {
    TerminalEnvironment environment;
    environment.stdout_is_terminal = ::isatty(STDOUT_FILENO) == 1;
    environment.no_color = read_environment("NO_COLOR");
    environment.term = read_environment("TERM");
    environment.style = read_environment("ECLAIR_SMI_STYLE");
    return environment;
}

StyleResolution resolve_style_mode(std::optional<StyleMode> requested,
                                   const TerminalEnvironment& environment,
                                   StyleMode build_default) {
    if (requested) {
        return {*requested, std::nullopt};
    }
    if (is_set(environment.style)) {
        if (const std::optional<StyleMode> mode = parse_style_mode(*environment.style)) {
            return {*mode, std::nullopt};
        }
        return {build_default,
                "ECLAIR_SMI_STYLE expects plain or pretty, not '" + *environment.style + "'"};
    }
    return {build_default, std::nullopt};
}

Style choose_style(StyleMode mode, ColorMode color, const TerminalEnvironment& environment) {
    const bool pretty = mode == StyleMode::Pretty;
    Style style;
    switch (color) {
        case ColorMode::Always:
            style.color = true;
            break;
        case ColorMode::Never:
            style.color = false;
            break;
        case ColorMode::Auto:
            style.color = pretty && environment.stdout_is_terminal &&
                          !is_set(environment.no_color) && environment.term.value_or("") != "dumb";
            break;
    }
    return style;
}

std::string paint(const Style& style, Tone tone, std::string_view text) {
    const std::string_view code = sgr(tone);
    if (!style.color || code.empty() || text.empty()) {
        return std::string(text);
    }
    std::string painted;
    painted.reserve(code.size() + text.size() + 4U);
    painted.append(code).append(text).append("\x1b[0m");
    return painted;
}

}  // namespace eclair::smi
