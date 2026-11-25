/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "options.hpp"

#include <cstddef>
#include <utility>

namespace eclair::smi {
namespace {

constexpr std::string_view kUsage =
    R"(Usage: eclair-smi [OPTIONS]

Show your GPU and the operations Eclair can run on it now: number formats the
hardware lacks, and the model kernels that inference engines call.

Options:
  -L, --list            List devices, one per line
  -i, --id DEVICE       Report one device: cuda:0, host, or an index such as 0
      --catalog         What each operation does, with its feature key
      --json            Machine-readable report (with --catalog, the catalog)
      --details         Add how each operation runs, unavailable operations, CPU
                        reference checks and diagnostics
      --disable LIST    Disable these comma-separated keys (fp8.* is a group)
                        as well as those in ECLAIR_DISABLE
      --plain           Plain text (the usual default)
      --pretty          The same text, with color on terminals
      --style STYLE     plain or pretty, else ECLAIR_SMI_STYLE, else the build
      --color WHEN      auto (the style decides), always, never. Honors NO_COLOR
      --no-color        Same as --color never
  -V, --version         Print the version
  -h, --help            Print this help

Runs as: "emulated on GPU" means Eclair builds the operation from other GPU
instructions because the hardware lacks its number format. "Native on GPU" means
the GPU's own units do the math. The CPU reference checks results and never
speeds up GPU work. GPU operations go through the installed NVIDIA driver.

Used by names the Eclair integrations that call an operation. A Ready operation
does not by itself make a model or framework work.

Exit status: 0 on success, 1 if output failed, 2 on a usage error.
)";

[[nodiscard]] std::optional<ColorMode> parse_color(std::string_view value) noexcept {
    if (value == "auto") {
        return ColorMode::Auto;
    }
    if (value == "always") {
        return ColorMode::Always;
    }
    if (value == "never") {
        return ColorMode::Never;
    }
    return std::nullopt;
}

class Parser {
public:
    explicit Parser(std::span<const std::string_view> arguments) : arguments_(arguments) {}

    [[nodiscard]] ParseResult run() {
        for (index_ = 0; index_ < arguments_.size() && !result_.error; ++index_) {
            parse_one(arguments_[index_]);
        }
        if (!result_.error) {
            finish();
        }
        return std::move(result_);
    }

private:
    void parse_one(std::string_view argument) {
        inline_value_.reset();
        if (argument.starts_with("--")) {
            const std::size_t equals = argument.find('=');
            if (equals != std::string_view::npos) {
                inline_value_ = argument.substr(equals + 1U);
                argument = argument.substr(0, equals);
            }
        }
        Options& options = result_.options;
        if (argument == "-h" || argument == "--help") {
            help_ = no_value(argument);
        } else if (argument == "-V" || argument == "--version") {
            version_ = no_value(argument);
        } else if (argument == "-L" || argument == "--list") {
            choose(Command::List, argument);
        } else if (argument == "--catalog") {
            choose(Command::Catalog, argument);
        } else if (argument == "--json") {
            options.json = no_value(argument);
        } else if (argument == "--details") {
            options.details = no_value(argument);
        } else if (argument == "--plain") {
            if (no_value(argument)) {
                options.style = StyleMode::Plain;
            }
        } else if (argument == "--pretty") {
            if (no_value(argument)) {
                options.style = StyleMode::Pretty;
            }
        } else if (argument == "--style") {
            if (const std::optional<std::string_view> value = take_value(argument)) {
                options.style = parse_style_mode(*value);
                if (!options.style) {
                    fail("--style expects plain or pretty, not '" + std::string(*value) + "'");
                }
            }
        } else if (argument == "--no-color") {
            if (no_value(argument)) {
                options.color = ColorMode::Never;
            }
        } else if (argument == "-i" || argument == "--id") {
            if (const std::optional<std::string_view> value = take_value(argument)) {
                options.device = std::string(*value);
            }
        } else if (argument == "--disable") {
            if (const std::optional<std::string_view> value = take_value(argument)) {
                options.disable_lists.emplace_back(*value);
            }
        } else if (argument == "--color") {
            if (const std::optional<std::string_view> value = take_value(argument)) {
                if (const std::optional<ColorMode> mode = parse_color(*value)) {
                    options.color = *mode;
                } else {
                    fail("--color expects auto, always, or never, not '" + std::string(*value) +
                         "'");
                }
            }
        } else if (argument.starts_with("-")) {
            fail("unknown option '" + std::string(argument) + "'");
        } else {
            fail("unexpected argument '" + std::string(argument) + "'");
        }
    }

    void finish() {
        Options& options = result_.options;
        if (help_) {
            options.command = Command::Help;
        } else if (version_) {
            options.command = Command::Version;
        } else if (options.json && options.command == Command::List) {
            fail("--json cannot be combined with --list");
        }
    }

    void choose(Command command, std::string_view argument) {
        if (!no_value(argument)) {
            return;
        }
        Options& options = result_.options;
        if (options.command != Command::Report && options.command != command) {
            fail("choose only one of --list and --catalog");
            return;
        }
        options.command = command;
    }

    [[nodiscard]] bool no_value(std::string_view argument) {
        if (inline_value_) {
            fail(std::string(argument) + " does not take a value");
            return false;
        }
        return true;
    }

    [[nodiscard]] std::optional<std::string_view> take_value(std::string_view argument) {
        std::optional<std::string_view> value = inline_value_;
        if (!value && index_ + 1U < arguments_.size()) {
            value = arguments_[++index_];
        }
        if (!value || value->empty()) {
            fail(std::string(argument) + " needs a value");
            return std::nullopt;
        }
        return value;
    }

    void fail(std::string message) {
        if (!result_.error) {
            result_.error = std::move(message);
        }
    }

    std::span<const std::string_view> arguments_;
    std::size_t index_ = 0;
    std::optional<std::string_view> inline_value_;
    bool help_ = false;
    bool version_ = false;
    ParseResult result_;
};

}  // namespace

ParseResult parse_options(std::span<const std::string_view> arguments) {
    return Parser(arguments).run();
}

std::string_view usage_text() noexcept {
    return kUsage;
}

std::optional<StyleMode> parse_style_mode(std::string_view value) noexcept {
    if (value == "plain") {
        return StyleMode::Plain;
    }
    if (value == "pretty") {
        return StyleMode::Pretty;
    }
    return std::nullopt;
}

}  // namespace eclair::smi
