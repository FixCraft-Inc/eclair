/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "json_report.hpp"
#include "options.hpp"
#include "system_report.hpp"
#include "terminal.hpp"
#include "text_report.hpp"

#include "eclair/capability/policy.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#ifndef ECLAIR_VERSION_STRING
#error "ECLAIR_VERSION_STRING must be defined by the build"
#endif
#ifndef ECLAIR_SMI_DEFAULT_PRETTY
#error "ECLAIR_SMI_DEFAULT_PRETTY must be defined by the build"
#endif

namespace {

constexpr int kExitOutputFailure = 1;
constexpr int kExitUsage = 2;
constexpr eclair::smi::StyleMode kBuildDefaultStyle =
    ECLAIR_SMI_DEFAULT_PRETTY ? eclair::smi::StyleMode::Pretty : eclair::smi::StyleMode::Plain;

int usage_error(std::string_view message) {
    std::cerr << "eclair-smi: " << message << "\nTry 'eclair-smi --help'.\n";
    return kExitUsage;
}

int finish_output() {
    std::cout.flush();
    return std::cout ? EXIT_SUCCESS : kExitOutputFailure;
}

// Builds the policy from ECLAIR_DISABLE and every --disable, in that order.
// Returns an error message for the first invalid entry.
std::optional<std::string> build_policy(const eclair::smi::Options& options,
                                        eclair::capability::FeaturePolicy& policy,
                                        std::string& source) {
    const std::string variable(eclair::capability::kDisableEnvironmentVariable);
    const char* environment = std::getenv(variable.c_str());
    if (environment != nullptr && *environment != '\0') {
        if (std::optional<std::string> error = policy.disable_list(environment)) {
            return variable + ": " + *error;
        }
        source = variable;
    }
    for (const std::string& list : options.disable_lists) {
        if (std::optional<std::string> error = policy.disable_list(list)) {
            return "--disable: " + *error;
        }
    }
    if (!options.disable_lists.empty()) {
        source += source.empty() ? "--disable" : " and --disable";
    }
    return std::nullopt;
}

int run(const eclair::smi::Options& options) {
    using eclair::smi::Command;

    if (options.command == Command::Help) {
        std::cout << eclair::smi::usage_text();
        return finish_output();
    }
    if (options.command == Command::Version) {
        std::cout << "eclair-smi " ECLAIR_VERSION_STRING "\n";
        return finish_output();
    }

    eclair::capability::FeaturePolicy policy;
    std::string policy_source;
    if (std::optional<std::string> error = build_policy(options, policy, policy_source)) {
        return usage_error(*error);
    }

    const eclair::smi::TerminalEnvironment terminal = eclair::smi::capture_environment();
    const eclair::smi::StyleResolution mode =
        eclair::smi::resolve_style_mode(options.style, terminal, kBuildDefaultStyle);
    if (mode.error) {
        return usage_error(*mode.error);
    }
    const eclair::smi::Style style = eclair::smi::choose_style(mode.mode, options.color, terminal);
    if (options.command == Command::Catalog) {
        if (options.json) {
            eclair::smi::render_json_catalog(std::cout);
        } else {
            eclair::smi::render_catalog(std::cout, style);
        }
        return finish_output();
    }

    eclair::smi::SystemReport report;
    report.version = ECLAIR_VERSION_STRING;
    report.backends = eclair::smi::probe_backends();
    eclair::smi::apply_policy(policy, policy_source, report);
    if (options.device) {
        if (std::optional<std::string> error =
                eclair::smi::select_device(*options.device, report)) {
            return usage_error(*error);
        }
    }

    if (options.json) {
        eclair::smi::render_json_report(std::cout, report);
    } else if (options.command == Command::List) {
        eclair::smi::render_list(std::cout, report, style);
    } else {
        eclair::smi::render_report(std::cout, report, style, options.details);
    }
    return finish_output();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<std::string_view> arguments(argv + 1, argv + argc);
        const eclair::smi::ParseResult parsed = eclair::smi::parse_options(arguments);
        if (parsed.error) {
            return usage_error(*parsed.error);
        }
        return run(parsed.options);
    } catch (const std::exception& error) {
        std::cerr << "eclair-smi: internal error: " << error.what() << '\n';
    } catch (...) {
        std::cerr << "eclair-smi: internal error\n";
    }
    return kExitOutputFailure;
}
