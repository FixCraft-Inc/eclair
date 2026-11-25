/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "system_report.hpp"
#include "terminal.hpp"

#include <ostream>

namespace eclair::smi {

// GPU operations first; details additionally show CPU checks, unavailable
// operations, machine keys and runtime diagnostics. Structured data is not filtered.
void render_report(std::ostream& out, const SystemReport& report, const Style& style,
                   bool details = false);

// One line per device, like nvidia-smi -L.
void render_list(std::ostream& out, const SystemReport& report, const Style& style);

// Every feature key with its title and summary.
void render_catalog(std::ostream& out, const Style& style);

}  // namespace eclair::smi
