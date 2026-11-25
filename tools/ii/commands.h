// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>

namespace ii::cli {

/// Each command receives its arguments (without the command name) and returns the process exit code.
int run_info(const std::vector<std::string>& args);

}  // namespace ii::cli
