// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>

namespace ii::cli {

/// Prepares the console for UTF-8 output and returns the command-line arguments as UTF-8 strings
/// (on Windows argv is in the ANSI code page, so the wide command line is used instead).
std::vector<std::string> init_console(int argc, char** argv);

/// True if standard output is an interactive terminal.
bool stdout_is_terminal();

}  // namespace ii::cli
