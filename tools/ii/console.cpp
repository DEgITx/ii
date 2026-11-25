// SPDX-License-Identifier: MIT
// Console setup for the CLI. This is the only platform-specific file of the CLI.
#include "console.h"

#if defined(_WIN32)
#include <windows.h>

#include <io.h>
#include <shellapi.h>
#else
#include <unistd.h>
#endif

#include <cstdio>

namespace ii::cli {

#if defined(_WIN32)
namespace {
std::string narrow(const wchar_t* wide) {
    const int length = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (length <= 1) {
        return {};
    }
    std::string result(static_cast<std::size_t>(length - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, result.data(), length, nullptr, nullptr);
    return result;
}
}  // namespace
#endif

std::vector<std::string> init_console(int argc, char** argv) {
    std::vector<std::string> args;
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    int count = 0;
    wchar_t** wide = CommandLineToArgvW(GetCommandLineW(), &count);
    if (wide != nullptr) {
        for (int i = 0; i < count; ++i) {
            args.push_back(narrow(wide[i]));
        }
        LocalFree(reinterpret_cast<HLOCAL>(wide));
        return args;
    }
#endif
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
}

bool stdout_is_terminal() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

}  // namespace ii::cli
