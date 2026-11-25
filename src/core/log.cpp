// SPDX-License-Identifier: MIT
#include <ii/log.h>

#include <cstdio>
#include <string>

namespace ii {

std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
    case LogLevel::kTrace:
        return "trace";
    case LogLevel::kDebug:
        return "debug";
    case LogLevel::kInfo:
        return "info";
    case LogLevel::kWarning:
        return "warning";
    case LogLevel::kError:
        return "error";
    case LogLevel::kOff:
        return "off";
    }
    return "unknown";
}

LogSink stderr_log_sink() {
    return [](LogLevel level, std::string_view message) {
        // Build the whole line first so concurrent messages are not interleaved within a line.
        std::string line;
        line.reserve(message.size() + 16);
        line += "[ii:";
        line += to_string(level);
        line += "] ";
        line += message;
        line += '\n';
        std::fwrite(line.data(), 1, line.size(), stderr);
    };
}

}  // namespace ii
