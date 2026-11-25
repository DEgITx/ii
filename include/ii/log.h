// SPDX-License-Identifier: MIT
/// @file
/// Logging. ii never writes to the console on its own initiative: messages go to a `Logger` owned by the
/// application (or by an ii `Context`), whose sink the application controls.
#pragma once

#include <ii/export.h>

#include <format>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace ii {

enum class LogLevel : int {
    kTrace = 0,
    kDebug,
    kInfo,
    kWarning,
    kError,
    kOff,  ///< Disables logging when used as the logger level.
};

/// Returns "trace", "debug", "info", "warning", "error" or "off".
II_API std::string_view to_string(LogLevel level) noexcept;

/// Receives formatted log messages. Must be thread-safe.
using LogSink = std::function<void(LogLevel level, std::string_view message)>;

/// Returns a sink that writes "[ii:<level>] <message>" lines to stderr.
II_API LogSink stderr_log_sink();

/// Filters messages by level and forwards them to a sink.
///
/// Formatting is skipped entirely for disabled levels. Configure the logger before sharing it between threads.
class Logger {
public:
    /// Logs `kInfo` and above to stderr.
    Logger() : Logger(LogLevel::kInfo, stderr_log_sink()) {}
    Logger(LogLevel level, LogSink sink) : level_(level), sink_(std::move(sink)) {}

    LogLevel level() const noexcept { return level_; }
    void set_level(LogLevel level) noexcept { level_ = level; }
    void set_sink(LogSink sink) { sink_ = std::move(sink); }

    bool enabled(LogLevel level) const noexcept { return level >= level_ && level != LogLevel::kOff && sink_; }

    void log(LogLevel level, std::string_view message) const {
        if (enabled(level)) {
            sink_(level, message);
        }
    }

    template <class... Args>
    void log(LogLevel level, std::format_string<Args...> format, Args&&... args) const {
        if (enabled(level)) {
            sink_(level, std::format(format, std::forward<Args>(args)...));
        }
    }

    template <class... Args>
    void trace(std::format_string<Args...> format, Args&&... args) const {
        log(LogLevel::kTrace, format, std::forward<Args>(args)...);
    }
    template <class... Args>
    void debug(std::format_string<Args...> format, Args&&... args) const {
        log(LogLevel::kDebug, format, std::forward<Args>(args)...);
    }
    template <class... Args>
    void info(std::format_string<Args...> format, Args&&... args) const {
        log(LogLevel::kInfo, format, std::forward<Args>(args)...);
    }
    template <class... Args>
    void warning(std::format_string<Args...> format, Args&&... args) const {
        log(LogLevel::kWarning, format, std::forward<Args>(args)...);
    }
    template <class... Args>
    void error(std::format_string<Args...> format, Args&&... args) const {
        log(LogLevel::kError, format, std::forward<Args>(args)...);
    }

private:
    LogLevel level_;
    LogSink sink_;
};

}  // namespace ii
