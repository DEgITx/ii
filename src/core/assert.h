// SPDX-License-Identifier: MIT
/// @file
/// Internal assertion macros.
///
///   II_CHECK(cond, ...)   always evaluated; terminates with a message when false. For states that must never occur
///                         even in release builds (they would otherwise cause memory corruption).
///   II_ASSERT(cond, ...)  evaluated only in debug builds (when NDEBUG is not defined). For internal invariants.
///   II_UNREACHABLE()      marks code that must never execute.
///
/// The optional arguments are a std::format format string and its arguments.
#pragma once

#include <format>
#include <string>
#include <string_view>

namespace ii::detail {

[[noreturn]] void fatal_error(const char* file, int line, std::string_view message) noexcept;

inline std::string check_message(const char* expression) {
    return std::string("check failed: ") + expression;
}

template <class... Args>
std::string check_message(const char* expression, std::format_string<Args...> format, Args&&... args) {
    return std::string("check failed: ") + expression + ": " + std::format(format, std::forward<Args>(args)...);
}

}  // namespace ii::detail

#define II_CHECK(cond, ...)                                                                                            \
    do {                                                                                                               \
        if (!(cond)) [[unlikely]] {                                                                                    \
            ::ii::detail::fatal_error(__FILE__, __LINE__,                                                              \
                                      ::ii::detail::check_message(#cond __VA_OPT__(, ) __VA_ARGS__));                  \
        }                                                                                                              \
    } while (false)

#if defined(NDEBUG)
#define II_ASSERT(cond, ...)                                                                                           \
    do {                                                                                                               \
    } while (false)
#else
#define II_ASSERT(cond, ...) II_CHECK(cond __VA_OPT__(, ) __VA_ARGS__)
#endif

#define II_UNREACHABLE() ::ii::detail::fatal_error(__FILE__, __LINE__, "unreachable code reached")
