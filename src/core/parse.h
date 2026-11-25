// SPDX-License-Identifier: MIT
/// @file
/// Locale-independent number parsing.
#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace ii {

/// Parses a complete decimal integer ("-12"). Returns nullopt on syntax errors or overflow.
template <class T = std::int64_t>
std::optional<T> parse_integer(std::string_view text) noexcept {
    if (!text.empty() && text.front() == '+') {
        text.remove_prefix(1);
    }
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size() || text.empty()) {
        return std::nullopt;
    }
    return value;
}

/// Parses a complete floating-point number ("1.5", "-2e-3", "inf", "-inf", "nan"). Returns nullopt on syntax errors.
inline std::optional<double> parse_double(std::string_view text) {
    if (text.empty()) {
        return std::nullopt;
    }
    if (text.front() == '+') {
        text.remove_prefix(1);
    }
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    {
        double parsed = 0;
        const auto [last, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
        const bool complete = last == text.data() + text.size();
        if (error == std::errc() && complete) {
            return parsed;
        }
        // Overflow / underflow fall through to strtod, which saturates to inf / 0.
        if (error != std::errc::result_out_of_range || !complete) {
            return std::nullopt;
        }
    }
#endif
    // strtod honours the C locale's decimal point; reject anything that is not plain ASCII number syntax first.
    for (const char c : text) {
        const bool ok = (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+' ||
                        c == 'i' || c == 'n' || c == 'f' || c == 'a' || c == 'I' || c == 'N' || c == 'F' || c == 'A';
        if (!ok) {
            return std::nullopt;
        }
    }
    const std::string copy(text);
    char* end = nullptr;
    const double value = std::strtod(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size()) {
        return std::nullopt;
    }
    return value;
}

}  // namespace ii
