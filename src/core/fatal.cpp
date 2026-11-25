// SPDX-License-Identifier: MIT
#include <ii/status.h>

#include "core/assert.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ii::detail {

void fatal_error(const char* file, int line, std::string_view message) noexcept {
    std::fprintf(stderr, "ii: fatal error at %s:%d: %.*s\n", file, line, static_cast<int>(message.size()),
                 message.data());
    std::fflush(stderr);
    std::abort();
}

void result_value_failed(const Status& status) noexcept {
    const std::string text = "Result::value() called on an error: " + status.to_string();
    std::fprintf(stderr, "ii: fatal error: %s\n", text.c_str());
    std::fflush(stderr);
    std::abort();
}

}  // namespace ii::detail
