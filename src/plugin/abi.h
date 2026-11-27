// SPDX-License-Identifier: MIT
/// @file
/// Helpers shared by both sides of the plugin interface (ii/plugin.h).
#pragma once

#include <ii/dtype.h>
#include <ii/log.h>
#include <ii/plugin.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include <algorithm>
#include <cstring>
#include <string_view>

namespace ii::plugin::abi {

// The interface passes these enumerations as integers.
static_assert(static_cast<int>(DType::kF32) == 1 && static_cast<int>(DType::kBool) == 13 &&
              static_cast<int>(DType::kQ4_0) == 14 && static_cast<int>(DType::kQ8_K) == 25);
static_assert(static_cast<int>(DeviceKind::kCpu) == 0 && static_cast<int>(DeviceKind::kCuda) == 1 &&
              static_cast<int>(DeviceKind::kTest) == 2);
static_assert(static_cast<int>(StatusCode::kUnavailable) == 8 && static_cast<int>(StatusCode::kInternal) == 11);
static_assert(static_cast<int>(LogLevel::kError) == II_PLUGIN_LOG_ERROR);

/// Fills `error` (truncating the message) and returns `code`.
inline int32_t set_error(ii_plugin_error* error, int32_t code, std::string_view message) noexcept {
    if (error != nullptr) {
        error->code = code;
        const std::size_t size = std::min<std::size_t>(message.size(), II_PLUGIN_ERROR_SIZE - 1);
        std::memcpy(error->message, message.data(), size);
        error->message[size] = '\0';
    }
    return code;
}

/// The status of a failed call (`code != 0`).
inline Status to_status(int32_t code, const ii_plugin_error& error) {
    StatusCode status = StatusCode::kInternal;
    if (code > 0 && code <= static_cast<int32_t>(StatusCode::kInternal)) {
        status = static_cast<StatusCode>(code);
    }
    const char* end = std::find(error.message, error.message + II_PLUGIN_ERROR_SIZE, '\0');
    std::string_view message(error.message, static_cast<std::size_t>(end - error.message));
    if (message.empty()) {
        message = "the plugin reported an error without a message";
    }
    return Status(status, std::string(message));
}

}  // namespace ii::plugin::abi
