// SPDX-License-Identifier: MIT
#include <ii/status.h>

namespace ii {

std::string_view to_string(StatusCode code) noexcept {
    switch (code) {
    case StatusCode::kOk:
        return "OK";
    case StatusCode::kCancelled:
        return "Cancelled";
    case StatusCode::kInvalidArgument:
        return "InvalidArgument";
    case StatusCode::kNotFound:
        return "NotFound";
    case StatusCode::kAlreadyExists:
        return "AlreadyExists";
    case StatusCode::kOutOfRange:
        return "OutOfRange";
    case StatusCode::kOutOfMemory:
        return "OutOfMemory";
    case StatusCode::kUnsupported:
        return "Unsupported";
    case StatusCode::kUnavailable:
        return "Unavailable";
    case StatusCode::kIoError:
        return "IoError";
    case StatusCode::kDataCorrupted:
        return "DataCorrupted";
    case StatusCode::kInternal:
        return "Internal";
    }
    return "Unknown";
}

}  // namespace ii
