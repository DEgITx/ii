// SPDX-License-Identifier: MIT
#include "core/platform/dynlib.h"
#include "core/platform/platform.h"
#include "core/platform/windows/win32_util.h"

#include <windows.h>

namespace ii::platform {

Result<DynLib> DynLib::open(const std::filesystem::path& path) {
    // Never show "missing DLL" dialogs: a plugin with an absent runtime must fail quietly.
    DWORD previous_mode = 0;
    SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &previous_mode);
    // LOAD_WITH_ALTERED_SEARCH_PATH: dependencies are searched in the library's own directory first, then in the
    // standard locations including PATH (where SDKs such as CUDA and TensorRT put their DLLs).
    const std::filesystem::path absolute = std::filesystem::absolute(path);
    HMODULE module = LoadLibraryExW(absolute.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    const DWORD error = GetLastError();
    SetThreadErrorMode(previous_mode, nullptr);

    if (module == nullptr) {
        const StatusCode code = (error == ERROR_MOD_NOT_FOUND && !std::filesystem::exists(absolute))
                                    ? StatusCode::kNotFound
                                    : StatusCode::kUnavailable;
        return Status(code, "cannot load '" + path_to_utf8(path) + "': " + win32::error_message(error));
    }
    return DynLib(reinterpret_cast<void*>(module));
}

Result<void*> DynLib::symbol(const char* name) const {
    if (handle_ == nullptr) {
        return errors::invalid_argument("library is not open");
    }
    FARPROC address = GetProcAddress(static_cast<HMODULE>(handle_), name);
    if (address == nullptr) {
        return errors::not_found("symbol '{}' not found", name);
    }
    return reinterpret_cast<void*>(address);
}

void DynLib::close() noexcept {
    if (handle_ != nullptr) {
        FreeLibrary(static_cast<HMODULE>(handle_));
        handle_ = nullptr;
    }
}

}  // namespace ii::platform
