// SPDX-License-Identifier: MIT
#include "core/platform/dynlib.h"
#include "core/platform/platform.h"

#include <dlfcn.h>

#include <string>

namespace ii::platform {

Result<DynLib> DynLib::open(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return errors::not_found("cannot load '{}': file not found", path_to_utf8(path));
    }
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        return errors::unavailable("cannot load '{}': {}", path_to_utf8(path), error ? error : "unknown error");
    }
    return DynLib(handle);
}

Result<void*> DynLib::symbol(const char* name) const {
    if (handle_ == nullptr) {
        return errors::invalid_argument("library is not open");
    }
    dlerror();  // clear any previous error
    void* address = dlsym(handle_, name);
    if (address == nullptr) {
        return errors::not_found("symbol '{}' not found", name);
    }
    return address;
}

void DynLib::close() noexcept {
    if (handle_ != nullptr) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

}  // namespace ii::platform
