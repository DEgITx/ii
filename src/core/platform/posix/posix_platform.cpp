// SPDX-License-Identifier: MIT
// Platform functions shared by POSIX systems (Linux, Android, macOS, iOS).
#include "core/platform/platform.h"

#include <dlfcn.h>
#include <pthread.h>

#include <cstdlib>
#include <string>

namespace ii::platform {

std::filesystem::path path_from_utf8(std::string_view utf8) {
    return std::filesystem::path(std::string(utf8));
}

std::string path_to_utf8(const std::filesystem::path& path) {
    return path.native();
}

std::optional<std::string> get_env(std::string_view name) {
    const char* value = std::getenv(std::string(name).c_str());
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string(value);
}

namespace {
/// Any object inside the ii binary; its address identifies the module.
const char kLibraryAnchor = 0;
}  // namespace

Result<std::filesystem::path> this_library_path() {
    Dl_info dl_info{};
    if (dladdr(&kLibraryAnchor, &dl_info) == 0 || dl_info.dli_fname == nullptr) {
        return errors::unavailable("dladdr failed to locate the ii binary");
    }
    return std::filesystem::path(dl_info.dli_fname);
}

void* aligned_allocate(std::size_t size, std::size_t alignment) noexcept {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
}

void aligned_free(void* ptr) noexcept {
    std::free(ptr);
}

void set_current_thread_name(std::string_view name) noexcept {
#if defined(__APPLE__)
    pthread_setname_np(std::string(name).c_str());
#else
    // Linux limits thread names to 15 characters plus the terminator.
    pthread_setname_np(pthread_self(), std::string(name.substr(0, 15)).c_str());
#endif
}

}  // namespace ii::platform
