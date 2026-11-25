// SPDX-License-Identifier: MIT
#pragma once

#include <ii/status.h>

#include <filesystem>
#include <utility>

namespace ii::platform {

/// A dynamically loaded shared library (used for backend plugins).
class DynLib {
public:
    /// Loads the library at `path`. Its dependencies are searched next to it first. Never shows OS error dialogs.
    static Result<DynLib> open(const std::filesystem::path& path);

    DynLib() noexcept = default;
    DynLib(DynLib&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    DynLib& operator=(DynLib&& other) noexcept {
        if (this != &other) {
            close();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    DynLib(const DynLib&) = delete;
    DynLib& operator=(const DynLib&) = delete;
    ~DynLib() { close(); }

    bool is_open() const noexcept { return handle_ != nullptr; }

    /// Looks up an exported symbol.
    Result<void*> symbol(const char* name) const;

    /// Looks up an exported function and casts it to `Fn` (a function pointer type).
    template <class Fn>
    Result<Fn> function(const char* name) const {
        II_ASSIGN_OR_RETURN(void* address, symbol(name));
        return reinterpret_cast<Fn>(address);
    }

private:
    explicit DynLib(void* handle) noexcept : handle_(handle) {}
    void close() noexcept;

    void* handle_ = nullptr;
};

}  // namespace ii::platform
