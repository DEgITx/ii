// SPDX-License-Identifier: MIT
#include "core/platform/platform.h"
#include "core/platform/windows/win32_util.h"

#include <malloc.h>
#include <windows.h>

#include <format>
#include <string>
#include <vector>

namespace ii::platform {

// --- win32 helpers -------------------------------------------------------------------------------------------------

namespace win32 {

std::wstring widen(std::string_view utf8) {
    if (utf8.empty()) {
        return {};
    }
    const int length = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), result.data(), length);
    return result;
}

std::string narrow(std::wstring_view wide) {
    if (wide.empty()) {
        return {};
    }
    const int length =
        WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<std::size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), result.data(), length, nullptr,
                        nullptr);
    return result;
}

std::string error_message(DWORD error) {
    wchar_t* buffer = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, error,
        MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::string message;
    if (length != 0 && buffer != nullptr) {
        message = narrow(std::wstring_view(buffer, length));
        while (!message.empty() &&
               (message.back() == '\n' || message.back() == '\r' || message.back() == ' ' || message.back() == '.')) {
            message.pop_back();
        }
    }
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    return std::format("{} (error {})", message.empty() ? "unknown error" : message, error);
}

std::string read_registry_string(const wchar_t* subkey, const wchar_t* value) {
    DWORD size = 0;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, subkey, value, RRF_RT_REG_SZ, nullptr, nullptr, &size) != ERROR_SUCCESS ||
        size == 0) {
        return {};
    }
    std::wstring buffer(size / sizeof(wchar_t), L'\0');
    if (RegGetValueW(HKEY_LOCAL_MACHINE, subkey, value, RRF_RT_REG_SZ, nullptr, buffer.data(), &size) !=
        ERROR_SUCCESS) {
        return {};
    }
    buffer.resize(wcsnlen(buffer.c_str(), buffer.size()));
    return narrow(buffer);
}

}  // namespace win32

// --- Strings and paths -----------------------------------------------------------------------------------------------

std::filesystem::path path_from_utf8(std::string_view utf8) {
    return std::filesystem::path(win32::widen(utf8));
}

std::string path_to_utf8(const std::filesystem::path& path) {
    return win32::narrow(path.native());
}

// --- Environment -----------------------------------------------------------------------------------------------------

std::optional<std::string> get_env(std::string_view name) {
    const std::wstring wide_name = win32::widen(name);
    SetLastError(ERROR_SUCCESS);
    const DWORD size = GetEnvironmentVariableW(wide_name.c_str(), nullptr, 0);
    if (size == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) {
            return std::nullopt;
        }
        return std::string();
    }
    std::wstring value(size, L'\0');
    const DWORD written = GetEnvironmentVariableW(wide_name.c_str(), value.data(), size);
    value.resize(written);
    return win32::narrow(value);
}

Result<std::filesystem::path> default_cache_dir() {
    if (auto dir = get_env("II_CACHE_DIR"); dir && !dir->empty()) {
        return path_from_utf8(*dir);
    }
    if (auto local = get_env("LOCALAPPDATA"); local && !local->empty()) {
        return path_from_utf8(*local) / "ii" / "cache";
    }
    return errors::unavailable("cannot determine the cache directory: neither II_CACHE_DIR nor LOCALAPPDATA is set");
}

namespace {
/// Any object inside the ii binary; its address identifies the module.
const char kLibraryAnchor = 0;
}  // namespace

Result<std::filesystem::path> this_library_path() {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&kLibraryAnchor), &module)) {
        return errors::io_error("GetModuleHandleExW failed: {}", win32::error_message(GetLastError()));
    }
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            return errors::io_error("GetModuleFileNameW failed: {}", win32::error_message(GetLastError()));
        }
        if (length < buffer.size()) {
            buffer.resize(length);
            return std::filesystem::path(buffer);
        }
        buffer.resize(buffer.size() * 2);
    }
}

// --- Memory ----------------------------------------------------------------------------------------------------------

void* aligned_allocate(std::size_t size, std::size_t alignment) noexcept {
    return _aligned_malloc(size, alignment);
}

void aligned_free(void* ptr) noexcept {
    _aligned_free(ptr);
}

// --- Threads ---------------------------------------------------------------------------------------------------------

void set_current_thread_name(std::string_view name) noexcept {
    // SetThreadDescription exists since Windows 10 1607; look it up dynamically.
    using SetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const auto set_description = win32::function_cast<SetThreadDescriptionFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription"));
    if (set_description != nullptr) {
        const std::wstring wide = win32::widen(name);
        set_description(GetCurrentThread(), wide.c_str());
    }
}

Status set_current_thread_affinity(std::span<const int> cpus) {
    if (cpus.empty()) {
        return errors::invalid_argument("empty CPU set");
    }
    const int group = cpus.front() / 64;
    KAFFINITY mask = 0;
    for (const int cpu : cpus) {
        if (cpu < 0 || cpu / 64 != group) {
            return errors::unsupported("thread affinity across processor groups is not supported (CPU {})", cpu);
        }
        mask |= KAFFINITY{1} << (cpu % 64);
    }
    GROUP_AFFINITY affinity{};
    affinity.Mask = mask;
    affinity.Group = static_cast<WORD>(group);
    if (!SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr)) {
        return errors::invalid_argument("SetThreadGroupAffinity failed: {}", win32::error_message(GetLastError()));
    }
    return {};
}

int current_cpu() noexcept {
    PROCESSOR_NUMBER number{};
    GetCurrentProcessorNumberEx(&number);
    return static_cast<int>(number.Group) * 64 + static_cast<int>(number.Number);
}

// --- System information ----------------------------------------------------------------------------------------------

OsInfo query_os_info() {
    // GetVersionEx lies for applications without a compatibility manifest; RtlGetVersion does not.
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    OSVERSIONINFOW version{};
    version.dwOSVersionInfoSize = sizeof(version);
    const auto rtl_get_version =
        win32::function_cast<RtlGetVersionFn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion"));
    if (rtl_get_version == nullptr || rtl_get_version(&version) != 0) {
        return {"Windows", "unknown"};
    }

    OsInfo info;
    if (version.dwMajorVersion == 10 && version.dwBuildNumber >= 22000) {
        info.name = "Windows 11";
    } else if (version.dwMajorVersion == 10) {
        info.name = "Windows 10";
    } else {
        info.name = std::format("Windows {}.{}", version.dwMajorVersion, version.dwMinorVersion);
    }
    const std::string display_version =
        win32::read_registry_string(L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", L"DisplayVersion");
    if (!display_version.empty()) {
        info.name += " " + display_version;
    }
    info.version = std::format("{}.{}.{}", version.dwMajorVersion, version.dwMinorVersion, version.dwBuildNumber);
    return info;
}

MemoryInfo query_memory_info() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) {
        return {};
    }
    return {status.ullTotalPhys, status.ullAvailPhys};
}

}  // namespace ii::platform
