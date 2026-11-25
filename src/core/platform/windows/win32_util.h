// SPDX-License-Identifier: MIT
/// @file
/// Helpers shared by the Windows platform implementation.
#pragma once

#include <windows.h>

#include <string>
#include <string_view>

namespace ii::platform::win32 {

std::wstring widen(std::string_view utf8);
std::string narrow(std::wstring_view wide);

/// Formats a Win32 error code (GetLastError()) as text.
std::string error_message(DWORD error);

/// Reads a REG_SZ value from HKEY_LOCAL_MACHINE; empty on failure.
std::string read_registry_string(const wchar_t* subkey, const wchar_t* value);

/// Casts a FARPROC to a concrete function pointer type without -Wcast-function-type warnings.
template <class Fn>
Fn function_cast(FARPROC proc) noexcept {
    return reinterpret_cast<Fn>(reinterpret_cast<void (*)()>(proc));
}

}  // namespace ii::platform::win32
