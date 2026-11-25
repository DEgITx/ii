// SPDX-License-Identifier: MIT
/// @file
/// The operating-system abstraction layer.
///
/// All OS-specific code of the library lives under src/core/platform/. Code elsewhere must use these functions
/// instead of OS APIs or platform #ifdefs.
#pragma once

#include <ii/status.h>
#include <ii/system_info.h>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ii::platform {

// --- Strings and paths -----------------------------------------------------------------------------------------------

/// Converts a UTF-8 string to a path (on Windows a plain std::string would be interpreted in the ANSI code page).
std::filesystem::path path_from_utf8(std::string_view utf8);

/// Converts a path to UTF-8.
std::string path_to_utf8(const std::filesystem::path& path);

// --- Environment -----------------------------------------------------------------------------------------------------

/// Returns the value of an environment variable (UTF-8), or nullopt if it is not set.
std::optional<std::string> get_env(std::string_view name);

/// Returns the directory for ii's persistent caches: II_CACHE_DIR if set, otherwise the platform convention
/// (%LOCALAPPDATA%\ii\cache, $XDG_CACHE_HOME/ii or ~/.cache/ii, ~/Library/Caches/ii). The directory may not exist.
Result<std::filesystem::path> default_cache_dir();

/// Returns the path of the binary that contains ii (the shared library, or the executable for static builds).
Result<std::filesystem::path> this_library_path();

// --- Memory ----------------------------------------------------------------------------------------------------------

/// Allocates `size` bytes aligned to `alignment` (a power of two >= alignof(std::max_align_t)). nullptr on failure.
void* aligned_allocate(std::size_t size, std::size_t alignment) noexcept;
void aligned_free(void* ptr) noexcept;

// --- Threads ---------------------------------------------------------------------------------------------------------

/// Sets the name of the calling thread as shown by debuggers and profilers. Best effort.
void set_current_thread_name(std::string_view name) noexcept;

/// Restricts the calling thread to the given logical CPUs (indices as in CpuCore::logical_cpus).
/// Returns kUnsupported on platforms without thread affinity (macOS, iOS).
Status set_current_thread_affinity(std::span<const int> cpus);

/// Index of the logical CPU the calling thread is running on, or -1 if unknown.
int current_cpu() noexcept;

// --- System information ----------------------------------------------------------------------------------------------

OsInfo query_os_info();
MemoryInfo query_memory_info();

/// Fills CPU vendor, brand and features (architecture-specific detection).
void detect_cpu_features(CpuInfo& info);

/// Fills core counts, the core topology and cache sizes (OS-specific detection).
void detect_cpu_topology(CpuInfo& info);

}  // namespace ii::platform
