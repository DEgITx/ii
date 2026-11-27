// SPDX-License-Identifier: MIT
/// @file
/// Autotuning of ii-cuda: choices measured on the device (the fastest cuBLASLt algorithm of a product shape, the
/// warps per feature of a weight product, the device's real launch cost and bandwidths for the planner) are kept in
/// a cache file, so later processes reuse them instead of measuring again.
///
/// The file (JSON) belongs to one GPU and one build: it records the device, the cuBLASLt version and the ii version,
/// and a file recorded with others is ignored (and rewritten). Entries are hints: a missing or stale entry costs a
/// measurement, never a wrong result.
#pragma once

#include <ii/status.h>

#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace ii::cuda {

class TuningCache {
public:
    /// The cache of the device `identity` in file `path` (empty: in memory only).
    TuningCache(std::filesystem::path path, std::string identity);

    std::optional<std::string> find(std::string_view key) const;

    /// Adds an entry and rewrites the file (failures to write are ignored: the cache is an optimization).
    void store(const std::string& key, const std::string& value);

    const std::filesystem::path& path() const noexcept { return path_; }
    std::size_t size() const;

private:
    void save() const;

    mutable std::mutex mutex_;
    std::filesystem::path path_;
    std::string identity_;
    std::map<std::string, std::string, std::less<>> entries_;
};

/// Measured figures of a device for the planner (`backend::DeviceInfo`).
struct DeviceFigures {
    double launch_us = 0;        ///< one kernel of a recorded run (CUDA graph)
    double bandwidth_gbs = 0;    ///< device memory (a copy reads and writes)
    double link_gbs = 0;         ///< copies between pageable host memory and the device
    double link_latency_us = 0;  ///< a small copy and the wait for it

    std::string to_string() const;
    static std::optional<DeviceFigures> parse(std::string_view text);
};

/// Measures the current device (about half a second; uses 256 MB of device memory meanwhile).
Result<DeviceFigures> measure_device();

/// The cache file of a device for the option value `option` ("cuda.cache"): empty means the default cache directory
/// (%LOCALAPPDATA%\ii\cache, $XDG_CACHE_HOME/ii, ...), "none" no file, anything else the file to use.
std::filesystem::path tuning_cache_path(std::string_view option, std::string_view device_key);

/// Lowercase hexadecimal digits of bytes, and back (empty optional for malformed text).
std::string to_hex(const void* data, std::size_t size);
bool from_hex(std::string_view text, void* data, std::size_t size);

}  // namespace ii::cuda
