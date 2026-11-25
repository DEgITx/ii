// SPDX-License-Identifier: MIT
// System information and topology for macOS and iOS (sysctl based).
#include "core/platform/platform.h"

#include <TargetConditionals.h>
#include <mach/mach.h>
#include <mach/vm_page_size.h>
#include <sys/sysctl.h>

#include <cstdint>
#include <string>

namespace ii::platform {

namespace {

std::string sysctl_string(const char* name) {
    std::size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
        return {};
    }
    const auto end = value.find('\0');
    value.resize(end == std::string::npos ? size : end);
    return value;
}

/// Reads an integer sysctl of 4 or 8 bytes; returns 0 if it does not exist.
std::uint64_t sysctl_uint(const char* name) {
    std::uint64_t value64 = 0;
    std::size_t size = sizeof(value64);
    if (sysctlbyname(name, &value64, &size, nullptr, 0) != 0) {
        return 0;
    }
    if (size == sizeof(std::uint32_t)) {
        std::uint32_t value32 = 0;
        size = sizeof(value32);
        if (sysctlbyname(name, &value32, &size, nullptr, 0) != 0) {
            return 0;
        }
        return value32;
    }
    return value64;
}

}  // namespace

Result<std::filesystem::path> default_cache_dir() {
    if (auto dir = get_env("II_CACHE_DIR"); dir && !dir->empty()) {
        return std::filesystem::path(*dir);
    }
    if (auto home = get_env("HOME"); home && !home->empty()) {
        return std::filesystem::path(*home) / "Library" / "Caches" / "ii";
    }
    return errors::unavailable("cannot determine the cache directory: neither II_CACHE_DIR nor HOME is set");
}

Status set_current_thread_affinity(std::span<const int> /*cpus*/) {
    return errors::unsupported("thread affinity is not supported on Apple platforms (use QoS classes instead)");
}

int current_cpu() noexcept {
    return -1;
}

OsInfo query_os_info() {
    OsInfo info;
    const std::string product_version = sysctl_string("kern.osproductversion");
#if TARGET_OS_IPHONE
    info.name = "iOS";
#else
    info.name = "macOS";
#endif
    if (!product_version.empty()) {
        info.name += " " + product_version;
    }
    info.version = sysctl_string("kern.osversion");
    return info;
}

MemoryInfo query_memory_info() {
    MemoryInfo info;
    info.total_bytes = sysctl_uint("hw.memsize");

    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count) ==
        KERN_SUCCESS) {
        const std::uint64_t page_size = vm_kernel_page_size;
        info.available_bytes =
            (static_cast<std::uint64_t>(stats.free_count) + stats.inactive_count + stats.purgeable_count) * page_size;
    }
    return info;
}

void detect_cpu_topology(CpuInfo& info) {
    info.logical_cpus = static_cast<int>(sysctl_uint("hw.logicalcpu"));
    info.physical_cores = static_cast<int>(sysctl_uint("hw.physicalcpu"));

    // Apple Silicon reports performance levels: perflevel0 is the fastest.
    const std::uint64_t levels = sysctl_uint("hw.nperflevels");
    if (levels >= 2) {
        info.performance_cores = static_cast<int>(sysctl_uint("hw.perflevel0.physicalcpu"));
        info.efficiency_cores = info.physical_cores - info.performance_cores;
        info.caches.l1d = sysctl_uint("hw.perflevel0.l1dcachesize");
        info.caches.l2 = sysctl_uint("hw.perflevel0.l2cachesize");
    } else {
        info.performance_cores = info.physical_cores;
        info.caches.l1d = sysctl_uint("hw.l1dcachesize");
        info.caches.l2 = sysctl_uint("hw.l2cachesize");
    }
    info.caches.l3 = sysctl_uint("hw.l3cachesize");
    // The core-to-CPU mapping is not exposed on Apple platforms; info.cores stays empty.
}

}  // namespace ii::platform
