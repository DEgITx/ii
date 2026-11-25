// SPDX-License-Identifier: MIT
// System information, topology and affinity for Linux and Android (sysfs / procfs based).
#include "core/platform/platform.h"

#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ii::platform {

namespace {

std::optional<std::string> read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return std::nullopt;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) {
        text.pop_back();
    }
    return text;
}

std::optional<long long> parse_integer(std::string_view text) {
    long long value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc()) {
        return std::nullopt;
    }
    return value;
}

/// Parses the kernel CPU list format, e.g. "0-3,8,10-11".
std::vector<int> parse_cpu_list(std::string_view text) {
    std::vector<int> cpus;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const std::string_view item = text.substr(0, comma);
        const auto dash = item.find('-');
        const auto first = parse_integer(item.substr(0, dash));
        const auto last = dash == std::string_view::npos ? first : parse_integer(item.substr(dash + 1));
        if (first && last && *first <= *last) {
            for (long long cpu = *first; cpu <= *last; ++cpu) {
                cpus.push_back(static_cast<int>(cpu));
            }
        }
        if (comma == std::string_view::npos) {
            break;
        }
        text.remove_prefix(comma + 1);
    }
    return cpus;
}

/// Parses cache sizes such as "48K", "3072K", "30M".
std::uint64_t parse_size(std::string_view text) {
    if (text.empty()) {
        return 0;
    }
    std::uint64_t multiplier = 1;
    if (text.back() == 'K') {
        multiplier = 1024;
        text.remove_suffix(1);
    } else if (text.back() == 'M') {
        multiplier = 1024 * 1024;
        text.remove_suffix(1);
    }
    const auto value = parse_integer(text);
    return value ? static_cast<std::uint64_t>(*value) * multiplier : 0;
}

std::string cpu_dir(int cpu) {
    return "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
}

}  // namespace

Result<std::filesystem::path> default_cache_dir() {
    if (auto dir = get_env("II_CACHE_DIR"); dir && !dir->empty()) {
        return std::filesystem::path(*dir);
    }
#if defined(__ANDROID__)
    return errors::unavailable("Android has no default cache directory; set II_CACHE_DIR or configure the context");
#else
    if (auto xdg = get_env("XDG_CACHE_HOME"); xdg && !xdg->empty()) {
        return std::filesystem::path(*xdg) / "ii";
    }
    if (auto home = get_env("HOME"); home && !home->empty()) {
        return std::filesystem::path(*home) / ".cache" / "ii";
    }
    return errors::unavailable("cannot determine the cache directory: neither II_CACHE_DIR nor HOME is set");
#endif
}

Status set_current_thread_affinity(std::span<const int> cpus) {
    if (cpus.empty()) {
        return errors::invalid_argument("empty CPU set");
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int cpu : cpus) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            return errors::invalid_argument("CPU index {} is out of range", cpu);
        }
        CPU_SET(cpu, &set);
    }
    // pid 0 = the calling thread.
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        return errors::invalid_argument("sched_setaffinity failed");
    }
    return {};
}

int current_cpu() noexcept {
    return sched_getcpu();
}

OsInfo query_os_info() {
    OsInfo info;
    utsname uts{};
    if (uname(&uts) == 0) {
        info.version = uts.release;
    }
#if defined(__ANDROID__)
    char release[PROP_VALUE_MAX] = {};
    info.name = __system_property_get("ro.build.version.release", release) > 0 ? std::string("Android ") + release
                                                                               : std::string("Android");
#else
    info.name = "Linux";
    std::ifstream os_release("/etc/os-release");
    std::string line;
    while (std::getline(os_release, line)) {
        if (line.starts_with("PRETTY_NAME=")) {
            std::string value = line.substr(12);
            if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'')) {
                value = value.substr(1, value.size() - 2);
            }
            if (!value.empty()) {
                info.name = value;
            }
            break;
        }
    }
#endif
    return info;
}

MemoryInfo query_memory_info() {
    MemoryInfo info;
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    while (meminfo >> key >> value) {
        std::getline(meminfo, unit);
        if (key == "MemTotal:") {
            info.total_bytes = value * 1024;
        } else if (key == "MemAvailable:") {
            info.available_bytes = value * 1024;
        }
    }
    if (info.total_bytes == 0) {
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long page_size = sysconf(_SC_PAGESIZE);
        if (pages > 0 && page_size > 0) {
            info.total_bytes = static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
        }
    }
    return info;
}

void detect_cpu_topology(CpuInfo& info) {
    std::vector<int> online;
    if (auto text = read_file("/sys/devices/system/cpu/online")) {
        online = parse_cpu_list(*text);
    }
    // Only CPUs this process may run on (containers and cgroups often restrict the set).
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        std::erase_if(online, [&](int cpu) { return cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed); });
    }
    if (online.empty()) {
        return;  // sysfs unavailable (restricted sandbox); the caller falls back to hardware_concurrency()
    }

    // Efficiency classes. Intel hybrid CPUs expose separate PMUs for P-cores ("cpu_core") and E-cores ("cpu_atom");
    // Arm big.LITTLE systems expose per-CPU capacity.
    std::map<int, int> cpu_class;
    if (auto core_cpus = read_file("/sys/devices/cpu_core/cpus")) {
        for (const int cpu : parse_cpu_list(*core_cpus)) {
            cpu_class[cpu] = 1;
        }
    } else {
        std::map<int, long long> capacity;
        std::set<long long> capacities;
        for (const int cpu : online) {
            if (auto text = read_file(cpu_dir(cpu) + "/cpu_capacity")) {
                if (auto value = parse_integer(*text)) {
                    capacity[cpu] = *value;
                    capacities.insert(*value);
                }
            }
        }
        if (capacities.size() > 1) {
            const std::vector<long long> ranks(capacities.begin(), capacities.end());
            for (const auto& [cpu, value] : capacity) {
                cpu_class[cpu] = static_cast<int>(std::lower_bound(ranks.begin(), ranks.end(), value) - ranks.begin());
            }
        }
    }

    // Group logical CPUs into physical cores by their sibling list.
    std::map<std::string, CpuCore> cores;
    std::vector<std::string> order;
    for (const int cpu : online) {
        auto siblings = read_file(cpu_dir(cpu) + "/topology/core_cpus_list");
        if (!siblings) {
            siblings = read_file(cpu_dir(cpu) + "/topology/thread_siblings_list");
        }
        const std::string key = siblings ? *siblings : std::to_string(cpu);
        auto [it, inserted] = cores.try_emplace(key);
        if (inserted) {
            order.push_back(key);
            const auto cls = cpu_class.find(cpu);
            it->second.efficiency_class = cls != cpu_class.end() ? cls->second : 0;
        }
        it->second.logical_cpus.push_back(cpu);
    }
    for (const std::string& key : order) {
        info.cores.push_back(std::move(cores[key]));
    }

    int max_class = 0;
    int min_class = info.cores.front().efficiency_class;
    for (const CpuCore& core : info.cores) {
        info.logical_cpus += static_cast<int>(core.logical_cpus.size());
        max_class = std::max(max_class, core.efficiency_class);
        min_class = std::min(min_class, core.efficiency_class);
    }
    info.physical_cores = static_cast<int>(info.cores.size());
    info.performance_cores = static_cast<int>(std::count_if(
        info.cores.begin(), info.cores.end(), [&](const CpuCore& core) { return core.efficiency_class == max_class; }));
    info.efficiency_cores = (max_class != min_class) ? info.physical_cores - info.performance_cores : 0;
    if (info.efficiency_cores == 0) {
        info.performance_cores = info.physical_cores;
    }

    // Caches of a performance core.
    const auto fast_core = std::find_if(info.cores.begin(), info.cores.end(),
                                        [&](const CpuCore& core) { return core.efficiency_class == max_class; });
    const int cpu = fast_core->logical_cpus.front();
    for (int index = 0; index < 16; ++index) {
        const std::string dir = cpu_dir(cpu) + "/cache/index" + std::to_string(index);
        const auto level = read_file(dir + "/level");
        if (!level) {
            break;
        }
        const std::string type = read_file(dir + "/type").value_or("");
        const std::uint64_t size = parse_size(read_file(dir + "/size").value_or(""));
        if (*level == "1" && (type == "Data" || type == "Unified")) {
            info.caches.l1d = size;
        } else if (*level == "2") {
            info.caches.l2 = size;
        } else if (*level == "3") {
            info.caches.l3 = size;
        }
    }
}

}  // namespace ii::platform
