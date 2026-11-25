// SPDX-License-Identifier: MIT
#include "core/platform/platform.h"

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ii::platform {

namespace {

struct CacheEntry {
    int level;
    PROCESSOR_CACHE_TYPE type;
    std::uint64_t size;
    WORD group;
    KAFFINITY mask;
};

std::vector<std::uint64_t> query_processor_information(LOGICAL_PROCESSOR_RELATIONSHIP relationship) {
    DWORD length = 0;
    GetLogicalProcessorInformationEx(relationship, nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) {
        return {};
    }
    // std::uint64_t storage keeps the variable-length records suitably aligned.
    std::vector<std::uint64_t> buffer((length + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t));
    if (!GetLogicalProcessorInformationEx(
            relationship, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()), &length)) {
        return {};
    }
    buffer.resize((length + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t));
    return buffer;
}

template <class Fn>
void for_each_record(const std::vector<std::uint64_t>& buffer, Fn&& fn) {
    const auto* begin = reinterpret_cast<const std::byte*>(buffer.data());
    const auto* end = begin + buffer.size() * sizeof(std::uint64_t);
    for (const std::byte* p = begin; p < end;) {
        const auto* record = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
        if (record->Size == 0) {
            break;
        }
        fn(*record);
        p += record->Size;
    }
}

}  // namespace

void detect_cpu_topology(CpuInfo& info) {
    const std::vector<std::uint64_t> buffer = query_processor_information(RelationAll);

    std::vector<CacheEntry> caches;
    for_each_record(buffer, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& record) {
        if (record.Relationship == RelationProcessorCore) {
            CpuCore core;
            core.efficiency_class = record.Processor.EfficiencyClass;
            for (WORD g = 0; g < record.Processor.GroupCount; ++g) {
                const GROUP_AFFINITY& affinity = record.Processor.GroupMask[g];
                for (int bit = 0; bit < 64; ++bit) {
                    if ((affinity.Mask >> bit) & 1u) {
                        core.logical_cpus.push_back(static_cast<int>(affinity.Group) * 64 + bit);
                    }
                }
            }
            info.cores.push_back(std::move(core));
        } else if (record.Relationship == RelationCache) {
            const CACHE_RELATIONSHIP& cache = record.Cache;
            caches.push_back({cache.Level, cache.Type, cache.CacheSize, cache.GroupMask.Group, cache.GroupMask.Mask});
        }
    });

    if (info.cores.empty()) {
        return;  // the caller falls back to std::thread::hardware_concurrency()
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

    // Report the caches seen by a performance core.
    const auto fast_core = std::find_if(info.cores.begin(), info.cores.end(),
                                        [&](const CpuCore& core) { return core.efficiency_class == max_class; });
    if (fast_core == info.cores.end() || fast_core->logical_cpus.empty()) {
        return;
    }
    const int cpu = fast_core->logical_cpus.front();
    for (const CacheEntry& cache : caches) {
        if (cache.group != cpu / 64 || ((cache.mask >> (cpu % 64)) & 1u) == 0) {
            continue;
        }
        if (cache.level == 1 && (cache.type == CacheData || cache.type == CacheUnified)) {
            info.caches.l1d = cache.size;
        } else if (cache.level == 2) {
            info.caches.l2 = cache.size;
        } else if (cache.level == 3) {
            info.caches.l3 = cache.size;
        }
    }
}

}  // namespace ii::platform
