// SPDX-License-Identifier: MIT
#include <ii/system_info.h>

#include "core/isa.h"
#include "core/platform/platform.h"

#include <algorithm>
#include <thread>

namespace ii {

std::string_view to_string(CpuArch arch) noexcept {
    switch (arch) {
    case CpuArch::kX86_64:
        return "x86_64";
    case CpuArch::kArm64:
        return "arm64";
    case CpuArch::kUnknown:
        break;
    }
    return "unknown";
}

std::string_view to_string(CpuFeature feature) noexcept {
    switch (feature) {
    case CpuFeature::kSse2:
        return "sse2";
    case CpuFeature::kSse3:
        return "sse3";
    case CpuFeature::kSsse3:
        return "ssse3";
    case CpuFeature::kSse41:
        return "sse4.1";
    case CpuFeature::kSse42:
        return "sse4.2";
    case CpuFeature::kAvx:
        return "avx";
    case CpuFeature::kAvx2:
        return "avx2";
    case CpuFeature::kFma:
        return "fma";
    case CpuFeature::kF16c:
        return "f16c";
    case CpuFeature::kAvxVnni:
        return "avx_vnni";
    case CpuFeature::kAvx512F:
        return "avx512f";
    case CpuFeature::kAvx512Bw:
        return "avx512bw";
    case CpuFeature::kAvx512Vl:
        return "avx512vl";
    case CpuFeature::kAvx512Dq:
        return "avx512dq";
    case CpuFeature::kAvx512Cd:
        return "avx512cd";
    case CpuFeature::kAvx512Vnni:
        return "avx512_vnni";
    case CpuFeature::kAvx512Bf16:
        return "avx512_bf16";
    case CpuFeature::kAvx512Fp16:
        return "avx512_fp16";
    case CpuFeature::kAmxTile:
        return "amx_tile";
    case CpuFeature::kAmxInt8:
        return "amx_int8";
    case CpuFeature::kAmxBf16:
        return "amx_bf16";
    case CpuFeature::kNeon:
        return "neon";
    case CpuFeature::kArmFp16:
        return "fp16";
    case CpuFeature::kArmDotProd:
        return "dotprod";
    case CpuFeature::kArmI8mm:
        return "i8mm";
    case CpuFeature::kArmBf16:
        return "bf16";
    case CpuFeature::kSve:
        return "sve";
    case CpuFeature::kSve2:
        return "sve2";
    case CpuFeature::kSme:
        return "sme";
    case CpuFeature::kCount:
        break;
    }
    return "unknown";
}

namespace {

CpuInfo detect_cpu() {
    CpuInfo info;
    platform::detect_cpu_features(info);
    platform::detect_cpu_topology(info);

    // Fallbacks when the OS does not expose the topology.
    if (info.logical_cpus <= 0) {
        info.logical_cpus = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    if (info.physical_cores <= 0) {
        info.physical_cores = info.logical_cpus;
    }
    if (info.performance_cores <= 0) {
        info.performance_cores = info.physical_cores - info.efficiency_cores;
    }
    return info;
}

}  // namespace

const CpuInfo& cpu_info() noexcept {
    // Hardware facts are immutable for the lifetime of the process, so caching them is not "global state".
    static const CpuInfo info = detect_cpu();
    return info;
}

OsInfo os_info() {
    return platform::query_os_info();
}

MemoryInfo memory_info() {
    return platform::query_memory_info();
}

std::vector<IsaSupport> isa_support() {
    std::vector<IsaSupport> result;
    for (const Isa isa : target_isas()) {
        result.push_back({std::string(isa_name(isa)), isa_compiled(isa), isa_supported(isa, cpu_info())});
    }
    return result;
}

Result<std::string> selected_isa() {
    II_ASSIGN_OR_RETURN(const Isa isa, select_isa_from_environment(cpu_info()));
    return std::string(isa_name(isa));
}

}  // namespace ii
