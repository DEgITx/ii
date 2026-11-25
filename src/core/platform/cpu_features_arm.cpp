// SPDX-License-Identifier: MIT
// CPU feature detection for Arm64. NEON (Advanced SIMD) is part of the Armv8-A baseline; optional extensions are
// queried from the OS, since user space cannot read the ID registers portably.
#include "core/platform/platform.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>

#include <fstream>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace ii::platform {

namespace {

#if defined(__linux__) || defined(__ANDROID__)
// Bit positions from the Linux uapi <asm/hwcap.h> for arm64 (defined here because older headers lack some of them).
constexpr unsigned long kHwcapAsimdHp = 1ul << 10;
constexpr unsigned long kHwcapAsimdDp = 1ul << 20;
constexpr unsigned long kHwcapSve = 1ul << 22;
constexpr unsigned long kHwcap2Sve2 = 1ul << 1;
constexpr unsigned long kHwcap2I8mm = 1ul << 13;
constexpr unsigned long kHwcap2Bf16 = 1ul << 14;
constexpr unsigned long kHwcap2Sme = 1ul << 23;

std::string arm_implementer_name(unsigned value) {
    switch (value) {
    case 0x41:
        return "ARM";
    case 0x42:
        return "Broadcom";
    case 0x46:
        return "Fujitsu";
    case 0x48:
        return "HiSilicon";
    case 0x4E:
        return "NVIDIA";
    case 0x51:
        return "Qualcomm";
    case 0x53:
        return "Samsung";
    case 0x61:
        return "Apple";
    case 0x6D:
        return "Microsoft";
    case 0xC0:
        return "Ampere";
    default:
        return "unknown";
    }
}

void read_proc_cpuinfo(CpuInfo& info) {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, colon);
        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) {
            key.pop_back();
        }
        std::string value = line.substr(colon + 1);
        while (!value.empty() && value.front() == ' ') {
            value.erase(value.begin());
        }
        if ((key == "model name" || key == "Hardware") && info.brand.empty()) {
            info.brand = value;
        } else if (key == "CPU implementer" && info.vendor.empty()) {
            info.vendor = arm_implementer_name(static_cast<unsigned>(std::strtoul(value.c_str(), nullptr, 0)));
        }
    }
}
#elif defined(__APPLE__)
bool sysctl_flag(const char* name) {
    int value = 0;
    std::size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0;
}

std::string sysctl_string(const char* name) {
    std::size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return {};
    }
    std::string value(size, '\0');
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) {
        return {};
    }
    value.resize(value.find('\0') == std::string::npos ? size : value.find('\0'));
    return value;
}
#endif

}  // namespace

void detect_cpu_features(CpuInfo& info) {
    info.arch = CpuArch::kArm64;
    CpuFeatureSet& f = info.features;
    f.set(CpuFeature::kNeon);

#if defined(__linux__) || defined(__ANDROID__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    f.set(CpuFeature::kArmFp16, (hwcap & kHwcapAsimdHp) != 0);
    f.set(CpuFeature::kArmDotProd, (hwcap & kHwcapAsimdDp) != 0);
    f.set(CpuFeature::kSve, (hwcap & kHwcapSve) != 0);
    f.set(CpuFeature::kSve2, (hwcap2 & kHwcap2Sve2) != 0);
    f.set(CpuFeature::kArmI8mm, (hwcap2 & kHwcap2I8mm) != 0);
    f.set(CpuFeature::kArmBf16, (hwcap2 & kHwcap2Bf16) != 0);
    f.set(CpuFeature::kSme, (hwcap2 & kHwcap2Sme) != 0);
    read_proc_cpuinfo(info);
#elif defined(__APPLE__)
    info.vendor = "Apple";
    info.brand = sysctl_string("machdep.cpu.brand_string");
    f.set(CpuFeature::kArmFp16, sysctl_flag("hw.optional.arm.FEAT_FP16"));
    f.set(CpuFeature::kArmDotProd, sysctl_flag("hw.optional.arm.FEAT_DotProd"));
    f.set(CpuFeature::kArmI8mm, sysctl_flag("hw.optional.arm.FEAT_I8MM"));
    f.set(CpuFeature::kArmBf16, sysctl_flag("hw.optional.arm.FEAT_BF16"));
    f.set(CpuFeature::kSme, sysctl_flag("hw.optional.arm.FEAT_SME"));
#elif defined(_WIN32)
    // PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE; other extensions are not reliably reported by Windows yet.
    f.set(CpuFeature::kArmDotProd, IsProcessorFeaturePresent(43) != 0);
#endif
}

}  // namespace ii::platform
