// SPDX-License-Identifier: MIT
// CPU feature detection for x86-64 via CPUID and XGETBV. Works on every OS; OS support for extended register state is
// verified through XCR0 (with a macOS-specific exception for AVX-512, see below).
#include "core/platform/platform.h"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace ii::platform {

namespace {

struct CpuidRegisters {
    std::uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
};

CpuidRegisters cpuid(std::uint32_t leaf, std::uint32_t subleaf = 0) {
    CpuidRegisters r;
#if defined(_MSC_VER) && !defined(__clang__)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<std::uint32_t>(regs[0]);
    r.ebx = static_cast<std::uint32_t>(regs[1]);
    r.ecx = static_cast<std::uint32_t>(regs[2]);
    r.edx = static_cast<std::uint32_t>(regs[3]);
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    r.eax = a;
    r.ebx = b;
    r.ecx = c;
    r.edx = d;
#endif
    return r;
}

std::uint64_t xgetbv0() {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    std::uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

bool bit(std::uint32_t value, int index) {
    return ((value >> index) & 1u) != 0;
}

#if defined(__APPLE__)
// macOS enables the AVX-512 register state lazily on first use, so XCR0 may not report it. Ask the kernel instead.
bool apple_sysctl_flag(const char* name) {
    int value = 0;
    std::size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0;
}
#endif

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(' ');
    if (first == std::string::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(' ');
    return text.substr(first, last - first + 1);
}

}  // namespace

void detect_cpu_features(CpuInfo& info) {
    info.arch = CpuArch::kX86_64;

    const CpuidRegisters leaf0 = cpuid(0);
    const std::uint32_t max_leaf = leaf0.eax;
    char vendor[13] = {};
    std::memcpy(vendor + 0, &leaf0.ebx, 4);
    std::memcpy(vendor + 4, &leaf0.edx, 4);
    std::memcpy(vendor + 8, &leaf0.ecx, 4);
    info.vendor = vendor;

    const std::uint32_t max_extended_leaf = cpuid(0x80000000u).eax;
    if (max_extended_leaf >= 0x80000004u) {
        std::array<char, 49> brand{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            const CpuidRegisters r = cpuid(0x80000002u + i);
            std::memcpy(brand.data() + i * 16 + 0, &r.eax, 4);
            std::memcpy(brand.data() + i * 16 + 4, &r.ebx, 4);
            std::memcpy(brand.data() + i * 16 + 8, &r.ecx, 4);
            std::memcpy(brand.data() + i * 16 + 12, &r.edx, 4);
        }
        info.brand = trim(brand.data());
    }

    if (max_leaf < 1) {
        return;
    }
    const CpuidRegisters leaf1 = cpuid(1);
    CpuFeatureSet& f = info.features;
    f.set(CpuFeature::kSse2, bit(leaf1.edx, 26));
    f.set(CpuFeature::kSse3, bit(leaf1.ecx, 0));
    f.set(CpuFeature::kSsse3, bit(leaf1.ecx, 9));
    f.set(CpuFeature::kSse41, bit(leaf1.ecx, 19));
    f.set(CpuFeature::kSse42, bit(leaf1.ecx, 20));

    // Extended register state must be enabled by the OS (XCR0) before AVX/AVX-512/AMX instructions can be used.
    const bool osxsave = bit(leaf1.ecx, 27);
    const std::uint64_t xcr0 = osxsave ? xgetbv0() : 0;
    const bool os_avx = (xcr0 & 0x6) == 0x6;          // XMM | YMM
    bool os_avx512 = (xcr0 & 0xE6) == 0xE6;           // XMM | YMM | opmask | ZMM_Hi256 | Hi16_ZMM
    const bool os_amx = (xcr0 & 0x60000) == 0x60000;  // XTILECFG | XTILEDATA
#if defined(__APPLE__)
    if (!os_avx512 && os_avx) {
        os_avx512 = apple_sysctl_flag("hw.optional.avx512f");
    }
#endif

    if (os_avx) {
        f.set(CpuFeature::kAvx, bit(leaf1.ecx, 28));
        f.set(CpuFeature::kFma, bit(leaf1.ecx, 12));
        f.set(CpuFeature::kF16c, bit(leaf1.ecx, 29));
    }

    if (max_leaf < 7) {
        return;
    }
    const CpuidRegisters leaf7 = cpuid(7, 0);
    const CpuidRegisters leaf7_1 = leaf7.eax >= 1 ? cpuid(7, 1) : CpuidRegisters{};

    if (os_avx) {
        f.set(CpuFeature::kAvx2, bit(leaf7.ebx, 5));
        f.set(CpuFeature::kAvxVnni, bit(leaf7_1.eax, 4));
    }
    if (os_avx512) {
        f.set(CpuFeature::kAvx512F, bit(leaf7.ebx, 16));
        f.set(CpuFeature::kAvx512Dq, bit(leaf7.ebx, 17));
        f.set(CpuFeature::kAvx512Cd, bit(leaf7.ebx, 28));
        f.set(CpuFeature::kAvx512Bw, bit(leaf7.ebx, 30));
        f.set(CpuFeature::kAvx512Vl, bit(leaf7.ebx, 31));
        f.set(CpuFeature::kAvx512Vnni, bit(leaf7.ecx, 11));
        f.set(CpuFeature::kAvx512Fp16, bit(leaf7.edx, 23));
        f.set(CpuFeature::kAvx512Bf16, bit(leaf7_1.eax, 5));
    }
    // Note: on Linux, AMX tile data additionally requires a per-process permission request
    // (arch_prctl(ARCH_REQ_XCOMP_PERM)); kernels that use AMX request it before first use.
    if (os_amx) {
        f.set(CpuFeature::kAmxBf16, bit(leaf7.edx, 22));
        f.set(CpuFeature::kAmxTile, bit(leaf7.edx, 24));
        f.set(CpuFeature::kAmxInt8, bit(leaf7.edx, 25));
    }
}

}  // namespace ii::platform
