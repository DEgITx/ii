// SPDX-License-Identifier: MIT
#include "core/isa.h"

#include "core/platform/platform.h"

#include <array>
#include <string>

namespace ii {

namespace {

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
constexpr std::array kTargetIsas{Isa::kGeneric, Isa::kAvx2, Isa::kAvx512};
#elif defined(__aarch64__) || defined(_M_ARM64)
constexpr std::array kTargetIsas{Isa::kGeneric, Isa::kNeon};
#else
constexpr std::array kTargetIsas{Isa::kGeneric};
#endif

}  // namespace

std::string_view isa_name(Isa isa) noexcept {
    switch (isa) {
    case Isa::kGeneric:
        return "generic";
    case Isa::kAvx2:
        return "avx2";
    case Isa::kAvx512:
        return "avx512";
    case Isa::kNeon:
        return "neon";
    }
    return "unknown";
}

std::optional<Isa> parse_isa(std::string_view name) noexcept {
    for (const Isa isa : {Isa::kGeneric, Isa::kAvx2, Isa::kAvx512, Isa::kNeon}) {
        if (isa_name(isa) == name) {
            return isa;
        }
    }
    return std::nullopt;
}

std::span<const Isa> target_isas() noexcept {
    return kTargetIsas;
}

bool isa_compiled(Isa isa) noexcept {
    // II_HAVE_ISA_<NAME> is defined by cmake/IiIsa.cmake for every compiled variant.
#if defined(II_HAVE_ISA_AVX2)
    constexpr bool kHaveAvx2 = true;
#else
    constexpr bool kHaveAvx2 = false;
#endif
#if defined(II_HAVE_ISA_AVX512)
    constexpr bool kHaveAvx512 = true;
#else
    constexpr bool kHaveAvx512 = false;
#endif
#if defined(II_HAVE_ISA_NEON)
    constexpr bool kHaveNeon = true;
#else
    constexpr bool kHaveNeon = false;
#endif
    switch (isa) {
    case Isa::kGeneric:
        return true;
    case Isa::kAvx2:
        return kHaveAvx2;
    case Isa::kAvx512:
        return kHaveAvx512;
    case Isa::kNeon:
        return kHaveNeon;
    }
    return false;
}

bool isa_supported(Isa isa, const CpuInfo& cpu) noexcept {
    switch (isa) {
    case Isa::kGeneric:
        return true;
    case Isa::kAvx2:
        return cpu.has(CpuFeature::kAvx2) && cpu.has(CpuFeature::kFma) && cpu.has(CpuFeature::kF16c);
    case Isa::kAvx512:
        return cpu.has(CpuFeature::kAvx512F) && cpu.has(CpuFeature::kAvx512Bw) && cpu.has(CpuFeature::kAvx512Vl) &&
               cpu.has(CpuFeature::kAvx512Dq) && isa_supported(Isa::kAvx2, cpu);
    case Isa::kNeon:
        return cpu.has(CpuFeature::kNeon);
    }
    return false;
}

std::vector<Isa> usable_isas(const CpuInfo& cpu) {
    std::vector<Isa> result;
    for (const Isa isa : target_isas()) {
        if (isa_compiled(isa) && isa_supported(isa, cpu)) {
            result.push_back(isa);
        }
    }
    return result;
}

Result<Isa> select_isa(const CpuInfo& cpu, std::optional<std::string_view> forced) {
    if (forced && !forced->empty()) {
        const std::optional<Isa> isa = parse_isa(*forced);
        bool for_target = false;
        for (const Isa candidate : target_isas()) {
            for_target = for_target || (isa && candidate == *isa);
        }
        if (!isa || !for_target) {
            return errors::invalid_argument("unknown CPU ISA '{}' for this architecture", *forced);
        }
        if (!isa_compiled(*isa)) {
            return errors::unavailable("CPU ISA '{}' is not compiled into this build", *forced);
        }
        if (!isa_supported(*isa, cpu)) {
            return errors::unsupported("CPU ISA '{}' is not supported by this CPU", *forced);
        }
        return *isa;
    }
    return usable_isas(cpu).back();
}

Result<Isa> select_isa_from_environment(const CpuInfo& cpu) {
    const std::optional<std::string> forced = platform::get_env("II_CPU_ISA");
    if (!forced) {
        return select_isa(cpu, std::nullopt);
    }
    Result<Isa> result = select_isa(cpu, *forced);
    if (!result.ok()) {
        Status status = std::move(result).status();
        status.add_context("II_CPU_ISA");
        return status;
    }
    return result;
}

}  // namespace ii
