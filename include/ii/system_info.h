// SPDX-License-Identifier: MIT
/// @file
/// Information about the machine: operating system, CPU (instruction sets, cores, caches) and memory.
#pragma once

#include <ii/export.h>
#include <ii/status.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ii {

enum class CpuArch {
    kUnknown,
    kX86_64,
    kArm64,
};

II_API std::string_view to_string(CpuArch arch) noexcept;

/// CPU features relevant for kernel selection. A feature is reported only when both the CPU and the operating
/// system support it (e.g. AVX requires the OS to save YMM registers).
enum class CpuFeature : std::uint8_t {
    // x86-64
    kSse2,
    kSse3,
    kSsse3,
    kSse41,
    kSse42,
    kAvx,
    kAvx2,
    kFma,
    kF16c,
    kAvxVnni,
    kAvx512F,
    kAvx512Bw,
    kAvx512Vl,
    kAvx512Dq,
    kAvx512Cd,
    kAvx512Vnni,
    kAvx512Bf16,
    kAvx512Fp16,
    kAmxTile,
    kAmxInt8,
    kAmxBf16,
    // Arm64
    kNeon,
    kArmFp16,
    kArmDotProd,
    kArmI8mm,
    kArmBf16,
    kSve,
    kSve2,
    kSme,

    kCount,
};

/// Returns the conventional lowercase name of a feature, e.g. "avx512f", "dotprod".
II_API std::string_view to_string(CpuFeature feature) noexcept;

/// A set of `CpuFeature`s.
class CpuFeatureSet {
public:
    constexpr bool has(CpuFeature feature) const noexcept { return (bits_ >> static_cast<unsigned>(feature)) & 1u; }
    constexpr void set(CpuFeature feature, bool value = true) noexcept {
        const std::uint64_t mask = std::uint64_t{1} << static_cast<unsigned>(feature);
        bits_ = value ? (bits_ | mask) : (bits_ & ~mask);
    }
    constexpr std::uint64_t bits() const noexcept { return bits_; }

private:
    static_assert(static_cast<unsigned>(CpuFeature::kCount) <= 64);
    std::uint64_t bits_ = 0;
};

/// One physical core and the logical CPUs (hardware threads) it runs.
struct CpuCore {
    /// Relative performance class. Higher is faster; all cores of a non-hybrid CPU have the same class.
    int efficiency_class = 0;
    /// Operating-system indices of the core's logical CPUs (usable for thread affinity).
    std::vector<int> logical_cpus;
};

/// Cache sizes in bytes as seen by one performance core; 0 when unknown or absent.
struct CpuCaches {
    std::uint64_t l1d = 0;  ///< Per-core L1 data cache.
    std::uint64_t l2 = 0;   ///< L2 cache (per core or per cluster).
    std::uint64_t l3 = 0;   ///< Last-level shared cache.
};

struct CpuInfo {
    CpuArch arch = CpuArch::kUnknown;
    std::string vendor;  ///< E.g. "GenuineIntel", "AuthenticAMD", "Apple", "ARM".
    std::string brand;   ///< Marketing name, e.g. "Intel(R) Core(TM) Ultra 7 265KF"; may be empty.
    CpuFeatureSet features;

    int logical_cpus = 0;       ///< Hardware threads available to the process.
    int physical_cores = 0;     ///< Physical cores.
    int performance_cores = 0;  ///< Cores of the highest efficiency class (all cores on non-hybrid CPUs).
    int efficiency_cores = 0;   ///< Remaining cores on hybrid CPUs; 0 otherwise.
    CpuCaches caches;

    /// Per-core topology. Empty when the OS does not expose a core-to-CPU mapping (e.g. macOS).
    std::vector<CpuCore> cores;

    bool has(CpuFeature feature) const noexcept { return features.has(feature); }
    bool is_hybrid() const noexcept { return efficiency_cores > 0; }
};

struct OsInfo {
    std::string name;     ///< E.g. "Windows 11 24H2", "Ubuntu 24.04.1 LTS", "macOS 15.1", "Android 14".
    std::string version;  ///< Kernel or build version, e.g. "10.0.26100", "6.8.0-45-generic".
};

struct MemoryInfo {
    std::uint64_t total_bytes = 0;      ///< Installed physical memory.
    std::uint64_t available_bytes = 0;  ///< Memory available to new allocations without swapping (estimate).
};

/// Describes the CPU. Detected once on first use; the reference stays valid for the lifetime of the process.
II_API const CpuInfo& cpu_info() noexcept;

/// Describes the operating system.
II_API OsInfo os_info();

/// Current physical memory state.
II_API MemoryInfo memory_info();

/// Availability of one CPU kernel instruction-set variant.
struct IsaSupport {
    std::string name;        ///< E.g. "generic", "avx2", "avx512", "neon".
    bool compiled = false;   ///< Kernels for this variant are compiled into the library.
    bool supported = false;  ///< The CPU (and OS) can execute this variant.
};

/// CPU kernel variants for the target architecture, from the baseline to the most advanced.
II_API std::vector<IsaSupport> isa_support();

/// The kernel variant ii uses on this machine: the most advanced usable one, or the one forced with the
/// `II_CPU_ISA` environment variable. Fails if the forced variant is unknown, not compiled or not supported.
II_API Result<std::string> selected_isa();

}  // namespace ii
