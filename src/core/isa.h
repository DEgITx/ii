// SPDX-License-Identifier: MIT
/// @file
/// CPU kernel instruction-set variants and runtime selection.
///
/// Kernels are compiled once per variant (see cmake/IiIsa.cmake). At runtime the best variant that is both compiled
/// in and supported by the CPU is chosen; `II_CPU_ISA=<name>` forces a specific variant, which is how tests exercise
/// every code path on a single machine.
#pragma once

#include <ii/status.h>
#include <ii/system_info.h>

#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ii {

/// Kernel ISA variants, ordered from baseline to most advanced within each architecture.
enum class Isa : std::uint8_t {
    kGeneric,      ///< Portable C++ compiled for the architecture baseline (x86-64: SSE2, Arm64: NEON).
    kAvx2,         ///< x86-64: AVX2 + FMA + F16C.
    kAvx512,       ///< x86-64: AVX-512 F/BW/VL/DQ.
    kNeon,         ///< Arm64: explicit NEON intrinsics.
    kAvxVnni,      ///< x86-64: the AVX2 kernels plus AVX-VNNI int8 dot products (vpdpbusd).
    kNeonDotprod,  ///< Arm64: the NEON kernels plus int8 dot products (sdot, Armv8.2 dotprod).
};

std::string_view isa_name(Isa isa) noexcept;
std::optional<Isa> parse_isa(std::string_view name) noexcept;

/// Variants that exist for the target architecture, from baseline to most advanced.
std::span<const Isa> target_isas() noexcept;

/// True if kernels for `isa` are compiled into this build.
bool isa_compiled(Isa isa) noexcept;

/// True if `cpu` can execute `isa` kernels.
bool isa_supported(Isa isa, const CpuInfo& cpu) noexcept;

/// Variants that are both compiled and supported, from baseline to most advanced. Never empty.
std::vector<Isa> usable_isas(const CpuInfo& cpu);

/// Selects the variant to use: `forced` if given (it must be compiled and supported), else the most advanced usable.
Result<Isa> select_isa(const CpuInfo& cpu, std::optional<std::string_view> forced);

/// `select_isa` with the forced variant taken from the II_CPU_ISA environment variable.
Result<Isa> select_isa_from_environment(const CpuInfo& cpu);

}  // namespace ii
