// SPDX-License-Identifier: MIT
/// @file
/// Declarations of the per-ISA FMA throughput kernels used by the roofline measurement.
///
/// This header is included by ISA-specific translation units, so it must not contain inline functions or templates
/// (see the ODR rules in cmake/IiIsa.cmake).
#pragma once

#include <cstdint>

namespace ii::perf {

// Each kernel runs `iterations` rounds of x = x * a + b over independent accumulators (enough of them to hide FMA
// latency) and returns the sum of all accumulator lanes so the work cannot be optimized away.
// With a = 1 and accumulators initialized to 0, 1, 2, ..., the result is exactly
//   lanes * (sum of initial values) + lanes * accumulators * iterations * b
// for small iteration counts, which makes the kernels testable.

#define II_DECLARE_FMA_KERNEL(isa)                                                                                     \
    namespace isa {                                                                                                    \
    double fma_f32(std::int64_t iterations, float a, float b) noexcept;                                                \
    int fma_f32_accumulators() noexcept;                                                                               \
    int fma_f32_lanes() noexcept;                                                                                      \
    }

II_DECLARE_FMA_KERNEL(generic)
#if defined(II_HAVE_ISA_AVX2)
II_DECLARE_FMA_KERNEL(avx2)
#endif
#if defined(II_HAVE_ISA_AVX512)
II_DECLARE_FMA_KERNEL(avx512)
#endif
#if defined(II_HAVE_ISA_NEON)
II_DECLARE_FMA_KERNEL(neon)
#endif

#undef II_DECLARE_FMA_KERNEL

}  // namespace ii::perf
