// SPDX-License-Identifier: MIT
// AVX2 + FMA throughput kernel: 12 independent 256-bit accumulators (16 YMM registers minus the two constants).
#include "core/perf/fma_kernels.h"

#include <immintrin.h>

namespace ii::perf::II_ISA_NS {

namespace {
constexpr int kAccumulators = 12;
constexpr int kLanes = 8;
}  // namespace

double fma_f32(std::int64_t iterations, float a, float b) noexcept {
    const __m256 va = _mm256_set1_ps(a);
    const __m256 vb = _mm256_set1_ps(b);
    __m256 x0 = _mm256_set1_ps(0.0f), x1 = _mm256_set1_ps(1.0f), x2 = _mm256_set1_ps(2.0f);
    __m256 x3 = _mm256_set1_ps(3.0f), x4 = _mm256_set1_ps(4.0f), x5 = _mm256_set1_ps(5.0f);
    __m256 x6 = _mm256_set1_ps(6.0f), x7 = _mm256_set1_ps(7.0f), x8 = _mm256_set1_ps(8.0f);
    __m256 x9 = _mm256_set1_ps(9.0f), x10 = _mm256_set1_ps(10.0f), x11 = _mm256_set1_ps(11.0f);
    for (std::int64_t it = 0; it < iterations; ++it) {
        x0 = _mm256_fmadd_ps(x0, va, vb);
        x1 = _mm256_fmadd_ps(x1, va, vb);
        x2 = _mm256_fmadd_ps(x2, va, vb);
        x3 = _mm256_fmadd_ps(x3, va, vb);
        x4 = _mm256_fmadd_ps(x4, va, vb);
        x5 = _mm256_fmadd_ps(x5, va, vb);
        x6 = _mm256_fmadd_ps(x6, va, vb);
        x7 = _mm256_fmadd_ps(x7, va, vb);
        x8 = _mm256_fmadd_ps(x8, va, vb);
        x9 = _mm256_fmadd_ps(x9, va, vb);
        x10 = _mm256_fmadd_ps(x10, va, vb);
        x11 = _mm256_fmadd_ps(x11, va, vb);
    }
    const __m256 sum = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3)),
                                                   _mm256_add_ps(_mm256_add_ps(x4, x5), _mm256_add_ps(x6, x7))),
                                     _mm256_add_ps(_mm256_add_ps(x8, x9), _mm256_add_ps(x10, x11)));
    alignas(32) float lanes[kLanes];
    _mm256_store_ps(lanes, sum);
    double total = 0;
    for (const float lane : lanes) {
        total += lane;
    }
    return total;
}

int fma_f32_accumulators() noexcept {
    return kAccumulators;
}

int fma_f32_lanes() noexcept {
    return kLanes;
}

}  // namespace ii::perf::II_ISA_NS
