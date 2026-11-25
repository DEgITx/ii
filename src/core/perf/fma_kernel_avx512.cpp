// SPDX-License-Identifier: MIT
// AVX-512 throughput kernel: 16 independent 512-bit accumulators (covers 2 FMA ports x 4+ cycles latency).
#include "core/perf/fma_kernels.h"

#include <immintrin.h>

namespace ii::perf::II_ISA_NS {

namespace {
constexpr int kAccumulators = 16;
constexpr int kLanes = 16;
}  // namespace

double fma_f32(std::int64_t iterations, float a, float b) noexcept {
    const __m512 va = _mm512_set1_ps(a);
    const __m512 vb = _mm512_set1_ps(b);
    __m512 x0 = _mm512_set1_ps(0.0f), x1 = _mm512_set1_ps(1.0f), x2 = _mm512_set1_ps(2.0f);
    __m512 x3 = _mm512_set1_ps(3.0f), x4 = _mm512_set1_ps(4.0f), x5 = _mm512_set1_ps(5.0f);
    __m512 x6 = _mm512_set1_ps(6.0f), x7 = _mm512_set1_ps(7.0f), x8 = _mm512_set1_ps(8.0f);
    __m512 x9 = _mm512_set1_ps(9.0f), x10 = _mm512_set1_ps(10.0f), x11 = _mm512_set1_ps(11.0f);
    __m512 x12 = _mm512_set1_ps(12.0f), x13 = _mm512_set1_ps(13.0f), x14 = _mm512_set1_ps(14.0f);
    __m512 x15 = _mm512_set1_ps(15.0f);
    for (std::int64_t it = 0; it < iterations; ++it) {
        x0 = _mm512_fmadd_ps(x0, va, vb);
        x1 = _mm512_fmadd_ps(x1, va, vb);
        x2 = _mm512_fmadd_ps(x2, va, vb);
        x3 = _mm512_fmadd_ps(x3, va, vb);
        x4 = _mm512_fmadd_ps(x4, va, vb);
        x5 = _mm512_fmadd_ps(x5, va, vb);
        x6 = _mm512_fmadd_ps(x6, va, vb);
        x7 = _mm512_fmadd_ps(x7, va, vb);
        x8 = _mm512_fmadd_ps(x8, va, vb);
        x9 = _mm512_fmadd_ps(x9, va, vb);
        x10 = _mm512_fmadd_ps(x10, va, vb);
        x11 = _mm512_fmadd_ps(x11, va, vb);
        x12 = _mm512_fmadd_ps(x12, va, vb);
        x13 = _mm512_fmadd_ps(x13, va, vb);
        x14 = _mm512_fmadd_ps(x14, va, vb);
        x15 = _mm512_fmadd_ps(x15, va, vb);
    }
    const __m512 s0 = _mm512_add_ps(_mm512_add_ps(x0, x1), _mm512_add_ps(x2, x3));
    const __m512 s1 = _mm512_add_ps(_mm512_add_ps(x4, x5), _mm512_add_ps(x6, x7));
    const __m512 s2 = _mm512_add_ps(_mm512_add_ps(x8, x9), _mm512_add_ps(x10, x11));
    const __m512 s3 = _mm512_add_ps(_mm512_add_ps(x12, x13), _mm512_add_ps(x14, x15));
    const __m512 sum = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    alignas(64) float lanes[kLanes];
    _mm512_store_ps(lanes, sum);
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
