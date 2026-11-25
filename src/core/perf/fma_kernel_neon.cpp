// SPDX-License-Identifier: MIT
// NEON throughput kernel: 16 independent 128-bit accumulators (covers 4 FMA pipes x 4 cycles latency on Apple
// and Neoverse cores).
#include "core/perf/fma_kernels.h"

#include <arm_neon.h>

namespace ii::perf::II_ISA_NS {

namespace {
constexpr int kAccumulators = 16;
constexpr int kLanes = 4;
}  // namespace

double fma_f32(std::int64_t iterations, float a, float b) noexcept {
    const float32x4_t va = vdupq_n_f32(a);
    const float32x4_t vb = vdupq_n_f32(b);
    float32x4_t x0 = vdupq_n_f32(0.0f), x1 = vdupq_n_f32(1.0f), x2 = vdupq_n_f32(2.0f), x3 = vdupq_n_f32(3.0f);
    float32x4_t x4 = vdupq_n_f32(4.0f), x5 = vdupq_n_f32(5.0f), x6 = vdupq_n_f32(6.0f), x7 = vdupq_n_f32(7.0f);
    float32x4_t x8 = vdupq_n_f32(8.0f), x9 = vdupq_n_f32(9.0f), x10 = vdupq_n_f32(10.0f), x11 = vdupq_n_f32(11.0f);
    float32x4_t x12 = vdupq_n_f32(12.0f), x13 = vdupq_n_f32(13.0f), x14 = vdupq_n_f32(14.0f);
    float32x4_t x15 = vdupq_n_f32(15.0f);
    for (std::int64_t it = 0; it < iterations; ++it) {
        // vfmaq_f32(acc, m, n) = acc + m * n, i.e. x = b + x * a.
        x0 = vfmaq_f32(vb, x0, va);
        x1 = vfmaq_f32(vb, x1, va);
        x2 = vfmaq_f32(vb, x2, va);
        x3 = vfmaq_f32(vb, x3, va);
        x4 = vfmaq_f32(vb, x4, va);
        x5 = vfmaq_f32(vb, x5, va);
        x6 = vfmaq_f32(vb, x6, va);
        x7 = vfmaq_f32(vb, x7, va);
        x8 = vfmaq_f32(vb, x8, va);
        x9 = vfmaq_f32(vb, x9, va);
        x10 = vfmaq_f32(vb, x10, va);
        x11 = vfmaq_f32(vb, x11, va);
        x12 = vfmaq_f32(vb, x12, va);
        x13 = vfmaq_f32(vb, x13, va);
        x14 = vfmaq_f32(vb, x14, va);
        x15 = vfmaq_f32(vb, x15, va);
    }
    const float32x4_t s0 = vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3));
    const float32x4_t s1 = vaddq_f32(vaddq_f32(x4, x5), vaddq_f32(x6, x7));
    const float32x4_t s2 = vaddq_f32(vaddq_f32(x8, x9), vaddq_f32(x10, x11));
    const float32x4_t s3 = vaddq_f32(vaddq_f32(x12, x13), vaddq_f32(x14, x15));
    const float32x4_t sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    float lanes[kLanes];
    vst1q_f32(lanes, sum);
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
