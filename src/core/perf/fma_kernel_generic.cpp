// SPDX-License-Identifier: MIT
// Portable FMA throughput kernel (scalar code; the compiler may vectorize it for the baseline ISA).
#include "core/perf/fma_kernels.h"

namespace ii::perf::II_ISA_NS {

namespace {
constexpr int kAccumulators = 16;
}  // namespace

double fma_f32(std::int64_t iterations, float a, float b) noexcept {
    float x[kAccumulators];
    for (int i = 0; i < kAccumulators; ++i) {
        x[i] = static_cast<float>(i);
    }
    for (std::int64_t it = 0; it < iterations; ++it) {
        for (int i = 0; i < kAccumulators; ++i) {
            x[i] = x[i] * a + b;
        }
    }
    double sum = 0;
    for (int i = 0; i < kAccumulators; ++i) {
        sum += x[i];
    }
    return sum;
}

int fma_f32_accumulators() noexcept {
    return kAccumulators;
}

int fma_f32_lanes() noexcept {
    return 1;
}

}  // namespace ii::perf::II_ISA_NS
