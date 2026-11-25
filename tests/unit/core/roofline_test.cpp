// SPDX-License-Identifier: MIT
#include <ii/roofline.h>
#include <ii/system_info.h>

#include "core/isa.h"
#include "core/perf/fma_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace ii {
namespace {

struct KernelUnderTest {
    Isa isa;
    double (*run)(std::int64_t, float, float) noexcept;
    int accumulators;
    int lanes;
};

std::vector<KernelUnderTest> usable_kernels() {
    std::vector<KernelUnderTest> kernels;
    kernels.push_back({Isa::kGeneric, &perf::generic::fma_f32, perf::generic::fma_f32_accumulators(),
                       perf::generic::fma_f32_lanes()});
#if defined(II_HAVE_ISA_AVX2)
    kernels.push_back(
        {Isa::kAvx2, &perf::avx2::fma_f32, perf::avx2::fma_f32_accumulators(), perf::avx2::fma_f32_lanes()});
#endif
#if defined(II_HAVE_ISA_AVX512)
    kernels.push_back(
        {Isa::kAvx512, &perf::avx512::fma_f32, perf::avx512::fma_f32_accumulators(), perf::avx512::fma_f32_lanes()});
#endif
#if defined(II_HAVE_ISA_NEON)
    kernels.push_back(
        {Isa::kNeon, &perf::neon::fma_f32, perf::neon::fma_f32_accumulators(), perf::neon::fma_f32_lanes()});
#endif
    std::erase_if(kernels, [](const KernelUnderTest& k) { return !isa_supported(k.isa, cpu_info()); });
    return kernels;
}

TEST(FmaKernels, ComputeExpectedValue) {
    for (const KernelUnderTest& kernel : usable_kernels()) {
        constexpr std::int64_t kIterations = 10;
        // Accumulators start at 0, 1, 2, ...; with a = 1 each grows by b per iteration.
        const double initial_sum = kernel.accumulators * (kernel.accumulators - 1) / 2.0;
        const double expected = kernel.lanes * (initial_sum + kernel.accumulators * kIterations * 0.5);
        EXPECT_EQ(kernel.run(kIterations, 1.0f, 0.5f), expected) << isa_name(kernel.isa);
    }
}

TEST(Roofline, QuickMeasurementProducesPositiveNumbers) {
    const Result<RooflineResult> result =
        measure_roofline({.threads = 2, .seconds_per_measurement = 0.01, .bandwidth_array_bytes = 4 << 20});
    ASSERT_TRUE(result.ok()) << result.status().to_string();
    EXPECT_EQ(result->threads, 2);
    EXPECT_FALSE(result->isa.empty());
    EXPECT_GT(result->f32_gflops_single, 0.0);
    EXPECT_GT(result->f32_gflops_multi, 0.0);
    EXPECT_GT(result->read_bandwidth_gbs, 0.0);
    EXPECT_GT(result->triad_bandwidth_gbs, 0.0);
}

TEST(Roofline, RejectsInvalidOptions) {
    EXPECT_FALSE(measure_roofline({.threads = -1}).ok());
    EXPECT_FALSE(measure_roofline({.seconds_per_measurement = 0}).ok());
}

}  // namespace
}  // namespace ii
