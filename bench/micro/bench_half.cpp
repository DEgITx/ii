// SPDX-License-Identifier: MIT
// Throughput of the portable scalar f16/bf16 conversions (vectorized conversions come with the CPU kernels).
#include <ii/half.h>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <random>
#include <vector>

namespace ii {
namespace {

std::vector<float> make_inputs(std::size_t n) {
    std::mt19937 rng(42);
    std::normal_distribution<float> distribution(0.0f, 10.0f);
    std::vector<float> values(n);
    for (float& v : values) {
        v = distribution(rng);
    }
    return values;
}

template <class Half>
void BM_FloatToHalf(benchmark::State& state) {
    const std::vector<float> input = make_inputs(4096);
    std::vector<Half> output(input.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = Half(input[i]);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(input.size()));
}

template <class Half>
void BM_HalfToFloat(benchmark::State& state) {
    const std::vector<float> values = make_inputs(4096);
    std::vector<Half> input(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        input[i] = Half(values[i]);
    }
    std::vector<float> output(input.size());
    for (auto _ : state) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = input[i].to_float();
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(input.size()));
}

BENCHMARK(BM_FloatToHalf<f16>);
BENCHMARK(BM_FloatToHalf<bf16>);
BENCHMARK(BM_HalfToFloat<f16>);
BENCHMARK(BM_HalfToFloat<bf16>);

}  // namespace
}  // namespace ii
