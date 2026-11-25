// SPDX-License-Identifier: MIT
// Runtime overheads: per-step dispatch cost of Session::run, re-planning on shape changes, memory planning, and the
// speed of reference kernels (a baseline the optimized backends are compared against).
#include <ii/builder.h>
#include <ii/model.h>

#include "runtime/memory_planner.h"

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

namespace {

ii::Context bench_context() {
    ii::ContextOptions options;
    options.num_threads = 1;
    return ii::Context::create(std::move(options)).value();
}

/// A chain of `steps` elementwise operations on a tiny tensor: measures per-step runtime overhead.
void BM_SessionStepOverhead(benchmark::State& state) {
    const ii::Context context = bench_context();
    ii::GraphBuilder b;
    ii::Var x = b.input("x", ii::DType::kF32, {4});
    for (int i = 0; i < state.range(0); ++i) {
        x = (i % 2 == 0) ? ii::relu(x) : ii::neg(x);
    }
    b.output("y", x);
    const ii::Program program = ii::Model::from_builder(context, std::move(b)).value().compile().value();
    ii::Session session = program.new_session().value();
    float data[4] = {1, -2, 3, -4};
    (void)session.set_input("x", ii::Tensor::wrap(data, ii::DType::kF32, ii::Shape{4}));
    for (auto _ : state) {
        benchmark::DoNotOptimize(session.run());
    }
    state.counters["ns_per_step"] = benchmark::Counter(static_cast<double>(state.iterations() * state.range(0)),
                                                       benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}
BENCHMARK(BM_SessionStepOverhead)->Arg(16)->Arg(256);

/// Alternating input shapes force a new execution plan on every run.
void BM_SessionReplan(benchmark::State& state) {
    const ii::Context context = bench_context();
    ii::GraphBuilder b;
    const ii::Var x = b.input("x", ii::DType::kF32, {"N", 8});
    b.output("y", ii::softmax(ii::relu(x) * 2.0));
    const ii::Program program = ii::Model::from_builder(context, std::move(b)).value().compile().value();
    ii::Session session = program.new_session().value();
    const ii::Tensor a = ii::Tensor::zeros(ii::DType::kF32, ii::Shape{2, 8}).value();
    const ii::Tensor c = ii::Tensor::zeros(ii::DType::kF32, ii::Shape{3, 8}).value();
    bool flip = false;
    for (auto _ : state) {
        (void)session.set_input("x", flip ? a : c);
        flip = !flip;
        benchmark::DoNotOptimize(session.run());
    }
}
BENCHMARK(BM_SessionReplan);

void BM_MemoryPlanner(benchmark::State& state) {
    std::mt19937 rng(1);
    std::vector<ii::runtime::MemoryRequest> requests;
    for (int i = 0; i < state.range(0); ++i) {
        requests.push_back({static_cast<std::size_t>(rng() % 100000), i, i + static_cast<int>(rng() % 8)});
    }
    for (auto _ : state) {
        benchmark::DoNotOptimize(ii::runtime::plan_memory(requests));
    }
}
BENCHMARK(BM_MemoryPlanner)->Arg(100)->Arg(1000);

/// Reference matmul: the correctness oracle is deliberately simple; this documents how slow it is.
void BM_RefMatmul(benchmark::State& state) {
    const ii::Context context = bench_context();
    const std::int64_t n = state.range(0);
    ii::GraphBuilder b;
    const ii::Var x = b.input("a", ii::DType::kF32, {n, n});
    const ii::Var y = b.input("b", ii::DType::kF32, {n, n});
    b.output("c", ii::matmul(x, y));
    const ii::Program program = ii::Model::from_builder(context, std::move(b)).value().compile().value();
    ii::Session session = program.new_session().value();
    const ii::Tensor m = ii::Tensor::zeros(ii::DType::kF32, ii::Shape{n, n}).value();
    (void)session.set_input("a", m);
    (void)session.set_input("b", m);
    for (auto _ : state) {
        benchmark::DoNotOptimize(session.run());
    }
    state.counters["GFLOP/s"] =
        benchmark::Counter(2.0 * static_cast<double>(n * n * n) * static_cast<double>(state.iterations()) / 1e9,
                           benchmark::Counter::kIsRate);
}
BENCHMARK(BM_RefMatmul)->Arg(64)->Arg(128);

}  // namespace
