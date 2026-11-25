// SPDX-License-Identifier: MIT
// ii-cpu kernels end to end (a compiled one-node model per benchmark, so runtime overhead is included):
// GEMM / GEMV with prepacked weights and with two runtime operands, elementwise chains, softmax and normalization.
// The ISA variant follows II_CPU_ISA; compare with OpenBLAS via scripts/bench_openblas.py.
#include <ii/builder.h>
#include <ii/model.h>

#include <benchmark/benchmark.h>

#include <random>
#include <thread>
#include <vector>

namespace {

ii::Context cpu_context(int threads) {
    ii::ContextOptions options;
    options.num_threads = threads;
    return ii::Context::create(std::move(options)).value();
}

ii::Tensor random_f32(ii::Shape shape, std::uint32_t seed) {
    ii::Tensor tensor = ii::Tensor::empty(ii::DType::kF32, shape).value();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto* data = static_cast<float*>(tensor.data());
    for (std::int64_t i = 0; i < tensor.numel(); ++i) {
        data[i] = dist(rng);
    }
    return tensor;
}

ii::Tensor cast_to(const ii::Tensor& tensor, ii::DType dtype) {
    if (dtype == ii::DType::kF32) {
        return tensor;
    }
    const ii::Context context = cpu_context(1);
    ii::GraphBuilder b;
    const ii::Var x = b.input("x", ii::DType::kF32, ii::Dims(tensor.shape().begin(), tensor.shape().end()));
    b.output("y", ii::cast(x, dtype));
    ii::Session session =
        ii::Model::from_builder(context, std::move(b)).value().compile().value().new_session().value();
    return session.run({{"x", tensor}}).value()[0];
}

/// Runs `session` in the benchmark loop.
void run_loop(benchmark::State& state, ii::Session& session) {
    for (auto _ : state) {
        const ii::Status status = session.run();
        if (!status.ok()) {
            state.SkipWithError(status.to_string());
            break;
        }
    }
}

int threads_arg(std::int64_t value) {
    return value == 0 ? static_cast<int>(std::thread::hardware_concurrency()) : static_cast<int>(value);
}

/// y[M, N] = x[M, K] @ w[N, K]^T with constant (prepacked) weights: the layout of a linear layer.
void BM_Linear(benchmark::State& state, ii::DType dtype) {
    const std::int64_t m = state.range(0);
    const std::int64_t n = state.range(1);
    const std::int64_t k = state.range(2);
    const ii::Context context = cpu_context(threads_arg(state.range(3)));
    ii::GraphBuilder b;
    const ii::Var x = b.input("x", dtype, {m, k});
    const ii::Var w = b.constant(cast_to(random_f32(ii::Shape{n, k}, 2), dtype));
    b.output("y", ii::linear(x, w));
    const ii::Program program = ii::Model::from_builder(context, std::move(b)).value().compile().value();
    ii::Session session = program.new_session().value();
    (void)session.set_input("x", cast_to(random_f32(ii::Shape{m, k}, 1), dtype));
    run_loop(state, session);
    state.counters["GFLOP/s"] =
        benchmark::Counter(2.0 * static_cast<double>(m * n * k), benchmark::Counter::kIsIterationInvariantRate);
    state.counters["GB/s"] =
        benchmark::Counter(static_cast<double>(n * k * static_cast<std::int64_t>(ii::element_size(dtype))),
                           benchmark::Counter::kIsIterationInvariantRate);
}

/// c[M, N] = a[M, K] @ b[K, N] with both operands bound at run time (B packed on every run).
void BM_Matmul(benchmark::State& state) {
    const std::int64_t m = state.range(0);
    const std::int64_t n = state.range(1);
    const std::int64_t k = state.range(2);
    const ii::Context context = cpu_context(threads_arg(state.range(3)));
    ii::GraphBuilder b;
    const ii::Var x = b.input("a", ii::DType::kF32, {m, k});
    const ii::Var y = b.input("b", ii::DType::kF32, {k, n});
    b.output("c", ii::matmul(x, y));
    const ii::Program program = ii::Model::from_builder(context, std::move(b)).value().compile().value();
    ii::Session session = program.new_session().value();
    (void)session.set_input("a", random_f32(ii::Shape{m, k}, 1));
    (void)session.set_input("b", random_f32(ii::Shape{k, n}, 2));
    run_loop(state, session);
    state.counters["GFLOP/s"] =
        benchmark::Counter(2.0 * static_cast<double>(m * n * k), benchmark::Counter::kIsIterationInvariantRate);
}

void gemm_args(benchmark::Benchmark* bench) {
    bench->ArgNames({"M", "N", "K", "threads"});
    for (const std::int64_t threads : {1, 0}) {
        for (const std::int64_t size : {256, 512, 1024, 2048}) {
            bench->Args({size, size, size, threads});
        }
        bench->Args({1, 4096, 4096, threads});    // LLM decode (GEMV)
        bench->Args({64, 4096, 4096, threads});   // small batch / prefill chunk
        bench->Args({512, 4096, 4096, threads});  // prefill
    }
    bench->UseRealTime();
}

BENCHMARK_CAPTURE(BM_Linear, f32, ii::DType::kF32)->Apply(gemm_args);
BENCHMARK_CAPTURE(BM_Linear, f16, ii::DType::kF16)
    ->Args({1, 4096, 4096, 1})
    ->Args({1, 4096, 4096, 0})
    ->Args({512, 4096, 4096, 0})
    ->ArgNames({"M", "N", "K", "threads"})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_Linear, bf16, ii::DType::kBF16)
    ->Args({1, 4096, 4096, 1})
    ->Args({1, 4096, 4096, 0})
    ->Args({512, 4096, 4096, 0})
    ->ArgNames({"M", "N", "K", "threads"})
    ->UseRealTime();
BENCHMARK(BM_Matmul)
    ->Args({1024, 1024, 1024, 1})
    ->Args({1024, 1024, 1024, 0})
    ->ArgNames({"M", "N", "K", "threads"})
    ->UseRealTime();

/// Elementwise: a single op, and a chain of 4 ops (fused into one pass when fusion is enabled).
void BM_Elementwise(benchmark::State& state) {
    const std::int64_t n = state.range(0);
    const bool chain = state.range(1) != 0;
    const ii::Context context = cpu_context(threads_arg(state.range(2)));
    ii::GraphBuilder b;
    const ii::Var x = b.input("x", ii::DType::kF32, {n});
    const ii::Var y = b.input("y", ii::DType::kF32, {n});
    b.output("z", chain ? ii::gelu(x * y + x) * 0.5 : x + y);
    const ii::Program program = ii::Model::from_builder(context, std::move(b)).value().compile().value();
    ii::Session session = program.new_session().value();
    (void)session.set_input("x", random_f32(ii::Shape{n}, 1));
    (void)session.set_input("y", random_f32(ii::Shape{n}, 2));
    run_loop(state, session);
    state.counters["GB/s"] =
        benchmark::Counter(12.0 * static_cast<double>(n), benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Elementwise)
    ->ArgNames({"n", "chain", "threads"})
    ->Args({1 << 16, 0, 1})
    ->Args({1 << 22, 0, 1})
    ->Args({1 << 22, 0, 0})
    ->Args({1 << 16, 1, 1})
    ->Args({1 << 22, 1, 1})
    ->Args({1 << 22, 1, 0})
    ->UseRealTime();

void BM_Softmax(benchmark::State& state) {
    const std::int64_t rows = state.range(0);
    const std::int64_t cols = state.range(1);
    const ii::Context context = cpu_context(threads_arg(state.range(2)));
    ii::GraphBuilder b;
    const ii::Var x = b.input("x", ii::DType::kF32, {rows, cols});
    b.output("y", ii::softmax(x));
    ii::Session session =
        ii::Model::from_builder(context, std::move(b)).value().compile().value().new_session().value();
    (void)session.set_input("x", random_f32(ii::Shape{rows, cols}, 1));
    run_loop(state, session);
    state.counters["GB/s"] =
        benchmark::Counter(8.0 * static_cast<double>(rows * cols), benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Softmax)
    ->ArgNames({"rows", "cols", "threads"})
    ->Args({1024, 1024, 1})
    ->Args({1024, 1024, 0})
    ->Args({32, 32000, 1})
    ->UseRealTime();

void BM_LayerNorm(benchmark::State& state) {
    const std::int64_t rows = state.range(0);
    const std::int64_t cols = state.range(1);
    const ii::Context context = cpu_context(threads_arg(state.range(2)));
    ii::GraphBuilder b;
    const ii::Var x = b.input("x", ii::DType::kF32, {rows, cols});
    b.output("y",
             ii::layer_norm(x, b.constant(random_f32(ii::Shape{cols}, 3)), b.constant(random_f32(ii::Shape{cols}, 4))));
    ii::Session session =
        ii::Model::from_builder(context, std::move(b)).value().compile().value().new_session().value();
    (void)session.set_input("x", random_f32(ii::Shape{rows, cols}, 1));
    run_loop(state, session);
    state.counters["GB/s"] =
        benchmark::Counter(8.0 * static_cast<double>(rows * cols), benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_LayerNorm)
    ->ArgNames({"rows", "cols", "threads"})
    ->Args({1024, 4096, 1})
    ->Args({1024, 4096, 0})
    ->UseRealTime();

}  // namespace
