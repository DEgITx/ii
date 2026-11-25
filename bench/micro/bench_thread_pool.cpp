// SPDX-License-Identifier: MIT
// Thread pool dispatch overhead. Target (docs/testing-and-benchmarks.md): < 5 us per parallel_for launch.
#include "core/thread_pool.h"

#include <ii/system_info.h>

#include <benchmark/benchmark.h>

#include <atomic>
#include <cstdint>
#include <memory>

namespace ii {
namespace {

ThreadPool& pool_with(int threads) {
    // One pool per thread count, reused across benchmark repetitions.
    static std::unique_ptr<ThreadPool> pools[257];
    auto& pool = pools[threads];
    if (!pool) {
        pool = std::make_unique<ThreadPool>(ThreadPoolOptions{.num_threads = threads});
    }
    return *pool;
}

/// One empty chunk per thread: measures pure fork/join latency.
void BM_ParallelForEmpty(benchmark::State& state) {
    ThreadPool& pool = pool_with(static_cast<int>(state.range(0)));
    const std::int64_t n = pool.concurrency();
    for (auto _ : state) {
        pool.parallel_for(n, 1, [](std::int64_t begin, std::int64_t end) { benchmark::DoNotOptimize(begin + end); });
    }
    state.counters["threads"] = static_cast<double>(pool.concurrency());
}

/// Fork/join with a small amount of real work per element.
void BM_ParallelForSmallWork(benchmark::State& state) {
    ThreadPool& pool = pool_with(static_cast<int>(state.range(0)));
    const std::int64_t n = state.range(1);
    std::atomic<std::int64_t> sink{0};
    for (auto _ : state) {
        pool.parallel_for(n, 64, [&](std::int64_t begin, std::int64_t end) {
            std::int64_t local = 0;
            for (std::int64_t i = begin; i < end; ++i) {
                local += i * i;
            }
            sink.fetch_add(local, std::memory_order_relaxed);
        });
    }
    benchmark::DoNotOptimize(sink.load());
    state.SetItemsProcessed(state.iterations() * n);
}

void BM_RunOnAll(benchmark::State& state) {
    ThreadPool& pool = pool_with(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        pool.run_on_all([](int index) { benchmark::DoNotOptimize(index); });
    }
}

void thread_counts(benchmark::Benchmark* bench) {
    const int logical = cpu_info().logical_cpus;
    for (const int threads : {1, 2, 4, 8, 16, 32, 64}) {
        if (threads <= logical) {
            bench->Arg(threads);
        }
    }
    if (logical > 1 && (logical & (logical - 1)) != 0) {
        bench->Arg(logical);
    }
}

BENCHMARK(BM_ParallelForEmpty)->Apply(thread_counts)->UseRealTime();
BENCHMARK(BM_RunOnAll)->Apply(thread_counts)->UseRealTime();
BENCHMARK(BM_ParallelForSmallWork)->ArgsProduct({{1, 4, 8}, {4096, 65536, 1 << 20}})->UseRealTime();

}  // namespace
}  // namespace ii
