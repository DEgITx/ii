// SPDX-License-Identifier: MIT
#include <ii/roofline.h>
#include <ii/system_info.h>

#include "core/aligned_buffer.h"
#include "core/isa.h"
#include "core/perf/fma_kernels.h"
#include "core/stopwatch.h"
#include "core/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

namespace ii {

namespace {

struct FmaKernel {
    double (*run)(std::int64_t, float, float) noexcept;
    double flops_per_iteration;
};

#define II_FMA_KERNEL(isa)                                                                                             \
    FmaKernel {                                                                                                        \
        &perf::isa::fma_f32, 2.0 * perf::isa::fma_f32_accumulators() * perf::isa::fma_f32_lanes()                      \
    }

FmaKernel fma_kernel(Isa isa) noexcept {
    switch (isa) {
#if defined(II_HAVE_ISA_AVX2)
    case Isa::kAvx2:
        return II_FMA_KERNEL(avx2);
#endif
#if defined(II_HAVE_ISA_AVX512)
    case Isa::kAvx512:
        return II_FMA_KERNEL(avx512);
#endif
#if defined(II_HAVE_ISA_NEON)
    case Isa::kNeon:
        return II_FMA_KERNEL(neon);
#endif
    default:
        return II_FMA_KERNEL(generic);
    }
}

#undef II_FMA_KERNEL

/// Prevents the compiler from discarding benchmark results.
void consume(double value) {
    static std::atomic<double> sink{0};
    sink.store(value, std::memory_order_relaxed);
}

/// Runs `measure` (which returns a rate) repeatedly for about `seconds` (at least 3 times) and returns the best rate.
template <class Fn>
double best_rate(double seconds, Fn&& measure) {
    double best = 0;
    const Stopwatch total;
    for (int run = 0; run < 3 || total.elapsed_seconds() < seconds; ++run) {
        best = std::max(best, measure());
        if (run >= 1000) {
            break;
        }
    }
    return best;
}

/// Finds an iteration count for which one kernel call takes at least `target_seconds`.
std::int64_t calibrate_iterations(const FmaKernel& kernel, double target_seconds) {
    std::int64_t iterations = 1 << 12;
    for (;;) {
        const Stopwatch watch;
        consume(kernel.run(iterations, 1.0f, 0.0f));
        const double elapsed = watch.elapsed_seconds();
        if (elapsed >= target_seconds || iterations >= (std::int64_t{1} << 40)) {
            if (elapsed > 0) {
                iterations = std::max<std::int64_t>(
                    1, static_cast<std::int64_t>(static_cast<double>(iterations) * target_seconds / elapsed));
            }
            return iterations;
        }
        iterations *= elapsed < target_seconds / 16 ? 8 : 2;
    }
}

/// Splits [0, n) into `parts` contiguous ranges and returns the i-th one.
std::pair<std::size_t, std::size_t> partition(std::size_t n, int parts, int index) {
    const std::size_t base = n / static_cast<std::size_t>(parts);
    const std::size_t extra = n % static_cast<std::size_t>(parts);
    const auto i = static_cast<std::size_t>(index);
    const std::size_t begin = i * base + std::min(i, extra);
    return {begin, begin + base + (i < extra ? 1 : 0)};
}

std::uint64_t read_kernel(const std::uint64_t* data, std::size_t n) noexcept {
    std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += data[i];
        s1 += data[i + 1];
        s2 += data[i + 2];
        s3 += data[i + 3];
    }
    for (; i < n; ++i) {
        s0 += data[i];
    }
    return s0 ^ s1 ^ s2 ^ s3;
}

void triad_kernel(double* __restrict a, const double* __restrict b, const double* __restrict c, double scalar,
                  std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = b[i] + scalar * c[i];
    }
}

std::uint64_t choose_array_bytes(const RooflineOptions& options) {
    if (options.bandwidth_array_bytes > 0) {
        return options.bandwidth_array_bytes;
    }
    // Large enough to defeat the caches, small enough to keep three arrays well within memory.
    const std::uint64_t mib = 1024 * 1024;
    std::uint64_t bytes = std::max<std::uint64_t>(256 * mib, cpu_info().caches.l3 * 8);
    bytes = std::min<std::uint64_t>(bytes, 1024 * mib);
    const MemoryInfo memory = memory_info();
    if (memory.total_bytes > 0) {
        bytes = std::min<std::uint64_t>(bytes, memory.total_bytes / 16);
    }
    return std::max<std::uint64_t>(bytes, 16 * mib);
}

}  // namespace

Result<RooflineResult> measure_roofline(const RooflineOptions& options) {
    if (options.threads < 0 || options.seconds_per_measurement <= 0) {
        return errors::invalid_argument("invalid roofline options");
    }

    RooflineResult result;
    II_ASSIGN_OR_RETURN(const Isa isa, select_isa_from_environment(cpu_info()));
    result.isa = std::string(isa_name(isa));
    result.threads = options.threads > 0 ? options.threads : cpu_info().logical_cpus;

    ThreadPool pool(
        {.num_threads = result.threads, .spin_duration = std::chrono::microseconds(200), .name = "ii-roof"});
    const double seconds = options.seconds_per_measurement;

    // --- Peak FMA throughput ---
    const FmaKernel kernel = fma_kernel(isa);
    const std::int64_t iterations = calibrate_iterations(kernel, std::min(0.05, seconds / 4));
    const double flops_per_call = kernel.flops_per_iteration * static_cast<double>(iterations);

    result.f32_gflops_single = best_rate(seconds, [&] {
        const Stopwatch watch;
        consume(kernel.run(iterations, 1.0f, 0.0f));
        return flops_per_call / watch.elapsed_seconds() * 1e-9;
    });
    result.f32_gflops_multi = best_rate(seconds, [&] {
        const Stopwatch watch;
        pool.run_on_all([&](int) { consume(kernel.run(iterations, 1.0f, 0.0f)); });
        return flops_per_call * pool.concurrency() / watch.elapsed_seconds() * 1e-9;
    });

    // --- Memory bandwidth ---
    result.array_bytes = choose_array_bytes(options);
    const std::size_t n = static_cast<std::size_t>(result.array_bytes / sizeof(double));
    const std::size_t bytes = n * sizeof(double);
    II_ASSIGN_OR_RETURN(AlignedBuffer a_buffer, AlignedBuffer::allocate(bytes));
    II_ASSIGN_OR_RETURN(AlignedBuffer b_buffer, AlignedBuffer::allocate(bytes));
    II_ASSIGN_OR_RETURN(AlignedBuffer c_buffer, AlignedBuffer::allocate(bytes));
    auto* a = static_cast<double*>(a_buffer.data());
    auto* b = static_cast<double*>(b_buffer.data());
    auto* c = static_cast<double*>(c_buffer.data());
    const int threads = pool.concurrency();

    // First touch from the threads that will use the data (places pages on the right NUMA node).
    pool.run_on_all([&](int index) {
        const auto [begin, end] = partition(n, threads, index);
        for (std::size_t i = begin; i < end; ++i) {
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 2.0;
        }
    });

    result.read_bandwidth_gbs = best_rate(seconds, [&] {
        std::vector<std::uint64_t> partial(static_cast<std::size_t>(threads));
        const Stopwatch watch;
        pool.run_on_all([&](int index) {
            const auto [begin, end] = partition(n, threads, index);
            partial[static_cast<std::size_t>(index)] =
                read_kernel(reinterpret_cast<const std::uint64_t*>(b) + begin, end - begin);
        });
        const double elapsed = watch.elapsed_seconds();
        std::uint64_t checksum = 0;
        for (const std::uint64_t value : partial) {
            checksum ^= value;
        }
        consume(static_cast<double>(checksum));
        return static_cast<double>(bytes) / elapsed * 1e-9;
    });

    result.triad_bandwidth_gbs = best_rate(seconds, [&] {
        const Stopwatch watch;
        pool.run_on_all([&](int index) {
            const auto [begin, end] = partition(n, threads, index);
            triad_kernel(a + begin, b + begin, c + begin, 3.0, end - begin);
        });
        const double elapsed = watch.elapsed_seconds();
        consume(a[n / 2]);
        return 3.0 * static_cast<double>(bytes) / elapsed * 1e-9;
    });

    return result;
}

}  // namespace ii
