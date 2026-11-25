// SPDX-License-Identifier: MIT
// Quantized product kernels of ii-cpu, single thread: bytes of weights per second for one activation row (decode)
// and multiply-accumulates per second for four rows (prompt processing), row kernels against panel kernels.
#include "backends/cpu/cpu_backend.h"
#include "core/isa.h"
#include "tensor/quant.h"

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

namespace {

using namespace ii;
using namespace ii::cpu;

struct Fixture {
    DType dtype;
    QType qtype;
    std::int64_t n = 2048;
    std::int64_t k = 2048;
    std::vector<std::uint8_t> w;
    std::vector<std::uint8_t> act;
    std::int64_t a_bytes = 0;
    AlignedBuffer panels;

    Fixture(DType d, QType q, const KernelTable& table) : dtype(d), qtype(q) {
        std::mt19937 rng(1);
        std::normal_distribution<float> normal;
        std::vector<float> wf(static_cast<std::size_t>(n * k));
        for (float& v : wf) {
            v = normal(rng);
        }
        w.resize(storage_size(dtype, n * k));
        quant::quantize(dtype, wf.data(), w.data(), n * k);
        const bool k_layout = dtype_info(dtype).block_size == 256;
        a_bytes = k_layout ? k / 256 * 292 : k / 32 * 40;
        act.resize(static_cast<std::size_t>(a_bytes * 4));
        std::vector<float> x(static_cast<std::size_t>(k));
        for (int r = 0; r < 4; ++r) {
            for (float& v : x) {
                v = normal(rng);
            }
            (k_layout ? table.quantize_q8k : table.quantize_q8x32)(x.data(), act.data() + r * a_bytes, k);
        }
        if (table.qgemm8[static_cast<int>(qtype)] != nullptr) {
            panels = std::move(*pack_quantized_panels(qtype, dtype, w.data(), n, k));
        }
    }
};

void run(benchmark::State& state, DType dtype, QType qtype, bool panel, int rows) {
    const Result<Isa> isa = select_isa_from_environment(cpu_info());
    const KernelTable& table = kernel_table(*isa);
    Fixture f(dtype, qtype, table);
    if (panel && table.qgemm8[static_cast<int>(qtype)] == nullptr) {
        state.SkipWithError("no panel kernel in this ISA variant");
        return;
    }
    std::vector<float> out(static_cast<std::size_t>(f.n * 4));
    const std::size_t w_row = storage_size(dtype, f.k);
    const std::int64_t stride = quantized_panel_stride(qtype, f.k);
    for (auto _ : state) {
        if (panel) {
            for (std::int64_t c = 0; c < f.n; c += 8) {
                table.qgemm8[static_cast<int>(qtype)](
                    f.k, static_cast<const std::uint8_t*>(f.panels.data()) + c / 8 * stride, f.act.data(), f.a_bytes,
                    rows, out.data() + c, f.n);
            }
        } else {
            for (std::int64_t c = 0; c < f.n; ++c) {
                table.qdot[static_cast<int>(qtype)](f.k, f.w.data() + c * w_row, f.act.data(), f.a_bytes, rows,
                                                    out.data() + c, f.n);
            }
        }
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(f.w.size()));
    state.counters["GMAC/s"] = benchmark::Counter(static_cast<double>(state.iterations()) * static_cast<double>(f.n) *
                                                      static_cast<double>(f.k) * rows / 1e9,
                                                  benchmark::Counter::kIsRate);
}

#define II_QBENCH(type, qtype)                                                                                         \
    void BM_Quant_##type##_rows(benchmark::State& state) {                                                             \
        run(state, DType::k##type, QType::k##qtype, false, static_cast<int>(state.range(0)));                          \
    }                                                                                                                  \
    void BM_Quant_##type##_panels(benchmark::State& state) {                                                           \
        run(state, DType::k##type, QType::k##qtype, true, static_cast<int>(state.range(0)));                           \
    }                                                                                                                  \
    BENCHMARK(BM_Quant_##type##_rows)->Arg(1)->Arg(4);                                                                 \
    BENCHMARK(BM_Quant_##type##_panels)->Arg(1)->Arg(4);

II_QBENCH(Q4_0, Q4_0)
II_QBENCH(Q8_0, Q8_0)
II_QBENCH(Q4_K, Q4_K)
II_QBENCH(Q6_K, Q6_K)

}  // namespace
