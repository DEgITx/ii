// SPDX-License-Identifier: MIT
#include "backends/cpu/cpu_backend.h"

#include <ii/system_info.h>

#include "core/assert.h"
#include "ir/op_schema.h"

#include <algorithm>
#include <atomic>

namespace ii::cpu {

const KernelTable& kernel_table(Isa isa) noexcept {
    switch (isa) {
#if defined(II_HAVE_ISA_AVX2)
    case Isa::kAvx2:
        return avx2::kernel_table();
#endif
#if defined(II_HAVE_ISA_AVX512)
    case Isa::kAvx512:
        return avx512::kernel_table();
#endif
#if defined(II_HAVE_ISA_NEON)
    case Isa::kNeon:
        return neon::kernel_table();
#endif
#if defined(II_HAVE_ISA_AVX_VNNI)
    case Isa::kAvxVnni:
        return avx_vnni::kernel_table();
#endif
#if defined(II_HAVE_ISA_NEON_DOTPROD)
    case Isa::kNeonDotprod:
        return neon_dotprod::kernel_table();
#endif
    default:
        return generic::kernel_table();
    }
}

CpuConfig make_config(Isa isa, int threads) {
    CpuConfig config;
    config.isa = isa;
    config.table = &kernel_table(isa);
    config.threads = std::max(threads, 1);

    const CpuCaches& caches = cpu_info().caches;
    const std::int64_t l1 = caches.l1d != 0 ? static_cast<std::int64_t>(caches.l1d) : 32 * 1024;
    const std::int64_t l2 = caches.l2 != 0 ? static_cast<std::int64_t>(caches.l2) : 256 * 1024;
    const std::int64_t mr = config.table->gemm_mr;
    const std::int64_t nr = config.table->gemm_nr;
    // KC: one MR x KC panel of A and one KC x NR panel of B fit in about 3/4 of L1.
    config.gemm_kc = std::clamp<std::int64_t>(l1 * 3 / 4 / (4 * (mr + nr)) / 16 * 16, 128, 512);
    // MC: the packed MC x KC block of A takes about half of L2.
    config.gemm_mc = std::clamp<std::int64_t>(l2 / 2 / (4 * config.gemm_kc) / mr * mr, 4 * mr, 64 * mr);
    // NC: columns of one work item (the B block streams from L3 / memory, prepacked).
    config.gemm_nc = nr * std::max<std::int64_t>(1, 1024 / nr);
    return config;
}

namespace {

/// Single-precision operations per cycle and core of an ISA variant (FMA units x lanes x 2).
double flops_per_cycle(Isa isa) noexcept {
    switch (isa) {
    case Isa::kAvx2:
    case Isa::kAvxVnni:
        return 32;
    case Isa::kAvx512:
        return 64;
    case Isa::kNeon:
    case Isa::kNeonDotprod:
        return 16;
    case Isa::kGeneric:
        break;
    }
    return 8;
}

}  // namespace

CpuBackend::CpuBackend(CpuConfig config)
    : config_(config), info_(backend::host_device_info(config.threads, flops_per_cycle(config.isa))) {
    register_elementwise_kernels(*this);
    register_gemm_kernels(*this);
    register_reduction_kernels(*this);
    register_attention_kernels(*this);
    register_movement_kernels(*this);
    register_conv_kernels(*this);
    register_llm_kernels(*this);
}

Status CpuBackend::supports(const ir::Node& node) const {
    const auto it = kernels_.find(node.op_name());
    if (it == kernels_.end()) {
        return errors::unsupported("{}: no kernel for '{}'", name(), node.op_name());
    }
    if (it->second.check != nullptr) {
        if (Status status = it->second.check(node); !status.ok()) {
            return errors::unsupported("{}: {}", name(), status.message());
        }
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> CpuBackend::create_kernel(const ir::Node& node) const {
    II_TRY(supports(node));
    return kernels_.at(node.op_name()).factory(node, config_);
}

void CpuBackend::add_kernel(std::string_view op, Factory factory, SupportCheck check) {
    const ir::OpSchema& schema = ir::builtin_op(op);
    II_CHECK(!kernels_.contains(schema.name), "kernel for '{}' registered twice", op);
    kernels_.emplace(schema.name, Entry{factory, check});
}

Result<std::unique_ptr<backend::Backend>> create_cpu_backend(std::optional<std::string_view> isa, int threads) {
    Result<Isa> selected = isa ? select_isa(cpu_info(), *isa) : select_isa_from_environment(cpu_info());
    if (!selected.ok()) {
        Status status = std::move(selected).status();
        return status.add_context("ii-cpu");
    }
    return std::unique_ptr<backend::Backend>(std::make_unique<CpuBackend>(make_config(*selected, threads)));
}

std::optional<Elem> elem_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::kF32:
        return Elem::kF32;
    case DType::kF16:
        return Elem::kF16;
    case DType::kBF16:
        return Elem::kBF16;
    case DType::kBool:
        return Elem::kBool;
    default:
        return std::nullopt;
    }
}

void parallel_range(const KernelContext& context, std::int64_t n, double cost_per_item,
                    FunctionRef<void(std::int64_t begin, std::int64_t end)> fn) {
    // Below about 32K units of work a parallel loop costs more than it saves.
    constexpr double kMinWorkPerTask = 32768.0;
    if (n <= 0) {
        return;
    }
    const double total = static_cast<double>(n) * std::max(cost_per_item, 1.0);
    if (context.executor == nullptr || context.executor->concurrency() <= 1 || total < 2 * kMinWorkPerTask) {
        fn(0, n);
        return;
    }
    const auto grain =
        std::max<std::int64_t>(1, static_cast<std::int64_t>(kMinWorkPerTask / std::max(cost_per_item, 1.0)));
    context.executor->parallel_for(n, grain, fn);
}

void parallel_slots(const KernelContext& context, int slots, std::int64_t items,
                    FunctionRef<void(int slot, std::int64_t item)> fn) {
    if (items <= 0) {
        return;
    }
    int workers = std::max(1, slots);
    if (context.executor != nullptr) {
        workers = std::min(workers, context.executor->concurrency());
    }
    workers = static_cast<int>(std::min<std::int64_t>(workers, items));
    if (workers <= 1 || context.executor == nullptr) {
        for (std::int64_t item = 0; item < items; ++item) {
            fn(0, item);
        }
        return;
    }
    std::atomic<std::int64_t> next{0};
    context.executor->parallel_for(workers, 1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t slot = begin; slot < end; ++slot) {
            for (;;) {
                const std::int64_t item = next.fetch_add(1, std::memory_order_relaxed);
                if (item >= items) {
                    break;
                }
                fn(static_cast<int>(slot), item);
            }
        }
    });
}

void parallel_chunks(const KernelContext& context, int slots, std::int64_t items, double cost_per_item,
                     FunctionRef<void(int slot, std::int64_t begin, std::int64_t end)> fn) {
    if (items <= 0) {
        return;
    }
    const double total = static_cast<double>(items) * cost_per_item;
    int workers = std::max(1, slots);
    if (context.executor != nullptr) {
        workers = std::min(workers, context.executor->concurrency());
    }
    if (total < 32768.0 || workers <= 1) {
        fn(0, 0, items);
        return;
    }
    const std::int64_t chunks = std::min<std::int64_t>(items, 4 * static_cast<std::int64_t>(workers));
    const std::int64_t chunk = (items + chunks - 1) / chunks;
    parallel_slots(context, workers, (items + chunk - 1) / chunk,
                   [&](int slot, std::int64_t c) { fn(slot, c * chunk, std::min(items, (c + 1) * chunk)); });
}

}  // namespace ii::cpu
