// SPDX-License-Identifier: MIT
/// @file
/// ii-cpu: the optimized CPU backend.
///
/// Kernels are SIMD code compiled once per instruction-set variant (generic, avx2, avx512, neon); the backend uses
/// the variant selected at creation (`II_CPU_ISA` / `ContextOptions::cpu_isa`, else the best one the CPU supports).
/// Operations, dtypes or attributes it does not accelerate are left to ii-ref by the planner.
#pragma once

#include <ii/function_ref.h>

#include "backend/backend.h"
#include "backends/cpu/kernel_table.h"
#include "core/aligned_buffer.h"
#include "core/isa.h"

#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ii::cpu {

/// Name of the backend.
inline constexpr std::string_view kBackendName = "ii-cpu";

using backend::Kernel;
using backend::KernelContext;
using backend::TensorView;

/// Machine-dependent parameters shared by all kernels of a backend instance.
struct CpuConfig {
    Isa isa = Isa::kGeneric;
    const KernelTable* table = nullptr;
    int threads = 1;  ///< concurrency of the context's executor (sizes per-thread scratch memory)
    // GEMM cache blocking (see gemm.cpp).
    std::int64_t gemm_mc = 0;  ///< rows of A per packed block (L2)
    std::int64_t gemm_kc = 0;  ///< depth per packed block (L1)
    std::int64_t gemm_nc = 0;  ///< columns per work item
};

/// The kernel table of an ISA variant (which must be compiled in).
const KernelTable& kernel_table(Isa isa) noexcept;

/// The configuration ii-cpu uses for `isa` on this machine with `threads` threads.
CpuConfig make_config(Isa isa, int threads);

class CpuBackend final : public backend::Backend {
public:
    using Factory = Result<std::unique_ptr<Kernel>> (*)(const ir::Node& node, const CpuConfig& config);
    using SupportCheck = Status (*)(const ir::Node& node);

    explicit CpuBackend(CpuConfig config);

    std::string_view name() const noexcept override { return kBackendName; }
    Status supports(const ir::Node& node) const override;
    Result<std::unique_ptr<Kernel>> create_kernel(const ir::Node& node) const override;

    void add_kernel(std::string_view op, Factory factory, SupportCheck check = nullptr);
    const CpuConfig& config() const noexcept { return config_; }

private:
    struct Entry {
        Factory factory;
        SupportCheck check;
    };
    CpuConfig config_;
    std::unordered_map<std::string_view, Entry> kernels_;
};

/// Creates ii-cpu for `isa` (nullopt: II_CPU_ISA or the best usable variant) sized for an executor with `threads`
/// threads.
Result<std::unique_ptr<backend::Backend>> create_cpu_backend(std::optional<std::string_view> isa, int threads);

// --------------------------------------------------------------------------------------------------------------------
// Helpers for kernel drivers

/// The kernel element type of a dtype (f32, f16, bf16, bool), or nullopt.
std::optional<Elem> elem_of(DType dtype) noexcept;

/// Runs fn(begin, end) over [0, n); in parallel when the total work (`n * cost_per_item`, roughly in bytes or
/// flops) makes it worthwhile.
void parallel_range(const KernelContext& context, std::int64_t n, double cost_per_item,
                    FunctionRef<void(std::int64_t begin, std::int64_t end)> fn);

/// Runs fn(slot, item) for every item in [0, items) on at most `slots` concurrent workers; items are handed out
/// dynamically (good for hybrid CPUs). A slot never runs two items at once, so per-slot scratch memory is safe.
void parallel_slots(const KernelContext& context, int slots, std::int64_t items,
                    FunctionRef<void(int slot, std::int64_t item)> fn);

/// Runs fn(slot, begin, end) over chunks of [0, items) on at most `slots` workers (dynamic hand-out, ~4 chunks per
/// worker; serial when `items * cost_per_item` is small). Like parallel_slots, for many small items.
void parallel_chunks(const KernelContext& context, int slots, std::int64_t items, double cost_per_item,
                     FunctionRef<void(int slot, std::int64_t begin, std::int64_t end)> fn);

/// An elementwise program compiled for the kernel table's interpreter.
struct CompiledElementwise {
    std::vector<EwInstr> code;
    int result = 0;
    std::vector<Elem> input_elems;
    Elem out_elem = Elem::kF32;
};

/// Compiles a graph of elementwise operations (one output) into an interpreter program (elementwise.cpp).
Result<CompiledElementwise> compile_elementwise_graph(const ir::Graph& program);

/// A kernel implemented by a callable.
template <class Fn>
class LambdaKernel final : public Kernel {
public:
    explicit LambdaKernel(Fn fn) : fn_(std::move(fn)) {}
    Status run(const KernelContext& context) const override { return fn_(context); }

private:
    Fn fn_;
};

template <class Fn>
std::unique_ptr<Kernel> make_kernel(Fn fn) {
    return std::make_unique<LambdaKernel<Fn>>(std::move(fn));
}

/// Repacks rows [0, rows) (a multiple of 8) of block-quantized weights [N, K] into the panels of the GEMM kernels
/// (`QGemm8Fn`); each panel takes `quantized_panel_stride(type, k)` bytes.
Result<AlignedBuffer> pack_quantized_panels(QType type, DType dtype, const void* w, std::int64_t rows, std::int64_t k);
std::int64_t quantized_panel_stride(QType type, std::int64_t k);

/// linear / matmul with block-quantized weights (qmatmul.cpp); used by the GEMM registration.
Status check_quantized_matmul(const ir::Node& node);
Result<std::unique_ptr<Kernel>> create_quantized_matmul(const ir::Node& node, const CpuConfig& config);

// Registration of the kernel groups (one file each).
void register_elementwise_kernels(CpuBackend& backend);
void register_gemm_kernels(CpuBackend& backend);
void register_reduction_kernels(CpuBackend& backend);
void register_attention_kernels(CpuBackend& backend);
void register_movement_kernels(CpuBackend& backend);
void register_conv_kernels(CpuBackend& backend);
void register_llm_kernels(CpuBackend& backend);

}  // namespace ii::cpu
