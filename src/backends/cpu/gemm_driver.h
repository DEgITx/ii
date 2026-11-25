// SPDX-License-Identifier: MIT
/// @file
/// The blocked GEMM driver of ii-cpu, shared by matmul / linear (gemm.cpp) and convolution (conv.cpp).
///
/// BLIS-style blocking: a work item is an MC x NC block of C (of one batch matrix). It walks the depth in KC steps:
/// packs the MC x KC block of A into MR-row panels (per-thread scratch), takes the KC x NC block of B from panels
/// packed once at compile time (constant weights), packs it into scratch, or has a callback pack it (implicit im2col
/// for convolution), and runs the ISA macro-kernel. Items are handed out dynamically to the threads, which balances
/// hybrid CPUs.
#pragma once

#include <ii/function_ref.h>

#include "backends/cpu/cpu_backend.h"

#include <cstddef>
#include <cstdint>

namespace ii::cpu {

/// Block sizes of one GEMM call.
struct GemmPlan {
    std::int64_t m = 0, n = 0, k = 0;
    std::int64_t batches = 1;
    std::int64_t mc = 0, kc = 0, nc = 0;
    std::int64_t m_blocks = 0, n_blocks = 0;
    int slots = 1;
    std::size_t a_scratch = 0;  ///< bytes per slot
    std::size_t b_scratch = 0;  ///< bytes per slot (B packed at run time)
    std::size_t c_scratch = 0;  ///< bytes per slot (f32 accumulator for 16-bit outputs)

    std::size_t slot_bytes() const noexcept { return a_scratch + b_scratch + c_scratch; }
    std::int64_t items() const noexcept { return batches * m_blocks * n_blocks; }
};

GemmPlan plan_gemm(const CpuConfig& config, std::int64_t batches, std::int64_t m, std::int64_t n, std::int64_t k,
                   bool b_prepacked, bool c_is_f32);

/// The depth block (KC) plan_gemm chooses for an M x K matrix A (it does not depend on N).
std::int64_t gemm_depth_block(const CpuConfig& config, std::int64_t m, std::int64_t k);

/// One matrix operand: element (i, j) of batch matrix b at data + (offset(b) + i * rs + j * cs) elements.
struct Operand {
    const void* data = nullptr;
    Elem elem = Elem::kF32;
    std::int64_t rs = 0;
    std::int64_t cs = 0;
};

/// Packs rows [p0, p0 + kc) and columns [j0, j0 + nc) of B of batch matrix `batch` into GEMM panels (NR columns
/// each, the depth along a panel, missing columns zero) at `out`.
using PackBFn = FunctionRef<void(std::int64_t batch, std::int64_t p0, std::int64_t kc, std::int64_t j0, std::int64_t nc,
                                 void* out)>;

/// Called for every finished block of an f32 C: rows [i0, i0 + rows) and columns [j0, j0 + cols) of batch matrix
/// `batch`, at `c` with row stride `ldc`.
using EpilogueFn = FunctionRef<void(std::int64_t batch, std::int64_t i0, std::int64_t rows, std::int64_t j0,
                                    std::int64_t cols, float* c, std::int64_t ldc)>;

/// Arguments of a (batched) GEMM: C[b] = A[a_index(b)] * B[b_index(b)] + bias. Batch matrices of C are contiguous
/// (batch b starts at element b * m * n).
struct GemmCall {
    Operand a;                         ///< unused when `packed_a` is set
    const float* packed_a = nullptr;   ///< A packed by `prepack_a` (all batch matrices), or nullptr
    std::int64_t packed_a_stride = 0;  ///< floats per prepacked A matrix (a_offsets are then matrix indices)
    Operand b;                         ///< unused when `packed_b` or `pack_b` is set
    const void* packed_b = nullptr;    ///< prepacked panels of all B matrices, or nullptr
    std::int64_t packed_b_stride = 0;  ///< elements per prepacked B matrix
    const PackBFn* pack_b = nullptr;   ///< packs B blocks (implicit B, e.g. im2col), or nullptr
    void* c = nullptr;
    Elem c_elem = Elem::kF32;
    const float* bias = nullptr;           ///< f32, one value per column (n), or nullptr
    const EpilogueFn* epilogue = nullptr;  ///< applied to finished f32 blocks, or nullptr
    const float* row_bias = nullptr;       ///< f32, one value per row of each batch matrix (m), or nullptr
    std::int64_t row_bias_stride = 0;      ///< elements between the row biases of consecutive batches
    // Batch mapping (broadcasting): offsets in elements of each batch matrix.
    const std::int64_t* a_offsets = nullptr;
    const std::int64_t* b_offsets = nullptr;  ///< element offsets, or matrix indices for prepacked B
};

void run_gemm(const KernelContext& context, const CpuConfig& config, const GemmPlan& plan, const GemmCall& call);

/// Floats of an M x K matrix A packed by `prepack_a` for depth blocks of `kc`.
std::int64_t packed_a_size(const KernelTable& table, std::int64_t m, std::int64_t k);

/// Packs a constant f32 matrix A (M x K, element (i, p) at a[i * rs + p * cs]) once, in the layout run_gemm reads
/// for any row block: depth blocks of `kc`, each holding all MR-row panels. A plan uses it when its `kc` matches.
void prepack_a(const KernelTable& table, const float* a, std::int64_t rs, std::int64_t cs, std::int64_t m,
               std::int64_t k, std::int64_t kc, float* out);

}  // namespace ii::cpu
