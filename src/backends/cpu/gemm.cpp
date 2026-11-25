// SPDX-License-Identifier: MIT
// ii-cpu matrix multiplication: matmul and linear.
//
// BLIS-style blocking: a work item is an MC x NC block of C (of one batch matrix). It walks the depth in KC steps:
// packs the MC x KC block of A into MR-row panels (per-thread scratch), takes the KC x NC block of B from panels
// packed once at compile time (constant weights) or packs it into scratch, and runs the ISA macro-kernel. Items are
// handed out dynamically to the threads, which balances hybrid CPUs. Accumulation is in f32; f16 / bf16 operands
// are widened while packing (B stays 16-bit in its panels and is widened in the micro-kernel).
#include "backends/cpu/cpu_backend.h"
#include "core/aligned_buffer.h"
#include "core/assert.h"
#include "ir/op_schema.h"

#include <algorithm>
#include <vector>

namespace ii::cpu {

namespace {

std::int64_t ceil_div(std::int64_t a, std::int64_t b) {
    return (a + b - 1) / b;
}

std::int64_t round_up(std::int64_t a, std::int64_t b) {
    return ceil_div(a, b) * b;
}

std::int64_t elem_bytes(Elem elem) {
    return elem == Elem::kF32 ? 4 : 2;
}

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
                   bool b_prepacked, bool c_is_f32) {
    const std::int64_t mr = config.table->gemm_mr;
    const std::int64_t nr = config.table->gemm_nr;
    GemmPlan plan;
    plan.m = m;
    plan.n = n;
    plan.k = k;
    plan.batches = batches;
    // One row panel (GEMV-like): A is tiny, so stream every B panel over the whole depth instead of in KC blocks.
    plan.kc = std::max<std::int64_t>(1, m <= mr ? k : std::min(config.gemm_kc, k));
    plan.mc = std::min(config.gemm_mc, round_up(std::max<std::int64_t>(m, 1), mr));
    // Runtime-packed B blocks are kept small (they are packed per item); prepacked B is read in place.
    plan.nc = std::min(b_prepacked ? config.gemm_nc : 16 * nr, round_up(std::max<std::int64_t>(n, 1), nr));

    // Enough items for all threads: shrink the column blocks, then the row blocks.
    const double flops =
        2.0 * static_cast<double>(batches) * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const int threads = flops < 2e5 ? 1 : config.threads;
    // Several items per thread, so that dynamic hand-out can balance cores of different speed (P/E cores,
    // big.LITTLE) and the last items do not leave most threads idle.
    const std::int64_t wanted = 4 * static_cast<std::int64_t>(threads);
    auto items = [&] { return batches * ceil_div(m, plan.mc) * ceil_div(n, plan.nc); };
    while (items() < wanted && plan.nc > 4 * nr) {
        plan.nc = round_up(plan.nc / 2, nr);
    }
    while (items() < wanted && plan.mc > 2 * mr) {
        plan.mc = round_up(plan.mc / 2, mr);
    }
    plan.m_blocks = ceil_div(m, plan.mc);
    plan.n_blocks = ceil_div(n, plan.nc);
    plan.slots = static_cast<int>(std::min<std::int64_t>(threads, std::max<std::int64_t>(plan.items(), 1)));
    plan.a_scratch = static_cast<std::size_t>(round_up(plan.mc, mr) * plan.kc * 4);
    plan.b_scratch = b_prepacked ? 0 : static_cast<std::size_t>(round_up(plan.nc, nr) * plan.kc * 4);
    plan.c_scratch = c_is_f32 ? 0 : static_cast<std::size_t>(plan.mc * plan.nc * 4);
    // Keep every scratch area 64-byte aligned.
    plan.a_scratch = static_cast<std::size_t>(round_up(static_cast<std::int64_t>(plan.a_scratch), 64));
    plan.b_scratch = static_cast<std::size_t>(round_up(static_cast<std::int64_t>(plan.b_scratch), 64));
    plan.c_scratch = static_cast<std::size_t>(round_up(static_cast<std::int64_t>(plan.c_scratch), 64));
    return plan;
}

/// One matrix operand: element (i, j) of batch matrix b at data + (offset(b) + i * rs + j * cs) elements.
struct Operand {
    const void* data = nullptr;
    Elem elem = Elem::kF32;
    std::int64_t rs = 0;
    std::int64_t cs = 0;
};

/// Arguments of a (batched) GEMM: C[b] = A[a_index(b)] * B[b_index(b)] + bias.
struct GemmCall {
    Operand a;
    Operand b;                         ///< unused when `packed_b` is set
    const void* packed_b = nullptr;    ///< prepacked panels of all B matrices, or nullptr
    std::int64_t packed_b_stride = 0;  ///< elements per prepacked B matrix
    void* c = nullptr;
    Elem c_elem = Elem::kF32;
    const float* bias = nullptr;  ///< f32, n values, or nullptr
    // Batch mapping (broadcasting): offsets in elements of each batch matrix.
    const std::int64_t* a_offsets = nullptr;
    const std::int64_t* b_offsets = nullptr;  ///< element offsets, or matrix indices for prepacked B
};

void run_gemm(const KernelContext& context, const CpuConfig& config, const GemmPlan& plan, const GemmCall& call) {
    const KernelTable& table = *config.table;
    const std::int64_t nr = table.gemm_nr;
    const std::int64_t m = plan.m;
    const std::int64_t n = plan.n;
    const std::int64_t k = plan.k;
    const std::int64_t c_size = call.c_elem == Elem::kF32 ? 4 : 2;
    auto* workspace = context.workspace.data();

    parallel_slots(context, plan.slots, plan.items(), [&](int slot, std::int64_t item) {
        std::byte* scratch = workspace + static_cast<std::size_t>(slot) * plan.slot_bytes();
        auto* packed_a = reinterpret_cast<float*>(scratch);
        void* packed_b_scratch = scratch + plan.a_scratch;
        auto* c_scratch = reinterpret_cast<float*>(scratch + plan.a_scratch + plan.b_scratch);

        const std::int64_t ib = item % plan.m_blocks;
        const std::int64_t jb = (item / plan.m_blocks) % plan.n_blocks;
        const std::int64_t batch = item / (plan.m_blocks * plan.n_blocks);
        const std::int64_t i0 = ib * plan.mc;
        const std::int64_t j0 = jb * plan.nc;
        const std::int64_t mc = std::min(plan.mc, m - i0);
        const std::int64_t nc = std::min(plan.nc, n - j0);

        float* c_block;
        std::int64_t ldc;
        if (call.c_elem == Elem::kF32) {
            c_block = static_cast<float*>(call.c) + (batch * m + i0) * n + j0;
            ldc = n;
        } else {
            c_block = c_scratch;
            ldc = nc;
        }
        const float* bias = call.bias != nullptr ? call.bias + j0 : nullptr;

        if (k == 0) {
            for (std::int64_t i = 0; i < mc; ++i) {
                for (std::int64_t j = 0; j < nc; ++j) {
                    c_block[i * ldc + j] = bias != nullptr ? bias[j] : 0.0f;
                }
            }
        }
        const std::int64_t a_size = elem_bytes(call.a.elem);
        const auto* a_base = static_cast<const std::uint8_t*>(call.a.data) + call.a_offsets[batch] * a_size;
        for (std::int64_t p0 = 0; p0 < k; p0 += plan.kc) {
            const std::int64_t kc = std::min(plan.kc, k - p0);
            table.gemm_pack_a(a_base + (i0 * call.a.rs + p0 * call.a.cs) * a_size, call.a.elem, call.a.rs, call.a.cs,
                              mc, kc, packed_a);
            GemmMacroArgs args{};
            args.m = mc;
            args.n = nc;
            args.k = kc;
            args.a = packed_a;
            if (call.packed_b != nullptr) {
                const std::int64_t b_size = elem_bytes(call.b.elem);
                const auto* matrix = static_cast<const std::uint8_t*>(call.packed_b) +
                                     call.b_offsets[batch] * call.packed_b_stride * b_size;
                args.b = matrix + ((j0 / nr) * k * nr + p0 * nr) * b_size;
                args.b_panel_stride = k * nr;
            } else {
                const std::int64_t b_size = elem_bytes(call.b.elem);
                const auto* b_base = static_cast<const std::uint8_t*>(call.b.data) + call.b_offsets[batch] * b_size;
                table.gemm_pack_b(b_base + (p0 * call.b.rs + j0 * call.b.cs) * b_size, call.b.elem, call.b.rs,
                                  call.b.cs, kc, nc, packed_b_scratch);
                args.b = packed_b_scratch;
                args.b_panel_stride = kc * nr;
            }
            args.b_elem = call.b.elem;
            args.c = c_block;
            args.ldc = ldc;
            args.accumulate = p0 > 0;
            args.bias = p0 + kc == k ? bias : nullptr;
            table.gemm_macro(args);
        }
        if (call.c_elem != Elem::kF32) {
            auto* out = static_cast<std::uint8_t*>(call.c);
            for (std::int64_t i = 0; i < mc; ++i) {
                table.convert(c_block + i * ldc, Elem::kF32, out + ((batch * m + i0 + i) * n + j0) * c_size,
                              call.c_elem, nc);
            }
        }
    });
}

// --------------------------------------------------------------------------------------------------------------------
// Shapes

/// Geometry of a matmul / linear node for concrete shapes.
struct MatmulShape {
    std::int64_t m = 0, n = 0, k = 0;
    Operand a;
    Operand b;
    std::int64_t batches = 1;
    // Batch broadcasting: output batch dims and the element strides of A and B along them (0 = broadcast).
    int batch_rank = 0;
    std::int64_t batch_sizes[8] = {};
    std::int64_t a_batch_strides[8] = {};
    std::int64_t b_batch_strides[8] = {};   ///< in elements
    std::int64_t b_matrix_strides[8] = {};  ///< in matrices (for prepacked B)
};

MatmulShape matmul_shape(const Shape& sa, const Shape& sb, bool transpose_a, bool transpose_b) {
    MatmulShape s;
    const int ra = sa.rank();
    const int rb = sb.rank();
    // A as [batch..., rows, cols]
    const std::int64_t a_rows = ra == 1 ? 1 : sa[ra - 2];
    const std::int64_t a_cols = sa[ra - 1];
    if (ra >= 2 && transpose_a) {
        s.m = a_cols;
        s.k = a_rows;
        s.a.rs = 1;
        s.a.cs = a_cols;
    } else {
        s.m = a_rows;
        s.k = a_cols;
        s.a.rs = a_cols;
        s.a.cs = 1;
    }
    const std::int64_t b_rows = rb == 1 ? sb[0] : sb[rb - 2];
    const std::int64_t b_cols = rb == 1 ? 1 : sb[rb - 1];
    if (rb >= 2 && transpose_b) {
        s.n = b_rows;
        s.b.rs = 1;
        s.b.cs = b_cols;
    } else {
        s.n = b_cols;
        s.b.rs = b_cols;
        s.b.cs = 1;
    }
    const int ba = std::max(ra - 2, 0);
    const int bb = std::max(rb - 2, 0);
    s.batch_rank = std::max(ba, bb);
    std::int64_t a_stride = a_rows * a_cols;
    std::int64_t b_stride = b_rows * b_cols;
    std::int64_t b_matrices = 1;
    for (int d = s.batch_rank - 1; d >= 0; --d) {
        const int da = d - (s.batch_rank - ba);
        const int db = d - (s.batch_rank - bb);
        const std::int64_t size_a = da >= 0 ? sa[da] : 1;
        const std::int64_t size_b = db >= 0 ? sb[db] : 1;
        const std::int64_t size = std::max(size_a, size_b);
        s.batch_sizes[d] = size;
        s.a_batch_strides[d] = size_a == 1 ? 0 : a_stride;
        s.b_batch_strides[d] = size_b == 1 ? 0 : b_stride;
        s.b_matrix_strides[d] = size_b == 1 ? 0 : b_matrices;
        a_stride *= size_a;
        b_stride *= size_b;
        b_matrices *= size_b;
        s.batches *= size;
    }
    return s;
}

class GemmKernel final : public Kernel {
public:
    GemmKernel(const CpuConfig& config, Elem elem, bool linear, bool transpose_a, bool transpose_b, bool has_bias)
        : config_(config),
          elem_(elem),
          linear_(linear),
          transpose_a_(transpose_a),
          transpose_b_(transpose_b),
          has_bias_(has_bias) {}

    /// Packs a constant B (rank >= 2; all its batch matrices) at compile time.
    Status prepack(const Tensor& b) {
        II_ASSIGN_OR_RETURN(const Tensor contiguous, b.contiguous());
        const Shape& shape = contiguous.shape();
        const int rank = shape.rank();
        const std::int64_t rows = shape[rank - 2];
        const std::int64_t cols = shape[rank - 1];
        const std::int64_t k = transpose_b_ ? cols : rows;
        const std::int64_t n = transpose_b_ ? rows : cols;
        const std::int64_t rs = transpose_b_ ? 1 : cols;
        const std::int64_t cs = transpose_b_ ? cols : 1;
        const std::int64_t matrices = rows * cols == 0 ? 0 : shape.numel() / (rows * cols);
        const std::int64_t nr = config_.table->gemm_nr;
        const std::int64_t size = elem_bytes(elem_);
        packed_stride_ = round_up(n, nr) * k;
        II_ASSIGN_OR_RETURN(packed_b_, AlignedBuffer::allocate(static_cast<std::size_t>(
                                           std::max<std::int64_t>(packed_stride_ * matrices * size, 64))));
        for (std::int64_t i = 0; i < matrices; ++i) {
            config_.table->gemm_pack_b(static_cast<const std::uint8_t*>(contiguous.data()) + i * rows * cols * size,
                                       elem_, rs, cs, k, n,
                                       static_cast<std::uint8_t*>(packed_b_.data()) + i * packed_stride_ * size);
        }
        prepacked_ = true;
        return ok_status();
    }

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        (void)outputs;
        const MatmulShape s = geometry(inputs[0], inputs[1]);
        const GemmPlan plan = plan_gemm(config_, s.batches, s.m, s.n, s.k, prepacked_, elem_ == Elem::kF32);
        std::size_t size = static_cast<std::size_t>(plan.slots) * plan.slot_bytes();
        size += batch_tables_bytes(s.batches);
        if (has_bias_ && elem_ != Elem::kF32) {
            size += static_cast<std::size_t>(round_up(s.n * 4, 64));
        }
        return size;
    }

    Status run(const KernelContext& context) const override {
        const TensorView& a = context.input(0);
        const TensorView& b = context.input(1);
        const TensorView& y = context.output(0);
        const MatmulShape s = geometry(a.shape, b.shape);
        const GemmPlan plan = plan_gemm(config_, s.batches, s.m, s.n, s.k, prepacked_, elem_ == Elem::kF32);
        if (y.numel() == 0) {
            return ok_status();
        }

        // Workspace layout: [slots scratch][a offsets][b offsets][bias f32]
        std::byte* cursor = context.workspace.data() + static_cast<std::size_t>(plan.slots) * plan.slot_bytes();
        auto* a_offsets = reinterpret_cast<std::int64_t*>(cursor);
        auto* b_offsets = a_offsets + s.batches;
        cursor += batch_tables_bytes(s.batches);
        for (std::int64_t batch = 0; batch < s.batches; ++batch) {
            std::int64_t rest = batch;
            std::int64_t a_offset = 0;
            std::int64_t b_offset = 0;
            for (int d = s.batch_rank - 1; d >= 0; --d) {
                const std::int64_t index = rest % s.batch_sizes[d];
                rest /= s.batch_sizes[d];
                a_offset += index * s.a_batch_strides[d];
                b_offset += index * (prepacked_ ? s.b_matrix_strides[d] : s.b_batch_strides[d]);
            }
            a_offsets[batch] = a_offset;
            b_offsets[batch] = b_offset;
        }

        const float* bias = nullptr;
        if (has_bias_) {
            const TensorView& bias_view = context.input(2);
            if (elem_ == Elem::kF32) {
                bias = bias_view.as<float>();
            } else {
                auto* converted = reinterpret_cast<float*>(cursor);
                config_.table->convert(bias_view.data, elem_, converted, Elem::kF32, s.n);
                bias = converted;
            }
        }

        GemmCall call{};
        call.a = s.a;
        call.a.data = a.data;
        call.a.elem = elem_;
        call.b = s.b;
        call.b.data = b.data;
        call.b.elem = elem_;
        call.packed_b = prepacked_ ? packed_b_.data() : nullptr;
        call.packed_b_stride = packed_stride_;
        call.c = y.data;
        call.c_elem = elem_;
        call.bias = bias;
        call.a_offsets = a_offsets;
        call.b_offsets = b_offsets;
        run_gemm(context, config_, plan, call);
        return ok_status();
    }

private:
    MatmulShape geometry(const Shape& a, const Shape& b) const {
        if (!linear_) {
            return matmul_shape(a, b, transpose_a_, transpose_b_);
        }
        // linear: x [..., in] as a [rows, in] matrix times the 2-D weight.
        return matmul_shape(Shape{numel_without_last(a), a[a.rank() - 1]}, b, false, transpose_b_);
    }

    static std::int64_t numel_without_last(const Shape& shape) {
        std::int64_t n = 1;
        for (int d = 0; d + 1 < shape.rank(); ++d) {
            n *= shape[d];
        }
        return n;
    }

    static std::size_t batch_tables_bytes(std::int64_t batches) {
        return static_cast<std::size_t>(round_up(2 * batches * 8, 64));
    }

    CpuConfig config_;
    Elem elem_;
    bool linear_;
    bool transpose_a_;
    bool transpose_b_;
    bool has_bias_;
    bool prepacked_ = false;
    std::int64_t packed_stride_ = 0;
    AlignedBuffer packed_b_;
};

Status check_gemm(const ir::Node& node) {
    const DType dtype = node.output()->dtype();
    if (dtype != DType::kF32 && dtype != DType::kF16 && dtype != DType::kBF16) {
        return errors::unsupported("'{}' is accelerated for f32, f16 and bf16 only", node.op_name());
    }
    for (const ir::Value* input : node.inputs()) {
        if (input != nullptr && input->dtype() != dtype) {
            return errors::unsupported("'{}' with mixed or block-quantized operands", node.op_name());
        }
    }
    if (node.input(0)->rank() > 8 || node.input(1)->rank() > 8) {
        return errors::unsupported("rank above 8");
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_gemm(const ir::Node& node, const CpuConfig& config) {
    const bool linear = node.op_name() == "linear";
    const Elem elem = *elem_of(node.output()->dtype());
    const bool transpose_a = !linear && node.attrs().get_bool("transpose_a");
    const bool transpose_b = linear ? node.attrs().get_bool("transpose_w") : node.attrs().get_bool("transpose_b");
    const bool has_bias = linear && node.input(2) != nullptr;
    auto kernel = std::make_unique<GemmKernel>(config, elem, linear, transpose_a, transpose_b, has_bias);
    const ir::Value* b = node.input(1);
    if (b->kind() == ir::ValueKind::kConstant && b->rank() >= 2) {
        II_TRY(kernel->prepack(b->data()));
    }
    return std::unique_ptr<Kernel>(std::move(kernel));
}

}  // namespace

void register_gemm_kernels(CpuBackend& backend) {
    backend.add_kernel("matmul", create_gemm, check_gemm);
    backend.add_kernel("linear", create_gemm, check_gemm);
}

}  // namespace ii::cpu
