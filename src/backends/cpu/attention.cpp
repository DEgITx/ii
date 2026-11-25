// SPDX-License-Identifier: MIT
// ii-cpu scaled dot-product attention (f32): one query row at a time — scores, softmax and the weighted sum of
// values stay in a per-thread row of scratch memory, so the [Sq, Skv] attention matrix is never materialized.
#include "backends/cpu/cpu_backend.h"
#include "ir/op_schema.h"

#include <algorithm>
#include <cmath>

namespace ii::cpu {

namespace {

constexpr std::int64_t kRowsPerItem = 8;

Status check_sdpa(const ir::Node& node) {
    if (node.input(0)->dtype() != DType::kF32) {
        return errors::unsupported("'sdpa' is accelerated for f32 queries only");
    }
    const DType kv = node.input(1)->dtype();
    if ((kv != DType::kF32 && kv != DType::kF16) || node.input(2)->dtype() != kv) {
        return errors::unsupported("'sdpa' is accelerated for f32 or f16 keys and values");
    }
    if (const ir::Value* mask = node.input(3)) {
        if (mask->dtype() != DType::kF32 && mask->dtype() != DType::kBool) {
            return errors::unsupported("'sdpa' mask must be bool or f32");
        }
        if (mask->rank() > 4) {
            return errors::unsupported("'sdpa' mask of rank above 4");
        }
    }
    return ok_status();
}

class AttentionKernel final : public Kernel {
public:
    AttentionKernel(const CpuConfig& config, bool causal, double scale, std::int64_t window, double softcap)
        : table_(config.table),
          threads_(config.threads),
          causal_(causal),
          scale_(scale),
          window_(window),
          softcap_(softcap) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        (void)outputs;
        return static_cast<std::size_t>(threads_) * row_bytes(inputs[1][2]);
    }

    Status run(const KernelContext& context) const override {
        const TensorView& q = context.input(0);
        const TensorView& k = context.input(1);
        const TensorView& v = context.input(2);
        const TensorView& mask = context.input(3);
        const TensorView& y = context.output(0);
        const std::int64_t batch = q.shape[0];
        const std::int64_t hq = q.shape[1];
        const std::int64_t sq = q.shape[2];
        const std::int64_t d = q.shape[3];
        const std::int64_t hkv = k.shape[1];
        const std::int64_t skv = k.shape[2];
        const std::int64_t dv = v.shape[3];
        const std::int64_t group = hq / hkv;
        const float scale = static_cast<float>(scale_ != 0.0 ? scale_ : 1.0 / std::sqrt(static_cast<double>(d)));
        const std::size_t scratch_bytes = row_bytes(skv);
        // Keys in use: kv_len (a preallocated KV cache) or all.
        std::int64_t used = skv;
        if (context.inputs.size() > 4 && context.input(4).present()) {
            const TensorView& kv_len = context.input(4);
            used = kv_len.dtype == DType::kI32 ? *kv_len.as<const std::int32_t>() : *kv_len.as<const std::int64_t>();
            if (used < 0 || used > skv) {
                return errors::out_of_range("'sdpa' kv_len {} is outside [0, {}]", used, skv);
            }
        }
        const Elem kv_elem = k.dtype == DType::kF16 ? Elem::kF16 : Elem::kF32;
        const std::int64_t kv_size = kv_elem == Elem::kF16 ? 2 : 4;

        // Mask strides over [B, H, Sq, Skv] (right-aligned, 0 along broadcast dimensions).
        std::int64_t mask_strides[4] = {0, 0, 0, 0};
        MaskKind mask_kind = MaskKind::kNone;
        if (mask.present()) {
            mask_kind = mask.dtype == DType::kBool ? MaskKind::kBool : MaskKind::kAdditive;
            const int offset = 4 - mask.rank();
            std::int64_t stride = 1;
            for (int dim = mask.rank() - 1; dim >= 0; --dim) {
                mask_strides[dim + offset] = mask.shape[dim] == 1 ? 0 : stride;
                stride *= mask.shape[dim];
            }
        }
        const std::int64_t mask_elem = mask_kind == MaskKind::kBool ? 1 : 4;

        const std::int64_t rows = batch * hq * sq;
        const std::int64_t items = (rows + kRowsPerItem - 1) / kRowsPerItem;
        const int slots = rows * used * d < 65536 ? 1 : threads_;
        parallel_slots(context, slots, items, [&](int slot, std::int64_t item) {
            auto* scratch =
                reinterpret_cast<float*>(context.workspace.data() + static_cast<std::size_t>(slot) * scratch_bytes);
            const std::int64_t end = std::min(rows, (item + 1) * kRowsPerItem);
            for (std::int64_t row = item * kRowsPerItem; row < end; ++row) {
                const std::int64_t i = row % sq;
                const std::int64_t h = (row / sq) % hq;
                const std::int64_t b = row / (sq * hq);
                const std::int64_t hk = h / group;
                AttentionRowArgs args{};
                args.q = q.as<float>() + row * d;
                args.k = static_cast<const std::uint8_t*>(k.data) + (b * hkv + hk) * skv * d * kv_size;
                args.k_stride = d;
                args.v = static_cast<const std::uint8_t*>(v.data) + (b * hkv + hk) * skv * dv * kv_size;
                args.v_stride = dv;
                args.kv_elem = kv_elem;
                args.d = d;
                args.dv = dv;
                args.kv_len = used;
                const std::int64_t position = i + used - sq;  // of the query among the keys in use
                args.kv_limit = causal_ ? std::clamp<std::int64_t>(position + 1, 0, used) : used;
                args.kv_start = window_ > 0 ? std::clamp<std::int64_t>(position - window_ + 1, 0, used) : 0;
                args.scale = scale;
                args.softcap = static_cast<float>(softcap_);
                args.mask_kind = mask_kind;
                if (mask_kind != MaskKind::kNone) {
                    const std::int64_t offset = b * mask_strides[0] + h * mask_strides[1] + i * mask_strides[2];
                    args.mask = static_cast<const std::uint8_t*>(mask.data) + offset * mask_elem;
                    args.mask_stride = mask_strides[3];
                }
                args.scratch = scratch;
                args.out = y.as<float>() + row * dv;
                table_->attention_row(args);
            }
        });
        return ok_status();
    }

private:
    /// Per-thread scratch: scores of every key plus room for the output row.
    static std::size_t row_bytes(std::int64_t keys) { return (static_cast<std::size_t>(keys) * 4 + 63) / 64 * 64 + 64; }

    const KernelTable* table_;
    int threads_;
    bool causal_;
    double scale_;
    std::int64_t window_;
    double softcap_;
};

Result<std::unique_ptr<Kernel>> create_sdpa(const ir::Node& node, const CpuConfig& config) {
    return std::unique_ptr<Kernel>(
        std::make_unique<AttentionKernel>(config, node.attrs().get_bool("causal"), node.attrs().get_float("scale"),
                                          node.attrs().get_int("window"), node.attrs().get_float("softcap")));
}

}  // namespace

void register_attention_kernels(CpuBackend& backend) {
    backend.add_kernel("sdpa", create_sdpa, check_sdpa);
}

}  // namespace ii::cpu
