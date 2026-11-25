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
    for (int i = 0; i < 3; ++i) {
        if (node.input(i)->dtype() != DType::kF32) {
            return errors::unsupported("'sdpa' is accelerated for f32 only");
        }
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
    AttentionKernel(const CpuConfig& config, bool causal, double scale)
        : table_(config.table), threads_(config.threads), causal_(causal), scale_(scale) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        (void)outputs;
        const Shape& k = inputs[1];
        const auto kv_len = static_cast<std::size_t>(k[2]);
        return static_cast<std::size_t>(threads_) * ((kv_len * 4 + 63) / 64 * 64);
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
        const std::size_t row_bytes = (static_cast<std::size_t>(skv) * 4 + 63) / 64 * 64;

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
        const int slots = rows * skv * d < 65536 ? 1 : threads_;
        parallel_slots(context, slots, items, [&](int slot, std::int64_t item) {
            auto* scratch =
                reinterpret_cast<float*>(context.workspace.data() + static_cast<std::size_t>(slot) * row_bytes);
            const std::int64_t end = std::min(rows, (item + 1) * kRowsPerItem);
            for (std::int64_t row = item * kRowsPerItem; row < end; ++row) {
                const std::int64_t i = row % sq;
                const std::int64_t h = (row / sq) % hq;
                const std::int64_t b = row / (sq * hq);
                const std::int64_t hk = h / group;
                AttentionRowArgs args{};
                args.q = q.as<float>() + row * d;
                args.k = k.as<float>() + (b * hkv + hk) * skv * d;
                args.k_stride = d;
                args.v = v.as<float>() + (b * hkv + hk) * skv * dv;
                args.v_stride = dv;
                args.d = d;
                args.dv = dv;
                args.kv_len = skv;
                args.kv_limit = causal_ ? std::clamp<std::int64_t>(i + skv - sq + 1, 0, skv) : skv;
                args.scale = scale;
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
    const KernelTable* table_;
    int threads_;
    bool causal_;
    double scale_;
};

Result<std::unique_ptr<Kernel>> create_sdpa(const ir::Node& node, const CpuConfig& config) {
    return std::unique_ptr<Kernel>(
        std::make_unique<AttentionKernel>(config, node.attrs().get_bool("causal"), node.attrs().get_float("scale")));
}

}  // namespace

void register_attention_kernels(CpuBackend& backend) {
    backend.add_kernel("sdpa", create_sdpa, check_sdpa);
}

}  // namespace ii::cpu
