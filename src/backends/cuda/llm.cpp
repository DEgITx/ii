// SPDX-License-Identifier: MIT
// ii-cuda operations of language models: products with block-quantized and 16-bit weights, rope, slice_update
// (KV cache writes, in place), sdpa and dequantize.
#include "backends/cuda/gemm.h"

#include "ops/infer_util.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <format>
#include <optional>

namespace ii::cuda {

std::optional<WeightType> weight_type_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::kF32:
        return WeightType::kF32;
    case DType::kF16:
        return WeightType::kF16;
    case DType::kBF16:
        return WeightType::kBF16;
    case DType::kQ4_0:
        return WeightType::kQ4_0;
    case DType::kQ4_1:
        return WeightType::kQ4_1;
    case DType::kQ5_0:
        return WeightType::kQ5_0;
    case DType::kQ5_1:
        return WeightType::kQ5_1;
    case DType::kQ8_0:
        return WeightType::kQ8_0;
    case DType::kQ8_1:
        return WeightType::kQ8_1;
    case DType::kQ2_K:
        return WeightType::kQ2_K;
    case DType::kQ3_K:
        return WeightType::kQ3_K;
    case DType::kQ4_K:
        return WeightType::kQ4_K;
    case DType::kQ5_K:
        return WeightType::kQ5_K;
    case DType::kQ6_K:
        return WeightType::kQ6_K;
    case DType::kQ8_K:
        return WeightType::kQ8_K;
    default:
        return std::nullopt;
    }
}

namespace {

/// Weights decoded per GEMM of a prefill product are at most this large (the rest of the weights follow in further
/// products).
constexpr std::size_t kDecodedBytes = std::size_t{32} << 20;

int axis_of(std::int64_t axis, int rank) {
    return static_cast<int>(axis < 0 ? axis + rank : axis);
}

// --- Products with weights ---------------------------------------------------------------------------------------

/// y = x w^T (+ bias) for f32 x [..., k] and weights w [n, k] stored block-quantized, f16 or bf16 (`linear`, and
/// `matmul` with transpose_b). Few rows of x (decoding): weight_gemv reads each weight once and multiplies in f32.
/// More rows (prefill) run on cuBLASLt with f32 accumulation:
///   - block-quantized weights are decoded to f16 in slices of rows and multiply x rounded to f16 (still finer than
///     the 8-bit activations of ii-cpu's quantized products);
///   - 16-bit weights are used as stored; x is split into 16-bit parts that add up to it (split_rows: two f16 parts
///     with a power-of-two scale per row, or three bf16 parts) and the products of the parts are summed, which keeps
///     about f32 precision (the CPU multiplies these weights in f32).
class WeightProductKernel final : public Kernel {
public:
    bool can_record() const override { return true; }  // only enqueues work on the stream

    WeightProductKernel(const CudaDevice& device, DType weights) : gemm_(device), device_(device), weights_(weights) {
        compute_ = is_block_quantized(weights) ? DType::kF16 : weights;
        parts_ = weights == DType::kBF16 ? 3 : (weights == DType::kF16 ? 2 : 1);
        for (std::atomic<int>& split : splits_) {
            split.store(-1, std::memory_order_relaxed);
        }
    }

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape>) const override {
        const std::int64_t k = inputs[0][inputs[0].rank() - 1];
        const std::int64_t n = inputs[1][0];
        const std::int64_t m = k == 0 ? 0 : inputs[0].numel() / k;
        if (gemv(m, k)) {
            return 0;
        }
        std::size_t size = kBlasWorkspace + kAlign + region_bytes(compute_, parts_ * m * k);
        if (is_block_quantized(weights_)) {
            size += region_bytes(compute_, slice_rows(n, k) * k);
        } else if (weights_ == DType::kF16) {
            size += region_bytes(DType::kF32, m);  // the row scales
        }
        return size;
    }

    Status run(const KernelContext& context) const override {
        II_TRY(device_.activate());
        const TensorView& x = context.input(0);
        const TensorView& w = context.input(1);
        const TensorView& bias = context.input(2);
        const TensorView& y = context.output(0);
        const cudaStream_t stream = stream_of(context);
        const std::int64_t k = x.shape[x.rank() - 1];
        const std::int64_t n = w.shape[0];
        const std::int64_t m = k == 0 ? 0 : x.numel() / k;
        if (m == 0 || n == 0) {
            return ok_status();
        }
        const auto* bias_data = static_cast<const float*>(bias.present() ? bias.data : nullptr);
        if (gemv(m, k)) {
            auto launch = [&](int split) {
                return weight_gemv(y.as<float>(), x.as<const float>(), w.data, *weight_type_of(weights_), bias_data,
                                   static_cast<int>(m), n, k, stream, split);
            };
            return launched(launch(gemv_split(static_cast<int>(m), n, k, stream, launch)), "weight_gemv");
        }
        // After cuBLASLt's workspace: x in the compute type (its parts), then the decoded weights or the row scales.
        const std::span<std::byte> blas_workspace = context.workspace.first(kBlasWorkspace + kAlign);
        std::byte* x16 = align_up(context.workspace.data() + kBlasWorkspace + kAlign);
        std::byte* after = align_up(x16 + storage_size(compute_, parts_ * m * k));
        GemmCall call;
        call.dtype = DType::kF32;
        call.input = compute_;
        call.trans_b = true;
        call.m = m;
        call.k = k;
        call.lda = k;
        call.ldb = k;
        call.ldc = n;
        call.a = x16;
        if (!is_block_quantized(weights_)) {
            float* scales = weights_ == DType::kF16 ? reinterpret_cast<float*>(after) : nullptr;
            II_TRY(launched(split_rows(x16, *scalar_of(compute_), parts_, x.as<const float>(), scales, m, k, stream),
                            "split_rows"));
            call.n = n;
            call.b = w.data;
            call.c = y.data;
            for (int part = 0; part < parts_; ++part) {
                call.a = x16 + static_cast<std::size_t>(part) * storage_size(compute_, m * k);
                call.accumulate = part > 0;
                call.bias = part == 0 && scales == nullptr ? bias_data : nullptr;
                II_TRY(gemm_.run(call, blas_workspace, stream));
            }
            if (scales != nullptr) {
                II_TRY(launched(scale_rows(y.as<float>(), scales, bias_data, m, n, stream), "scale_rows"));
            }
            return ok_status();
        }
        II_TRY(launched(convert(x16, Scalar::kF16, x.data, Scalar::kF32, m * k, stream), "convert"));
        std::byte* decoded = after;
        const std::int64_t rows = slice_rows(n, k);
        for (std::int64_t first = 0; first < n; first += rows) {
            const std::int64_t count = std::min(rows, n - first);
            II_TRY(launched(
                dequantize_rows(decoded, Scalar::kF16, w.data, *weight_type_of(weights_), first, count, k, stream),
                "dequantize_rows"));
            call.n = count;
            call.b = decoded;
            call.c = y.as<float>() + first;
            call.bias = bias_data != nullptr ? bias_data + first : nullptr;
            II_TRY(gemm_.run(call, blas_workspace, stream));
        }
        return ok_status();
    }

private:
    bool gemv(std::int64_t m, std::int64_t k) const { return m <= kGemvRows && k % 32 == 0; }

    /// Warps per feature for `rows` rows of x: remembered, from the autotuning cache, measured (1, 2 or 4), or 0 (the
    /// launcher's choice from the shape).
    template <class Launch>
    int gemv_split(int rows, std::int64_t n, std::int64_t k, cudaStream_t stream, const Launch& launch) const {
        std::atomic<int>& known = splits_[static_cast<std::size_t>(rows)];
        if (const int split = known.load(std::memory_order_relaxed); split >= 0) {
            return split;
        }
        TuningCache* tuning = device_.tuning();
        cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
        if (tuning == nullptr || cudaStreamIsCapturing(stream, &capturing) != cudaSuccess ||
            capturing != cudaStreamCaptureStatusNone) {
            cudaGetLastError();
            return 0;
        }
        const std::string name = std::format("gemv {} rows {} {}x{}", to_string(weights_), rows, n, k);
        int split = 0;
        if (const std::optional<std::string> cached = tuning->find(name)) {
            split = std::atoi(cached->c_str());
        }
        if (split != 1 && split != 2 && split != 4) {
            static constexpr int kSplits[] = {1, 2, 4};
            const Result<int> best = fastest(3, stream, [&](int candidate) { return launch(kSplits[candidate]); });
            if (!best.ok()) {
                return 0;
            }
            split = kSplits[*best];
            tuning->store(name, std::to_string(split));
        }
        known.store(split, std::memory_order_relaxed);
        return split;
    }

    std::int64_t slice_rows(std::int64_t n, std::int64_t k) const {
        const auto bytes = static_cast<std::int64_t>(element_size(compute_)) * std::max<std::int64_t>(k, 1);
        // Whole slices of 8 rows keep the output columns of every product aligned.
        const std::int64_t rows = std::max<std::int64_t>(static_cast<std::int64_t>(kDecodedBytes) / bytes / 8 * 8, 8);
        return std::min(rows, n);
    }

    Gemm gemm_;
    const CudaDevice& device_;
    DType weights_;
    DType compute_;
    int parts_;  ///< 16-bit parts of x (1 for block-quantized weights)
    mutable std::array<std::atomic<int>, kGemvRows + 1> splits_{};  ///< per rows of x; -1: not chosen yet
};

/// Weight dtypes and layouts the weight product kernel handles.
bool weight_dtype(DType activations, DType weights) {
    return activations == DType::kF32 &&
           (is_block_quantized(weights) || weights == DType::kF16 || weights == DType::kBF16);
}

// --- rope -----------------------------------------------------------------------------------------------------------

Status check_rope(const ir::Node& node) {
    II_TRY(check_dtypes(node, {DType::kF32, DType::kF16, DType::kBF16}));
    for (int i = 0; i < 3; ++i) {
        if (node.input(i)->rank() > kMaxRank) {
            return errors::unsupported("rank above {}", kMaxRank);
        }
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_rope(const ir::Node& node, const CudaDevice& device) {
    const bool interleaved = node.attrs().get_bool("interleaved");
    return make_kernel(device, [interleaved](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& cos = context.input(1);
        RopeParams p;
        p.rank = x.rank();
        p.half = cos.shape[cos.rank() - 1];
        p.interleaved = interleaved;
        for (int d = 0; d < x.rank(); ++d) {
            p.dims[d] = x.shape[d];
        }
        // Strides of the table over x's dimensions, right-aligned (0 along broadcast dimensions).
        const int offset = x.rank() - cos.rank();
        std::int64_t stride = 1;
        for (int d = cos.rank() - 1; d >= 0; --d) {
            if (d + offset >= 0) {
                p.table_strides[d + offset] = cos.shape[d] == 1 ? 0 : stride;
            }
            stride *= cos.shape[d];
        }
        return launched(
            cuda::rope(context.output(0).data, x.data, cos.data, context.input(2).data, *scalar_of(x.dtype), p, stream),
            "rope");
    });
}

// --- slice_update ---------------------------------------------------------------------------------------------------

Status check_slice_update(const ir::Node& node) {
    const DType dtype = node.input(0)->dtype();
    if (!ops::is_plain(dtype) || node.input(0)->rank() > kMaxRank) {
        return errors::unsupported("'slice_update' of {} or rank above {}", to_string(dtype), kMaxRank);
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_slice_update(const ir::Node& node, const CudaDevice& device) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel(
        device,
        [axis_attr](const KernelContext& context, cudaStream_t stream) -> Status {
            const TensorView& x = context.input(0);
            const TensorView& update = context.input(1);
            const TensorView& start = context.input(2);
            const TensorView& y = context.output(0);
            if (y.data != x.data) {
                II_CUDA_TRY(cudaMemcpyAsync(y.data, x.data, x.nbytes(), cudaMemcpyDeviceToDevice, stream));
            }
            const int axis = axis_of(axis_attr, x.rank());
            SliceUpdateParams p;
            p.outer = 1;
            for (int d = 0; d < axis; ++d) {
                p.outer *= x.shape[d];
            }
            std::int64_t inner = 1;
            for (int d = axis + 1; d < x.rank(); ++d) {
                inner *= x.shape[d];
            }
            p.axis_size = x.shape[axis];
            p.length = update.shape[axis];
            p.inner_bytes = inner * static_cast<std::int64_t>(element_size(x.dtype));
            p.start = start.data;
            p.start64 = start.dtype == DType::kI64;
            p.error = error_of(context);
            return launched(cuda::slice_update(y.data, update.data, p, stream), "slice_update");
        },
        nullptr, 0);
}

// --- sdpa -----------------------------------------------------------------------------------------------------------

Status check_sdpa(const ir::Node& node) {
    const DType q = node.input(0)->dtype();
    const DType kv = node.input(1)->dtype();
    if (q != DType::kF32 && q != DType::kF16 && q != DType::kBF16) {
        return errors::unsupported("'sdpa' of {}", to_string(q));
    }
    if (kv != q && !(q == DType::kF32 && (kv == DType::kF16 || kv == DType::kBF16))) {
        return errors::unsupported("'sdpa' of {} queries with {} keys", to_string(q), to_string(kv));
    }
    for (int i = 1; i < 3; ++i) {
        const Dim& size = node.input(i)->shape()[3];
        if (!size.is_constant() || size.value() > kAttentionMaxHead) {
            return errors::unsupported("'sdpa' with heads of {} features (at most {})", size.to_string(),
                                       kAttentionMaxHead);
        }
    }
    if (const ir::Value* mask = node.input(3)) {
        if (mask->rank() > 4) {
            return errors::unsupported("'sdpa' mask of rank above 4");
        }
    }
    return ok_status();
}

/// The attention parameters for given shapes (pointers are set by the caller).
AttentionParams attention_params(const Shape& q, const Shape& k, const Shape& v, const Shape* mask, bool causal,
                                 double scale, std::int64_t window, double softcap) {
    AttentionParams p;
    p.batch = q[0];
    p.heads = q[1];
    p.sq = q[2];
    p.d = q[3];
    p.kv_heads = k[1];
    p.skv = k[2];
    p.dv = v[3];
    p.scale = static_cast<float>(scale != 0.0 ? scale : 1.0 / std::sqrt(static_cast<double>(p.d)));
    p.softcap = static_cast<float>(softcap);
    p.causal = causal;
    p.window = window;
    if (mask != nullptr) {
        const int offset = 4 - mask->rank();
        std::int64_t stride = 1;
        for (int d = mask->rank() - 1; d >= 0; --d) {
            p.mask_strides[d + offset] = (*mask)[d] == 1 ? 0 : stride;
            stride *= (*mask)[d];
        }
    }
    return p;
}

class AttentionKernel final : public Kernel {
public:
    bool can_record() const override { return true; }  // only enqueues work on the stream

    AttentionKernel(const CudaDevice& device, bool causal, double scale, std::int64_t window, double softcap)
        : device_(device), causal_(causal), scale_(scale), window_(window), softcap_(softcap) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape>) const override {
        const Shape* mask = inputs.size() > 3 && inputs[3].rank() > 0 ? &inputs[3] : nullptr;
        const AttentionParams p =
            attention_params(inputs[0], inputs[1], inputs[2], mask, causal_, scale_, window_, softcap_);
        const std::size_t bytes = attention_workspace(p, device_.sm_count());
        return bytes == 0 ? 0 : bytes + kAlign;
    }

    Status run(const KernelContext& context) const override {
        II_TRY(device_.activate());
        const TensorView& q = context.input(0);
        const TensorView& k = context.input(1);
        const TensorView& v = context.input(2);
        const TensorView& mask = context.input(3);
        const TensorView& kv_len = context.input(4);
        AttentionParams p = attention_params(q.shape, k.shape, v.shape, mask.present() ? &mask.shape : nullptr, causal_,
                                             scale_, window_, softcap_);
        if (mask.present()) {
            p.mask_kind = mask.dtype == DType::kBool ? 1 : 2;
        }
        if (kv_len.present()) {
            p.kv_len = kv_len.data;
            p.kv_len64 = kv_len.dtype == DType::kI64;
        }
        p.error = error_of(context);
        p.arch = device_.compute_capability();
        void* workspace = context.workspace.empty() ? nullptr : align_up(context.workspace.data());
        return launched(attention(context.output(0).data, q.data, k.data, v.data, mask.data, *scalar_of(q.dtype),
                                  *scalar_of(k.dtype), p, device_.sm_count(), workspace, stream_of(context)),
                        "attention");
    }

private:
    const CudaDevice& device_;
    bool causal_;
    double scale_;
    std::int64_t window_;
    double softcap_;
};

Result<std::unique_ptr<Kernel>> create_sdpa(const ir::Node& node, const CudaDevice& device) {
    return std::unique_ptr<Kernel>(
        std::make_unique<AttentionKernel>(device, node.attrs().get_bool("causal"), node.attrs().get_float("scale"),
                                          node.attrs().get_int("window"), node.attrs().get_float("softcap")));
}

// --- dequantize -----------------------------------------------------------------------------------------------------

Status check_dequantize(const ir::Node& node) {
    const DType dtype = node.input(0)->dtype();
    if (!is_block_quantized(dtype)) {
        return errors::unsupported("'dequantize' of {} (only block-quantized types)", to_string(dtype));
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_dequantize(const ir::Node& node, const CudaDevice& device) {
    (void)node;
    return make_kernel(device, [](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const std::int64_t k = x.rank() == 0 ? 1 : x.shape[x.rank() - 1];
        const std::int64_t rows = k == 0 ? 0 : x.numel() / k;
        return launched(
            dequantize_rows(context.output(0).data, Scalar::kF32, x.data, *weight_type_of(x.dtype), 0, rows, k, stream),
            "dequantize");
    });
}

}  // namespace

bool is_weight_product(const ir::Node& node) {
    if (node.op_name() == "linear") {
        return node.attrs().get_bool("transpose_w") && weight_dtype(node.input(0)->dtype(), node.input(1)->dtype());
    }
    if (node.op_name() == "matmul") {
        return node.attrs().get_bool("transpose_b") && !node.attrs().get_bool("transpose_a") &&
               node.input(1)->rank() == 2 && node.input(0)->rank() >= 1 &&
               weight_dtype(node.input(0)->dtype(), node.input(1)->dtype());
    }
    return false;
}

Status check_weight_product(const ir::Node& node) {
    if (const ir::Value* bias = node.input(2); node.op_name() == "linear" && bias != nullptr) {
        if (bias->dtype() != DType::kF32) {
            return errors::unsupported("'linear' with a {} bias", to_string(bias->dtype()));
        }
    }
    const Dim& k = node.input(1)->shape()[1];
    if (is_block_quantized(node.input(1)->dtype()) && (!k.is_constant() || k.value() % 32 != 0)) {
        return errors::unsupported("block-quantized weights with rows of {} elements", k.to_string());
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_weight_product(const ir::Node& node, const CudaDevice& device) {
    return std::unique_ptr<Kernel>(std::make_unique<WeightProductKernel>(device, node.input(1)->dtype()));
}

void register_llm_kernels(CudaBackend& backend) {
    backend.add_kernel("rope", create_rope, check_rope);
    backend.add_kernel("slice_update", create_slice_update, check_slice_update);
    backend.add_kernel("sdpa", create_sdpa, check_sdpa);
    backend.add_kernel("dequantize", create_dequantize, check_dequantize);
}

}  // namespace ii::cuda
