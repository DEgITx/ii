// SPDX-License-Identifier: MIT
// ii-cuda matrix products with cuBLASLt: matmul (batched, with broadcasting), linear (bias in the epilogue) and
// conv (1x1 convolutions directly, others through im2col; grouped convolutions with a direct kernel).
#include "backends/cuda/gemm.h"

#include "ops/fused.h"
#include "ops/infer_util.h"

#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <mutex>
#include <vector>

namespace ii::cuda {

namespace {

cudaDataType cuda_type(DType dtype) {
    switch (dtype) {
    case DType::kF16:
        return CUDA_R_16F;
    case DType::kBF16:
        return CUDA_R_16BF;
    default:
        return CUDA_R_32F;
    }
}

/// The largest power of two up to 256 dividing the address and the batch stride in bytes.
std::uint32_t alignment(const void* pointer, std::int64_t stride_bytes) {
    std::uintptr_t bits = reinterpret_cast<std::uintptr_t>(pointer) | kAlign;
    if (stride_bytes != 0) {
        bits |= static_cast<std::uintptr_t>(stride_bytes);
    }
    return static_cast<std::uint32_t>(bits & (~bits + 1));
}

}  // namespace

Result<int> fastest(int count, cudaStream_t stream, const std::function<cudaError_t(int)>& run) {
    struct Events {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        ~Events() {
            if (start != nullptr) {
                cudaEventDestroy(start);
            }
            if (stop != nullptr) {
                cudaEventDestroy(stop);
            }
        }
    } events;
    II_CUDA_TRY(cudaEventCreate(&events.start));
    II_CUDA_TRY(cudaEventCreate(&events.stop));
    // Average time of `repeats` runs of a candidate in milliseconds; negative when it fails.
    auto measure = [&](int candidate, int repeats) -> Result<float> {
        II_CUDA_TRY(cudaEventRecord(events.start, stream));
        bool ok = true;
        for (int repeat = 0; repeat < repeats && ok; ++repeat) {
            ok = run(candidate) == cudaSuccess;
        }
        II_CUDA_TRY(cudaEventRecord(events.stop, stream));
        II_CUDA_TRY(cudaEventSynchronize(events.stop));
        float ms = 0;
        II_CUDA_TRY(cudaEventElapsedTime(&ms, events.start, events.stop));
        cudaGetLastError();
        return ok ? ms / static_cast<float>(repeats) : -1.0f;
    };
    // Repetitions per measurement: about 0.2 ms of work (short kernels are noisy), between 3 and 64.
    std::vector<int> repeats(static_cast<std::size_t>(count), 0);
    for (int candidate = 0; candidate < count; ++candidate) {
        II_ASSIGN_OR_RETURN(const float once, measure(candidate, 1));  // also the warm-up
        if (once >= 0) {
            repeats[static_cast<std::size_t>(candidate)] =
                std::clamp(static_cast<int>(0.2f / std::max(once, 1e-4f)), 3, 64);
        }
    }
    // Two rounds over the candidates in turn (slow drifts of the clock affect all of them); the best average wins.
    std::vector<float> best_ms(static_cast<std::size_t>(count), -1.0f);
    for (int round = 0; round < 2; ++round) {
        for (int candidate = 0; candidate < count; ++candidate) {
            const int n = repeats[static_cast<std::size_t>(candidate)];
            if (n == 0) {
                continue;
            }
            II_ASSIGN_OR_RETURN(const float ms, measure(candidate, n));
            float& best = best_ms[static_cast<std::size_t>(candidate)];
            if (ms >= 0 && (best < 0 || ms < best)) {
                best = ms;
            }
        }
    }
    int best = -1;
    for (int candidate = 0; candidate < count; ++candidate) {
        const float ms = best_ms[static_cast<std::size_t>(candidate)];
        if (ms >= 0 && (best < 0 || ms < best_ms[static_cast<std::size_t>(best)])) {
            best = candidate;
        }
    }
    if (best < 0) {
        return errors::internal("autotuning: no candidate ran");
    }
    return best;
}

Status Gemm::run(const GemmCall& call, std::span<std::byte> workspace, cudaStream_t stream) const {
    if (call.m == 0 || call.n == 0 || call.batch == 0) {
        return ok_status();
    }
    const Libraries& l = device_.libraries();
    const DType input = call.input == DType::kInvalid ? call.dtype : call.input;
    const std::size_t element = element_size(call.dtype);
    const std::size_t input_element = element_size(input);
    if (call.k == 0) {
        II_CUDA_TRY(
            cudaMemsetAsync(call.c, 0, static_cast<std::size_t>(call.batch * call.m * call.ldc) * element, stream));
        return ok_status();
    }
    const cudaDataType type = cuda_type(input);
    const cublasComputeType_t compute =
        input == DType::kF32 && device_.options().tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    // Column-major view: C^T [n, m] = op(B)^T [n, k] * op(A)^T [k, m]; X is B's memory, Y is A's.
    cublasLtMatmulDescOpaque_t desc_storage{};
    auto* desc = reinterpret_cast<cublasLtMatmulDesc_t>(&desc_storage);
    II_TRY(blas(l.matmul_desc_init(desc, sizeof(desc_storage), compute, CUDA_R_32F), "matmul descriptor"));
    const cublasOperation_t op_x = call.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_y = call.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    II_TRY(blas(l.matmul_desc_set(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_x, sizeof(op_x)), "TRANSA"));
    II_TRY(blas(l.matmul_desc_set(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_y, sizeof(op_y)), "TRANSB"));
    if (call.bias != nullptr) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        II_TRY(blas(l.matmul_desc_set(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)), "EPILOGUE"));
        II_TRY(blas(l.matmul_desc_set(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &call.bias, sizeof(call.bias)),
                    "BIAS_POINTER"));
        const cudaDataType bias_type = cuda_type(call.dtype);
        II_TRY(blas(l.matmul_desc_set(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)),
                    "BIAS_DATA_TYPE"));
    }
    cublasLtMatrixLayoutOpaque_t x_storage{};
    cublasLtMatrixLayoutOpaque_t y_storage{};
    cublasLtMatrixLayoutOpaque_t d_storage{};
    auto* x = reinterpret_cast<cublasLtMatrixLayout_t>(&x_storage);
    auto* y = reinterpret_cast<cublasLtMatrixLayout_t>(&y_storage);
    auto* d = reinterpret_cast<cublasLtMatrixLayout_t>(&d_storage);
    II_TRY(layout(x, type, call.trans_b ? call.k : call.n, call.trans_b ? call.n : call.k, call.ldb, call.batch,
                  call.stride_b));
    II_TRY(layout(y, type, call.trans_a ? call.m : call.k, call.trans_a ? call.k : call.m, call.lda, call.batch,
                  call.stride_a));
    II_TRY(layout(d, cuda_type(call.dtype), call.n, call.m, call.ldc, call.batch, call.stride_c));

    std::byte* work = align_up(workspace.data());
    const std::size_t work_size =
        workspace.size() >= kBlasWorkspace + kAlign ? kBlasWorkspace : 0;  // too small: algorithms without it
    Key key{input,
            call.accumulate,
            call.trans_a,
            call.trans_b,
            call.bias != nullptr,
            call.m,
            call.n,
            call.k,
            call.lda,
            call.ldb,
            call.ldc,
            call.batch,
            call.stride_a,
            call.stride_b,
            call.stride_c,
            alignment(call.b, call.stride_b * static_cast<std::int64_t>(input_element)),
            alignment(call.a, call.stride_a * static_cast<std::int64_t>(input_element)),
            alignment(call.c, call.stride_c * static_cast<std::int64_t>(element)),
            work_size};
    const float alpha = 1;
    const float beta = call.accumulate ? 1.0f : 0.0f;
    auto launch = [&](const cublasLtMatmulAlgo_t& algo) {
        return l.matmul(device_.blas(), desc, &alpha, call.b, x, call.a, y, &beta, call.c, d, call.c, d, &algo, work,
                        work_size, stream);
    };
    II_ASSIGN_OR_RETURN(const cublasLtMatmulAlgo_t algo, algorithm(key, desc, x, y, d, launch, stream));
    II_TRY(blas(launch(algo), "cublasLtMatmul"));
    return ok_status();
}

Status Gemm::blas(cublasStatus_t status, std::string_view what) const {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return ok_status();
    }
    return errors::internal("cuBLASLt {}: {}", what, device_.libraries().lt_status_string(status));
}

Status Gemm::layout(cublasLtMatrixLayout_t layout, cudaDataType type, std::int64_t rows, std::int64_t cols,
                    std::int64_t ld, std::int64_t batch, std::int64_t stride) const {
    const Libraries& l = device_.libraries();
    II_TRY(blas(l.layout_init(layout, sizeof(cublasLtMatrixLayoutOpaque_t), type, static_cast<std::uint64_t>(rows),
                              static_cast<std::uint64_t>(cols), ld),
                "matrix layout"));
    if (batch > 1) {
        const auto count = static_cast<std::int32_t>(batch);
        II_TRY(blas(l.layout_set(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)), "batch"));
        II_TRY(blas(l.layout_set(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)),
                    "batch stride"));
    }
    return ok_status();
}

Result<cublasLtMatmulAlgo_t> Gemm::algorithm(const Key& key, cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t x,
                                             cublasLtMatrixLayout_t y, cublasLtMatrixLayout_t d,
                                             const std::function<cublasStatus_t(const cublasLtMatmulAlgo_t&)>& launch,
                                             cudaStream_t stream) const {
    {
        const std::lock_guard lock(mutex_);
        for (const auto& [cached, algo] : algos_) {
            if (cached == key) {
                return algo;
            }
        }
    }
    auto remember = [&](const cublasLtMatmulAlgo_t& algo) {
        const std::lock_guard lock(mutex_);
        algos_.emplace_back(key, algo);
        return algo;
    };
    TuningCache* tuning = device_.tuning();
    const std::string name =
        std::format("gemm {} {}{}{} {}x{}x{} ld {} {} {} batch {} stride {} {} {} align {} {} {} work {} tf32 {}",
                    to_string(key.input), key.trans_a ? 't' : 'n', key.trans_b ? 't' : 'n', key.bias ? 'b' : '-', key.m,
                    key.n, key.k, key.lda, key.ldb, key.ldc, key.batch, key.stride_a, key.stride_b, key.stride_c,
                    key.align_x, key.align_y, key.align_d, key.workspace, device_.options().tf32 ? 1 : 0);
    if (tuning != nullptr) {
        if (const std::optional<std::string> cached = tuning->find(name)) {
            cublasLtMatmulAlgo_t algo{};
            if (from_hex(*cached, &algo, sizeof(algo))) {
                return remember(algo);
            }
        }
    }
    const Libraries& l = device_.libraries();
    cublasLtMatmulPreference_t preference = nullptr;
    II_TRY(blas(l.preference_create(&preference), "preference"));
    struct Destroy {
        const Libraries& l;
        cublasLtMatmulPreference_t preference;
        ~Destroy() { l.preference_destroy(preference); }
    } destroy{l, preference};
    const std::uint64_t workspace = key.workspace;
    II_TRY(blas(l.preference_set(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)),
                "workspace preference"));
    II_TRY(blas(
        l.preference_set(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &key.align_x, sizeof(key.align_x)),
        "alignment preference"));
    II_TRY(blas(
        l.preference_set(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &key.align_y, sizeof(key.align_y)),
        "alignment preference"));
    II_TRY(blas(
        l.preference_set(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &key.align_d, sizeof(key.align_d)),
        "alignment preference"));
    II_TRY(blas(
        l.preference_set(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &key.align_d, sizeof(key.align_d)),
        "alignment preference"));
    constexpr int kCandidates = 8;
    std::array<cublasLtMatmulHeuristicResult_t, kCandidates> results{};
    int found = 0;
    II_TRY(blas(l.algo_heuristic(device_.blas(), desc, x, y, d, d, preference, tuning != nullptr ? kCandidates : 1,
                                 results.data(), &found),
                "algorithm heuristic"));
    if (found == 0) {
        return errors::unsupported("cuBLASLt has no algorithm for a {}x{}x{} product", key.m, key.n, key.k);
    }
    // Products that add to C are not measured (repeated runs would change C): cuBLASLt's first choice.
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (tuning == nullptr || found == 1 || key.accumulate || cudaStreamIsCapturing(stream, &capturing) != cudaSuccess ||
        capturing != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return remember(results[0].algo);  // cuBLASLt's first choice (no measurement while recording)
    }
    // Autotuning: run every candidate on the real arguments (the output is written again afterwards).
    II_ASSIGN_OR_RETURN(const int best, fastest(found, stream, [&](int candidate) {
                            return launch(results[static_cast<std::size_t>(candidate)].algo) == CUBLAS_STATUS_SUCCESS
                                       ? cudaSuccess
                                       : cudaErrorUnknown;
                        }));
    const cublasLtMatmulAlgo_t& algo = results[static_cast<std::size_t>(best)].algo;
    tuning->store(name, to_hex(&algo, sizeof(algo)));
    return remember(algo);
}

namespace {

Status check_float(const ir::Node& node) {
    return check_dtypes(node, {DType::kF32, DType::kF16, DType::kBF16});
}

// --- matmul -----------------------------------------------------------------------------------------------------

class MatmulKernel final : public Kernel {
public:
    bool can_record() const override { return true; }  // only enqueues work on the stream

    MatmulKernel(const CudaDevice& device, bool widen, bool trans_a, bool trans_b)
        : gemm_(device), device_(device), widen_(widen), trans_a_(trans_a), trans_b_(trans_b) {}

    std::size_t widened_bytes(const Shape& b) const {
        return widen_ ? static_cast<std::size_t>(b.numel()) * sizeof(float) + kAlign : 0;
    }

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape>) const override {
        return kBlasWorkspace + kAlign + widened_bytes(inputs[1]);
    }

    Status run(const KernelContext& context) const override {
        II_TRY(device_.activate());
        const TensorView& a = context.input(0);
        TensorView b = context.input(1);
        const TensorView& out = context.output(0);
        std::span<std::byte> workspace = context.workspace;
        if (widen_) {
            // f16 / bf16 weights with f32 activations: widen to f32 first.
            auto* wide = reinterpret_cast<float*>(align_up(workspace.data() + kBlasWorkspace + kAlign));
            II_TRY(launched(widen(wide, b.data, *scalar_of(b.dtype), b.numel(), stream_of(context)), "widen"));
            b.data = wide;
            workspace = workspace.first(kBlasWorkspace + kAlign);
        }
        GemmCall call;
        call.dtype = out.dtype;
        const int ra = a.rank();
        const int rb = b.rank();
        if (ra == 1) {
            call.m = 1;
            call.k = a.shape[0];
            call.lda = call.k;
        } else {
            call.trans_a = trans_a_;
            call.m = trans_a_ ? a.shape[ra - 1] : a.shape[ra - 2];
            call.k = trans_a_ ? a.shape[ra - 2] : a.shape[ra - 1];
            call.lda = a.shape[ra - 1];
        }
        if (rb == 1) {
            call.n = 1;
            call.ldb = 1;
        } else {
            call.trans_b = trans_b_;
            call.n = trans_b_ ? b.shape[rb - 2] : b.shape[rb - 1];
            call.ldb = b.shape[rb - 1];
        }
        call.ldc = call.n;
        // Batch dimensions: the output's leading dimensions, broadcast from those of a and b.
        const int batch_rank = std::max(ra - 2, 0) > std::max(rb - 2, 0) ? ra - 2 : std::max(rb - 2, 0);
        std::int64_t batch = 1;
        std::int64_t a_batch = 1;
        std::int64_t b_batch = 1;
        for (int d = 0; d < batch_rank; ++d) {
            batch *= out.shape[d];
        }
        for (int d = 0; d + 2 < ra; ++d) {
            a_batch *= a.shape[d];
        }
        for (int d = 0; d + 2 < rb; ++d) {
            b_batch *= b.shape[d];
        }
        const std::int64_t a_matrix = call.m * call.k;
        const std::int64_t b_matrix = call.k * call.n;
        const std::int64_t c_matrix = call.m * call.n;
        const std::size_t element = element_size(out.dtype);
        call.a = a.data;
        call.b = b.data;
        call.c = out.data;
        if (batch == 1 || (b_batch == 1 && a_batch == batch && !call.trans_a && ra >= 2)) {
            call.m *= batch;  // [batch * m, k] x [k, n]
            return gemm_.run(call, workspace, stream_of(context));
        }
        if ((a_batch == batch || a_batch == 1) && (b_batch == batch || b_batch == 1)) {
            call.batch = batch;
            call.stride_a = a_batch == 1 ? 0 : a_matrix;
            call.stride_b = b_batch == 1 ? 0 : b_matrix;
            call.stride_c = c_matrix;
            return gemm_.run(call, workspace, stream_of(context));
        }
        // General broadcasting of the batch dimensions: one product per batch element.
        std::array<std::int64_t, kMaxRank> a_strides{};
        std::array<std::int64_t, kMaxRank> b_strides{};
        {
            std::int64_t stride = a_matrix;
            for (int d = ra - 3; d >= 0; --d) {
                a_strides[static_cast<std::size_t>(batch_rank - (ra - 2) + d)] = a.shape[d] == 1 ? 0 : stride;
                stride *= a.shape[d];
            }
            stride = b_matrix;
            for (int d = rb - 3; d >= 0; --d) {
                b_strides[static_cast<std::size_t>(batch_rank - (rb - 2) + d)] = b.shape[d] == 1 ? 0 : stride;
                stride *= b.shape[d];
            }
        }
        for (std::int64_t i = 0; i < batch; ++i) {
            std::int64_t rest = i;
            std::int64_t a_offset = 0;
            std::int64_t b_offset = 0;
            for (int d = batch_rank - 1; d >= 0; --d) {
                const std::int64_t index = rest % out.shape[d];
                rest /= out.shape[d];
                a_offset += index * a_strides[static_cast<std::size_t>(d)];
                b_offset += index * b_strides[static_cast<std::size_t>(d)];
            }
            GemmCall one = call;
            one.a = static_cast<const std::byte*>(a.data) + a_offset * static_cast<std::int64_t>(element);
            one.b = static_cast<const std::byte*>(b.data) + b_offset * static_cast<std::int64_t>(element);
            one.c = static_cast<std::byte*>(out.data) + i * c_matrix * static_cast<std::int64_t>(element);
            II_TRY(gemm_.run(one, workspace, stream_of(context)));
        }
        return ok_status();
    }

private:
    Gemm gemm_;
    const CudaDevice& device_;
    bool widen_;  // b is f16 / bf16 and a is f32
    bool trans_a_;
    bool trans_b_;
};

Status check_matmul(const ir::Node& node) {
    if (is_weight_product(node)) {
        return check_weight_product(node);
    }
    II_TRY(check_float(node));
    const DType a = node.input(0)->dtype();
    const DType b = node.input(1)->dtype();
    if (a != b && a != DType::kF32) {
        return errors::unsupported("'matmul' of {} and {}", to_string(a), to_string(b));
    }
    if (node.input(0)->rank() > kMaxRank || node.input(1)->rank() > kMaxRank) {
        return errors::unsupported("rank above {}", kMaxRank);
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_matmul(const ir::Node& node, const CudaDevice& device) {
    if (is_weight_product(node)) {
        return create_weight_product(node, device);
    }
    const bool widen = node.input(0)->dtype() != node.input(1)->dtype();
    return std::unique_ptr<Kernel>(std::make_unique<MatmulKernel>(device, widen, node.attrs().get_bool("transpose_a"),
                                                                  node.attrs().get_bool("transpose_b")));
}

// --- linear -----------------------------------------------------------------------------------------------------

class LinearKernel final : public Kernel {
public:
    bool can_record() const override { return true; }  // only enqueues work on the stream

    LinearKernel(const CudaDevice& device, bool widen, bool transpose_w)
        : gemm_(device), device_(device), widen_(widen), transpose_w_(transpose_w) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape>) const override {
        return kBlasWorkspace + kAlign +
               (widen_ ? static_cast<std::size_t>(inputs[1].numel()) * sizeof(float) + kAlign : 0);
    }

    Status run(const KernelContext& context) const override {
        II_TRY(device_.activate());
        const TensorView& x = context.input(0);
        TensorView w = context.input(1);
        const TensorView& bias = context.input(2);
        const TensorView& out = context.output(0);
        std::span<std::byte> workspace = context.workspace;
        if (widen_) {
            auto* wide = reinterpret_cast<float*>(align_up(workspace.data() + kBlasWorkspace + kAlign));
            II_TRY(launched(widen(wide, w.data, *scalar_of(w.dtype), w.numel(), stream_of(context)), "widen"));
            w.data = wide;
            workspace = workspace.first(kBlasWorkspace + kAlign);
        }
        GemmCall call;
        call.dtype = out.dtype;
        call.k = x.shape[x.rank() - 1];
        call.n = transpose_w_ ? w.shape[0] : w.shape[1];
        call.m = call.n == 0 ? 0 : out.numel() / call.n;
        call.lda = call.k;
        call.trans_b = transpose_w_;
        call.ldb = transpose_w_ ? call.k : call.n;
        call.ldc = call.n;
        call.a = x.data;
        call.b = w.data;
        call.c = out.data;
        call.bias = bias.present() ? bias.data : nullptr;
        return gemm_.run(call, workspace, stream_of(context));
    }

private:
    Gemm gemm_;
    const CudaDevice& device_;
    bool widen_;
    bool transpose_w_;
};

Status check_linear(const ir::Node& node) {
    if (is_weight_product(node)) {
        return check_weight_product(node);
    }
    II_TRY(check_float(node));
    const DType dtype = node.input(0)->dtype();
    const DType weights = node.input(1)->dtype();
    if (weights != dtype && dtype != DType::kF32) {
        return errors::unsupported("'linear' of {} with {} weights", to_string(dtype), to_string(weights));
    }
    if (node.input(2) != nullptr && node.input(2)->dtype() != dtype) {
        return errors::unsupported("'linear' with a {} bias", to_string(node.input(2)->dtype()));
    }
    if (node.input(1)->rank() != 2) {
        return errors::unsupported("'linear' with weights of rank {}", node.input(1)->rank());
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_linear(const ir::Node& node, const CudaDevice& device) {
    if (is_weight_product(node)) {
        return create_weight_product(node, device);
    }
    const bool widen = node.input(0)->dtype() != node.input(1)->dtype();
    return std::unique_ptr<Kernel>(std::make_unique<LinearKernel>(device, widen, node.attrs().get_bool("transpose_w")));
}

// --- conv -------------------------------------------------------------------------------------------------------

/// Unfolded images are processed in chunks of at most this many bytes.
constexpr std::int64_t kColumnBytes = std::int64_t{256} << 20;

/// conv, and fused_conv: the convolution, then its elementwise epilogue over the result (in place when the result
/// has the output's dtype, else from a temporary in the workspace).
class ConvKernel final : public Kernel {
public:
    bool can_record() const override { return true; }  // only enqueues work on the stream

    ConvKernel(const CudaDevice& device, int spatial, ops::WindowAttrs window, std::int64_t groups, DType conv_dtype,
               const ElementwiseModule* epilogue, bool temporary)
        : gemm_(device),
          device_(device),
          spatial_(spatial),
          window_(std::move(window)),
          groups_(groups),
          conv_dtype_(conv_dtype),
          epilogue_(epilogue),
          temporary_(temporary) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        const ConvParams p = params(inputs[0], inputs[1], outputs[0]);
        std::size_t size = temporary_bytes(outputs[0]);
        if (p.groups != 1) {
            return size;
        }
        size += kBlasWorkspace + kAlign;
        if (!pointwise(p)) {
            size += static_cast<std::size_t>(chunk(p, inputs[0]) * column_bytes(p, inputs[0])) + kAlign;
        }
        return size;
    }

    Status run(const KernelContext& context) const override {
        II_TRY(device_.activate());
        const TensorView& x = context.input(0);
        const TensorView& w = context.input(1);
        const TensorView& bias = context.input(2);
        const TensorView& out = context.output(0);
        // The convolution's result: the output, or a temporary at the start of the workspace.
        TensorView y{out.data, conv_dtype_, out.shape};
        std::span<std::byte> workspace = context.workspace;
        if (temporary_) {
            const std::size_t bytes = temporary_bytes(out.shape);
            y.data = align_up(workspace.data());
            workspace = workspace.subspan(bytes);
        }
        II_TRY(convolve(x, w, bias, y, workspace, stream_of(context)));
        if (epilogue_ == nullptr) {
            return ok_status();
        }
        std::array<TensorView, 16> inputs;
        inputs[0] = y;
        const std::size_t count = context.inputs.size() < 3 ? 1 : context.inputs.size() - 2;
        for (std::size_t i = 1; i < count; ++i) {
            inputs[i] = context.inputs[i + 2];
        }
        return launch_elementwise(*epilogue_, std::span<const TensorView>(inputs.data(), count), out,
                                  stream_of(context));
    }

private:
    std::size_t temporary_bytes(const Shape& out) const {
        return temporary_ ? static_cast<std::size_t>(out.numel()) * element_size(conv_dtype_) + kAlign : 0;
    }

    Status convolve(const TensorView& x, const TensorView& w, const TensorView& bias, const TensorView& y,
                    std::span<std::byte> workspace, cudaStream_t stream) const {
        const ConvParams p = params(x.shape, w.shape, y.shape);
        const std::optional<Scalar> type = scalar_of(y.dtype);
        if (p.groups != 1) {
            return launched(conv_direct(y.data, x.data, w.data, bias.present() ? bias.data : nullptr, *type, p, stream),
                            "conv_direct");
        }
        const std::int64_t positions = p.out_h * p.out_w;
        const std::int64_t depth = p.channels * p.kernel_h * p.kernel_w;
        const auto element = static_cast<std::int64_t>(element_size(y.dtype));
        GemmCall call;
        call.dtype = y.dtype;
        call.m = p.out_channels;
        call.n = positions;
        call.k = depth;
        call.lda = depth;
        call.ldb = positions;
        call.ldc = positions;
        call.a = w.data;
        call.stride_a = 0;
        call.stride_c = p.out_channels * positions;
        if (pointwise(p)) {
            call.batch = p.batch;
            call.b = x.data;
            call.stride_b = p.channels * positions;
            call.c = y.data;
            II_TRY(gemm_.run(call, workspace, stream));
        } else {
            const std::int64_t images = chunk(p, x.shape);
            std::span<std::byte> blas_workspace = workspace.first(kBlasWorkspace + kAlign);
            std::byte* columns = align_up(workspace.data() + kBlasWorkspace + kAlign);
            for (std::int64_t first = 0; first < p.batch; first += images) {
                const std::int64_t count = std::min(images, p.batch - first);
                II_TRY(launched(im2col(columns, x.data, static_cast<int>(element), p, first, count, stream), "im2col"));
                call.batch = count;
                call.b = columns;
                call.stride_b = depth * positions;
                call.c = static_cast<std::byte*>(y.data) + first * p.out_channels * positions * element;
                II_TRY(gemm_.run(call, blas_workspace, stream));
            }
        }
        if (bias.present()) {
            II_TRY(
                launched(add_bias(y.data, bias.data, *type, p.batch, p.out_channels, positions, stream), "add_bias"));
        }
        return ok_status();
    }

    ConvParams params(const Shape& x, const Shape& w, const Shape& y) const {
        ConvParams p;
        p.batch = x[0];
        p.channels = x[1];
        p.out_channels = y[1];
        p.groups = groups_;
        const auto s = static_cast<std::size_t>(spatial_);
        for (int d = 0; d < spatial_; ++d) {
            const auto i = static_cast<std::size_t>(d);
            const ops::WindowGeometry g =
                ops::resolve_window(x[2 + d], w[2 + d], window_.strides[i], window_.dilations[i], window_.pads[i],
                                    window_.pads[s + i], window_.auto_pad);
            const bool width = d == spatial_ - 1;
            (width ? p.in_w : p.in_h) = x[2 + d];
            (width ? p.out_w : p.out_h) = y[2 + d];
            (width ? p.kernel_w : p.kernel_h) = g.kernel;
            (width ? p.stride_w : p.stride_h) = g.stride;
            (width ? p.pad_w : p.pad_h) = g.pad_begin;
            (width ? p.dilation_w : p.dilation_h) = g.dilation;
        }
        return p;
    }

    static bool pointwise(const ConvParams& p) {
        return p.kernel_h == 1 && p.kernel_w == 1 && p.stride_h == 1 && p.stride_w == 1 && p.pad_h == 0 &&
               p.pad_w == 0 && p.in_h == p.out_h && p.in_w == p.out_w;
    }

    std::int64_t column_bytes(const ConvParams& p, const Shape&) const {
        return p.channels * p.kernel_h * p.kernel_w * p.out_h * p.out_w * 4;
    }

    std::int64_t chunk(const ConvParams& p, const Shape& x) const {
        const std::int64_t per_image = std::max<std::int64_t>(column_bytes(p, x), 1);
        return std::clamp<std::int64_t>(kColumnBytes / per_image, 1, std::max<std::int64_t>(p.batch, 1));
    }

    Gemm gemm_;
    const CudaDevice& device_;
    int spatial_;
    ops::WindowAttrs window_;
    std::int64_t groups_;
    DType conv_dtype_;
    const ElementwiseModule* epilogue_;
    bool temporary_;
};

Status check_conv(const ir::Node& node) {
    const DType dtype = node.input(0)->dtype();
    if (dtype != DType::kF32 && dtype != DType::kF16 && dtype != DType::kBF16) {
        return errors::unsupported("'{}' of {}", node.op_name(), to_string(dtype));
    }
    const int rank = node.input(0)->rank();
    if (rank != 3 && rank != 4) {
        return errors::unsupported("'{}' with {} spatial dimensions", node.op_name(), rank - 2);
    }
    for (int i = 1; i < 3; ++i) {
        if (node.input(i) != nullptr && node.input(i)->dtype() != dtype) {
            return errors::unsupported("'{}' of mixed dtypes", node.op_name());
        }
    }
    if (node.op_name() == ops::kFusedConv) {
        if (node.num_inputs() > 3 + 15) {
            return errors::unsupported("'fused_conv' with more than 15 extra inputs");
        }
        II_ASSIGN_OR_RETURN(const ElementwiseModule* module, compile_epilogue(node, nullptr));
        (void)module;
    } else if (node.output()->dtype() != dtype) {
        return errors::unsupported("'conv' of mixed dtypes");
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_conv(const ir::Node& node, const CudaDevice& device) {
    const int spatial = node.input(0)->rank() - 2;
    const bool fused = node.op_name() == ops::kFusedConv;
    II_ASSIGN_OR_RETURN(ops::WindowAttrs window,
                        ops::parse_window_attrs(fused ? ops::conv_attributes(node.attrs()) : node.attrs(), spatial));
    const ElementwiseModule* epilogue = nullptr;
    if (fused) {
        II_ASSIGN_OR_RETURN(epilogue, compile_epilogue(node, &device));
    }
    const DType conv_dtype = node.input(0)->dtype();
    const bool temporary = fused && element_size(conv_dtype) != element_size(node.output()->dtype());
    return std::unique_ptr<Kernel>(std::make_unique<ConvKernel>(
        device, spatial, std::move(window), node.attrs().get_int("groups", 1), conv_dtype, epilogue, temporary));
}

}  // namespace

void register_gemm_kernels(CudaBackend& backend) {
    backend.add_kernel("matmul", create_matmul, check_matmul);
    backend.add_kernel("linear", create_linear, check_linear);
    backend.add_kernel("conv", create_conv, check_conv);
    backend.add_kernel(ops::kFusedConv, create_conv, check_conv);
}

}  // namespace ii::cuda
