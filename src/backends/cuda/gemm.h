// SPDX-License-Identifier: MIT
/// @file
/// Matrix products of ii-cuda on cuBLASLt (gemm.cpp), shared by the matmul / linear / conv kernels and the weight
/// products of language models (llm.cpp).
#pragma once

#include "backends/cuda/cuda_backend.h"

#include <cstddef>
#include <functional>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace ii::cuda {

/// Workspace of cuBLASLt (plus alignment slack: cuBLASLt wants 256-byte aligned workspace).
inline constexpr std::size_t kBlasWorkspace = std::size_t{32} << 20;
inline constexpr std::size_t kAlign = 256;

inline std::byte* align_up(std::byte* p) {
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    return p + ((kAlign - address % kAlign) % kAlign);
}

/// Bytes of `count` elements of `dtype` plus alignment slack (a region of a workspace).
inline std::size_t region_bytes(DType dtype, std::int64_t count) {
    return storage_size(dtype, count) + kAlign;
}

/// A row-major product C[m, n] = op(A) op(B) (A is [m, k], or [k, m] when trans_a), repeated `batch` times. A and B
/// have type `input` (the output's when invalid), C has type `dtype`; products accumulate in f32.
struct GemmCall {
    DType dtype = DType::kF32;
    DType input = DType::kInvalid;
    bool trans_a = false;
    bool trans_b = false;
    std::int64_t m = 0, n = 0, k = 0;
    std::int64_t lda = 0, ldb = 0, ldc = 0;
    std::int64_t batch = 1;
    std::int64_t stride_a = 0, stride_b = 0, stride_c = 0;  // elements
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    const void* bias = nullptr;  ///< n elements of `dtype` added to every row
    bool accumulate = false;     ///< C += op(A) op(B) instead of C =
};

/// Autotuning: the index of the fastest of `count` candidates, each enqueued by `run(i)` on `stream` (a warm-up, then
/// two rounds of about 0.2 ms of runs per candidate; waits for the stream). A candidate that fails is skipped.
Result<int> fastest(int count, cudaStream_t stream, const std::function<cudaError_t(int)>& run);

/// Runs GEMMs; caches the cuBLASLt algorithm of every shape it saw (shared by the sessions using the kernel).
class Gemm {
public:
    explicit Gemm(const CudaDevice& device) : device_(device) {}

    /// `workspace` of at least kBlasWorkspace + kAlign bytes lets cuBLASLt use its faster algorithms.
    Status run(const GemmCall& call, std::span<std::byte> workspace, cudaStream_t stream) const;

private:
    struct Key {
        DType input;
        bool accumulate, trans_a, trans_b, bias;
        std::int64_t m, n, k, lda, ldb, ldc, batch, stride_a, stride_b, stride_c;
        std::uint32_t align_x, align_y, align_d;
        std::size_t workspace;
        bool operator==(const Key&) const = default;
    };

    Status blas(cublasStatus_t status, std::string_view what) const;
    Status layout(cublasLtMatrixLayout_t layout, cudaDataType type, std::int64_t rows, std::int64_t cols,
                  std::int64_t ld, std::int64_t batch, std::int64_t stride) const;
    /// The algorithm for `key`: remembered, from the autotuning cache, or the fastest of cuBLASLt's candidates
    /// (each run once with `launch` on `stream`).
    Result<cublasLtMatmulAlgo_t> algorithm(const Key& key, cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t x,
                                           cublasLtMatrixLayout_t y, cublasLtMatrixLayout_t d,
                                           const std::function<cublasStatus_t(const cublasLtMatmulAlgo_t&)>& launch,
                                           cudaStream_t stream) const;

    const CudaDevice& device_;
    mutable std::mutex mutex_;
    mutable std::vector<std::pair<Key, cublasLtMatmulAlgo_t>> algos_;
};

}  // namespace ii::cuda
