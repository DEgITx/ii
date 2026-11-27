// SPDX-License-Identifier: MIT
/// @file
/// Kernels of ii-cuda compiled by nvcc (kernels_*.cu) and the host functions that launch them.
///
/// Shared by the CUDA sources and the C++ sources of the plugin, so it uses plain types only. Every launcher
/// enqueues its work on `stream` and returns the launch status; errors during execution surface when the stream is
/// synchronized. Pointers are device memory. Elementwise operations are not here: they are generated and compiled
/// with NVRTC (elementwise.cpp).
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ii::cuda {

inline constexpr int kMaxRank = 8;

/// Errors kernels report while they run (they cannot return a status): the first error wins. `DeviceError` lives in
/// host memory mapped into the device; the queue checks it when it synchronizes.
enum class DeviceErrorCode : long long { kNone = 0, kGatherIndex = 1, kSliceUpdateStart = 2, kKvLength = 3 };
struct DeviceError {
    long long code;   ///< DeviceErrorCode
    long long value;  ///< e.g. the index
    long long limit;  ///< e.g. the size of the axis
    long long extra;  ///< e.g. the length of a slice
};

/// Reports an error (the first one wins); a device function for the .cu files.
#ifdef __CUDACC__
__device__ inline void report_error(DeviceError* error, DeviceErrorCode code, long long value, long long limit,
                                    long long extra = 0) {
    if (atomicCAS(reinterpret_cast<unsigned long long*>(&error->code), 0ull, static_cast<unsigned long long>(code)) ==
        0ull) {
        error->value = value;
        error->limit = limit;
        error->extra = extra;
    }
}
#endif

/// Element types of the compute kernels.
enum class Scalar : int { kF32, kF16, kBF16, kI32, kI64 };

/// Storage of matrix weights: floats or the block-quantized GGUF layouts (ii::DType kQ4_0 ... kQ8_K).
enum class WeightType : int {
    kF32,
    kF16,
    kBF16,
    kQ4_0,
    kQ4_1,
    kQ5_0,
    kQ5_1,
    kQ8_0,
    kQ8_1,
    kQ2_K,
    kQ3_K,
    kQ4_K,
    kQ5_K,
    kQ6_K,
    kQ8_K
};

// --------------------------------------------------------------------------------------------------------------------
// Data movement (any element size of 1, 2, 4 or 8 bytes)

/// dst[index] = src[index] for every index of `dims`, both addressed with their own element strides (which may be
/// 0 or negative): permute, slice, expand, flip, concat and split are copies between such views.
struct CopyParams {
    int rank = 0;
    std::int64_t dims[kMaxRank] = {};
    std::int64_t src_strides[kMaxRank] = {};
    std::int64_t dst_strides[kMaxRank] = {};
};
cudaError_t copy_strided(void* dst, const void* src, int element_size, const CopyParams& params, cudaStream_t stream);

/// out[o, j, :] = data[o, indices[j], :] with slices of `slice_bytes` bytes; negative indices count from the end.
/// An index out of range is reported in `error` (its slice becomes zeros).
struct GatherParams {
    std::int64_t outer = 0;
    std::int64_t axis_size = 0;
    std::int64_t count = 0;  ///< number of indices
    std::int64_t slice_bytes = 0;
    bool index64 = true;  ///< i64 (else i32) indices
    DeviceError* error = nullptr;
};
cudaError_t gather(void* out, const void* data, const void* indices, const GatherParams& params, cudaStream_t stream);

/// Padding (and tiling): out[index] = in[index - pad_begin] where inside; outside, `mode` decides.
enum class PadMode : int { kConstant, kEdge, kReflect, kWrap };
struct PadParams {
    int rank = 0;
    PadMode mode = PadMode::kConstant;
    std::int64_t out_dims[kMaxRank] = {};
    std::int64_t in_dims[kMaxRank] = {};
    std::int64_t pad_begin[kMaxRank] = {};
    std::uint64_t fill = 0;  ///< kConstant: the bits of the fill value (the low `element_size` bytes)
};
cudaError_t pad(void* out, const void* in, int element_size, const PadParams& params, cudaStream_t stream);

/// A kernel that does nothing (measures the cost of a launch).
cudaError_t launch_empty(cudaStream_t stream);

/// Sets `count` elements to the bits of `fill` (the low `element_size` bytes).
cudaError_t fill(void* out, int element_size, std::uint64_t fill, std::int64_t count, cudaStream_t stream);

/// out[i] = start + step * (position of i along the middle dimension of [outer, size, inner]), converted with the
/// `cast` rules.
cudaError_t iota(void* out, Scalar type, double start, double step, std::int64_t count, std::int64_t size,
                 std::int64_t inner, cudaStream_t stream);

// --------------------------------------------------------------------------------------------------------------------
// Reductions

enum class ReduceOp : int { kSum, kMean, kMax, kMin, kProd };

/// out[k] = op over the reduced index space of in[kept offset(k) + reduced offset(r)]. Floats accumulate in f32,
/// integers in 64 bits; max / min propagate NaN.
struct ReduceParams {
    std::int64_t outputs = 0;
    std::int64_t count = 0;  ///< elements reduced into each output
    int kept_rank = 0;
    std::int64_t kept_dims[kMaxRank] = {};
    std::int64_t kept_strides[kMaxRank] = {};
    int reduced_rank = 0;
    std::int64_t reduced_dims[kMaxRank] = {};
    std::int64_t reduced_strides[kMaxRank] = {};
};
cudaError_t reduce(void* out, const void* in, Scalar type, ReduceOp op, const ReduceParams& params,
                   cudaStream_t stream);

/// argmax / argmin over the middle dimension of [outer, size, inner] into i64 [outer, inner]; the first NaN wins.
struct ArgParams {
    std::int64_t outer = 0;
    std::int64_t size = 0;
    std::int64_t inner = 0;
    bool max = true;
    bool select_last = false;
};
cudaError_t arg_reduce(std::int64_t* out, const void* in, Scalar type, const ArgParams& params, cudaStream_t stream);

// --------------------------------------------------------------------------------------------------------------------
// Normalization (floats)

/// softmax / log_softmax over the middle dimension of [outer, size, inner].
cudaError_t softmax(void* out, const void* in, Scalar type, std::int64_t outer, std::int64_t size, std::int64_t inner,
                    bool log, cudaStream_t stream);

/// layer_norm (rms: rms_norm) of rows of `size` elements; `scale` and `bias` (optional) have `size` elements.
cudaError_t normalize(void* out, const void* in, const void* scale, const void* bias, Scalar type, std::int64_t rows,
                      std::int64_t size, float epsilon, bool rms, cudaStream_t stream);

// --------------------------------------------------------------------------------------------------------------------
// Convolution and pooling (floats; 1-D and 2-D: 1-D has height 1)

struct ConvParams {
    std::int64_t batch = 0;
    std::int64_t channels = 0;
    std::int64_t out_channels = 0;
    std::int64_t groups = 1;
    std::int64_t in_h = 1, in_w = 1, out_h = 1, out_w = 1;
    std::int64_t kernel_h = 1, kernel_w = 1;
    std::int64_t stride_h = 1, stride_w = 1;
    std::int64_t pad_h = 0, pad_w = 0;  ///< padding before
    std::int64_t dilation_h = 1, dilation_w = 1;
};

/// Unfolds images [first, first + images) of x [batch, channels, in_h, in_w] (groups == 1) into
/// col [images, channels * kernel_h * kernel_w, out_h * out_w].
cudaError_t im2col(void* col, const void* x, int element_size, const ConvParams& params, std::int64_t first,
                   std::int64_t images, cudaStream_t stream);

/// A direct convolution (one thread per output element): grouped and depthwise convolutions.
cudaError_t conv_direct(void* y, const void* x, const void* w, const void* bias, Scalar type, const ConvParams& params,
                        cudaStream_t stream);

/// Converts `count` f16 / bf16 elements to f32.
cudaError_t widen(float* out, const void* in, Scalar type, std::int64_t count, cudaStream_t stream);

/// Converts `count` elements between float types (rounding to nearest even).
cudaError_t convert(void* out, Scalar out_type, const void* in, Scalar in_type, std::int64_t count,
                    cudaStream_t stream);

/// y[o, c, i] += bias[c] for y [outer, channels, inner].
cudaError_t add_bias(void* y, const void* bias, Scalar type, std::int64_t outer, std::int64_t channels,
                     std::int64_t inner, cudaStream_t stream);

struct PoolParams {
    std::int64_t planes = 0;  ///< batch * channels
    std::int64_t in_h = 1, in_w = 1, out_h = 1, out_w = 1;
    std::int64_t kernel_h = 1, kernel_w = 1;
    std::int64_t stride_h = 1, stride_w = 1;
    std::int64_t pad_h = 0, pad_w = 0, pad_h_end = 0, pad_w_end = 0;
    std::int64_t dilation_h = 1, dilation_w = 1;
    bool max = true;
    bool count_include_pad = false;
};
cudaError_t pool(void* y, const void* x, Scalar type, const PoolParams& params, cudaStream_t stream);

/// Resampling with precomputed taps: y[index] = sum over the taps of every dimension of weight * x[source index].
/// Dimension d has `taps[d]` taps per output position: source indices index[d][out * taps[d] + t] with weights
/// weight[d][...] (device arrays).
struct ResizeParams {
    int rank = 0;
    std::int64_t out_dims[kMaxRank] = {};
    std::int64_t in_strides[kMaxRank] = {};
    int taps[kMaxRank] = {};
    const std::int64_t* index[kMaxRank] = {};
    const float* weight[kMaxRank] = {};
};
cudaError_t resize(void* y, const void* x, Scalar type, const ResizeParams& params, cudaStream_t stream);

// --------------------------------------------------------------------------------------------------------------------
// Weights (kernels_quant.cu): rows of `k` elements (a multiple of 32, and of the block size)

/// y[m, j] = sum over i of x[m, i] * w[j, i] (+ bias[j]) for x f32 [rows, k], w [n, k], y f32 [rows, n] and
/// 1 <= rows <= kGemvRows: products whose time is reading the weights (decoding). Blocks are decoded to f32 and
/// multiplied in f32.
/// `split` warps share a feature: 1, 2 or 4 (autotuned), or 0 for a choice from the shape.
inline constexpr int kGemvRows = 8;
cudaError_t weight_gemv(float* y, const float* x, const void* w, WeightType type, const float* bias, int rows,
                        std::int64_t n, std::int64_t k, cudaStream_t stream, int split = 0);

/// Splits f32 rows x [rows, k] into `count` (2 or 3) parts of `type` (f16 or bf16) whose sum is x / scale: parts
/// [count, rows, k] (each part the rounding of what the previous ones left). With `scales` (f16), every row is first
/// divided by a power of two (written to scales[row]) that brings its largest magnitude near 2^14: no overflow, and
/// the low part stays in f16's normal range. Products of 16-bit weights with the parts on tensor cores then keep about
/// f32 precision.
cudaError_t split_rows(void* parts, Scalar type, int count, const float* x, float* scales, std::int64_t rows,
                       std::int64_t k, cudaStream_t stream);

/// c[i, j] = c[i, j] * scales[i] (+ bias[j]) for c [rows, n]; `scales` or `bias` may be null.
cudaError_t scale_rows(float* c, const float* scales, const float* bias, std::int64_t rows, std::int64_t n,
                       cudaStream_t stream);

/// Decodes rows [first, first + count) of w [*, k] into out [count, k] of `out_type` (f32, f16 or bf16).
cudaError_t dequantize_rows(void* out, Scalar out_type, const void* w, WeightType type, std::int64_t first,
                            std::int64_t count, std::int64_t k, cudaStream_t stream);

// --------------------------------------------------------------------------------------------------------------------
// Language models (kernels_llm.cu)

/// Rotary embedding (the `rope` operation) of rows of x [..., size]: pairs (i, i + half) or (2i, 2i + 1) of the first
/// 2 * half features rotate by the angles of the cos / sin row at `table offset(row)`, the rest is copied.
struct RopeParams {
    int rank = 0;                               ///< of x
    std::int64_t dims[kMaxRank] = {};           ///< x's shape
    std::int64_t table_strides[kMaxRank] = {};  ///< element strides of cos / sin over x's dimensions (0: broadcast)
    std::int64_t half = 0;
    bool interleaved = false;
};
cudaError_t rope(void* y, const void* x, const void* cos, const void* sin, Scalar type, const RopeParams& params,
                 cudaStream_t stream);

/// y[o, start + j, :] = update[o, j, :] for y [outer, axis_size, inner] (in place: y is x's memory), update
/// [outer, length, inner]; `start` is a device scalar. A start outside [0, axis_size - length] is reported.
struct SliceUpdateParams {
    std::int64_t outer = 0;
    std::int64_t axis_size = 0;
    std::int64_t length = 0;
    std::int64_t inner_bytes = 0;
    const void* start = nullptr;
    bool start64 = true;
    DeviceError* error = nullptr;
};
cudaError_t slice_update(void* y, const void* update, const SliceUpdateParams& params, cudaStream_t stream);

/// Scaled dot-product attention (the `sdpa` operation) for contiguous q [B, H, Sq, D], k [B, Hkv, Skv, D],
/// v [B, Hkv, Skv, Dv], out [B, H, Sq, Dv] with D, Dv <= 256: one pass over the keys with an online softmax
/// (flash attention); the scores are never stored. Short queries (decoding) split the keys over several blocks
/// whose partial results are merged (split-K); the split depends on the shapes only, so the launch can be recorded
/// in a CUDA graph. `kv_len` (a device scalar, optional) limits the keys; a value outside [0, Skv] is reported.
struct AttentionParams {
    std::int64_t batch = 0, heads = 0, kv_heads = 0, sq = 0, skv = 0, d = 0, dv = 0;
    float scale = 1;
    float softcap = 0;
    bool causal = false;
    std::int64_t window = 0;
    const void* kv_len = nullptr;
    bool kv_len64 = true;
    int mask_kind = 0;                  ///< 0: none, 1: bool (true = attend), 2: additive (q's type)
    std::int64_t mask_strides[4] = {};  ///< element strides over [B, H, Sq, Skv] (0: broadcast)
    DeviceError* error = nullptr;
    int arch = 0;  ///< compute capability of the device, e.g. 120: prompts use tensor cores from 80
};
inline constexpr int kAttentionMaxHead = 256;
/// Workspace bytes for the partial results of a split.
std::size_t attention_workspace(const AttentionParams& params, int sm_count);
cudaError_t attention(void* out, const void* q, const void* k, const void* v, const void* mask, Scalar q_type,
                      Scalar kv_type, const AttentionParams& params, int sm_count, void* workspace,
                      cudaStream_t stream);

/// The tensor-core kernel `attention` uses for prompts (kernels_flash.cu): compute capability 8.0, head sizes 64, 96
/// or 128 (equal for keys and values).
bool flash_attention_supported(const AttentionParams& params, Scalar q_type, Scalar kv_type, int arch);
cudaError_t flash_attention(void* out, const void* q, const void* k, const void* v, const void* mask, Scalar q_type,
                            Scalar kv_type, const AttentionParams& params, cudaStream_t stream);

}  // namespace ii::cuda
