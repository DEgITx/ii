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

#include <cstdint>

namespace ii::cuda {

inline constexpr int kMaxRank = 8;

/// Errors kernels report while they run (they cannot return a status): the first error wins. `DeviceError` lives in
/// host memory mapped into the device; the queue checks it when it synchronizes.
enum class DeviceErrorCode : long long { kNone = 0, kGatherIndex = 1 };
struct DeviceError {
    long long code;   ///< DeviceErrorCode
    long long value;  ///< e.g. the index
    long long limit;  ///< e.g. the size of the axis
};

/// Element types of the compute kernels.
enum class Scalar : int { kF32, kF16, kBF16, kI32, kI64 };

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

}  // namespace ii::cuda
