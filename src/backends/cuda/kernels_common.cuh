// SPDX-License-Identifier: MIT
// Helpers shared by the nvcc-compiled kernels of ii-cuda.
#pragma once

#include "backends/cuda/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ii::cuda {

inline constexpr int kThreads = 256;

/// Blocks for a grid-stride loop over `n` items.
inline unsigned blocks_for(std::int64_t n, int threads = kThreads) {
    const std::int64_t blocks = (n + threads - 1) / threads;
    return static_cast<unsigned>(blocks < 1 ? 1 : (blocks > 65535 * 8 ? 65535 * 8 : blocks));
}

/// Loads and stores of element types; floats compute in f32, integers in 64 bits.
template <class T>
struct Traits;
template <>
struct Traits<float> {
    using Acc = float;
    static __device__ __forceinline__ float load(float v) { return v; }
    static __device__ __forceinline__ float store(float v) { return v; }
};
template <>
struct Traits<__half> {
    using Acc = float;
    static __device__ __forceinline__ float load(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half store(float v) { return __float2half_rn(v); }
};
template <>
struct Traits<__nv_bfloat16> {
    using Acc = float;
    static __device__ __forceinline__ float load(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 store(float v) { return __float2bfloat16_rn(v); }
};
template <>
struct Traits<std::int32_t> {
    using Acc = long long;
    static __device__ __forceinline__ long long load(std::int32_t v) { return v; }
    static __device__ __forceinline__ std::int32_t store(long long v) {
        return static_cast<std::int32_t>(static_cast<unsigned long long>(v));
    }
};
template <>
struct Traits<std::int64_t> {
    using Acc = long long;
    static __device__ __forceinline__ long long load(std::int64_t v) { return v; }
    static __device__ __forceinline__ std::int64_t store(long long v) { return v; }
};

__device__ __forceinline__ bool is_nan(float v) {
    return v != v;
}
__device__ __forceinline__ bool is_nan(long long) {
    return false;
}
__device__ __forceinline__ float quiet_nan() {
    return __int_as_float(0x7fc00000);
}

/// Calls `fn(T{})` with the element type of a float scalar type; integer types return cudaErrorInvalidValue.
template <class Fn>
cudaError_t dispatch_float(Scalar type, Fn&& fn) {
    switch (type) {
    case Scalar::kF32:
        return fn(float{});
    case Scalar::kF16:
        return fn(__half{});
    case Scalar::kBF16:
        return fn(__nv_bfloat16{});
    default:
        return cudaErrorInvalidValue;
    }
}

/// Like dispatch_float, plus i32 and i64.
template <class Fn>
cudaError_t dispatch_all(Scalar type, Fn&& fn) {
    switch (type) {
    case Scalar::kI32:
        return fn(std::int32_t{});
    case Scalar::kI64:
        return fn(std::int64_t{});
    default:
        return dispatch_float(type, fn);
    }
}

/// Warp-wide reductions.
template <class T, class Op>
__device__ __forceinline__ T warp_reduce(T value, Op op) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = op(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

/// Block-wide reduction (blockDim.x a multiple of 32, at most 1024); every thread gets the result.
template <class T, class Op>
__device__ T block_reduce(T value, Op op, T identity) {
    __shared__ T partial[32];
    __shared__ T result;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    value = warp_reduce(value, op);
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        T v = lane < static_cast<int>(blockDim.x / 32) ? partial[lane] : identity;
        v = warp_reduce(v, op);
        if (lane == 0) {
            result = v;
        }
    }
    __syncthreads();
    const T r = result;
    __syncthreads();  // `partial` and `result` may be reused by the next call
    return r;
}

}  // namespace ii::cuda
