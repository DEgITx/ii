// SPDX-License-Identifier: MIT
// ii-cuda weight kernels: decoders of the float and block-quantized (GGUF) weight layouts, the matrix-vector
// product of decoding (weight_gemv) and dequantization (the `dequantize` operation and the decoded weights of prefill
// products, which then run on cuBLASLt).
//
// Rows are decoded in units of 32 consecutive elements (a classic block q4_0 ... q8_1 is one unit, a K-quant
// super-block of 256 elements is 8), and every unit by 4 neighbouring threads, 8 elements each: the threads of a warp
// read neighbouring bytes of the weights (coalesced loads). The decoders follow the scalar reference codecs
// (src/tensor/quant.cpp) and produce the same values; products accumulate in f32.
#include "backends/cuda/kernels_common.cuh"

#include <type_traits>

namespace ii::cuda {

namespace {

// --- Loads ----------------------------------------------------------------------------------------------------------

/// A 16-bit little-endian value at a 2-byte aligned address.
__device__ __forceinline__ unsigned u16_at(const std::uint8_t* p) {
    return *reinterpret_cast<const unsigned short*>(p);
}

/// A 32-bit value at an address aligned to `kAlign` bytes (2 or 4).
template <int kAlign>
__device__ __forceinline__ std::uint32_t u32_at(const std::uint8_t* p) {
    if constexpr (kAlign >= 4) {
        return *reinterpret_cast<const std::uint32_t*>(p);
    } else {
        return u16_at(p) | (u16_at(p + 2) << 16);
    }
}

__device__ __forceinline__ float f16_at(const std::uint8_t* p) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(u16_at(p))));
}

/// Byte `i` (0..3) of `word` minus `bias` as a float: the byte becomes the mantissa of 2^23 + byte, which one
/// subtraction turns into the value (faster than an integer conversion).
__device__ __forceinline__ float byte_float(std::uint32_t word, int i, float bias = 0.0f) {
    return __int_as_float(static_cast<int>(__byte_perm(word, 0x4B000000u, 0x7440u | static_cast<unsigned>(i)))) -
           (8388608.0f + bias);
}

/// The 4 bytes of `word` minus `bias`, times `scale`, plus `offset`, into v[at..at + 4).
__device__ __forceinline__ void four(std::uint32_t word, float bias, float scale, float offset, float* v) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = byte_float(word, i, bias) * scale + offset;
    }
}

// --- Layouts
// ----------------------------------------------------------------------------------------------------------
//
// A layout has kBlock elements in kBytes bytes and decodes 8 elements of a unit: decode(block, unit, part, v) with
// part 0..3. kSplit layouts (the classic 4- and 5-bit blocks, whose low nibbles hold elements 0..15 and high nibbles
// 16..31) yield elements [4 part, 4 part + 4) and [16 + 4 part, 16 + 4 part + 4) of the unit; the others yield
// [8 part, 8 part + 8).

struct F32Weights {
    static constexpr int kBlock = 32;
    static constexpr int kBytes = 128;
    static constexpr bool kSplit = false;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int, int part, float (&v)[8]) {
        const float4* p = reinterpret_cast<const float4*>(b) + 2 * part;
        const float4 a = p[0];
        const float4 c = p[1];
        v[0] = a.x;
        v[1] = a.y;
        v[2] = a.z;
        v[3] = a.w;
        v[4] = c.x;
        v[5] = c.y;
        v[6] = c.z;
        v[7] = c.w;
    }
};

template <class T>
struct HalfWeights {
    static constexpr int kBlock = 32;
    static constexpr int kBytes = 64;
    static constexpr bool kSplit = false;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int, int part, float (&v)[8]) {
        const uint4 q = reinterpret_cast<const uint4*>(b)[part];
        const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if constexpr (std::is_same_v<T, __half>) {
                v[2 * j] = __half2float(__ushort_as_half(static_cast<unsigned short>(words[j] & 0xFFFF)));
                v[2 * j + 1] = __half2float(__ushort_as_half(static_cast<unsigned short>(words[j] >> 16)));
            } else {
                v[2 * j] = __uint_as_float(words[j] << 16);
                v[2 * j + 1] = __uint_as_float(words[j] & 0xFFFF0000u);
            }
        }
    }
};

/// q4_0 (d, qs[16]) and q4_1 (d, m, qs[16]).
template <bool kMin>
struct Q4 {
    static constexpr int kBlock = 32;
    static constexpr int kBytes = kMin ? 20 : 18;
    static constexpr bool kSplit = true;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int, int part, float (&v)[8]) {
        const float d = f16_at(b);
        const float m = kMin ? f16_at(b + 2) : 0.0f;
        const std::uint32_t word = u32_at<kMin ? 4 : 2>(b + (kMin ? 4 : 2) + 4 * part);
        four(word & 0x0F0F0F0Fu, kMin ? 0.0f : 8.0f, d, m, v);
        four((word >> 4) & 0x0F0F0F0Fu, kMin ? 0.0f : 8.0f, d, m, v + 4);
    }
};

/// q5_0 (d, qh, qs[16]) and q5_1 (d, m, qh, qs[16]): a fifth bit from qh (bit j for element j).
template <bool kMin>
struct Q5 {
    static constexpr int kBlock = 32;
    static constexpr int kBytes = kMin ? 24 : 22;
    static constexpr bool kSplit = true;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int, int part, float (&v)[8]) {
        const float d = f16_at(b);
        const float m = kMin ? f16_at(b + 2) : 0.0f;
        const std::uint32_t qh = u32_at<2>(b + (kMin ? 4 : 2));
        const std::uint32_t word = u32_at<kMin ? 4 : 2>(b + (kMin ? 8 : 6) + 4 * part);
        std::uint32_t lo = word & 0x0F0F0F0Fu;
        std::uint32_t hi = (word >> 4) & 0x0F0F0F0Fu;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lo |= ((qh >> (4 * part + i)) & 1u) << (8 * i + 4);
            hi |= ((qh >> (4 * part + i + 16)) & 1u) << (8 * i + 4);
        }
        four(lo, kMin ? 0.0f : 16.0f, d, m, v);
        four(hi, kMin ? 0.0f : 16.0f, d, m, v + 4);
    }
};

/// q8_0 (d, qs[32]) and q8_1 (d, s, qs[32]).
template <bool kSum>
struct Q8 {
    static constexpr int kBlock = 32;
    static constexpr int kBytes = kSum ? 36 : 34;
    static constexpr bool kSplit = false;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int, int part, float (&v)[8]) {
        constexpr int kAlign = kSum ? 4 : 2;
        const float d = f16_at(b);
        const std::uint8_t* q = b + (kSum ? 4 : 2) + 8 * part;
        // int8 + 128 as bytes, then 128 off.
        four(u32_at<kAlign>(q) ^ 0x80808080u, 128.0f, d, 0.0f, v);
        four(u32_at<kAlign>(q + 4) ^ 0x80808080u, 128.0f, d, 0.0f, v + 4);
    }
};

struct Q2_K {  // scales[16], qs[64], d, dmin
    static constexpr int kBlock = 256;
    static constexpr int kBytes = 84;
    static constexpr bool kSplit = false;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int unit, int part, float (&v)[8]) {
        const int half = unit >> 2;
        const int shift = 2 * (unit & 3);
        const unsigned sc = b[8 * half + 2 * (unit & 3) + (part >> 1)];
        const float dl = f16_at(b + 80) * static_cast<float>(sc & 0x0F);
        const float ml = f16_at(b + 82) * static_cast<float>(sc >> 4);
        const std::uint8_t* q = b + 16 + 32 * half + 8 * part;
        four((u32_at<4>(q) >> shift) & 0x03030303u, 0.0f, dl, -ml, v);
        four((u32_at<4>(q + 4) >> shift) & 0x03030303u, 0.0f, dl, -ml, v + 4);
    }
};

struct Q3_K {  // hmask[32], qs[64], scales[12], d
    static constexpr int kBlock = 256;
    static constexpr int kBytes = 110;
    static constexpr bool kSplit = false;
    static __device__ __forceinline__ int scale(const std::uint8_t* packed, int index) {
        const int byte = index & 3;
        const int group = index >> 2;
        const int low = group < 2 ? packed[4 * group + byte] & 0x0F : packed[4 * (group - 2) + byte] >> 4;
        const int high = (packed[8 + byte] >> (2 * group)) & 3;
        return (low | (high << 4)) - 32;
    }
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int unit, int part, float (&v)[8]) {
        const int half = unit >> 2;
        const int j = unit & 3;
        const int shift = 2 * j;
        const int mask_shift = 4 * half + j;
        const float dl = f16_at(b + 108) * static_cast<float>(scale(b + 96, 8 * half + 2 * j + (part >> 1)));
        const std::uint8_t* q = b + 32 + 32 * half + 8 * part;
        const std::uint8_t* hm = b + 8 * part;
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const std::uint32_t low = (u32_at<2>(q + 4 * w) >> shift) & 0x03030303u;
            // 4 is subtracted where the high bit is clear: add 4 where it is set, subtract 4 always.
            const std::uint32_t high = (u32_at<2>(hm + 4 * w) >> mask_shift) & 0x01010101u;
            four(low + (high << 2), 4.0f, dl, 0.0f, v + 4 * w);
        }
    }
};

/// The 6-bit scale and minimum of sub-block j of q4_k / q5_k.
__device__ __forceinline__ void scale_min_k4(int j, const std::uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// q4_k / q5_k: d, dmin, scales[12], (q5_k: qh[32]), qs[128].
template <bool kFive>
struct QK45 {
    static constexpr int kBlock = 256;
    static constexpr int kBytes = kFive ? 176 : 144;
    static constexpr bool kSplit = false;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int unit, int part, float (&v)[8]) {
        int sc = 0;
        int mn = 0;
        scale_min_k4(unit, b + 4, sc, mn);
        const float dl = f16_at(b) * static_cast<float>(sc);
        const float ml = f16_at(b + 2) * static_cast<float>(mn);
        const std::uint8_t* q = b + (kFive ? 48 : 16) + 32 * (unit >> 1) + 8 * part;
        const int shift = 4 * (unit & 1);
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            std::uint32_t word = (u32_at<4>(q + 4 * w) >> shift) & 0x0F0F0F0Fu;
            if constexpr (kFive) {
                word |= ((u32_at<4>(b + 16 + 8 * part + 4 * w) >> unit) & 0x01010101u) << 4;
            }
            four(word, 0.0f, dl, -ml, v + 4 * w);
        }
    }
};

struct Q6_K {  // ql[128], qh[64], scales[16] (int8), d
    static constexpr int kBlock = 256;
    static constexpr int kBytes = 210;
    static constexpr bool kSplit = false;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int unit, int part, float (&v)[8]) {
        const int half = unit >> 2;
        const int t = unit & 3;
        const std::uint8_t* ql = b + 64 * half + 32 * (t & 1) + 8 * part;
        const std::uint8_t* qh = b + 128 + 32 * half + 8 * part;
        const auto scale =
            static_cast<float>(reinterpret_cast<const std::int8_t*>(b + 192 + 8 * half)[(part >> 1) + 2 * t]);
        const float dl = f16_at(b + 208) * scale;
        const int low_shift = 4 * (t >> 1);
        const int high_shift = 2 * t;
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const std::uint32_t low = (u32_at<2>(ql + 4 * w) >> low_shift) & 0x0F0F0F0Fu;
            const std::uint32_t high = (u32_at<2>(qh + 4 * w) >> high_shift) & 0x03030303u;
            four(low | (high << 4), 32.0f, dl, 0.0f, v + 4 * w);
        }
    }
};

struct Q8_K {  // d (f32), qs[256], bsums[16]
    static constexpr int kBlock = 256;
    static constexpr int kBytes = 292;
    static constexpr bool kSplit = false;
    static __device__ __forceinline__ void decode(const std::uint8_t* b, int unit, int part, float (&v)[8]) {
        const float d = *reinterpret_cast<const float*>(b);
        const std::uint8_t* q = b + 4 + 32 * unit + 8 * part;
        four(u32_at<4>(q) ^ 0x80808080u, 128.0f, d, 0.0f, v);
        four(u32_at<4>(q + 4) ^ 0x80808080u, 128.0f, d, 0.0f, v + 4);
    }
};

template <class Fn>
cudaError_t dispatch_weights(WeightType type, Fn&& fn) {
    switch (type) {
    case WeightType::kF32:
        return fn(F32Weights{});
    case WeightType::kF16:
        return fn(HalfWeights<__half>{});
    case WeightType::kBF16:
        return fn(HalfWeights<__nv_bfloat16>{});
    case WeightType::kQ4_0:
        return fn(Q4<false>{});
    case WeightType::kQ4_1:
        return fn(Q4<true>{});
    case WeightType::kQ5_0:
        return fn(Q5<false>{});
    case WeightType::kQ5_1:
        return fn(Q5<true>{});
    case WeightType::kQ8_0:
        return fn(Q8<false>{});
    case WeightType::kQ8_1:
        return fn(Q8<true>{});
    case WeightType::kQ2_K:
        return fn(Q2_K{});
    case WeightType::kQ3_K:
        return fn(Q3_K{});
    case WeightType::kQ4_K:
        return fn(QK45<false>{});
    case WeightType::kQ5_K:
        return fn(QK45<true>{});
    case WeightType::kQ6_K:
        return fn(Q6_K{});
    case WeightType::kQ8_K:
        return fn(Q8_K{});
    }
    return cudaErrorInvalidValue;
}

/// Decodes part `part` of unit `unit` of a row.
template <class L>
__device__ __forceinline__ void decode_part(const std::uint8_t* row, std::int64_t unit, int part, float (&v)[8]) {
    constexpr int kUnits = L::kBlock / 32;
    L::decode(row + (unit / kUnits) * L::kBytes, static_cast<int>(unit % kUnits), part, v);
}

/// Offsets within the unit of the two groups of 4 elements a part holds.
template <class L>
__device__ __forceinline__ int first_group(int part) {
    return L::kSplit ? 4 * part : 8 * part;
}
template <class L>
__device__ __forceinline__ int second_group(int part) {
    return L::kSplit ? 16 + 4 * part : 8 * part + 4;
}

// --- weight_gemv ----------------------------------------------------------------------------------------------------

constexpr int kGemvWarps = 4;

/// Each output feature gets `split` warps of a block (1, 2 or 4: more for long rows of few features, so that the GPU
/// has enough warps). A step of a warp covers 8 consecutive units of the weight row (4 threads per unit); every decoded
/// element is multiplied with the matching element of every row of x.
template <class L, int kRows>
__global__ void __launch_bounds__(kGemvWarps * 32)
    gemv_kernel(float* __restrict__ y, const float* __restrict__ x, const std::uint8_t* __restrict__ w,
                const float* __restrict__ bias, std::int64_t n, std::int64_t k, std::int64_t row_bytes, int split) {
    __shared__ float partial[kGemvWarps][kRows];
    const int lane = threadIdx.x % 32;
    const int warp = static_cast<int>(threadIdx.x / 32);
    const int part = lane & 3;
    const int sub = warp % split;
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * (kGemvWarps / split) + warp / split;
    const bool valid = j < n;
    const std::uint8_t* row = w + (valid ? j : 0) * row_bytes;
    const std::int64_t units = k / 32;
    const int first = first_group<L>(part);
    const int second = second_group<L>(part);
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc[r] = 0.0f;
    }
    for (std::int64_t u = valid ? sub * 8 + (lane >> 2) : units; u < units; u += 8 * split) {
        float v[8];
        decode_part<L>(row, u, part, v);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const float* xr = x + r * k + u * 32;
            const float4 a = *reinterpret_cast<const float4*>(xr + first);
            const float4 c = *reinterpret_cast<const float4*>(xr + second);
            float sum = v[0] * a.x;
            sum = fmaf(v[1], a.y, sum);
            sum = fmaf(v[2], a.z, sum);
            sum = fmaf(v[3], a.w, sum);
            sum = fmaf(v[4], c.x, sum);
            sum = fmaf(v[5], c.y, sum);
            sum = fmaf(v[6], c.z, sum);
            sum = fmaf(v[7], c.w, sum);
            acc[r] += sum;
        }
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc[r] = warp_reduce(acc[r], [](float a, float b) { return a + b; });
    }
    if (split > 1) {
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                partial[warp][r] = acc[r];
            }
        }
        __syncthreads();
        if (sub != 0) {
            return;
        }
        for (int other = 1; other < split; ++other) {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                acc[r] += partial[warp + other][r];
            }
        }
    }
    if (valid && lane == 0) {
        const float b = bias != nullptr ? bias[j] : 0.0f;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            y[r * n + j] = acc[r] + b;
        }
    }
}

template <class L, int kRows>
cudaError_t launch_gemv(float* y, const float* x, const void* w, const float* bias, int rows, std::int64_t n,
                        std::int64_t k, cudaStream_t stream, int split) {
    if constexpr (kRows > 1) {
        if (rows < kRows) {
            return launch_gemv<L, kRows - 1>(y, x, w, bias, rows, n, k, stream, split);
        }
    }
    if (split == 0) {
        // More warps per feature while the GPU has few warps and each still gets a step of 8 units.
        const std::int64_t units = k / 32;
        split = 1;
        while (split < kGemvWarps && n * split < 2048 && units >= 16 * split) {
            split *= 2;
        }
    }
    const std::int64_t row_bytes = k / L::kBlock * L::kBytes;
    const std::int64_t per_block = kGemvWarps / split;
    const auto blocks = static_cast<unsigned>((n + per_block - 1) / per_block);
    gemv_kernel<L, kRows><<<blocks, kGemvWarps * 32, 0, stream>>>(y, x, static_cast<const std::uint8_t*>(w), bias, n, k,
                                                                  row_bytes, split);
    return cudaGetLastError();
}

// --- dequantize_rows ------------------------------------------------------------------------------------------------

template <class T>
__device__ __forceinline__ void store4(T* out, const float* v) {
    if constexpr (std::is_same_v<T, float>) {
        *reinterpret_cast<float4*>(out) = make_float4(v[0], v[1], v[2], v[3]);
    } else {
        std::uint32_t words[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const T a = Traits<T>::store(v[2 * e]);
            const T b = Traits<T>::store(v[2 * e + 1]);
            words[e] = static_cast<std::uint32_t>(*reinterpret_cast<const unsigned short*>(&a)) |
                       (static_cast<std::uint32_t>(*reinterpret_cast<const unsigned short*>(&b)) << 16);
        }
        *reinterpret_cast<uint2*>(out) = make_uint2(words[0], words[1]);
    }
}

/// One thread per part (8 elements) of a unit.
template <class L, class T>
__global__ void dequantize_kernel(T* __restrict__ out, const std::uint8_t* __restrict__ w, std::int64_t first,
                                  std::int64_t units_per_row, std::int64_t row_bytes, std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int part = static_cast<int>(i & 3);
        const std::int64_t unit_index = i >> 2;
        const std::int64_t row = unit_index / units_per_row;
        const std::int64_t unit = unit_index % units_per_row;
        float v[8];
        decode_part<L>(w + (first + row) * row_bytes, unit, part, v);
        T* o = out + unit_index * 32;
        store4(o + first_group<L>(part), v);
        store4(o + second_group<L>(part), v + 4);
    }
}

/// One block per row.
template <class T>
__global__ void split_rows_kernel(T* __restrict__ parts, int count, const float* __restrict__ x,
                                  float* __restrict__ scales, std::int64_t rows, std::int64_t k) {
    const std::int64_t row = blockIdx.x;
    const float* in = x + row * k;
    float factor = 1.0f;
    if (scales != nullptr) {
        float top = 0.0f;
        for (std::int64_t i = threadIdx.x; i < k; i += blockDim.x) {
            top = fmaxf(top, fabsf(in[i]));
        }
        top = block_reduce(top, [](float a, float b) { return fmaxf(a, b); }, 0.0f);
        // scale = 2^(e - 14) with top < 2^e: the scaled row stays below 2^14 (f16 reaches 65504).
        int exponent = 0;
        if (top > 0.0f && top == top && top < infinity()) {
            frexpf(top, &exponent);
        }
        const float scale = ldexpf(1.0f, exponent - 14);
        factor = 1.0f / scale;
        if (threadIdx.x == 0) {
            scales[row] = scale;
        }
    }
    for (std::int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        float rest = in[i] * factor;
        for (int p = 0; p < count; ++p) {
            const T part = Traits<T>::store(rest);
            parts[(p * rows + row) * k + i] = part;
            rest -= Traits<T>::load(part);
        }
    }
}

__global__ void scale_rows_kernel(float* c, const float* scales, const float* bias, std::int64_t n,
                                  std::int64_t total) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        float value = c[i];
        if (scales != nullptr) {
            value *= scales[i / n];
        }
        if (bias != nullptr) {
            value += bias[i % n];
        }
        c[i] = value;
    }
}

template <class T>
__global__ void convert_kernel(void* out, const void* in, Scalar in_type, std::int64_t n) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        float value = 0;
        switch (in_type) {
        case Scalar::kF16:
            value = __half2float(static_cast<const __half*>(in)[i]);
            break;
        case Scalar::kBF16:
            value = __bfloat162float(static_cast<const __nv_bfloat16*>(in)[i]);
            break;
        default:
            value = static_cast<const float*>(in)[i];
            break;
        }
        static_cast<T*>(out)[i] = Traits<T>::store(value);
    }
}

}  // namespace

cudaError_t weight_gemv(float* y, const float* x, const void* w, WeightType type, const float* bias, int rows,
                        std::int64_t n, std::int64_t k, cudaStream_t stream, int split) {
    if (rows < 1 || rows > kGemvRows || k % 32 != 0 || (split != 0 && split != 1 && split != 2 && split != 4)) {
        return cudaErrorInvalidValue;
    }
    if (n == 0) {
        return cudaSuccess;
    }
    return dispatch_weights(type, [&](auto layout) {
        using L = decltype(layout);
        return launch_gemv<L, kGemvRows>(y, x, w, bias, rows, n, k, stream, split);
    });
}

cudaError_t dequantize_rows(void* out, Scalar out_type, const void* w, WeightType type, std::int64_t first,
                            std::int64_t count, std::int64_t k, cudaStream_t stream) {
    if (k % 32 != 0) {
        return cudaErrorInvalidValue;
    }
    const std::int64_t n = count * (k / 32) * 4;
    if (n == 0) {
        return cudaSuccess;
    }
    return dispatch_weights(type, [&](auto layout) {
        using L = decltype(layout);
        const std::int64_t row_bytes = k / L::kBlock * L::kBytes;
        return dispatch_float(out_type, [&](auto tag) {
            using T = decltype(tag);
            dequantize_kernel<L, T><<<blocks_for(n), kThreads, 0, stream>>>(
                static_cast<T*>(out), static_cast<const std::uint8_t*>(w), first, k / 32, row_bytes, n);
            return cudaGetLastError();
        });
    });
}

cudaError_t split_rows(void* parts, Scalar type, int count, const float* x, float* scales, std::int64_t rows,
                       std::int64_t k, cudaStream_t stream) {
    if (rows == 0 || k == 0) {
        return cudaSuccess;
    }
    auto launch = [&](auto tag) {
        using T = decltype(tag);
        split_rows_kernel<T>
            <<<static_cast<unsigned>(rows), kThreads, 0, stream>>>(static_cast<T*>(parts), count, x, scales, rows, k);
        return cudaGetLastError();
    };
    switch (type) {
    case Scalar::kF16:
        return launch(__half{});
    case Scalar::kBF16:
        return launch(__nv_bfloat16{});
    default:
        return cudaErrorInvalidValue;
    }
}

cudaError_t scale_rows(float* c, const float* scales, const float* bias, std::int64_t rows, std::int64_t n,
                       cudaStream_t stream) {
    const std::int64_t total = rows * n;
    if (total == 0 || (scales == nullptr && bias == nullptr)) {
        return cudaSuccess;
    }
    scale_rows_kernel<<<blocks_for(total), kThreads, 0, stream>>>(c, scales, bias, n, total);
    return cudaGetLastError();
}

cudaError_t convert(void* out, Scalar out_type, const void* in, Scalar in_type, std::int64_t count,
                    cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }
    return dispatch_float(out_type, [&](auto tag) {
        using T = decltype(tag);
        convert_kernel<T><<<blocks_for(count), kThreads, 0, stream>>>(out, in, in_type, count);
        return cudaGetLastError();
    });
}

}  // namespace ii::cuda
