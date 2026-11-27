// SPDX-License-Identifier: MIT
// ii-cuda attention on tensor cores for prompts (FlashAttention-2): a block takes 64 query rows of one head (16 per
// warp) and walks over the keys in tiles of 32; scores, the online softmax and the output stay in the registers of
// mma.sync m16n8k16 fragments. Operands are f16 (bf16 for bf16 models); f32 operands are split into a high and a low
// f16 part whose products are added (three products instead of one), so results keep about f32 precision.
// Needs compute capability 8.0 (the launcher in kernels_llm.cu falls back to the CUDA-core kernel otherwise).
#include "backends/cuda/kernels_common.cuh"

namespace ii::cuda {

namespace {

constexpr int kFlashWarps = 4;
constexpr int kFlashRows = 16 * kFlashWarps;
[[maybe_unused]] constexpr int kFlashKeys = 32;

/// Two floats as a packed pair of f16 (or bf16), the first in the low half.
template <bool kBf16>
__device__ __forceinline__ std::uint32_t pack(float a, float b) {
    if constexpr (kBf16) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<const std::uint32_t*>(&v);
    } else {
        const __half2 v = __floats2half2_rn(a, b);
        return *reinterpret_cast<const std::uint32_t*>(&v);
    }
}

/// What the packed pair loses: the pair of (a - high(a), b - high(b)).
template <bool kBf16>
__device__ __forceinline__ std::uint32_t pack_rest(float a, float b, std::uint32_t high) {
    if constexpr (kBf16) {
        const __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(&high);
        return pack<true>(a - __low2float(h), b - __high2float(h));
    } else {
        const __half2 h = *reinterpret_cast<const __half2*>(&high);
        return pack<false>(a - __low2float(h), b - __high2float(h));
    }
}

/// c += a b for a 16x16 A (row-major fragment) and a 16x8 B (column fragment), f32 accumulators.
template <bool kBf16>
__device__ __forceinline__ void mma(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (kBf16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
#else
    (void)c;
    (void)a;
    (void)b0;
    (void)b1;
#endif
}

/// The keys [begin, end) query row i attends among `used` keys (see `sdpa`).
struct Range {
    std::int64_t begin;
    std::int64_t end;
};
[[maybe_unused]] __device__ __forceinline__ Range keys_of(const AttentionParams& p, std::int64_t i, std::int64_t used) {
    const std::int64_t position = i + used - p.sq;
    std::int64_t end = used;
    if (p.causal) {
        end = position + 1 < 0 ? 0 : (position + 1 > used ? used : position + 1);
    }
    std::int64_t begin = 0;
    if (p.window > 0) {
        const std::int64_t first = position - p.window + 1;
        begin = first < 0 ? 0 : (first > used ? used : first);
    }
    return {begin, end};
}

/// kSplitQ: q (and so the probabilities) carry a low part; kSplitKV: k and v do.
template <class TQ, class TKV, int kD, bool kBf16, bool kSplitQ, bool kSplitKV>
__global__ void __launch_bounds__(kFlashWarps * 32)
    flash_kernel(TQ* __restrict__ out, const TQ* __restrict__ q, const TKV* __restrict__ k, const TKV* __restrict__ v,
                 const void* __restrict__ mask, AttentionParams p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int kParts = kSplitKV ? 2 : 1;
    constexpr int kKStride = kD + 8;          // halves per key row (padding: no bank conflicts)
    constexpr int kVStride = kFlashKeys + 8;  // halves per row of the transposed values
    constexpr int kSteps = kD / 16;           // k-steps of the score product
    constexpr int kOut = kD / 8;              // output fragments per warp
    __shared__ alignas(16) std::uint16_t ks[kParts][kFlashKeys * kKStride];
    __shared__ alignas(16) std::uint16_t vt[kParts][kD * kVStride];

    const int warp = static_cast<int>(threadIdx.x / 32);
    const int lane = static_cast<int>(threadIdx.x % 32);
    const int g = lane >> 2;
    const int t = lane & 3;
    const std::int64_t row_tiles = (p.sq + kFlashRows - 1) / kFlashRows;
    std::int64_t block = blockIdx.x;
    const std::int64_t tile = block % row_tiles;
    block /= row_tiles;
    const std::int64_t h = block % p.heads;
    const std::int64_t b = block / p.heads;
    const std::int64_t hk = h / (p.heads / p.kv_heads);
    const std::int64_t first = tile * kFlashRows;
    const std::int64_t rows[2] = {first + warp * 16 + g, first + warp * 16 + g + 8};

    // The query fragments (rows r0 / r1, columns 2t, 2t + 1, 2t + 8, 2t + 9 of every 16 features).
    const TQ* qb = q + (b * p.heads + h) * p.sq * kD;
    std::uint32_t qh[kSteps][4];
    [[maybe_unused]] std::uint32_t ql[kSplitQ ? kSteps : 1][4];
#pragma unroll
    for (int kk = 0; kk < kSteps; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const std::int64_t row = rows[r & 1];
            const int col = kk * 16 + 2 * t + (r & 2 ? 8 : 0);
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (row < p.sq) {
                x0 = Traits<TQ>::load(qb[row * kD + col]);
                x1 = Traits<TQ>::load(qb[row * kD + col + 1]);
            }
            qh[kk][r] = pack<kBf16>(x0, x1);
            if constexpr (kSplitQ) {
                ql[kk][r] = pack_rest<kBf16>(x0, x1, qh[kk][r]);
            }
        }
    }

    std::int64_t used = p.skv;
    if (p.kv_len != nullptr) {
        used = p.kv_len64 ? *static_cast<const std::int64_t*>(p.kv_len) : *static_cast<const std::int32_t*>(p.kv_len);
        if (used < 0 || used > p.skv) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                report_error(p.error, DeviceErrorCode::kKvLength, used, p.skv);
            }
            used = used < 0 ? 0 : p.skv;
        }
    }
    const std::int64_t last = (first + kFlashRows < p.sq ? first + kFlashRows : p.sq) - 1;
    const std::int64_t begin = keys_of(p, first, used).begin;
    const std::int64_t end = keys_of(p, last, used).end;
    Range mine[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        mine[r] = rows[r] < p.sq ? keys_of(p, rows[r], used) : Range{0, 0};
    }

    const TKV* kh = k + (b * p.kv_heads + hk) * p.skv * kD;
    const TKV* vh = v + (b * p.kv_heads + hk) * p.skv * kD;
    float o[kOut][4];
#pragma unroll
    for (int f = 0; f < kOut; ++f) {
        o[f][0] = o[f][1] = o[f][2] = o[f][3] = 0.0f;
    }
    float m[2] = {-infinity(), -infinity()};
    float l[2] = {0.0f, 0.0f};

    for (std::int64_t t0 = begin; t0 < end; t0 += kFlashKeys) {
        const int count = static_cast<int>(end - t0 < kFlashKeys ? end - t0 : kFlashKeys);
        __syncthreads();  // the previous tile is consumed
        // Keys [t0, t0 + count) as rows of halves; values transposed (features x keys). Missing keys are zeros.
        for (int e = static_cast<int>(threadIdx.x); e < kFlashKeys * kD / 2; e += kFlashWarps * 32) {
            const int key = e / (kD / 2);
            const int d = 2 * (e % (kD / 2));
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (key < count) {
                x0 = Traits<TKV>::load(kh[(t0 + key) * kD + d]);
                x1 = Traits<TKV>::load(kh[(t0 + key) * kD + d + 1]);
            }
            const std::uint32_t high = pack<kBf16>(x0, x1);
            *reinterpret_cast<std::uint32_t*>(&ks[0][key * kKStride + d]) = high;
            if constexpr (kSplitKV) {
                *reinterpret_cast<std::uint32_t*>(&ks[1][key * kKStride + d]) = pack_rest<kBf16>(x0, x1, high);
            }
        }
        for (int e = static_cast<int>(threadIdx.x); e < kFlashKeys * kD / 2; e += kFlashWarps * 32) {
            // Two keys of one feature per thread: a packed pair of the transposed tile.
            const int pair = e / kD;
            const int d = e % kD;
            const int key = 2 * pair;
            const float x0 = key < count ? Traits<TKV>::load(vh[(t0 + key) * kD + d]) : 0.0f;
            const float x1 = key + 1 < count ? Traits<TKV>::load(vh[(t0 + key + 1) * kD + d]) : 0.0f;
            const std::uint32_t high = pack<kBf16>(x0, x1);
            *reinterpret_cast<std::uint32_t*>(&vt[0][d * kVStride + key]) = high;
            if constexpr (kSplitKV) {
                *reinterpret_cast<std::uint32_t*>(&vt[1][d * kVStride + key]) = pack_rest<kBf16>(x0, x1, high);
            }
        }
        __syncthreads();

        // Scores of the warp's 16 rows and the tile's 32 keys: 4 fragments of 8 keys.
        float s[4][4];
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            s[f][0] = s[f][1] = s[f][2] = s[f][3] = 0.0f;
        }
#pragma unroll
        for (int kk = 0; kk < kSteps; ++kk) {
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int at = (f * 8 + g) * kKStride + kk * 16 + 2 * t;
                const std::uint32_t b0 = *reinterpret_cast<const std::uint32_t*>(&ks[0][at]);
                const std::uint32_t b1 = *reinterpret_cast<const std::uint32_t*>(&ks[0][at + 8]);
                mma<kBf16>(s[f], qh[kk], b0, b1);
                if constexpr (kSplitKV) {
                    mma<kBf16>(s[f], qh[kk], *reinterpret_cast<const std::uint32_t*>(&ks[1][at]),
                               *reinterpret_cast<const std::uint32_t*>(&ks[1][at + 8]));
                }
                if constexpr (kSplitQ) {
                    mma<kBf16>(s[f], ql[kk], b0, b1);
                }
            }
        }
        // Scale, soft cap and masks: element e of fragment f is row rows[e >> 1], key t0 + 8 f + 2 t + (e & 1).
#pragma unroll
        for (int f = 0; f < 4; ++f) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = e >> 1;
                const std::int64_t key = t0 + f * 8 + 2 * t + (e & 1);
                float value = s[f][e] * p.scale;
                if (p.softcap > 0.0f) {
                    value = p.softcap * tanhf(value / p.softcap);
                }
                bool keep = key < t0 + count && key >= mine[r].begin && key < mine[r].end;
                if (keep && p.mask_kind != 0) {
                    const std::int64_t offset = b * p.mask_strides[0] + h * p.mask_strides[1] +
                                                rows[r] * p.mask_strides[2] + key * p.mask_strides[3];
                    if (p.mask_kind == 1) {
                        keep = static_cast<const bool*>(mask)[offset];
                    } else {
                        value += Traits<TQ>::load(static_cast<const TQ*>(mask)[offset]);
                    }
                }
                s[f][e] = keep ? value : -infinity();
            }
        }
        // Online softmax per row; a row's values are spread over the 4 threads of a quad.
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float top = -infinity();
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                top = fmaxf(top, fmaxf(s[f][2 * r], s[f][2 * r + 1]));
            }
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
            const float m_new = fmaxf(m[r], top);
            const float correction = m_new == -infinity() ? 1.0f : expf(m[r] - m_new);
            float sum = 0.0f;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
#pragma unroll
                for (int e = 2 * r; e < 2 * r + 2; ++e) {
                    const float pe = s[f][e] == -infinity() ? 0.0f : expf(s[f][e] - m_new);
                    s[f][e] = pe;
                    sum += pe;
                }
            }
            l[r] = l[r] * correction + sum;
            m[r] = m_new;
#pragma unroll
            for (int f = 0; f < kOut; ++f) {
                o[f][2 * r] *= correction;
                o[f][2 * r + 1] *= correction;
            }
        }
        // o += P V: the score fragments are the A fragments of P (2 steps of 16 keys).
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            std::uint32_t ph[4];
            [[maybe_unused]] std::uint32_t pl[4];
            ph[0] = pack<kBf16>(s[2 * kk][0], s[2 * kk][1]);
            ph[1] = pack<kBf16>(s[2 * kk][2], s[2 * kk][3]);
            ph[2] = pack<kBf16>(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            ph[3] = pack<kBf16>(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            if constexpr (kSplitQ) {
                pl[0] = pack_rest<kBf16>(s[2 * kk][0], s[2 * kk][1], ph[0]);
                pl[1] = pack_rest<kBf16>(s[2 * kk][2], s[2 * kk][3], ph[1]);
                pl[2] = pack_rest<kBf16>(s[2 * kk + 1][0], s[2 * kk + 1][1], ph[2]);
                pl[3] = pack_rest<kBf16>(s[2 * kk + 1][2], s[2 * kk + 1][3], ph[3]);
            }
#pragma unroll
            for (int f = 0; f < kOut; ++f) {
                const int at = (f * 8 + g) * kVStride + kk * 16 + 2 * t;
                const std::uint32_t b0 = *reinterpret_cast<const std::uint32_t*>(&vt[0][at]);
                const std::uint32_t b1 = *reinterpret_cast<const std::uint32_t*>(&vt[0][at + 8]);
                mma<kBf16>(o[f], ph, b0, b1);
                if constexpr (kSplitKV) {
                    mma<kBf16>(o[f], ph, *reinterpret_cast<const std::uint32_t*>(&vt[1][at]),
                               *reinterpret_cast<const std::uint32_t*>(&vt[1][at + 8]));
                }
                if constexpr (kSplitQ) {
                    mma<kBf16>(o[f], pl, b0, b1);
                }
            }
        }
    }
    // Normalize by the row sums (quad-wide); fully masked rows produce NaN.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float total = l[r];
        total += __shfl_xor_sync(0xffffffffu, total, 1);
        total += __shfl_xor_sync(0xffffffffu, total, 2);
        if (rows[r] >= p.sq) {
            continue;
        }
        TQ* orow = out + ((b * p.heads + h) * p.sq + rows[r]) * kD;
        const float inverse = 1.0f / total;
#pragma unroll
        for (int f = 0; f < kOut; ++f) {
            const int col = f * 8 + 2 * t;
            orow[col] = Traits<TQ>::store(total == 0.0f ? quiet_nan() : o[f][2 * r] * inverse);
            orow[col + 1] = Traits<TQ>::store(total == 0.0f ? quiet_nan() : o[f][2 * r + 1] * inverse);
        }
    }
#else
    (void)out;
    (void)q;
    (void)k;
    (void)v;
    (void)mask;
    (void)p;
#endif
}

template <class TQ, class TKV, bool kBf16, bool kSplitQ, bool kSplitKV>
cudaError_t launch_flash(void* out, const void* q, const void* k, const void* v, const void* mask,
                         const AttentionParams& p, cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>(p.batch * p.heads * ((p.sq + kFlashRows - 1) / kFlashRows));
    auto launch = [&](auto kernel) {
        kernel<<<blocks, kFlashWarps * 32, 0, stream>>>(static_cast<TQ*>(out), static_cast<const TQ*>(q),
                                                        static_cast<const TKV*>(k), static_cast<const TKV*>(v), mask,
                                                        p);
        return cudaGetLastError();
    };
    switch (p.d) {
    case 64:
        return launch(flash_kernel<TQ, TKV, 64, kBf16, kSplitQ, kSplitKV>);
    case 96:
        return launch(flash_kernel<TQ, TKV, 96, kBf16, kSplitQ, kSplitKV>);
    case 128:
        return launch(flash_kernel<TQ, TKV, 128, kBf16, kSplitQ, kSplitKV>);
    default:
        return cudaErrorInvalidValue;
    }
}

}  // namespace

bool flash_attention_supported(const AttentionParams& p, Scalar q_type, Scalar kv_type, int arch) {
    if (arch < 80 || p.d != p.dv || (p.d != 64 && p.d != 96 && p.d != 128)) {
        return false;
    }
    return q_type == Scalar::kF32 || q_type == kv_type;
}

cudaError_t flash_attention(void* out, const void* q, const void* k, const void* v, const void* mask, Scalar q_type,
                            Scalar kv_type, const AttentionParams& p, cudaStream_t stream) {
    if (q_type == Scalar::kF16 && kv_type == Scalar::kF16) {
        return launch_flash<__half, __half, false, false, false>(out, q, k, v, mask, p, stream);
    }
    if (q_type == Scalar::kBF16 && kv_type == Scalar::kBF16) {
        return launch_flash<__nv_bfloat16, __nv_bfloat16, true, false, false>(out, q, k, v, mask, p, stream);
    }
    if (q_type != Scalar::kF32) {
        return cudaErrorInvalidValue;
    }
    switch (kv_type) {
    case Scalar::kF32:
        return launch_flash<float, float, false, true, true>(out, q, k, v, mask, p, stream);
    case Scalar::kF16:
        return launch_flash<float, __half, false, true, false>(out, q, k, v, mask, p, stream);
    case Scalar::kBF16:
        return launch_flash<float, __nv_bfloat16, false, true, true>(out, q, k, v, mask, p, stream);
    default:
        return cudaErrorInvalidValue;
    }
}

}  // namespace ii::cuda
