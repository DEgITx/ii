// SPDX-License-Identifier: MIT
// Panel kernels of the quantized products: 8 weight rows (repacked panels, see kernel_table.h) times up to 4
// activation rows. AVX2 (with vpdpbusd in the AVX-VNNI variant) and Armv8.2 dot products; other variants have none and
// use the row kernels of quant.cpp.
#include "backends/cpu/isa/isa_kernels.h"
#include "backends/cpu/isa/simd.h"

namespace ii::cpu::II_ISA_NS {

namespace {

[[maybe_unused]] constexpr std::int64_t kQ8x32Bytes = 40;
[[maybe_unused]] constexpr std::int64_t kQ8KBytes = 292;

[[maybe_unused]] II_SIMD_INLINE float load_f32(const std::uint8_t* p) {
    float f;
    std::memcpy(&f, p, 4);
    return f;
}

[[maybe_unused]] II_SIMD_INLINE std::int32_t load_i32(const std::uint8_t* p) {
    std::int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

#if defined(II_ISA_IS_AVX2) || defined(II_ISA_IS_AVX512)
#define II_QGEMM_AVX2 1

/// acc + sums of 4 products of unsigned bytes u and signed bytes s per int32 lane (u * s pairs below 2^15 without
/// VNNI).
II_SIMD_INLINE __m256i dot_add(__m256i acc, __m256i u, __m256i s) {
#if defined(II_ISA_IS_AVX_VNNI)
    return _mm256_dpbusd_avx_epi32(acc, u, s);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));
#endif
}

/// The 32 nibbles of a step: rows 0-3 from the low halves of 16 bytes, rows 4-7 from the high halves.
II_SIMD_INLINE __m256i step_nibbles(const std::uint8_t* p) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m256i both = _mm256_insertf128_si256(_mm256_castsi128_si256(bytes), _mm_srli_epi16(bytes, 4), 1);
    return _mm256_and_si256(both, _mm256_set1_epi8(0x0F));
}

/// 0xFF in byte i where bit i of the 32-bit word at p is set.
II_SIMD_INLINE __m256i step_bits(const std::uint8_t* p) {
    const __m256i shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(load_i32(p)), shuffle);
    bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

II_SIMD_INLINE __m256 load_u8_as_f32(const std::uint8_t* p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}

II_SIMD_INLINE __m256 load_i8_as_f32(const std::uint8_t* p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}

/// The 32-value types (q4_0, q5_0, q8_0): `decode(block, step)` gives the 32 unsigned weight bytes of a step,
/// weights being `unsigned - offset`.
template <int R, int kBlockBytes, int kOffset, class Decode>
void gemm32(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
            std::int64_t out_stride, Decode decode) {
    __m256 facc[R];
    for (int r = 0; r < R; ++r) {
        facc[r] = _mm256_setzero_ps();
    }
    for (std::int64_t b = 0; b < k / 32; ++b) {
        const std::uint8_t* p = panel + b * kBlockBytes;
        __m256i acc[R];
        for (int r = 0; r < R; ++r) {
            acc[r] = _mm256_setzero_si256();
        }
        const std::uint8_t* ab = a + b * kQ8x32Bytes + 8;
        for (int s = 0; s < 8; ++s) {
            const __m256i w = decode(p, s);
            for (int r = 0; r < R; ++r) {
                acc[r] = dot_add(acc[r], w, _mm256_set1_epi32(load_i32(ab + r * a_stride + 4 * s)));
            }
        }
        const __m256 d = _mm256_loadu_ps(reinterpret_cast<const float*>(p));
        for (int r = 0; r < R; ++r) {
            const std::uint8_t* head = a + r * a_stride + b * kQ8x32Bytes;
            facc[r] =
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[r]), _mm256_mul_ps(d, _mm256_set1_ps(load_f32(head))), facc[r]);
            if constexpr (kOffset != 0) {
                facc[r] =
                    _mm256_fnmadd_ps(d, _mm256_set1_ps(static_cast<float>(kOffset) * load_f32(head + 4)), facc[r]);
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        _mm256_storeu_ps(out + r * out_stride, facc[r]);
    }
}

template <int R>
void gemm_q4_0(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    gemm32<R, 160, 8>(k, panel, a, a_stride, out, out_stride,
                      [](const std::uint8_t* p, int s) { return step_nibbles(p + 32 + 16 * s); });
}

template <int R>
void gemm_q5_0(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    gemm32<R, 192, 16>(k, panel, a, a_stride, out, out_stride, [](const std::uint8_t* p, int s) {
        const __m256i high = _mm256_and_si256(step_bits(p + 160 + 4 * s), _mm256_set1_epi8(0x10));
        return _mm256_or_si256(step_nibbles(p + 32 + 16 * s), high);
    });
}

template <int R>
void gemm_q8_0(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
#if defined(II_ISA_IS_AVX_VNNI)
    gemm32<R, 288, 128>(k, panel, a, a_stride, out, out_stride, [](const std::uint8_t* p, int s) {
        return _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32 + 32 * s)),
                                _mm256_set1_epi8(static_cast<char>(0x80)));
    });
#else
    // Signed weights without VNNI: |w| times the activations with w's sign.
    __m256 facc[R];
    for (int r = 0; r < R; ++r) {
        facc[r] = _mm256_setzero_ps();
    }
    const __m256i ones = _mm256_set1_epi16(1);
    for (std::int64_t b = 0; b < k / 32; ++b) {
        const std::uint8_t* p = panel + b * 288;
        __m256i acc[R];
        for (int r = 0; r < R; ++r) {
            acc[r] = _mm256_setzero_si256();
        }
        const std::uint8_t* ab = a + b * kQ8x32Bytes + 8;
        for (int s = 0; s < 8; ++s) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32 + 32 * s));
            const __m256i aw = _mm256_sign_epi8(w, w);
            for (int r = 0; r < R; ++r) {
                const __m256i x = _mm256_sign_epi8(_mm256_set1_epi32(load_i32(ab + r * a_stride + 4 * s)), w);
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, x), ones));
            }
        }
        const __m256 d = _mm256_loadu_ps(reinterpret_cast<const float*>(p));
        for (int r = 0; r < R; ++r) {
            const float da = load_f32(a + r * a_stride + b * kQ8x32Bytes);
            facc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[r]), _mm256_mul_ps(d, _mm256_set1_ps(da)), facc[r]);
        }
    }
    for (int r = 0; r < R; ++r) {
        _mm256_storeu_ps(out + r * out_stride, facc[r]);
    }
#endif
}

template <int R>
void gemm_q4_k(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    __m256 facc[R];
    for (int r = 0; r < R; ++r) {
        facc[r] = _mm256_setzero_ps();
    }
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* p = panel + b * 1216;
        const __m256 d = _mm256_loadu_ps(reinterpret_cast<const float*>(p));
        const __m256 dmin = _mm256_loadu_ps(reinterpret_cast<const float*>(p + 32));
        const std::uint8_t* ab = a + b * kQ8KBytes;
        for (int sub = 0; sub < 8; ++sub) {
            __m256i acc[R];
            for (int r = 0; r < R; ++r) {
                acc[r] = _mm256_setzero_si256();
            }
            for (int s = 0; s < 8; ++s) {
                const int step = sub * 8 + s;
                const __m256i w = step_nibbles(p + 192 + 16 * step);
                for (int r = 0; r < R; ++r) {
                    acc[r] = dot_add(acc[r], w, _mm256_set1_epi32(load_i32(ab + r * a_stride + 4 + 4 * step)));
                }
            }
            const __m256 scale = _mm256_mul_ps(d, load_u8_as_f32(p + 64 + 8 * sub));
            const __m256 minimum = _mm256_mul_ps(dmin, load_u8_as_f32(p + 128 + 8 * sub));
            for (int r = 0; r < R; ++r) {
                const std::uint8_t* head = ab + r * a_stride;
                std::int16_t sums[2];
                std::memcpy(sums, head + 260 + 4 * sub, 4);
                const float da = load_f32(head);
                facc[r] =
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[r]), _mm256_mul_ps(scale, _mm256_set1_ps(da)), facc[r]);
                facc[r] =
                    _mm256_fnmadd_ps(minimum, _mm256_set1_ps(da * static_cast<float>(sums[0] + sums[1])), facc[r]);
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        _mm256_storeu_ps(out + r * out_stride, facc[r]);
    }
}

template <int R>
void gemm_q6_k(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    __m256 facc[R];
    for (int r = 0; r < R; ++r) {
        facc[r] = _mm256_setzero_ps();
    }
    const __m256i m3 = _mm256_set1_epi8(3);
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* p = panel + b * 1696;
        const __m256 d = _mm256_loadu_ps(reinterpret_cast<const float*>(p));
        const std::uint8_t* ab = a + b * kQ8KBytes;
        for (int g = 0; g < 16; ++g) {
            const std::uint8_t* group = p + 160 + 96 * g;
            const __m256i high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group + 64));
            __m256i acc[R];
            for (int r = 0; r < R; ++r) {
                acc[r] = _mm256_setzero_si256();
            }
            for (int t = 0; t < 4; ++t) {
                const __m256i h = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high, 2 * t), m3), 4);
                const __m256i w = _mm256_or_si256(step_nibbles(group + 16 * t), h);
                const int step = 4 * g + t;
                for (int r = 0; r < R; ++r) {
                    acc[r] = dot_add(acc[r], w, _mm256_set1_epi32(load_i32(ab + r * a_stride + 4 + 4 * step)));
                }
            }
            const __m256 scale = _mm256_mul_ps(d, load_i8_as_f32(p + 32 + 8 * g));
            for (int r = 0; r < R; ++r) {
                const std::uint8_t* head = ab + r * a_stride;
                std::int16_t sum;
                std::memcpy(&sum, head + 260 + 2 * g, 2);
                const float da = load_f32(head);
                // (q - 32) * a: the offset through the sum of the group's activations.
                facc[r] =
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[r]), _mm256_mul_ps(scale, _mm256_set1_ps(da)), facc[r]);
                facc[r] = _mm256_fnmadd_ps(scale, _mm256_set1_ps(32.0f * da * static_cast<float>(sum)), facc[r]);
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        _mm256_storeu_ps(out + r * out_stride, facc[r]);
    }
}

#elif defined(II_ISA_IS_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define II_QGEMM_NEON 1

/// The two halves of a step: rows 0-3 and rows 4-7, 4 values each, as signed bytes minus `offset`.
II_SIMD_INLINE void step_nibbles(const std::uint8_t* p, int offset, int8x16_t& lo, int8x16_t& hi) {
    const uint8x16_t bytes = vld1q_u8(p);
    lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(0x0F))), vdupq_n_s8(static_cast<std::int8_t>(offset)));
    hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), vdupq_n_s8(static_cast<std::int8_t>(offset)));
}

/// 0x10 in byte i where bit i of the 32-bit word at p is set (first 16 and last 16 bytes).
II_SIMD_INLINE void step_bits16(const std::uint8_t* p, uint8x16_t& lo, uint8x16_t& hi) {
    const uint8x16_t mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x8_t b0 = vdup_n_u8(p[0]);
    const uint8x8_t b1 = vdup_n_u8(p[1]);
    const uint8x8_t b2 = vdup_n_u8(p[2]);
    const uint8x8_t b3 = vdup_n_u8(p[3]);
    lo = vandq_u8(vtstq_u8(vcombine_u8(b0, b1), mask), vdupq_n_u8(0x10));
    hi = vandq_u8(vtstq_u8(vcombine_u8(b2, b3), mask), vdupq_n_u8(0x10));
}

template <int R>
II_SIMD_INLINE void store_rows(float32x4_t (&lo)[R], float32x4_t (&hi)[R], float* out, std::int64_t out_stride) {
    for (int r = 0; r < R; ++r) {
        vst1q_f32(out + r * out_stride, lo[r]);
        vst1q_f32(out + r * out_stride + 4, hi[r]);
    }
}

/// q4_0 / q5_0 / q8_0: per block of 32 values, d[8] and 8 steps.
template <int R, int kBlockBytes, class Decode>
void gemm32(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
            std::int64_t out_stride, Decode decode) {
    float32x4_t flo[R];
    float32x4_t fhi[R];
    for (int r = 0; r < R; ++r) {
        flo[r] = vdupq_n_f32(0);
        fhi[r] = vdupq_n_f32(0);
    }
    for (std::int64_t b = 0; b < k / 32; ++b) {
        const std::uint8_t* p = panel + b * kBlockBytes;
        int32x4_t lo_acc[R];
        int32x4_t hi_acc[R];
        for (int r = 0; r < R; ++r) {
            lo_acc[r] = vdupq_n_s32(0);
            hi_acc[r] = vdupq_n_s32(0);
        }
        const std::uint8_t* ab = a + b * kQ8x32Bytes + 8;
        for (int half = 0; half < 2; ++half) {
            int8x16_t x[R];
            for (int r = 0; r < R; ++r) {
                x[r] = vld1q_s8(reinterpret_cast<const std::int8_t*>(ab + r * a_stride + 16 * half));
            }
#define II_STEP(lane)                                                                                                  \
    {                                                                                                                  \
        int8x16_t lo;                                                                                                  \
        int8x16_t hi;                                                                                                  \
        decode(p, 4 * half + (lane), lo, hi);                                                                          \
        for (int r = 0; r < R; ++r) {                                                                                  \
            lo_acc[r] = vdotq_laneq_s32(lo_acc[r], lo, x[r], lane);                                                    \
            hi_acc[r] = vdotq_laneq_s32(hi_acc[r], hi, x[r], lane);                                                    \
        }                                                                                                              \
    }
            II_STEP(0)
            II_STEP(1)
            II_STEP(2)
            II_STEP(3)
#undef II_STEP
        }
        const float32x4_t dlo = vld1q_f32(reinterpret_cast<const float*>(p));
        const float32x4_t dhi = vld1q_f32(reinterpret_cast<const float*>(p) + 4);
        for (int r = 0; r < R; ++r) {
            const float da = load_f32(a + r * a_stride + b * kQ8x32Bytes);
            flo[r] = vfmaq_f32(flo[r], vcvtq_f32_s32(lo_acc[r]), vmulq_n_f32(dlo, da));
            fhi[r] = vfmaq_f32(fhi[r], vcvtq_f32_s32(hi_acc[r]), vmulq_n_f32(dhi, da));
        }
    }
    store_rows<R>(flo, fhi, out, out_stride);
}

template <int R>
void gemm_q4_0(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    gemm32<R, 160>(
        k, panel, a, a_stride, out, out_stride,
        [](const std::uint8_t* p, int s, int8x16_t& lo, int8x16_t& hi) { step_nibbles(p + 32 + 16 * s, 8, lo, hi); });
}

template <int R>
void gemm_q5_0(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    gemm32<R, 192>(
        k, panel, a, a_stride, out, out_stride, [](const std::uint8_t* p, int s, int8x16_t& lo, int8x16_t& hi) {
            const uint8x16_t bytes = vld1q_u8(p + 32 + 16 * s);
            uint8x16_t hlo;
            uint8x16_t hhi;
            step_bits16(p + 160 + 4 * s, hlo, hhi);
            lo = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(bytes, vdupq_n_u8(0x0F)), hlo)), vdupq_n_s8(16));
            hi = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(bytes, 4), hhi)), vdupq_n_s8(16));
        });
}

template <int R>
void gemm_q8_0(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    gemm32<R, 288>(k, panel, a, a_stride, out, out_stride,
                   [](const std::uint8_t* p, int s, int8x16_t& lo, int8x16_t& hi) {
                       lo = vld1q_s8(reinterpret_cast<const std::int8_t*>(p + 32 + 32 * s));
                       hi = vld1q_s8(reinterpret_cast<const std::int8_t*>(p + 32 + 32 * s + 16));
                   });
}

template <int R>
void gemm_q4_k(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    float32x4_t flo[R];
    float32x4_t fhi[R];
    for (int r = 0; r < R; ++r) {
        flo[r] = vdupq_n_f32(0);
        fhi[r] = vdupq_n_f32(0);
    }
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* p = panel + b * 1216;
        const float32x4_t dlo = vld1q_f32(reinterpret_cast<const float*>(p));
        const float32x4_t dhi = vld1q_f32(reinterpret_cast<const float*>(p) + 4);
        const float32x4_t mlo = vld1q_f32(reinterpret_cast<const float*>(p + 32));
        const float32x4_t mhi = vld1q_f32(reinterpret_cast<const float*>(p + 32) + 4);
        const std::uint8_t* ab = a + b * kQ8KBytes;
        for (int sub = 0; sub < 8; ++sub) {
            int32x4_t lo_acc[R];
            int32x4_t hi_acc[R];
            for (int r = 0; r < R; ++r) {
                lo_acc[r] = vdupq_n_s32(0);
                hi_acc[r] = vdupq_n_s32(0);
            }
            for (int half = 0; half < 2; ++half) {
                int8x16_t x[R];
                for (int r = 0; r < R; ++r) {
                    x[r] = vld1q_s8(reinterpret_cast<const std::int8_t*>(ab + r * a_stride + 4 + 32 * sub + 16 * half));
                }
#define II_STEP(lane)                                                                                                  \
    {                                                                                                                  \
        int8x16_t lo;                                                                                                  \
        int8x16_t hi;                                                                                                  \
        step_nibbles(p + 192 + 16 * (sub * 8 + 4 * half + (lane)), 0, lo, hi);                                         \
        for (int r = 0; r < R; ++r) {                                                                                  \
            lo_acc[r] = vdotq_laneq_s32(lo_acc[r], lo, x[r], lane);                                                    \
            hi_acc[r] = vdotq_laneq_s32(hi_acc[r], hi, x[r], lane);                                                    \
        }                                                                                                              \
    }
                II_STEP(0)
                II_STEP(1)
                II_STEP(2)
                II_STEP(3)
#undef II_STEP
            }
            const uint8x8_t sc8 = vld1_u8(p + 64 + 8 * sub);
            const uint8x8_t mn8 = vld1_u8(p + 128 + 8 * sub);
            const uint16x8_t sc16 = vmovl_u8(sc8);
            const uint16x8_t mn16 = vmovl_u8(mn8);
            const float32x4_t slo = vmulq_f32(dlo, vcvtq_f32_u32(vmovl_u16(vget_low_u16(sc16))));
            const float32x4_t shi = vmulq_f32(dhi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(sc16))));
            const float32x4_t nlo = vmulq_f32(mlo, vcvtq_f32_u32(vmovl_u16(vget_low_u16(mn16))));
            const float32x4_t nhi = vmulq_f32(mhi, vcvtq_f32_u32(vmovl_u16(vget_high_u16(mn16))));
            for (int r = 0; r < R; ++r) {
                const std::uint8_t* head = ab + r * a_stride;
                std::int16_t sums[2];
                std::memcpy(sums, head + 260 + 4 * sub, 4);
                const float da = load_f32(head);
                const float ms = da * static_cast<float>(sums[0] + sums[1]);
                flo[r] =
                    vfmsq_f32(vfmaq_f32(flo[r], vcvtq_f32_s32(lo_acc[r]), vmulq_n_f32(slo, da)), nlo, vdupq_n_f32(ms));
                fhi[r] =
                    vfmsq_f32(vfmaq_f32(fhi[r], vcvtq_f32_s32(hi_acc[r]), vmulq_n_f32(shi, da)), nhi, vdupq_n_f32(ms));
            }
        }
    }
    store_rows<R>(flo, fhi, out, out_stride);
}

template <int R>
void gemm_q6_k(std::int64_t k, const std::uint8_t* panel, const std::uint8_t* a, std::int64_t a_stride, float* out,
               std::int64_t out_stride) {
    float32x4_t flo[R];
    float32x4_t fhi[R];
    for (int r = 0; r < R; ++r) {
        flo[r] = vdupq_n_f32(0);
        fhi[r] = vdupq_n_f32(0);
    }
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* p = panel + b * 1696;
        const float32x4_t dlo = vld1q_f32(reinterpret_cast<const float*>(p));
        const float32x4_t dhi = vld1q_f32(reinterpret_cast<const float*>(p) + 4);
        const std::uint8_t* ab = a + b * kQ8KBytes;
        for (int g = 0; g < 16; ++g) {
            const std::uint8_t* group = p + 160 + 96 * g;
            const uint8x16_t high_lo = vld1q_u8(group + 64);
            const uint8x16_t high_hi = vld1q_u8(group + 80);
            int32x4_t lo_acc[R];
            int32x4_t hi_acc[R];
            int8x16_t x[R];
            for (int r = 0; r < R; ++r) {
                lo_acc[r] = vdupq_n_s32(0);
                hi_acc[r] = vdupq_n_s32(0);
                x[r] = vld1q_s8(reinterpret_cast<const std::int8_t*>(ab + r * a_stride + 4 + 16 * g));
            }
            const uint8x16_t m3 = vdupq_n_u8(3);
#define II_STEP(t)                                                                                                     \
    {                                                                                                                  \
        const uint8x16_t bytes = vld1q_u8(group + 16 * (t));                                                           \
        const uint8x16_t hl =                                                                                          \
            vshlq_n_u8(vandq_u8(vshlq_u8(high_lo, vdupq_n_s8(static_cast<std::int8_t>(-2 * (t)))), m3), 4);            \
        const uint8x16_t hh =                                                                                          \
            vshlq_n_u8(vandq_u8(vshlq_u8(high_hi, vdupq_n_s8(static_cast<std::int8_t>(-2 * (t)))), m3), 4);            \
        const int8x16_t lo =                                                                                           \
            vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(bytes, vdupq_n_u8(0x0F)), hl)), vdupq_n_s8(32));            \
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(bytes, 4), hh)), vdupq_n_s8(32));        \
        for (int r = 0; r < R; ++r) {                                                                                  \
            lo_acc[r] = vdotq_laneq_s32(lo_acc[r], lo, x[r], t);                                                       \
            hi_acc[r] = vdotq_laneq_s32(hi_acc[r], hi, x[r], t);                                                       \
        }                                                                                                              \
    }
            II_STEP(0)
            II_STEP(1)
            II_STEP(2)
            II_STEP(3)
#undef II_STEP
            const int16x8_t sc16 = vmovl_s8(vld1_s8(reinterpret_cast<const std::int8_t*>(p + 32 + 8 * g)));
            const float32x4_t slo = vmulq_f32(dlo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(sc16))));
            const float32x4_t shi = vmulq_f32(dhi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(sc16))));
            for (int r = 0; r < R; ++r) {
                const float da = load_f32(ab + r * a_stride);
                flo[r] = vfmaq_f32(flo[r], vcvtq_f32_s32(lo_acc[r]), vmulq_n_f32(slo, da));
                fhi[r] = vfmaq_f32(fhi[r], vcvtq_f32_s32(hi_acc[r]), vmulq_n_f32(shi, da));
            }
        }
    }
    store_rows<R>(flo, fhi, out, out_stride);
}

#endif

#if defined(II_QGEMM_AVX2) || defined(II_QGEMM_NEON)

#define II_QGEMM_ENTRY(type)                                                                                           \
    void qgemm8_##type(std::int64_t k, const void* panel, const void* a, std::int64_t a_stride, int rows, float* out,  \
                       std::int64_t out_stride) {                                                                      \
        const auto* p = static_cast<const std::uint8_t*>(panel);                                                       \
        const auto* x = static_cast<const std::uint8_t*>(a);                                                           \
        switch (rows) {                                                                                                \
        case 4:                                                                                                        \
            gemm_##type<4>(k, p, x, a_stride, out, out_stride);                                                        \
            break;                                                                                                     \
        case 3:                                                                                                        \
            gemm_##type<3>(k, p, x, a_stride, out, out_stride);                                                        \
            break;                                                                                                     \
        case 2:                                                                                                        \
            gemm_##type<2>(k, p, x, a_stride, out, out_stride);                                                        \
            break;                                                                                                     \
        default:                                                                                                       \
            gemm_##type<1>(k, p, x, a_stride, out, out_stride);                                                        \
            break;                                                                                                     \
        }                                                                                                              \
    }

II_QGEMM_ENTRY(q4_0)
II_QGEMM_ENTRY(q5_0)
II_QGEMM_ENTRY(q8_0)
II_QGEMM_ENTRY(q4_k)
II_QGEMM_ENTRY(q6_k)

#undef II_QGEMM_ENTRY

#endif

}  // namespace

#if defined(II_QGEMM_AVX2) || defined(II_QGEMM_NEON)
const QGemm8Fn kQGemm8[static_cast<int>(QType::kCount)] = {
    qgemm8_q4_0, nullptr, qgemm8_q5_0, nullptr, qgemm8_q8_0, nullptr, nullptr, qgemm8_q4_k, nullptr, qgemm8_q6_k,
};
#else
const QGemm8Fn kQGemm8[static_cast<int>(QType::kCount)] = {};
#endif

}  // namespace ii::cpu::II_ISA_NS
