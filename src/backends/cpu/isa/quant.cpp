// SPDX-License-Identifier: MIT
// Quantized dot products: weight rows in GGUF block layouts times activation rows quantized to 8 bits.
//
// Every type has a portable scalar implementation (the generic variant, and the fallback of rare types); q4_0, q5_0,
// q8_0, q4_K, q5_K and q6_K have AVX2 and NEON versions. A weight block is decoded once and multiplied with up to
// four activation rows, which keeps prompt processing compute-bound on the integer units instead of on decoding.
#include "backends/cpu/isa/isa_kernels.h"
#include "backends/cpu/isa/simd.h"

namespace ii::cpu::II_ISA_NS {

namespace {

[[maybe_unused]] constexpr int kMaxRows = 4;

// Activation block layouts (see kernel_table.h).
constexpr std::int64_t kQ8x32Bytes = 40;
constexpr std::int64_t kQ8KBytes = 292;

// Weight block sizes.
constexpr std::int64_t kQ4_0Bytes = 18;
constexpr std::int64_t kQ4_1Bytes = 20;
constexpr std::int64_t kQ5_0Bytes = 22;
constexpr std::int64_t kQ5_1Bytes = 24;
constexpr std::int64_t kQ8_0Bytes = 34;
constexpr std::int64_t kQ2_KBytes = 84;
constexpr std::int64_t kQ3_KBytes = 110;
constexpr std::int64_t kQ4_KBytes = 144;
constexpr std::int64_t kQ5_KBytes = 176;
constexpr std::int64_t kQ6_KBytes = 210;

II_SIMD_INLINE float half_to_float(const std::uint8_t* p) {
    std::uint16_t h;
    std::memcpy(&h, p, 2);
#if defined(II_ISA_IS_AVX2) || defined(II_ISA_IS_AVX512)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#elif defined(II_ISA_IS_NEON)
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
#else
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1fu;
    std::uint32_t mantissa = h & 0x3ffu;
    std::uint32_t bits;
    if (exponent == 0x1f) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else if (exponent != 0) {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    } else if (mantissa == 0) {
        bits = sign;
    } else {
        std::uint32_t e = 113;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --e;
        }
        bits = sign | (e << 23) | ((mantissa & 0x3ffu) << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
#endif
}

II_SIMD_INLINE float load_f32(const std::uint8_t* p) {
    float f;
    std::memcpy(&f, p, 4);
    return f;
}

II_SIMD_INLINE float abs_f(float x) {
    return x < 0 ? -x : x;
}

/// 6-bit scales and minimums of a q4_K / q5_K super-block: sc[0..7], mn[0..7].
II_SIMD_INLINE void k4_scales(const std::uint8_t* q, std::uint8_t* sc, std::uint8_t* mn) {
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = q[j] & 63;
            mn[j] = q[j + 4] & 63;
        } else {
            sc[j] = static_cast<std::uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
            mn[j] = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }
}

}  // namespace

// =====================================================================================================================
// Activation quantization

void quantize_q8x32(const float* x, void* out, std::int64_t k) {
    auto* o = static_cast<std::uint8_t*>(out);
    for (std::int64_t b = 0; b < k / 32; ++b) {
        const float* v = x + b * 32;
        VF amax_v = vzero();
        for (int i = 0; i < 32; i += kLanes) {
            amax_v = vmax(amax_v, vabs(vload(v + i)));
        }
        const float amax = vreduce_max(amax_v);
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        alignas(64) float bits[32];
        const VF scale = vset1(id);
        for (int i = 0; i < 32; i += kLanes) {
            vstore(bits + i, vfloat_bits(vcvt_round(vmul(vload(v + i), scale))));
        }
        std::int32_t q[32];
        std::memcpy(q, bits, sizeof q);
        std::uint8_t* block = o + b * kQ8x32Bytes;
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            block[8 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q[i]));
            sum += q[i];
        }
        const float s = d * static_cast<float>(sum);
        std::memcpy(block, &d, 4);
        std::memcpy(block + 4, &s, 4);
    }
}

void quantize_q8k(const float* x, void* out, std::int64_t k) {
    auto* o = static_cast<std::uint8_t*>(out);
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const float* v = x + b * 256;
        std::uint8_t* block = o + b * kQ8KBytes;
        // The value with the largest magnitude maps to -127 (ggml's q8_K).
        float max = 0;
        float amax = 0;
        for (int i = 0; i < 256; ++i) {
            const float a = abs_f(v[i]);
            if (a > amax) {
                amax = a;
                max = v[i];
            }
        }
        if (amax == 0.0f) {
            std::memset(block, 0, kQ8KBytes);
            continue;
        }
        const float iscale = -127.0f / max;
        alignas(64) float bits[256];
        const VF scale = vset1(iscale);
        for (int i = 0; i < 256; i += kLanes) {
            vstore(bits + i, vfloat_bits(vcvt_round(vmul(vload(v + i), scale))));
        }
        std::int32_t q[256];
        std::memcpy(q, bits, sizeof q);
        for (int i = 0; i < 256; ++i) {
            block[4 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q[i] > 127 ? 127 : q[i]));
        }
        for (int g = 0; g < 16; ++g) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) {
                sum += static_cast<std::int8_t>(block[4 + g * 16 + i]);
            }
            const auto s16 = static_cast<std::int16_t>(sum);
            std::memcpy(block + 260 + 2 * g, &s16, 2);
        }
        const float d = 1.0f / iscale;
        std::memcpy(block, &d, 4);
    }
}

// =====================================================================================================================
// Scalar dot products (all types)

namespace {

[[maybe_unused]] void scalar_q4_0(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 32; ++b, w += kQ4_0Bytes, a += kQ8x32Bytes) {
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 8);
        int s = 0;
        for (int j = 0; j < 16; ++j) {
            s += ((w[2 + j] & 0x0F) - 8) * qa[j] + ((w[2 + j] >> 4) - 8) * qa[j + 16];
        }
        sum += half_to_float(w) * load_f32(a) * static_cast<float>(s);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q4_1(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 32; ++b, w += kQ4_1Bytes, a += kQ8x32Bytes) {
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 8);
        int s = 0;
        for (int j = 0; j < 16; ++j) {
            s += (w[4 + j] & 0x0F) * qa[j] + (w[4 + j] >> 4) * qa[j + 16];
        }
        sum += half_to_float(w) * load_f32(a) * static_cast<float>(s) + half_to_float(w + 2) * load_f32(a + 4);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q5_0(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 32; ++b, w += kQ5_0Bytes, a += kQ8x32Bytes) {
        std::uint32_t qh;
        std::memcpy(&qh, w + 2, 4);
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 8);
        int s = 0;
        for (int j = 0; j < 16; ++j) {
            const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10);
            const int h1 = static_cast<int>((qh >> (j + 12)) & 0x10);
            s += (((w[6 + j] & 0x0F) | h0) - 16) * qa[j] + (((w[6 + j] >> 4) | h1) - 16) * qa[j + 16];
        }
        sum += half_to_float(w) * load_f32(a) * static_cast<float>(s);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q5_1(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 32; ++b, w += kQ5_1Bytes, a += kQ8x32Bytes) {
        std::uint32_t qh;
        std::memcpy(&qh, w + 4, 4);
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 8);
        int s = 0;
        for (int j = 0; j < 16; ++j) {
            const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10);
            const int h1 = static_cast<int>((qh >> (j + 12)) & 0x10);
            s += ((w[8 + j] & 0x0F) | h0) * qa[j] + ((w[8 + j] >> 4) | h1) * qa[j + 16];
        }
        sum += half_to_float(w) * load_f32(a) * static_cast<float>(s) + half_to_float(w + 2) * load_f32(a + 4);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q8_0(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 32; ++b, w += kQ8_0Bytes, a += kQ8x32Bytes) {
        const auto* qw = reinterpret_cast<const std::int8_t*>(w + 2);
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 8);
        int s = 0;
        for (int j = 0; j < 32; ++j) {
            s += qw[j] * qa[j];
        }
        sum += half_to_float(w) * load_f32(a) * static_cast<float>(s);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q2_k(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 256; ++b, w += kQ2_KBytes, a += kQ8KBytes) {
        const std::uint8_t* scales = w;
        const std::uint8_t* q = w + 16;
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 4);
        std::int16_t bsums[16];
        std::memcpy(bsums, a + 260, 32);
        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += bsums[j] * (scales[j] >> 4);
        }
        int isum = 0;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int d = scales[is++] & 0x0F;
                    int s = 0;
                    for (int l = 0; l < 16; ++l) {
                        s += qa[n + j * 32 + half * 16 + l] * ((q[half * 16 + l] >> shift) & 3);
                    }
                    isum += d * s;
                }
                shift += 2;
            }
            q += 32;
        }
        const float da = load_f32(a);
        sum += half_to_float(w + 80) * da * static_cast<float>(isum) -
               half_to_float(w + 82) * da * static_cast<float>(summs);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q3_k(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 256; ++b, w += kQ3_KBytes, a += kQ8KBytes) {
        const std::uint8_t* hm = w;
        const std::uint8_t* q = w + 32;
        // Unpack the 16 6-bit scales.
        std::uint32_t aux[4];
        std::memcpy(aux, w + 96, 12);
        const std::uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
        aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
        std::int8_t scales[16];
        std::memcpy(scales, aux, 16);
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 4);
        int isum = 0;
        int is = 0;
        std::uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const int d = scales[is++] - 32;
                    int s = 0;
                    for (int l = 0; l < 16; ++l) {
                        const int idx = half * 16 + l;
                        const int value = ((q[idx] >> shift) & 3) - ((hm[idx] & m) != 0 ? 0 : 4);
                        s += qa[n + j * 32 + idx] * value;
                    }
                    isum += d * s;
                }
                shift += 2;
                m = static_cast<std::uint8_t>(m << 1);
            }
            q += 32;
        }
        sum += half_to_float(w + 108) * load_f32(a) * static_cast<float>(isum);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q4_k(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 256; ++b, w += kQ4_KBytes, a += kQ8KBytes) {
        std::uint8_t sc[8];
        std::uint8_t mn[8];
        k4_scales(w + 4, sc, mn);
        const std::uint8_t* q = w + 16;
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 4);
        std::int16_t bsums[16];
        std::memcpy(bsums, a + 260, 32);
        int isum = 0;
        int msum = 0;
        for (int j = 0; j < 4; ++j) {
            int s0 = 0;
            int s1 = 0;
            for (int l = 0; l < 32; ++l) {
                s0 += (q[l] & 0x0F) * qa[j * 64 + l];
                s1 += (q[l] >> 4) * qa[j * 64 + 32 + l];
            }
            isum += sc[2 * j] * s0 + sc[2 * j + 1] * s1;
            msum +=
                mn[2 * j] * (bsums[4 * j] + bsums[4 * j + 1]) + mn[2 * j + 1] * (bsums[4 * j + 2] + bsums[4 * j + 3]);
            q += 32;
        }
        const float da = load_f32(a);
        sum += half_to_float(w) * da * static_cast<float>(isum) - half_to_float(w + 2) * da * static_cast<float>(msum);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q5_k(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 256; ++b, w += kQ5_KBytes, a += kQ8KBytes) {
        std::uint8_t sc[8];
        std::uint8_t mn[8];
        k4_scales(w + 4, sc, mn);
        const std::uint8_t* qh = w + 16;
        const std::uint8_t* ql = w + 48;
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 4);
        std::int16_t bsums[16];
        std::memcpy(bsums, a + 260, 32);
        int isum = 0;
        int msum = 0;
        std::uint8_t u1 = 1;
        std::uint8_t u2 = 2;
        for (int j = 0; j < 4; ++j) {
            int s0 = 0;
            int s1 = 0;
            for (int l = 0; l < 32; ++l) {
                s0 += ((ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0)) * qa[j * 64 + l];
                s1 += ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) * qa[j * 64 + 32 + l];
            }
            isum += sc[2 * j] * s0 + sc[2 * j + 1] * s1;
            msum +=
                mn[2 * j] * (bsums[4 * j] + bsums[4 * j + 1]) + mn[2 * j + 1] * (bsums[4 * j + 2] + bsums[4 * j + 3]);
            ql += 32;
            u1 = static_cast<std::uint8_t>(u1 << 2);
            u2 = static_cast<std::uint8_t>(u2 << 2);
        }
        const float da = load_f32(a);
        sum += half_to_float(w) * da * static_cast<float>(isum) - half_to_float(w + 2) * da * static_cast<float>(msum);
    }
    *out = sum;
}

[[maybe_unused]] void scalar_q6_k(std::int64_t k, const std::uint8_t* w, const std::uint8_t* a, float* out) {
    float sum = 0;
    for (std::int64_t b = 0; b < k / 256; ++b, w += kQ6_KBytes, a += kQ8KBytes) {
        const std::uint8_t* ql = w;
        const std::uint8_t* qh = w + 128;
        const auto* sc = reinterpret_cast<const std::int8_t*>(w + 192);
        const auto* qa = reinterpret_cast<const std::int8_t*>(a + 4);
        int isum = 0;
        for (int n = 0; n < 256; n += 128) {
            int s[8] = {};
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                s[is + 0] += q1 * qa[n + l];
                s[is + 2] += q2 * qa[n + l + 32];
                s[is + 4] += q3 * qa[n + l + 64];
                s[is + 6] += q4 * qa[n + l + 96];
            }
            for (int g = 0; g < 8; ++g) {
                isum += sc[g] * s[g];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
        sum += half_to_float(w + 208) * load_f32(a) * static_cast<float>(isum);
    }
    *out = sum;
}

using ScalarDot = void (*)(std::int64_t, const std::uint8_t*, const std::uint8_t*, float*);

void scalar_rows(ScalarDot dot, std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows,
                 float* out, std::int64_t out_stride) {
    for (int r = 0; r < rows; ++r) {
        dot(k, static_cast<const std::uint8_t*>(w), static_cast<const std::uint8_t*>(a) + r * a_stride,
            out + r * out_stride);
    }
}

// =====================================================================================================================
// AVX2

#if defined(II_ISA_IS_AVX2) || defined(II_ISA_IS_AVX512)
#define II_QUANT_VECTOR 1

II_SIMD_INLINE float hsum(__m256 v) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

/// 8 int32 sums of products of signed bytes.
[[maybe_unused]] II_SIMD_INLINE __m256i dot_i8(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), _mm256_set1_epi16(1));
}

/// 32 nibbles: the low halves of 16 bytes, then the high halves.
II_SIMD_INLINE __m256i nibbles(const std::uint8_t* p) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m256i both = _mm256_insertf128_si256(_mm256_castsi128_si256(bytes), _mm_srli_epi16(bytes, 4), 1);
    return _mm256_and_si256(both, _mm256_set1_epi8(0x0F));
}

/// 0xFF in byte i where bit i of the 32-bit word at p is set.
II_SIMD_INLINE __m256i bits_to_bytes(const std::uint8_t* p) {
    std::uint32_t x32;
    std::memcpy(&x32, p, 4);
    const __m256i shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(x32)), shuffle);
    bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

/// Shared loop of the 32-element types: `decode(block)` gives 32 signed weight bytes.
template <class Decode>
[[maybe_unused]] II_SIMD_INLINE void dot32_rows(std::int64_t k, const std::uint8_t* w, std::int64_t block_bytes,
                                                const std::uint8_t* a, std::int64_t a_stride, int rows, float* out,
                                                std::int64_t out_stride, Decode decode) {
    __m256 acc[kMaxRows] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (std::int64_t b = 0; b < k / 32; ++b) {
        const std::uint8_t* block = w + b * block_bytes;
        const __m256i qw = decode(block);
        const float dw = half_to_float(block);
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8x32Bytes;
            const __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 8));
            const __m256 scale = _mm256_set1_ps(dw * load_f32(ab));
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_i8(qw, qa)), scale, acc[r]);
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = hsum(acc[r]);
    }
}

/// 8 int32 sums of products of unsigned bytes u and signed bytes s (no saturation with VNNI; without it u * s
/// pairs must stay below 2^15, i.e. u <= 128).
II_SIMD_INLINE __m256i dot_u8_s8(__m256i u, __m256i s) {
#if defined(II_ISA_IS_AVX_VNNI)
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1));
#endif
}

/// The 32-element types with weights decoded to unsigned bytes u = w + offset: the dot product of each block is
/// dw * da * sum(u * a) - offset * dw * (da * sum(a)), with da * sum(a) stored in the activation block.
template <class Decode>
II_SIMD_INLINE void dot32_unsigned_rows(std::int64_t k, const std::uint8_t* w, std::int64_t block_bytes, int offset,
                                        const std::uint8_t* a, std::int64_t a_stride, int rows, float* out,
                                        std::int64_t out_stride, Decode decode) {
    __m256 acc[kMaxRows] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    float correction[kMaxRows] = {};
    for (std::int64_t b = 0; b < k / 32; ++b) {
        const std::uint8_t* block = w + b * block_bytes;
        const __m256i qw = decode(block);
        const float dw = half_to_float(block);
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8x32Bytes;
            const __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 8));
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_u8_s8(qw, qa)), _mm256_set1_ps(dw * load_f32(ab)), acc[r]);
            correction[r] += dw * load_f32(ab + 4);
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = hsum(acc[r]) - static_cast<float>(offset) * correction[r];
    }
}

void vector_q4_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    dot32_unsigned_rows(k, static_cast<const std::uint8_t*>(w), kQ4_0Bytes, 8, static_cast<const std::uint8_t*>(a),
                        a_stride, rows, out, out_stride, [](const std::uint8_t* block) { return nibbles(block + 2); });
}

void vector_q5_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    dot32_unsigned_rows(k, static_cast<const std::uint8_t*>(w), kQ5_0Bytes, 16, static_cast<const std::uint8_t*>(a),
                        a_stride, rows, out, out_stride, [](const std::uint8_t* block) {
                            // nibble | high bit << 4, unsigned 0..31
                            const __m256i high = _mm256_and_si256(bits_to_bytes(block + 2), _mm256_set1_epi8(0x10));
                            return _mm256_or_si256(nibbles(block + 6), high);
                        });
}

void vector_q8_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
#if defined(II_ISA_IS_AVX_VNNI)
    // w + 128 as unsigned bytes (no saturation with vpdpbusd).
    dot32_unsigned_rows(k, static_cast<const std::uint8_t*>(w), kQ8_0Bytes, 128, static_cast<const std::uint8_t*>(a),
                        a_stride, rows, out, out_stride, [](const std::uint8_t* block) {
                            return _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 2)),
                                                    _mm256_set1_epi8(static_cast<char>(0x80)));
                        });
#else
    dot32_rows(k, static_cast<const std::uint8_t*>(w), kQ8_0Bytes, static_cast<const std::uint8_t*>(a), a_stride, rows,
               out, out_stride, [](const std::uint8_t* block) {
                   return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 2));
               });
#endif
}

/// The 8 sub-block minimums of a q4_K / q5_K super-block as int16.
II_SIMD_INLINE __m128i k4_mins16(const std::uint8_t* mn) {
    std::uint64_t m8;
    std::memcpy(&m8, mn, 8);
    return _mm_cvtepu8_epi16(_mm_cvtsi64_si128(static_cast<long long>(m8)));
}

/// acc_m += dmin * sum over groups of min * (sum of the group's activations), for one activation row.
II_SIMD_INLINE void k4_minimums(__m128i mins16, const std::uint8_t* ab, float dmin, __m128& acc_m) {
    const __m256i sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 260));
    // Pairs of 16-element sums -> 8 sums of 32 elements.
    const __m128i pair = _mm_hadd_epi16(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(_mm_madd_epi16(mins16, pair)), acc_m);
}

II_SIMD_INLINE float hsum128(__m128 r) {
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

void vector_q4_k(std::int64_t k, const void* wv, const void* av, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    const auto* w = static_cast<const std::uint8_t*>(wv);
    const auto* a = static_cast<const std::uint8_t*>(av);
    __m256 acc[kMaxRows] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    __m128 acc_m[kMaxRows] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* block = w + b * kQ4_KBytes;
        std::uint8_t sc[8];
        std::uint8_t mn[8];
        k4_scales(block + 4, sc, mn);
        const float d = half_to_float(block);
        const float dmin = -half_to_float(block + 2);
        const __m128i mins16 = k4_mins16(mn);
        __m256i q[8];
        __m256i scale[8];
        for (int j = 0; j < 4; ++j) {
            const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 16 + 32 * j));
            q[2 * j] = _mm256_and_si256(bits, m4);
            q[2 * j + 1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            scale[2 * j] = _mm256_set1_epi16(sc[2 * j]);
            scale[2 * j + 1] = _mm256_set1_epi16(sc[2 * j + 1]);
        }
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8KBytes;
            const float da = load_f32(ab);
            k4_minimums(mins16, ab, dmin * da, acc_m[r]);
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            for (int g = 0; g < 8; g += 2) {
                const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 4 + 32 * g));
                const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 4 + 32 * g + 32));
                sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(scale[g], _mm256_maddubs_epi16(q[g], a0)));
                sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(scale[g + 1], _mm256_maddubs_epi16(q[g + 1], a1)));
            }
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d * da), _mm256_cvtepi32_ps(_mm256_add_epi32(sum0, sum1)), acc[r]);
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = hsum(acc[r]) + hsum128(acc_m[r]);
    }
}

void vector_q5_k(std::int64_t k, const void* wv, const void* av, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    const auto* w = static_cast<const std::uint8_t*>(wv);
    const auto* a = static_cast<const std::uint8_t*>(av);
    __m256 acc[kMaxRows] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    __m128 acc_m[kMaxRows] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* block = w + b * kQ5_KBytes;
        std::uint8_t sc[8];
        std::uint8_t mn[8];
        k4_scales(block + 4, sc, mn);
        const float d = half_to_float(block);
        const float dmin = -half_to_float(block + 2);
        const __m256i hbits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 16));
        __m256i q5[8];
        __m256i hmask = _mm256_set1_epi8(1);
        for (int j = 0; j < 4; ++j) {
            const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + 48 + 32 * j));
            // High bit 2j (low nibbles) and 2j + 1 (high nibbles) of every byte of qh, moved to bit 4.
            const __m256i h0 = _mm256_and_si256(hbits, hmask);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i h1 = _mm256_and_si256(hbits, hmask);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i bit0 =
                _mm256_andnot_si256(_mm256_cmpeq_epi8(h0, _mm256_setzero_si256()), _mm256_set1_epi8(16));
            const __m256i bit1 =
                _mm256_andnot_si256(_mm256_cmpeq_epi8(h1, _mm256_setzero_si256()), _mm256_set1_epi8(16));
            q5[2 * j] = _mm256_or_si256(_mm256_and_si256(bits, m4), bit0);
            q5[2 * j + 1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), bit1);
        }
        const __m128i mins16 = k4_mins16(mn);
        __m256i scale[8];
        for (int j = 0; j < 8; ++j) {
            scale[j] = _mm256_set1_epi16(sc[j]);
        }
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8KBytes;
            const float da = load_f32(ab);
            k4_minimums(mins16, ab, dmin * da, acc_m[r]);
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            for (int j = 0; j < 8; j += 2) {
                const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 4 + 32 * j));
                const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 4 + 32 * j + 32));
                sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(scale[j], _mm256_maddubs_epi16(q5[j], a0)));
                sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(scale[j + 1], _mm256_maddubs_epi16(q5[j + 1], a1)));
            }
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d * da), _mm256_cvtepi32_ps(_mm256_add_epi32(sum0, sum1)), acc[r]);
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = hsum(acc[r]) + hsum128(acc_m[r]);
    }
}

void vector_q6_k(std::int64_t k, const void* wv, const void* av, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    const auto* w = static_cast<const std::uint8_t*>(wv);
    const auto* a = static_cast<const std::uint8_t*>(av);
    __m256 acc[kMaxRows] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m2 = _mm256_set1_epi8(3);
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* block = w + b * kQ6_KBytes;
        const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
        const float d = half_to_float(block + 208);
        // Unsigned 6-bit values 0..63 of the 8 groups of 32, in activation order.
        __m256i q6[8];
        for (int n = 0; n < 2; ++n) {
            const std::uint8_t* ql = block + 64 * n;
            const std::uint8_t* qh = block + 128 + 32 * n;
            const __m256i bits1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i bits2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
            const __m256i h0 = _mm256_slli_epi16(_mm256_and_si256(high, m2), 4);
            const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high, 2), m2), 4);
            const __m256i h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high, 4), m2), 4);
            const __m256i h3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(high, 6), m2), 4);
            q6[4 * n + 0] = _mm256_or_si256(_mm256_and_si256(bits1, m4), h0);
            q6[4 * n + 1] = _mm256_or_si256(_mm256_and_si256(bits2, m4), h1);
            q6[4 * n + 2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4), h2);
            q6[4 * n + 3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4), h3);
        }
        // Per group of 32: 16 int16 scales, the first 8 for elements 0..15, the last 8 for 16..31.
        __m256i sc16[8];
        for (int g = 0; g < 8; ++g) {
            sc16[g] = _mm256_setr_epi16(scales[2 * g], scales[2 * g], scales[2 * g], scales[2 * g], scales[2 * g],
                                        scales[2 * g], scales[2 * g], scales[2 * g], scales[2 * g + 1],
                                        scales[2 * g + 1], scales[2 * g + 1], scales[2 * g + 1], scales[2 * g + 1],
                                        scales[2 * g + 1], scales[2 * g + 1], scales[2 * g + 1]);
        }
        // sum (q - 32) * a * sc = sum q * a * sc - 32 * sum sc * (sum of each group of 16 activations)
        const __m256i scales16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(scales)));
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8KBytes;
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            for (int g = 0; g < 8; g += 2) {
                const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 4 + 32 * g));
                const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 4 + 32 * g + 32));
                sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(sc16[g], _mm256_maddubs_epi16(q6[g], a0)));
                sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(sc16[g + 1], _mm256_maddubs_epi16(q6[g + 1], a1)));
            }
            const __m256i bsums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ab + 260));
            const __m256i offset = _mm256_slli_epi32(_mm256_madd_epi16(scales16, bsums), 5);
            const __m256i sumi = _mm256_sub_epi32(_mm256_add_epi32(sum0, sum1), offset);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d * load_f32(ab)), _mm256_cvtepi32_ps(sumi), acc[r]);
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = hsum(acc[r]);
    }
}

// =====================================================================================================================
// NEON

#elif defined(II_ISA_IS_NEON)
#define II_QUANT_VECTOR 1

/// acc + 4 int32 sums of products of 16 signed bytes (groups of 4 adjacent bytes with dot product instructions).
II_SIMD_INLINE int32x4_t dot_i8(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vpadalq_s16(vpadalq_s16(acc, lo), hi);
#endif
}

/// Shared loop of the 32-element types: `decode(block, low, high)` gives 32 signed weight bytes.
template <class Decode>
II_SIMD_INLINE void dot32_rows(std::int64_t k, const std::uint8_t* w, std::int64_t block_bytes, const std::uint8_t* a,
                               std::int64_t a_stride, int rows, float* out, std::int64_t out_stride, Decode decode) {
    float32x4_t acc[kMaxRows] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    for (std::int64_t b = 0; b < k / 32; ++b) {
        const std::uint8_t* block = w + b * block_bytes;
        int8x16_t lo;
        int8x16_t hi;
        decode(block, lo, hi);
        const float dw = half_to_float(block);
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8x32Bytes;
            const int8x16_t a0 = vld1q_s8(reinterpret_cast<const std::int8_t*>(ab + 8));
            const int8x16_t a1 = vld1q_s8(reinterpret_cast<const std::int8_t*>(ab + 24));
            const int32x4_t s = dot_i8(dot_i8(vdupq_n_s32(0), lo, a0), hi, a1);
            acc[r] = vmlaq_n_f32(acc[r], vcvtq_f32_s32(s), dw * load_f32(ab));
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = vaddvq_f32(acc[r]);
    }
}

void vector_q4_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    dot32_rows(k, static_cast<const std::uint8_t*>(w), kQ4_0Bytes, static_cast<const std::uint8_t*>(a), a_stride, rows,
               out, out_stride, [](const std::uint8_t* block, int8x16_t& lo, int8x16_t& hi) {
                   const uint8x16_t bytes = vld1q_u8(block + 2);
                   lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
                   hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), vdupq_n_s8(8));
               });
}

void vector_q5_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    dot32_rows(k, static_cast<const std::uint8_t*>(w), kQ5_0Bytes, static_cast<const std::uint8_t*>(a), a_stride, rows,
               out, out_stride, [](const std::uint8_t* block, int8x16_t& lo, int8x16_t& hi) {
                   std::uint32_t qh;
                   std::memcpy(&qh, block + 2, 4);
                   // Expand the 32 high bits to bytes (0x10 where set).
                   alignas(16) std::uint8_t h[32];
                   for (int j = 0; j < 32; ++j) {
                       h[j] = static_cast<std::uint8_t>(((qh >> j) & 1u) << 4);
                   }
                   const uint8x16_t bytes = vld1q_u8(block + 6);
                   const uint8x16_t l = vorrq_u8(vandq_u8(bytes, vdupq_n_u8(0x0F)), vld1q_u8(h));
                   const uint8x16_t u = vorrq_u8(vshrq_n_u8(bytes, 4), vld1q_u8(h + 16));
                   lo = vsubq_s8(vreinterpretq_s8_u8(l), vdupq_n_s8(16));
                   hi = vsubq_s8(vreinterpretq_s8_u8(u), vdupq_n_s8(16));
               });
}

void vector_q8_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    dot32_rows(k, static_cast<const std::uint8_t*>(w), kQ8_0Bytes, static_cast<const std::uint8_t*>(a), a_stride, rows,
               out, out_stride, [](const std::uint8_t* block, int8x16_t& lo, int8x16_t& hi) {
                   lo = vld1q_s8(reinterpret_cast<const std::int8_t*>(block + 2));
                   hi = vld1q_s8(reinterpret_cast<const std::int8_t*>(block + 18));
               });
}

II_SIMD_INLINE std::int32_t k4_minimum_sum(const std::uint8_t* mn, const std::uint8_t* ab) {
    std::int16_t bsums[16];
    std::memcpy(bsums, ab + 260, 32);
    std::int32_t sum = 0;
    for (int j = 0; j < 8; ++j) {
        sum += mn[j] * (bsums[2 * j] + bsums[2 * j + 1]);
    }
    return sum;
}

void vector_q4_k(std::int64_t k, const void* wv, const void* av, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    const auto* w = static_cast<const std::uint8_t*>(wv);
    const auto* a = static_cast<const std::uint8_t*>(av);
    float sum[kMaxRows] = {};
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* block = w + b * kQ4_KBytes;
        std::uint8_t sc[8];
        std::uint8_t mn[8];
        k4_scales(block + 4, sc, mn);
        const float d = half_to_float(block);
        const float dmin = half_to_float(block + 2);
        int8x16_t q[16];  // groups of 32 as two halves
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t b0 = vld1q_u8(block + 16 + 32 * j);
            const uint8x16_t b1 = vld1q_u8(block + 16 + 32 * j + 16);
            q[4 * j + 0] = vreinterpretq_s8_u8(vandq_u8(b0, vdupq_n_u8(0x0F)));
            q[4 * j + 1] = vreinterpretq_s8_u8(vandq_u8(b1, vdupq_n_u8(0x0F)));
            q[4 * j + 2] = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
            q[4 * j + 3] = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
        }
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8KBytes;
            const auto* qa = reinterpret_cast<const std::int8_t*>(ab + 4);
            std::int32_t isum = 0;
            for (int g = 0; g < 8; ++g) {
                const int32x4_t s = dot_i8(dot_i8(vdupq_n_s32(0), q[2 * g], vld1q_s8(qa + 32 * g)), q[2 * g + 1],
                                           vld1q_s8(qa + 32 * g + 16));
                isum += sc[g] * vaddvq_s32(s);
            }
            const float da = load_f32(ab);
            sum[r] += d * da * static_cast<float>(isum) - dmin * da * static_cast<float>(k4_minimum_sum(mn, ab));
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = sum[r];
    }
}

void vector_q5_k(std::int64_t k, const void* wv, const void* av, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    const auto* w = static_cast<const std::uint8_t*>(wv);
    const auto* a = static_cast<const std::uint8_t*>(av);
    float sum[kMaxRows] = {};
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* block = w + b * kQ5_KBytes;
        std::uint8_t sc[8];
        std::uint8_t mn[8];
        k4_scales(block + 4, sc, mn);
        const float d = half_to_float(block);
        const float dmin = half_to_float(block + 2);
        const uint8x16_t h0 = vld1q_u8(block + 16);
        const uint8x16_t h1 = vld1q_u8(block + 32);
        int8x16_t q[16];
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t b0 = vld1q_u8(block + 48 + 32 * j);
            const uint8x16_t b1 = vld1q_u8(block + 48 + 32 * j + 16);
            const uint8x16_t bit_lo = vdupq_n_u8(static_cast<std::uint8_t>(1u << (2 * j)));
            const uint8x16_t bit_hi = vdupq_n_u8(static_cast<std::uint8_t>(2u << (2 * j)));
            const uint8x16_t sixteen = vdupq_n_u8(16);
            q[4 * j + 0] =
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(b0, vdupq_n_u8(0x0F)), vandq_u8(vtstq_u8(h0, bit_lo), sixteen)));
            q[4 * j + 1] =
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(b1, vdupq_n_u8(0x0F)), vandq_u8(vtstq_u8(h1, bit_lo), sixteen)));
            q[4 * j + 2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(b0, 4), vandq_u8(vtstq_u8(h0, bit_hi), sixteen)));
            q[4 * j + 3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(b1, 4), vandq_u8(vtstq_u8(h1, bit_hi), sixteen)));
        }
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8KBytes;
            const auto* qa = reinterpret_cast<const std::int8_t*>(ab + 4);
            std::int32_t isum = 0;
            for (int g = 0; g < 8; ++g) {
                const int32x4_t s = dot_i8(dot_i8(vdupq_n_s32(0), q[2 * g], vld1q_s8(qa + 32 * g)), q[2 * g + 1],
                                           vld1q_s8(qa + 32 * g + 16));
                isum += sc[g] * vaddvq_s32(s);
            }
            const float da = load_f32(ab);
            sum[r] += d * da * static_cast<float>(isum) - dmin * da * static_cast<float>(k4_minimum_sum(mn, ab));
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = sum[r];
    }
}

void vector_q6_k(std::int64_t k, const void* wv, const void* av, std::int64_t a_stride, int rows, float* out,
                 std::int64_t out_stride) {
    const auto* w = static_cast<const std::uint8_t*>(wv);
    const auto* a = static_cast<const std::uint8_t*>(av);
    float sum[kMaxRows] = {};
    for (std::int64_t b = 0; b < k / 256; ++b) {
        const std::uint8_t* block = w + b * kQ6_KBytes;
        const auto* scales = reinterpret_cast<const std::int8_t*>(block + 192);
        const float d = half_to_float(block + 208);
        int8x16_t q[16];  // signed values -32..31, 16 per register, in activation order
        for (int n = 0; n < 2; ++n) {
            const std::uint8_t* ql = block + 64 * n;
            const std::uint8_t* qh = block + 128 + 32 * n;
            for (int half = 0; half < 2; ++half) {
                const uint8x16_t l0 = vld1q_u8(ql + 16 * half);
                const uint8x16_t l1 = vld1q_u8(ql + 32 + 16 * half);
                const uint8x16_t h = vld1q_u8(qh + 16 * half);
                const uint8x16_t m4 = vdupq_n_u8(0x0F);
                const uint8x16_t m3 = vdupq_n_u8(3);
                const uint8x16_t v0 = vorrq_u8(vandq_u8(l0, m4), vshlq_n_u8(vandq_u8(h, m3), 4));
                const uint8x16_t v1 = vorrq_u8(vandq_u8(l1, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 2), m3), 4));
                const uint8x16_t v2 = vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h, 4), m3), 4));
                const uint8x16_t v3 = vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vshrq_n_u8(h, 6), 4));
                const int8x16_t offset = vdupq_n_s8(32);
                q[8 * n + 0 + half] = vsubq_s8(vreinterpretq_s8_u8(v0), offset);
                q[8 * n + 2 + half] = vsubq_s8(vreinterpretq_s8_u8(v1), offset);
                q[8 * n + 4 + half] = vsubq_s8(vreinterpretq_s8_u8(v2), offset);
                q[8 * n + 6 + half] = vsubq_s8(vreinterpretq_s8_u8(v3), offset);
            }
        }
        for (int r = 0; r < rows; ++r) {
            const std::uint8_t* ab = a + r * a_stride + b * kQ8KBytes;
            const auto* qa = reinterpret_cast<const std::int8_t*>(ab + 4);
            std::int32_t isum = 0;
            for (int g = 0; g < 16; ++g) {
                isum += scales[g] * vaddvq_s32(dot_i8(vdupq_n_s32(0), q[g], vld1q_s8(qa + 16 * g)));
            }
            sum[r] += d * load_f32(ab) * static_cast<float>(isum);
        }
    }
    for (int r = 0; r < rows; ++r) {
        out[r * out_stride] = sum[r];
    }
}

#endif

}  // namespace

// =====================================================================================================================
// Entry points

#if defined(II_QUANT_VECTOR)
void qdot_q4_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    vector_q4_0(k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q5_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    vector_q5_0(k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q8_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    vector_q8_0(k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q4_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    vector_q4_k(k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q5_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    vector_q5_k(k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q6_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    vector_q6_k(k, w, a, a_stride, rows, out, out_stride);
}
#else
void qdot_q4_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q4_0, k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q5_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q5_0, k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q8_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q8_0, k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q4_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q4_k, k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q5_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q5_k, k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q6_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q6_k, k, w, a, a_stride, rows, out, out_stride);
}
#endif

void qdot_q4_1(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q4_1, k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q5_1(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q5_1, k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q2_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q2_k, k, w, a, a_stride, rows, out, out_stride);
}
void qdot_q3_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride) {
    scalar_rows(scalar_q3_k, k, w, a, a_stride, rows, out, out_stride);
}

}  // namespace ii::cpu::II_ISA_NS
