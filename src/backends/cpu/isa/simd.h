// SPDX-License-Identifier: MIT
/// @file
/// The SIMD layer of ii-cpu: a thin vector abstraction over the intrinsics of one instruction set.
///
/// Included only by ISA-specific translation units (src/backends/cpu/isa/*.cpp), which are compiled once per ISA
/// variant with II_ISA_NS defined to the variant name and II_ISA_IS_<NAME> defined for it. Everything here lives in
/// `ii::cpu::II_ISA_NS`, so the code instantiated with different instruction sets never meets in the linker
/// (see the "ODR trap" in docs/architecture.md). For the same reason this header must not use inline functions or
/// templates of shared headers (the C++ standard library included) — only intrinsics.
///
/// Types (`kLanes` floats per vector):
///   VF  float vector         VI  int32 vector (same lane count)         VM  lane mask (result of comparisons)
///   VD  double vector with kLanes / 2 lanes (for accurate accumulation)
///
/// Semantics shared by all ISAs:
///   * vmin / vmax: the result for NaN operands is unspecified (callers handle NaN explicitly where it matters);
///   * vround rounds half to even; vfloor / vceil / vtrunc are exact for every float;
///   * vfma(a, b, c) = a * b + c (fused where the ISA has FMA, so results may differ in the last bit between ISAs);
///   * vcvt_round: float to int32 rounding half to even, for |x| < 2^31.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if !defined(II_ISA_NS)
#error "simd.h is for ISA-specific translation units only"
#endif

#if defined(II_ISA_IS_AVX2) || defined(II_ISA_IS_AVX512)
#include <immintrin.h>
#elif defined(II_ISA_IS_NEON)
#include <arm_neon.h>
#else
#include <cmath>  // generic only: compiled with the baseline flags, so library code is safe
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define II_SIMD_INLINE __forceinline
#else
#define II_SIMD_INLINE inline __attribute__((always_inline))
#endif

namespace ii::cpu::II_ISA_NS {

// =====================================================================================================================
#if defined(II_ISA_IS_AVX512)

inline constexpr int kLanes = 16;
inline constexpr const char* kIsaName = "avx512";

struct VF {
    __m512 v;
};
struct VI {
    __m512i v;
};
struct VM {
    __mmask16 m;
};
struct VD {
    __m512d v;
};

II_SIMD_INLINE VF vzero() {
    return {_mm512_setzero_ps()};
}
II_SIMD_INLINE VF vset1(float x) {
    return {_mm512_set1_ps(x)};
}
II_SIMD_INLINE VF vload(const float* p) {
    return {_mm512_loadu_ps(p)};
}
II_SIMD_INLINE void vstore(float* p, VF a) {
    _mm512_storeu_ps(p, a.v);
}
II_SIMD_INLINE VF vadd(VF a, VF b) {
    return {_mm512_add_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vsub(VF a, VF b) {
    return {_mm512_sub_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vmul(VF a, VF b) {
    return {_mm512_mul_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vdiv(VF a, VF b) {
    return {_mm512_div_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vfma(VF a, VF b, VF c) {
    return {_mm512_fmadd_ps(a.v, b.v, c.v)};
}
II_SIMD_INLINE VF vmin(VF a, VF b) {
    return {_mm512_min_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vmax(VF a, VF b) {
    return {_mm512_max_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vsqrt(VF a) {
    return {_mm512_sqrt_ps(a.v)};
}
II_SIMD_INLINE VF vround(VF a) {
    return {_mm512_roundscale_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
II_SIMD_INLINE VF vfloor(VF a) {
    return {_mm512_roundscale_ps(a.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)};
}
II_SIMD_INLINE VF vceil(VF a) {
    return {_mm512_roundscale_ps(a.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)};
}
II_SIMD_INLINE VF vtrunc(VF a) {
    return {_mm512_roundscale_ps(a.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
}

II_SIMD_INLINE VI vbits(VF a) {
    return {_mm512_castps_si512(a.v)};
}
II_SIMD_INLINE VF vfloat_bits(VI a) {
    return {_mm512_castsi512_ps(a.v)};
}
II_SIMD_INLINE VI vseti(std::int32_t x) {
    return {_mm512_set1_epi32(x)};
}
II_SIMD_INLINE VI vaddi(VI a, VI b) {
    return {_mm512_add_epi32(a.v, b.v)};
}
II_SIMD_INLINE VI vsubi(VI a, VI b) {
    return {_mm512_sub_epi32(a.v, b.v)};
}
II_SIMD_INLINE VI vandi(VI a, VI b) {
    return {_mm512_and_si512(a.v, b.v)};
}
II_SIMD_INLINE VI vori(VI a, VI b) {
    return {_mm512_or_si512(a.v, b.v)};
}
II_SIMD_INLINE VI vxori(VI a, VI b) {
    return {_mm512_xor_si512(a.v, b.v)};
}
template <int kShift>
II_SIMD_INLINE VI vshli(VI a) {
    return {_mm512_slli_epi32(a.v, kShift)};
}
template <int kShift>
II_SIMD_INLINE VI vsrli(VI a) {
    return {_mm512_srli_epi32(a.v, kShift)};
}
template <int kShift>
II_SIMD_INLINE VI vsrai(VI a) {
    return {_mm512_srai_epi32(a.v, kShift)};
}
II_SIMD_INLINE VI vcvt_round(VF a) {
    return {_mm512_cvtps_epi32(a.v)};
}
II_SIMD_INLINE VF vcvt_int(VI a) {
    return {_mm512_cvtepi32_ps(a.v)};
}
II_SIMD_INLINE VM veqi(VI a, VI b) {
    return {_mm512_cmpeq_epi32_mask(a.v, b.v)};
}
II_SIMD_INLINE VM vgti(VI a, VI b) {
    return {_mm512_cmpgt_epi32_mask(a.v, b.v)};
}

II_SIMD_INLINE VM vlt(VF a, VF b) {
    return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ)};
}
II_SIMD_INLINE VM vle(VF a, VF b) {
    return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ)};
}
II_SIMD_INLINE VM vgt(VF a, VF b) {
    return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ)};
}
II_SIMD_INLINE VM vge(VF a, VF b) {
    return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ)};
}
II_SIMD_INLINE VM veq(VF a, VF b) {
    return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ)};
}
II_SIMD_INLINE VM vne(VF a, VF b) {
    return {_mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_UQ)};
}
II_SIMD_INLINE VM visnan(VF a) {
    return {_mm512_cmp_ps_mask(a.v, a.v, _CMP_UNORD_Q)};
}
II_SIMD_INLINE VF vselect(VM m, VF a, VF b) {
    return {_mm512_mask_blend_ps(m.m, b.v, a.v)};
}
II_SIMD_INLINE VI vselecti(VM m, VI a, VI b) {
    return {_mm512_mask_blend_epi32(m.m, b.v, a.v)};
}
II_SIMD_INLINE VM vmand(VM a, VM b) {
    return {static_cast<__mmask16>(a.m & b.m)};
}
II_SIMD_INLINE VM vmor(VM a, VM b) {
    return {static_cast<__mmask16>(a.m | b.m)};
}
II_SIMD_INLINE VM vmxor(VM a, VM b) {
    return {static_cast<__mmask16>(a.m ^ b.m)};
}
II_SIMD_INLINE VM vmnot(VM a) {
    return {static_cast<__mmask16>(~a.m)};
}

II_SIMD_INLINE float vreduce_add(VF a) {
    return _mm512_reduce_add_ps(a.v);
}
II_SIMD_INLINE float vreduce_max(VF a) {
    return _mm512_reduce_max_ps(a.v);
}
II_SIMD_INLINE float vreduce_min(VF a) {
    return _mm512_reduce_min_ps(a.v);
}

II_SIMD_INLINE VF vload_f16(const std::uint16_t* p) {
    return {_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)))};
}
II_SIMD_INLINE void vstore_f16(std::uint16_t* p, VF a) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _mm512_cvtps_ph(a.v, _MM_FROUND_TO_NEAREST_INT));
}
II_SIMD_INLINE VI vload_u16(const std::uint16_t* p) {
    return {_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)))};
}
II_SIMD_INLINE void vstore_u16(std::uint16_t* p, VI a) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _mm512_cvtepi32_epi16(a.v));
}

inline constexpr int kDLanes = 8;
II_SIMD_INLINE VD vd_zero() {
    return {_mm512_setzero_pd()};
}
II_SIMD_INLINE VD vd_add(VD a, VD b) {
    return {_mm512_add_pd(a.v, b.v)};
}
II_SIMD_INLINE VD vd_mul(VD a, VD b) {
    return {_mm512_mul_pd(a.v, b.v)};
}
II_SIMD_INLINE VD vd_fma(VD a, VD b, VD c) {
    return {_mm512_fmadd_pd(a.v, b.v, c.v)};
}
II_SIMD_INLINE VD vd_lo(VF a) {
    return {_mm512_cvtps_pd(_mm512_castps512_ps256(a.v))};
}
II_SIMD_INLINE VD vd_hi(VF a) {
    return {_mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a.v), 1)))};
}
II_SIMD_INLINE double vd_reduce_add(VD a) {
    return _mm512_reduce_add_pd(a.v);
}

// =====================================================================================================================
#elif defined(II_ISA_IS_AVX2)

inline constexpr int kLanes = 8;
inline constexpr const char* kIsaName = "avx2";

struct VF {
    __m256 v;
};
struct VI {
    __m256i v;
};
struct VM {
    __m256 m;  // all-ones lanes
};
struct VD {
    __m256d v;
};

II_SIMD_INLINE VF vzero() {
    return {_mm256_setzero_ps()};
}
II_SIMD_INLINE VF vset1(float x) {
    return {_mm256_set1_ps(x)};
}
II_SIMD_INLINE VF vload(const float* p) {
    return {_mm256_loadu_ps(p)};
}
II_SIMD_INLINE void vstore(float* p, VF a) {
    _mm256_storeu_ps(p, a.v);
}
II_SIMD_INLINE VF vadd(VF a, VF b) {
    return {_mm256_add_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vsub(VF a, VF b) {
    return {_mm256_sub_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vmul(VF a, VF b) {
    return {_mm256_mul_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vdiv(VF a, VF b) {
    return {_mm256_div_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vfma(VF a, VF b, VF c) {
    return {_mm256_fmadd_ps(a.v, b.v, c.v)};
}
II_SIMD_INLINE VF vmin(VF a, VF b) {
    return {_mm256_min_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vmax(VF a, VF b) {
    return {_mm256_max_ps(a.v, b.v)};
}
II_SIMD_INLINE VF vsqrt(VF a) {
    return {_mm256_sqrt_ps(a.v)};
}
II_SIMD_INLINE VF vround(VF a) {
    return {_mm256_round_ps(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
II_SIMD_INLINE VF vfloor(VF a) {
    return {_mm256_round_ps(a.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)};
}
II_SIMD_INLINE VF vceil(VF a) {
    return {_mm256_round_ps(a.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)};
}
II_SIMD_INLINE VF vtrunc(VF a) {
    return {_mm256_round_ps(a.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)};
}

II_SIMD_INLINE VI vbits(VF a) {
    return {_mm256_castps_si256(a.v)};
}
II_SIMD_INLINE VF vfloat_bits(VI a) {
    return {_mm256_castsi256_ps(a.v)};
}
II_SIMD_INLINE VI vseti(std::int32_t x) {
    return {_mm256_set1_epi32(x)};
}
II_SIMD_INLINE VI vaddi(VI a, VI b) {
    return {_mm256_add_epi32(a.v, b.v)};
}
II_SIMD_INLINE VI vsubi(VI a, VI b) {
    return {_mm256_sub_epi32(a.v, b.v)};
}
II_SIMD_INLINE VI vandi(VI a, VI b) {
    return {_mm256_and_si256(a.v, b.v)};
}
II_SIMD_INLINE VI vori(VI a, VI b) {
    return {_mm256_or_si256(a.v, b.v)};
}
II_SIMD_INLINE VI vxori(VI a, VI b) {
    return {_mm256_xor_si256(a.v, b.v)};
}
template <int kShift>
II_SIMD_INLINE VI vshli(VI a) {
    return {_mm256_slli_epi32(a.v, kShift)};
}
template <int kShift>
II_SIMD_INLINE VI vsrli(VI a) {
    return {_mm256_srli_epi32(a.v, kShift)};
}
template <int kShift>
II_SIMD_INLINE VI vsrai(VI a) {
    return {_mm256_srai_epi32(a.v, kShift)};
}
II_SIMD_INLINE VI vcvt_round(VF a) {
    return {_mm256_cvtps_epi32(a.v)};
}
II_SIMD_INLINE VF vcvt_int(VI a) {
    return {_mm256_cvtepi32_ps(a.v)};
}
II_SIMD_INLINE VM veqi(VI a, VI b) {
    return {_mm256_castsi256_ps(_mm256_cmpeq_epi32(a.v, b.v))};
}
II_SIMD_INLINE VM vgti(VI a, VI b) {
    return {_mm256_castsi256_ps(_mm256_cmpgt_epi32(a.v, b.v))};
}

II_SIMD_INLINE VM vlt(VF a, VF b) {
    return {_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)};
}
II_SIMD_INLINE VM vle(VF a, VF b) {
    return {_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)};
}
II_SIMD_INLINE VM vgt(VF a, VF b) {
    return {_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)};
}
II_SIMD_INLINE VM vge(VF a, VF b) {
    return {_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ)};
}
II_SIMD_INLINE VM veq(VF a, VF b) {
    return {_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ)};
}
II_SIMD_INLINE VM vne(VF a, VF b) {
    return {_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_UQ)};
}
II_SIMD_INLINE VM visnan(VF a) {
    return {_mm256_cmp_ps(a.v, a.v, _CMP_UNORD_Q)};
}
II_SIMD_INLINE VF vselect(VM m, VF a, VF b) {
    return {_mm256_blendv_ps(b.v, a.v, m.m)};
}
II_SIMD_INLINE VI vselecti(VM m, VI a, VI b) {
    return {_mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(b.v), _mm256_castsi256_ps(a.v), m.m))};
}
II_SIMD_INLINE VM vmand(VM a, VM b) {
    return {_mm256_and_ps(a.m, b.m)};
}
II_SIMD_INLINE VM vmor(VM a, VM b) {
    return {_mm256_or_ps(a.m, b.m)};
}
II_SIMD_INLINE VM vmxor(VM a, VM b) {
    return {_mm256_xor_ps(a.m, b.m)};
}
II_SIMD_INLINE VM vmnot(VM a) {
    return {_mm256_xor_ps(a.m, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))};
}

II_SIMD_INLINE float vreduce_add(VF a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
II_SIMD_INLINE float vreduce_max(VF a) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
II_SIMD_INLINE float vreduce_min(VF a) {
    __m128 s = _mm_min_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
    s = _mm_min_ps(s, _mm_movehl_ps(s, s));
    s = _mm_min_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

II_SIMD_INLINE VF vload_f16(const std::uint16_t* p) {
    return {_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)))};
}
II_SIMD_INLINE void vstore_f16(std::uint16_t* p, VF a) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_cvtps_ph(a.v, _MM_FROUND_TO_NEAREST_INT));
}
II_SIMD_INLINE VI vload_u16(const std::uint16_t* p) {
    return {_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)))};
}
II_SIMD_INLINE void vstore_u16(std::uint16_t* p, VI a) {
    // Lanes hold values < 2^16: pack with unsigned saturation, then restore the lane order.
    const __m256i packed = _mm256_packus_epi32(a.v, a.v);
    const __m256i ordered = _mm256_permute4x64_epi64(packed, 0x08);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_castsi256_si128(ordered));
}

inline constexpr int kDLanes = 4;
II_SIMD_INLINE VD vd_zero() {
    return {_mm256_setzero_pd()};
}
II_SIMD_INLINE VD vd_add(VD a, VD b) {
    return {_mm256_add_pd(a.v, b.v)};
}
II_SIMD_INLINE VD vd_mul(VD a, VD b) {
    return {_mm256_mul_pd(a.v, b.v)};
}
II_SIMD_INLINE VD vd_fma(VD a, VD b, VD c) {
    return {_mm256_fmadd_pd(a.v, b.v, c.v)};
}
II_SIMD_INLINE VD vd_lo(VF a) {
    return {_mm256_cvtps_pd(_mm256_castps256_ps128(a.v))};
}
II_SIMD_INLINE VD vd_hi(VF a) {
    return {_mm256_cvtps_pd(_mm256_extractf128_ps(a.v, 1))};
}
II_SIMD_INLINE double vd_reduce_add(VD a) {
    __m128d s = _mm_add_pd(_mm256_castpd256_pd128(a.v), _mm256_extractf128_pd(a.v, 1));
    s = _mm_add_sd(s, _mm_unpackhi_pd(s, s));
    return _mm_cvtsd_f64(s);
}

// =====================================================================================================================
#elif defined(II_ISA_IS_NEON)

inline constexpr int kLanes = 4;
inline constexpr const char* kIsaName = "neon";

struct VF {
    float32x4_t v;
};
struct VI {
    int32x4_t v;
};
struct VM {
    uint32x4_t m;
};
struct VD {
    float64x2_t v;
};

II_SIMD_INLINE VF vzero() {
    return {vdupq_n_f32(0.0f)};
}
II_SIMD_INLINE VF vset1(float x) {
    return {vdupq_n_f32(x)};
}
II_SIMD_INLINE VF vload(const float* p) {
    return {vld1q_f32(p)};
}
II_SIMD_INLINE void vstore(float* p, VF a) {
    vst1q_f32(p, a.v);
}
II_SIMD_INLINE VF vadd(VF a, VF b) {
    return {vaddq_f32(a.v, b.v)};
}
II_SIMD_INLINE VF vsub(VF a, VF b) {
    return {vsubq_f32(a.v, b.v)};
}
II_SIMD_INLINE VF vmul(VF a, VF b) {
    return {vmulq_f32(a.v, b.v)};
}
II_SIMD_INLINE VF vdiv(VF a, VF b) {
    return {vdivq_f32(a.v, b.v)};
}
II_SIMD_INLINE VF vfma(VF a, VF b, VF c) {
    return {vfmaq_f32(c.v, a.v, b.v)};
}
II_SIMD_INLINE VF vmin(VF a, VF b) {
    return {vminq_f32(a.v, b.v)};
}
II_SIMD_INLINE VF vmax(VF a, VF b) {
    return {vmaxq_f32(a.v, b.v)};
}
II_SIMD_INLINE VF vsqrt(VF a) {
    return {vsqrtq_f32(a.v)};
}
II_SIMD_INLINE VF vround(VF a) {
    return {vrndnq_f32(a.v)};
}
II_SIMD_INLINE VF vfloor(VF a) {
    return {vrndmq_f32(a.v)};
}
II_SIMD_INLINE VF vceil(VF a) {
    return {vrndpq_f32(a.v)};
}
II_SIMD_INLINE VF vtrunc(VF a) {
    return {vrndq_f32(a.v)};
}

II_SIMD_INLINE VI vbits(VF a) {
    return {vreinterpretq_s32_f32(a.v)};
}
II_SIMD_INLINE VF vfloat_bits(VI a) {
    return {vreinterpretq_f32_s32(a.v)};
}
II_SIMD_INLINE VI vseti(std::int32_t x) {
    return {vdupq_n_s32(x)};
}
II_SIMD_INLINE VI vaddi(VI a, VI b) {
    return {vaddq_s32(a.v, b.v)};
}
II_SIMD_INLINE VI vsubi(VI a, VI b) {
    return {vsubq_s32(a.v, b.v)};
}
II_SIMD_INLINE VI vandi(VI a, VI b) {
    return {vandq_s32(a.v, b.v)};
}
II_SIMD_INLINE VI vori(VI a, VI b) {
    return {vorrq_s32(a.v, b.v)};
}
II_SIMD_INLINE VI vxori(VI a, VI b) {
    return {veorq_s32(a.v, b.v)};
}
template <int kShift>
II_SIMD_INLINE VI vshli(VI a) {
    return {vshlq_n_s32(a.v, kShift)};
}
template <int kShift>
II_SIMD_INLINE VI vsrli(VI a) {
    return {vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a.v), kShift))};
}
template <int kShift>
II_SIMD_INLINE VI vsrai(VI a) {
    return {vshrq_n_s32(a.v, kShift)};
}
II_SIMD_INLINE VI vcvt_round(VF a) {
    return {vcvtnq_s32_f32(a.v)};
}
II_SIMD_INLINE VF vcvt_int(VI a) {
    return {vcvtq_f32_s32(a.v)};
}
II_SIMD_INLINE VM veqi(VI a, VI b) {
    return {vceqq_s32(a.v, b.v)};
}
II_SIMD_INLINE VM vgti(VI a, VI b) {
    return {vcgtq_s32(a.v, b.v)};
}

II_SIMD_INLINE VM vlt(VF a, VF b) {
    return {vcltq_f32(a.v, b.v)};
}
II_SIMD_INLINE VM vle(VF a, VF b) {
    return {vcleq_f32(a.v, b.v)};
}
II_SIMD_INLINE VM vgt(VF a, VF b) {
    return {vcgtq_f32(a.v, b.v)};
}
II_SIMD_INLINE VM vge(VF a, VF b) {
    return {vcgeq_f32(a.v, b.v)};
}
II_SIMD_INLINE VM veq(VF a, VF b) {
    return {vceqq_f32(a.v, b.v)};
}
II_SIMD_INLINE VM vne(VF a, VF b) {
    return {vmvnq_u32(vceqq_f32(a.v, b.v))};
}
II_SIMD_INLINE VM visnan(VF a) {
    return {vmvnq_u32(vceqq_f32(a.v, a.v))};
}
II_SIMD_INLINE VF vselect(VM m, VF a, VF b) {
    return {vbslq_f32(m.m, a.v, b.v)};
}
II_SIMD_INLINE VI vselecti(VM m, VI a, VI b) {
    return {vbslq_s32(m.m, a.v, b.v)};
}
II_SIMD_INLINE VM vmand(VM a, VM b) {
    return {vandq_u32(a.m, b.m)};
}
II_SIMD_INLINE VM vmor(VM a, VM b) {
    return {vorrq_u32(a.m, b.m)};
}
II_SIMD_INLINE VM vmxor(VM a, VM b) {
    return {veorq_u32(a.m, b.m)};
}
II_SIMD_INLINE VM vmnot(VM a) {
    return {vmvnq_u32(a.m)};
}

II_SIMD_INLINE float vreduce_add(VF a) {
    return vaddvq_f32(a.v);
}
II_SIMD_INLINE float vreduce_max(VF a) {
    return vmaxvq_f32(a.v);
}
II_SIMD_INLINE float vreduce_min(VF a) {
    return vminvq_f32(a.v);
}

II_SIMD_INLINE VF vload_f16(const std::uint16_t* p) {
    return {vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)))};
}
II_SIMD_INLINE void vstore_f16(std::uint16_t* p, VF a) {
    vst1_u16(p, vreinterpret_u16_f16(vcvt_f16_f32(a.v)));
}
II_SIMD_INLINE VI vload_u16(const std::uint16_t* p) {
    return {vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p)))};
}
II_SIMD_INLINE void vstore_u16(std::uint16_t* p, VI a) {
    vst1_u16(p, vmovn_u32(vreinterpretq_u32_s32(a.v)));
}

inline constexpr int kDLanes = 2;
II_SIMD_INLINE VD vd_zero() {
    return {vdupq_n_f64(0.0)};
}
II_SIMD_INLINE VD vd_add(VD a, VD b) {
    return {vaddq_f64(a.v, b.v)};
}
II_SIMD_INLINE VD vd_mul(VD a, VD b) {
    return {vmulq_f64(a.v, b.v)};
}
II_SIMD_INLINE VD vd_fma(VD a, VD b, VD c) {
    return {vfmaq_f64(c.v, a.v, b.v)};
}
II_SIMD_INLINE VD vd_lo(VF a) {
    return {vcvt_f64_f32(vget_low_f32(a.v))};
}
II_SIMD_INLINE VD vd_hi(VF a) {
    return {vcvt_high_f64_f32(a.v)};
}
II_SIMD_INLINE double vd_reduce_add(VD a) {
    return vaddvq_f64(a.v);
}

// =====================================================================================================================
#else  // generic: portable C++ over small arrays (auto-vectorized for the baseline: SSE2 / NEON)

inline constexpr int kLanes = 4;
inline constexpr const char* kIsaName = "generic";

struct VF {
    float v[4];
};
struct VI {
    std::int32_t v[4];
};
struct VM {
    bool m[4];
};
struct VD {
    double v[2];
};

#define II_GENERIC_MAP(Type, expression)                                                                               \
    Type r{};                                                                                                          \
    for (int i = 0; i < 4; ++i) {                                                                                      \
        r.v[i] = (expression);                                                                                         \
    }                                                                                                                  \
    return r
#define II_GENERIC_MASK(expression)                                                                                    \
    VM r{};                                                                                                            \
    for (int i = 0; i < 4; ++i) {                                                                                      \
        r.m[i] = (expression);                                                                                         \
    }                                                                                                                  \
    return r

II_SIMD_INLINE float generic_bits_to_float(std::int32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
II_SIMD_INLINE std::int32_t generic_float_to_bits(float f) {
    std::int32_t bits;
    std::memcpy(&bits, &f, sizeof f);
    return bits;
}

II_SIMD_INLINE VF vzero() {
    return VF{{0.0f, 0.0f, 0.0f, 0.0f}};
}
II_SIMD_INLINE VF vset1(float x) {
    return VF{{x, x, x, x}};
}
II_SIMD_INLINE VF vload(const float* p) {
    VF r{};
    std::memcpy(r.v, p, sizeof r.v);
    return r;
}
II_SIMD_INLINE void vstore(float* p, VF a) {
    std::memcpy(p, a.v, sizeof a.v);
}
II_SIMD_INLINE VF vadd(VF a, VF b) {
    II_GENERIC_MAP(VF, a.v[i] + b.v[i]);
}
II_SIMD_INLINE VF vsub(VF a, VF b) {
    II_GENERIC_MAP(VF, a.v[i] - b.v[i]);
}
II_SIMD_INLINE VF vmul(VF a, VF b) {
    II_GENERIC_MAP(VF, a.v[i] * b.v[i]);
}
II_SIMD_INLINE VF vdiv(VF a, VF b) {
    II_GENERIC_MAP(VF, a.v[i] / b.v[i]);
}
II_SIMD_INLINE VF vfma(VF a, VF b, VF c) {
    II_GENERIC_MAP(VF, a.v[i] * b.v[i] + c.v[i]);
}
II_SIMD_INLINE VF vmin(VF a, VF b) {
    II_GENERIC_MAP(VF, a.v[i] < b.v[i] ? a.v[i] : b.v[i]);
}
II_SIMD_INLINE VF vmax(VF a, VF b) {
    II_GENERIC_MAP(VF, a.v[i] > b.v[i] ? a.v[i] : b.v[i]);
}
// The generic variant is compiled without extra instruction-set flags, so the C++ library is safe to use here.
II_SIMD_INLINE VF vsqrt(VF a) {
    II_GENERIC_MAP(VF, std::sqrt(a.v[i]));
}
II_SIMD_INLINE VF vround(VF a) {
    II_GENERIC_MAP(VF, std::nearbyint(a.v[i]));
}  // default mode: nearest even
II_SIMD_INLINE VF vfloor(VF a) {
    II_GENERIC_MAP(VF, std::floor(a.v[i]));
}
II_SIMD_INLINE VF vceil(VF a) {
    II_GENERIC_MAP(VF, std::ceil(a.v[i]));
}
II_SIMD_INLINE VF vtrunc(VF a) {
    II_GENERIC_MAP(VF, std::trunc(a.v[i]));
}

II_SIMD_INLINE VI vbits(VF a) {
    II_GENERIC_MAP(VI, generic_float_to_bits(a.v[i]));
}
II_SIMD_INLINE VF vfloat_bits(VI a) {
    II_GENERIC_MAP(VF, generic_bits_to_float(a.v[i]));
}
II_SIMD_INLINE VI vseti(std::int32_t x) {
    return VI{{x, x, x, x}};
}
II_SIMD_INLINE VI vaddi(VI a, VI b) {
    II_GENERIC_MAP(VI,
                   static_cast<std::int32_t>(static_cast<std::uint32_t>(a.v[i]) + static_cast<std::uint32_t>(b.v[i])));
}
II_SIMD_INLINE VI vsubi(VI a, VI b) {
    II_GENERIC_MAP(VI,
                   static_cast<std::int32_t>(static_cast<std::uint32_t>(a.v[i]) - static_cast<std::uint32_t>(b.v[i])));
}
II_SIMD_INLINE VI vandi(VI a, VI b) {
    II_GENERIC_MAP(VI, a.v[i] & b.v[i]);
}
II_SIMD_INLINE VI vori(VI a, VI b) {
    II_GENERIC_MAP(VI, a.v[i] | b.v[i]);
}
II_SIMD_INLINE VI vxori(VI a, VI b) {
    II_GENERIC_MAP(VI, a.v[i] ^ b.v[i]);
}
template <int kShift>
II_SIMD_INLINE VI vshli(VI a) {
    II_GENERIC_MAP(VI, static_cast<std::int32_t>(static_cast<std::uint32_t>(a.v[i]) << kShift));
}
template <int kShift>
II_SIMD_INLINE VI vsrli(VI a) {
    II_GENERIC_MAP(VI, static_cast<std::int32_t>(static_cast<std::uint32_t>(a.v[i]) >> kShift));
}
template <int kShift>
II_SIMD_INLINE VI vsrai(VI a) {
    II_GENERIC_MAP(VI, a.v[i] >> kShift);
}
II_SIMD_INLINE VI vcvt_round(VF a) {
    II_GENERIC_MAP(VI, static_cast<std::int32_t>(std::nearbyint(a.v[i])));
}
II_SIMD_INLINE VF vcvt_int(VI a) {
    II_GENERIC_MAP(VF, static_cast<float>(a.v[i]));
}
II_SIMD_INLINE VM veqi(VI a, VI b) {
    II_GENERIC_MASK(a.v[i] == b.v[i]);
}
II_SIMD_INLINE VM vgti(VI a, VI b) {
    II_GENERIC_MASK(a.v[i] > b.v[i]);
}

II_SIMD_INLINE VM vlt(VF a, VF b) {
    II_GENERIC_MASK(a.v[i] < b.v[i]);
}
II_SIMD_INLINE VM vle(VF a, VF b) {
    II_GENERIC_MASK(a.v[i] <= b.v[i]);
}
II_SIMD_INLINE VM vgt(VF a, VF b) {
    II_GENERIC_MASK(a.v[i] > b.v[i]);
}
II_SIMD_INLINE VM vge(VF a, VF b) {
    II_GENERIC_MASK(a.v[i] >= b.v[i]);
}
II_SIMD_INLINE VM veq(VF a, VF b) {
    II_GENERIC_MASK(a.v[i] == b.v[i]);
}
II_SIMD_INLINE VM vne(VF a, VF b) {
    II_GENERIC_MASK(!(a.v[i] == b.v[i]));
}
II_SIMD_INLINE VM visnan(VF a) {
    II_GENERIC_MASK(a.v[i] != a.v[i]);
}
II_SIMD_INLINE VF vselect(VM m, VF a, VF b) {
    II_GENERIC_MAP(VF, m.m[i] ? a.v[i] : b.v[i]);
}
II_SIMD_INLINE VI vselecti(VM m, VI a, VI b) {
    II_GENERIC_MAP(VI, m.m[i] ? a.v[i] : b.v[i]);
}
II_SIMD_INLINE VM vmand(VM a, VM b) {
    II_GENERIC_MASK(a.m[i] && b.m[i]);
}
II_SIMD_INLINE VM vmor(VM a, VM b) {
    II_GENERIC_MASK(a.m[i] || b.m[i]);
}
II_SIMD_INLINE VM vmxor(VM a, VM b) {
    II_GENERIC_MASK(a.m[i] != b.m[i]);
}
II_SIMD_INLINE VM vmnot(VM a) {
    II_GENERIC_MASK(!a.m[i]);
}

II_SIMD_INLINE float vreduce_add(VF a) {
    return (a.v[0] + a.v[2]) + (a.v[1] + a.v[3]);
}
II_SIMD_INLINE float vreduce_max(VF a) {
    const float x = a.v[0] > a.v[2] ? a.v[0] : a.v[2];
    const float y = a.v[1] > a.v[3] ? a.v[1] : a.v[3];
    return x > y ? x : y;
}
II_SIMD_INLINE float vreduce_min(VF a) {
    const float x = a.v[0] < a.v[2] ? a.v[0] : a.v[2];
    const float y = a.v[1] < a.v[3] ? a.v[1] : a.v[3];
    return x < y ? x : y;
}

II_SIMD_INLINE VI vload_u16(const std::uint16_t* p) {
    II_GENERIC_MAP(VI, static_cast<std::int32_t>(p[i]));
}
II_SIMD_INLINE void vstore_u16(std::uint16_t* p, VI a) {
    for (int i = 0; i < 4; ++i) {
        p[i] = static_cast<std::uint16_t>(a.v[i]);
    }
}

inline constexpr int kDLanes = 2;
II_SIMD_INLINE VD vd_zero() {
    return VD{{0.0, 0.0}};
}
II_SIMD_INLINE VD vd_add(VD a, VD b) {
    return VD{{a.v[0] + b.v[0], a.v[1] + b.v[1]}};
}
II_SIMD_INLINE VD vd_mul(VD a, VD b) {
    return VD{{a.v[0] * b.v[0], a.v[1] * b.v[1]}};
}
II_SIMD_INLINE VD vd_fma(VD a, VD b, VD c) {
    return VD{{a.v[0] * b.v[0] + c.v[0], a.v[1] * b.v[1] + c.v[1]}};
}
II_SIMD_INLINE VD vd_lo(VF a) {
    return VD{{a.v[0], a.v[1]}};
}
II_SIMD_INLINE VD vd_hi(VF a) {
    return VD{{a.v[2], a.v[3]}};
}
II_SIMD_INLINE double vd_reduce_add(VD a) {
    return a.v[0] + a.v[1];
}

#undef II_GENERIC_MAP
#undef II_GENERIC_MASK

#endif
// =====================================================================================================================
// Portable helpers built on the primitives above.

/// Hints the hardware to fetch the cache line at `p` (no effect on correctness).
II_SIMD_INLINE void vprefetch(const void* p) {
#if defined(II_ISA_IS_AVX2) || defined(II_ISA_IS_AVX512)
    _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p);
#else
    (void)p;
#endif
}

II_SIMD_INLINE VF vneg(VF a) {
    return vfloat_bits(vxori(vbits(a), vseti(static_cast<std::int32_t>(0x80000000u))));
}
II_SIMD_INLINE VF vabs(VF a) {
    return vfloat_bits(vandi(vbits(a), vseti(0x7fffffff)));
}
/// `a` with the sign of `b`.
II_SIMD_INLINE VF vcopysign(VF a, VF b) {
    const VI sign = vseti(static_cast<std::int32_t>(0x80000000u));
    return vfloat_bits(vori(vandi(vbits(a), vseti(0x7fffffff)), vandi(vbits(b), sign)));
}
/// 1.0f where the mask is set, 0.0f elsewhere.
II_SIMD_INLINE VF vmask_to_float(VM m) {
    return vselect(m, vset1(1.0f), vzero());
}
/// Lanes that are "true" (non-zero, NaN included) — the bool interpretation of a float.
II_SIMD_INLINE VM vtruthy(VF a) {
    return vne(a, vzero());
}
/// NaN-propagating minimum and maximum (as in ii-ref).
II_SIMD_INLINE VF vmin_nan(VF a, VF b) {
    return vselect(vmor(visnan(a), visnan(b)), vadd(a, b), vmin(a, b));
}
II_SIMD_INLINE VF vmax_nan(VF a, VF b) {
    return vselect(vmor(visnan(a), visnan(b)), vadd(a, b), vmax(a, b));
}

/// Loads `n < kLanes` floats (the rest zero).
II_SIMD_INLINE VF vload_partial(const float* p, std::int64_t n) {
    alignas(64) float buffer[kLanes] = {};
    for (std::int64_t i = 0; i < n; ++i) {
        buffer[i] = p[i];
    }
    return vload(buffer);
}
II_SIMD_INLINE void vstore_partial(float* p, VF a, std::int64_t n) {
    alignas(64) float buffer[kLanes];
    vstore(buffer, a);
    for (std::int64_t i = 0; i < n; ++i) {
        p[i] = buffer[i];
    }
}

// --- f16 / bf16 conversions (round to nearest even; NaN stays NaN) ---

#if defined(II_ISA_IS_AVX2) || defined(II_ISA_IS_AVX512) || defined(II_ISA_IS_NEON)
// Hardware conversions above.
#else
II_SIMD_INLINE float generic_f16_to_f32(std::uint16_t h) {
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
        // Subnormal: normalize.
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
}
II_SIMD_INLINE std::uint16_t generic_f32_to_f16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t abs = bits & 0x7fffffffu;
    if (abs >= 0x7f800000u) {
        return static_cast<std::uint16_t>(sign | 0x7c00u | (abs > 0x7f800000u ? 0x200u : 0u));
    }
    if (abs >= 0x477ff000u) {  // rounds to >= 65520: overflow to infinity
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (abs < 0x38800000u) {  // subnormal or zero in f16
        // Shift the implicit-one mantissa into place with round to nearest even.
        if (abs < 0x33000000u) {
            return static_cast<std::uint16_t>(sign);  // < half of the smallest subnormal
        }
        const std::uint32_t exponent = abs >> 23;
        const std::uint32_t mantissa = (abs & 0x7fffffu) | 0x800000u;
        const std::uint32_t shift = 126u - exponent;  // 14..24
        const std::uint32_t half = 1u << (shift - 1);
        const std::uint32_t rest = mantissa & ((1u << shift) - 1u);
        std::uint32_t result = mantissa >> shift;
        if (rest > half || (rest == half && (result & 1u) != 0)) {
            ++result;
        }
        return static_cast<std::uint16_t>(sign | result);
    }
    const std::uint32_t rounded = abs + 0xfffu + ((abs >> 13) & 1u);
    return static_cast<std::uint16_t>(sign | ((rounded - 0x38000000u) >> 13));
}
II_SIMD_INLINE VF vload_f16(const std::uint16_t* p) {
    VF r{};
    for (int i = 0; i < 4; ++i) {
        r.v[i] = generic_f16_to_f32(p[i]);
    }
    return r;
}
II_SIMD_INLINE void vstore_f16(std::uint16_t* p, VF a) {
    for (int i = 0; i < 4; ++i) {
        p[i] = generic_f32_to_f16(a.v[i]);
    }
}
#endif

II_SIMD_INLINE VF vload_bf16(const std::uint16_t* p) {
    return vfloat_bits(vshli<16>(vload_u16(p)));
}
II_SIMD_INLINE void vstore_bf16(std::uint16_t* p, VF a) {
    const VI bits = vbits(a);
    // Round to nearest even: add 0x7fff plus the lowest kept bit; NaN becomes a quiet NaN.
    const VI lsb = vandi(vsrli<16>(bits), vseti(1));
    const VI rounded = vsrli<16>(vaddi(vaddi(bits, vseti(0x7fff)), lsb));
    const VI result = vselecti(visnan(a), vseti(0x7fc0), rounded);
    vstore_u16(p, vandi(result, vseti(0xffff)));
}

}  // namespace ii::cpu::II_ISA_NS
