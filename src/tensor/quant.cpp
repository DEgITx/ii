// SPDX-License-Identifier: MIT
// Reference codecs of the ggml block-quantized types. Blocks are read and written through byte pointers (no
// alignment assumptions); the field offsets are those of ggml's block structs.
#include "tensor/quant.h"

#include <ii/half.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ii::quant {

namespace {

constexpr int kQK = 32;    // elements per block of the classic types
constexpr int kQKK = 256;  // elements per K-quant super-block
constexpr float kGroupMaxEps = 1e-15f;

float load_f16(const std::uint8_t* p) noexcept {
    std::uint16_t bits = 0;
    std::memcpy(&bits, p, 2);
    return f16::from_bits(bits).to_float();
}

void store_f16(std::uint8_t* p, float value) noexcept {
    const std::uint16_t bits = f16(value).bits;
    std::memcpy(p, &bits, 2);
}

float load_f32(const std::uint8_t* p) noexcept {
    float value = 0;
    std::memcpy(&value, p, 4);
    return value;
}

/// Truncation toward zero, as ggml's casts `(int8_t)(x + 8.5f)` do (the arguments are non-negative there).
int truncate(float value) noexcept {
    return static_cast<int>(value);
}

/// Round to nearest, ties to even, for |value| < 2^22 (ggml's `nearest_int`).
int nearest_int(float value) noexcept {
    const float shifted = value + 12582912.f;
    std::int32_t bits = 0;
    std::memcpy(&bits, &shifted, 4);
    return (bits & 0x007fffff) - 0x00400000;
}

// ------------------------------------------------------------------------------------------------------------------
// Classic 32-element blocks

void dequantize_q4_0(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f16(b);
    const std::uint8_t* qs = b + 2;
    for (int j = 0; j < kQK / 2; ++j) {
        y[j] = static_cast<float>((qs[j] & 0x0F) - 8) * d;
        y[j + kQK / 2] = static_cast<float>((qs[j] >> 4) - 8) * d;
    }
}

void quantize_q4_0(const float* x, std::uint8_t* b) noexcept {
    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < kQK; ++j) {
        if (amax < std::fabs(x[j])) {
            amax = std::fabs(x[j]);
            max = x[j];
        }
    }
    const float d = max / -8;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    store_f16(b, d);
    for (int j = 0; j < kQK / 2; ++j) {
        const auto q0 = static_cast<std::uint8_t>(std::min(15, truncate(x[j] * id + 8.5f)));
        const auto q1 = static_cast<std::uint8_t>(std::min(15, truncate(x[j + kQK / 2] * id + 8.5f)));
        b[2 + j] = static_cast<std::uint8_t>(q0 | (q1 << 4));
    }
}

void dequantize_q4_1(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f16(b);
    const float m = load_f16(b + 2);
    const std::uint8_t* qs = b + 4;
    for (int j = 0; j < kQK / 2; ++j) {
        y[j] = static_cast<float>(qs[j] & 0x0F) * d + m;
        y[j + kQK / 2] = static_cast<float>(qs[j] >> 4) * d + m;
    }
}

void quantize_q4_1(const float* x, std::uint8_t* b) noexcept {
    float min = x[0];
    float max = x[0];
    for (int j = 1; j < kQK; ++j) {
        min = std::min(min, x[j]);
        max = std::max(max, x[j]);
    }
    const float d = (max - min) / 15;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    store_f16(b, d);
    store_f16(b + 2, min);
    for (int j = 0; j < kQK / 2; ++j) {
        const auto q0 = static_cast<std::uint8_t>(std::min(15, truncate((x[j] - min) * id + 0.5f)));
        const auto q1 = static_cast<std::uint8_t>(std::min(15, truncate((x[j + kQK / 2] - min) * id + 0.5f)));
        b[4 + j] = static_cast<std::uint8_t>(q0 | (q1 << 4));
    }
}

void dequantize_q5_0(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f16(b);
    std::uint32_t qh = 0;
    std::memcpy(&qh, b + 2, 4);
    const std::uint8_t* qs = b + 6;
    for (int j = 0; j < kQK / 2; ++j) {
        const auto h0 = static_cast<std::uint8_t>(((qh >> j) << 4) & 0x10);
        const auto h1 = static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
        y[j] = static_cast<float>(((qs[j] & 0x0F) | h0) - 16) * d;
        y[j + kQK / 2] = static_cast<float>(((qs[j] >> 4) | h1) - 16) * d;
    }
}

void quantize_q5_0(const float* x, std::uint8_t* b) noexcept {
    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < kQK; ++j) {
        if (amax < std::fabs(x[j])) {
            amax = std::fabs(x[j]);
            max = x[j];
        }
    }
    const float d = max / -16;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    store_f16(b, d);
    std::uint32_t qh = 0;
    for (int j = 0; j < kQK / 2; ++j) {
        const auto q0 = static_cast<std::uint8_t>(std::min(31, truncate(x[j] * id + 16.5f)));
        const auto q1 = static_cast<std::uint8_t>(std::min(31, truncate(x[j + kQK / 2] * id + 16.5f)));
        b[6 + j] = static_cast<std::uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        qh |= static_cast<std::uint32_t>((q0 & 0x10) >> 4) << j;
        qh |= static_cast<std::uint32_t>((q1 & 0x10) >> 4) << (j + kQK / 2);
    }
    std::memcpy(b + 2, &qh, 4);
}

void dequantize_q5_1(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f16(b);
    const float m = load_f16(b + 2);
    std::uint32_t qh = 0;
    std::memcpy(&qh, b + 4, 4);
    const std::uint8_t* qs = b + 8;
    for (int j = 0; j < kQK / 2; ++j) {
        const auto h0 = static_cast<std::uint8_t>(((qh >> j) << 4) & 0x10);
        const auto h1 = static_cast<std::uint8_t>((qh >> (j + 12)) & 0x10);
        y[j] = static_cast<float>((qs[j] & 0x0F) | h0) * d + m;
        y[j + kQK / 2] = static_cast<float>((qs[j] >> 4) | h1) * d + m;
    }
}

void quantize_q5_1(const float* x, std::uint8_t* b) noexcept {
    float min = x[0];
    float max = x[0];
    for (int j = 1; j < kQK; ++j) {
        min = std::min(min, x[j]);
        max = std::max(max, x[j]);
    }
    const float d = (max - min) / 31;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    store_f16(b, d);
    store_f16(b + 2, min);
    std::uint32_t qh = 0;
    for (int j = 0; j < kQK / 2; ++j) {
        const auto q0 = static_cast<std::uint8_t>(truncate((x[j] - min) * id + 0.5f));
        const auto q1 = static_cast<std::uint8_t>(truncate((x[j + kQK / 2] - min) * id + 0.5f));
        b[8 + j] = static_cast<std::uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        qh |= static_cast<std::uint32_t>((q0 & 0x10) >> 4) << j;
        qh |= static_cast<std::uint32_t>((q1 & 0x10) >> 4) << (j + kQK / 2);
    }
    std::memcpy(b + 4, &qh, 4);
}

void dequantize_q8_0(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f16(b);
    for (int j = 0; j < kQK; ++j) {
        y[j] = static_cast<float>(static_cast<std::int8_t>(b[2 + j])) * d;
    }
}

void quantize_q8_0(const float* x, std::uint8_t* b) noexcept {
    float amax = 0.0f;
    for (int j = 0; j < kQK; ++j) {
        amax = std::max(amax, std::fabs(x[j]));
    }
    const float d = amax / 127;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    store_f16(b, d);
    for (int j = 0; j < kQK; ++j) {
        b[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::round(x[j] * id)));
    }
}

void dequantize_q8_1(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f16(b);
    for (int j = 0; j < kQK; ++j) {
        y[j] = static_cast<float>(static_cast<std::int8_t>(b[4 + j])) * d;
    }
}

void quantize_q8_1(const float* x, std::uint8_t* b) noexcept {
    float amax = 0.0f;
    for (int j = 0; j < kQK; ++j) {
        amax = std::max(amax, std::fabs(x[j]));
    }
    const float d = amax / 127;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int sum = 0;
    for (int j = 0; j < kQK; ++j) {
        const auto q = static_cast<std::int8_t>(std::round(x[j] * id));
        b[4 + j] = static_cast<std::uint8_t>(q);
        sum += q;
    }
    store_f16(b, d);
    store_f16(b + 2, static_cast<float>(sum) * d);
}

// ------------------------------------------------------------------------------------------------------------------
// K-quants: helpers of the reference quantizers

/// Asymmetric quantization of `n` values to [0, nmax] with an iterative refinement of scale and minimum.
/// Returns the scale and stores the (negated, non-negative) minimum in `the_min`.
float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, std::uint8_t* L, float* the_min,
                       std::uint8_t* Laux, float rmin, float rdelta, int nstep, bool use_mad) noexcept {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        min = std::min(min, x[i]);
        max = std::max(max, x[i]);
        const float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max == min) {
        std::fill(L, L + n, std::uint8_t{0});
        *the_min = -min;
        return 0.0f;
    }
    float iscale = static_cast<float>(nmax) / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        const int l = nearest_int(iscale * (x[i] - min));
        L[i] = static_cast<std::uint8_t>(std::max(0, std::min(nmax, l)));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * static_cast<float>(is) + static_cast<float>(nmax)) / (max - min);
        float sum_l = 0;
        float sum_l2 = 0;
        float sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * (x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = static_cast<std::uint8_t>(l);
            const float w = weights[i];
            const auto fl = static_cast<float>(l);
            sum_l += w * fl;
            sum_l2 += w * fl * fl;
            sum_xl += w * fl * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? std::fabs(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                std::copy(Laux, Laux + n, L);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

/// Symmetric quantization of `n` values to [-nmax, nmax - 1] (stored offset by nmax), weighted by x² (rmse_type 1).
float make_qx_quants(int n, int nmax, const float* x, std::int8_t* L) noexcept {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        const float ax = std::fabs(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < kGroupMaxEps) {
        std::fill(L, L + n, std::int8_t{0});
        return 0.0f;
    }
    float iscale = -static_cast<float>(nmax) / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = static_cast<std::int8_t>(l + nmax);
        const float w = x[i] * x[i];
        sumlx += w * x[i] * static_cast<float>(l);
        suml2 += w * static_cast<float>(l) * static_cast<float>(l);
    }
    float scale = suml2 != 0.0f ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(static_cast<float>(nmax) + 0.1f * static_cast<float>(is)) / max;
        sumlx = 0;
        suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            const float w = x[i] * x[i];
            sumlx += w * x[i] * static_cast<float>(l);
            suml2 += w * static_cast<float>(l) * static_cast<float>(l);
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                const int l = nearest_int(iscale * x[i]);
                L[i] = static_cast<std::int8_t>(nmax + std::max(-nmax, std::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

/// Symmetric 3-bit quantization with coordinate-descent refinement (q3_k).
float make_q3_quants(int n, int nmax, const float* x, std::int8_t* L) noexcept {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        const float ax = std::fabs(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < kGroupMaxEps) {
        std::fill(L, L + n, std::int8_t{0});
        return 0.0f;
    }
    const float iscale = -static_cast<float>(nmax) / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = static_cast<std::int8_t>(l);
        const float w = x[i] * x[i];
        sumlx += w * x[i] * static_cast<float>(l);
        suml2 += w * static_cast<float>(l) * static_cast<float>(l);
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = x[i] * x[i];
            float slx = sumlx - w * x[i] * static_cast<float>(L[i]);
            if (slx > 0) {
                float sl2 = suml2 - w * static_cast<float>(L[i]) * static_cast<float>(L[i]);
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = std::max(-nmax, std::min(nmax - 1, new_l));
                if (new_l != L[i]) {
                    slx += w * x[i] * static_cast<float>(new_l);
                    sl2 += w * static_cast<float>(new_l) * static_cast<float>(new_l);
                    if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = static_cast<std::int8_t>(new_l);
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (n_changed == 0) {
            break;
        }
    }
    for (int i = 0; i < n; ++i) {
        L[i] = static_cast<std::int8_t>(L[i] + nmax);
    }
    return sumlx / suml2;
}

/// 6-bit scale and minimum of sub-block `j` of a q4_k / q5_k super-block.
void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) noexcept {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = static_cast<std::uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

// ------------------------------------------------------------------------------------------------------------------
// q2_k: scales[16] qs[64] d dmin

void dequantize_q2_k(const std::uint8_t* b, float* y) noexcept {
    const std::uint8_t* scales = b;
    const std::uint8_t* q = b + 16;
    const float d = load_f16(b + 80);
    const float min = load_f16(b + 82);
    int is = 0;
    for (int n = 0; n < kQKK; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            std::uint8_t sc = scales[is++];
            float dl = d * static_cast<float>(sc & 0x0F);
            float ml = min * static_cast<float>(sc >> 4);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * static_cast<float>((q[l] >> shift) & 3) - ml;
            }
            sc = scales[is++];
            dl = d * static_cast<float>(sc & 0x0F);
            ml = min * static_cast<float>(sc >> 4);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * static_cast<float>((q[l + 16] >> shift) & 3) - ml;
            }
            shift += 2;
        }
        q += 32;
    }
}

void quantize_q2_k(const float* x, std::uint8_t* b) noexcept {
    std::uint8_t L[kQKK];
    std::uint8_t Laux[16];
    float weights[16];
    float mins[kQKK / 16];
    float scales[kQKK / 16];
    constexpr float kQ4Scale = 15.f;
    std::uint8_t* y_scales = b;
    std::uint8_t* qs = b + 16;
    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < kQKK / 16; ++j) {
        for (int l = 0; l < 16; ++l) {
            weights[l] = std::fabs(x[16 * j + l]);
        }
        scales[j] = make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
        max_scale = std::max(max_scale, scales[j]);
        max_min = std::max(max_min, mins[j]);
    }
    if (max_scale > 0) {
        const float iscale = kQ4Scale / max_scale;
        for (int j = 0; j < kQKK / 16; ++j) {
            y_scales[j] = static_cast<std::uint8_t>(nearest_int(iscale * scales[j]));
        }
        store_f16(b + 80, max_scale / kQ4Scale);
    } else {
        std::fill(y_scales, y_scales + 16, std::uint8_t{0});
        store_f16(b + 80, 0.0f);
    }
    if (max_min > 0) {
        const float iscale = kQ4Scale / max_min;
        for (int j = 0; j < kQKK / 16; ++j) {
            y_scales[j] = static_cast<std::uint8_t>(y_scales[j] | (nearest_int(iscale * mins[j]) << 4));
        }
        store_f16(b + 82, max_min / kQ4Scale);
    } else {
        store_f16(b + 82, 0.0f);
    }
    const float d_all = load_f16(b + 80);
    const float dmin_all = load_f16(b + 82);
    for (int j = 0; j < kQKK / 16; ++j) {
        const float d = d_all * static_cast<float>(y_scales[j] & 0x0F);
        if (d == 0.0f) {
            continue;
        }
        const float dm = dmin_all * static_cast<float>(y_scales[j] >> 4);
        for (int ii = 0; ii < 16; ++ii) {
            const int l = nearest_int((x[16 * j + ii] + dm) / d);
            L[16 * j + ii] = static_cast<std::uint8_t>(std::max(0, std::min(3, l)));
        }
    }
    for (int j = 0; j < kQKK; j += 128) {
        for (int l = 0; l < 32; ++l) {
            qs[j / 4 + l] = static_cast<std::uint8_t>(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) |
                                                      (L[j + l + 96] << 6));
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// q3_k: hmask[32] qs[64] scales[12] d

void q3_k_scales(const std::uint8_t* packed, std::int8_t* scales) noexcept {
    constexpr std::uint32_t kMask1 = 0x03030303;
    constexpr std::uint32_t kMask2 = 0x0f0f0f0f;
    std::uint32_t aux[4];
    std::memcpy(aux, packed, 12);
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kMask2) | (((tmp >> 4) & kMask1) << 4);
    aux[3] = ((aux[1] >> 4) & kMask2) | (((tmp >> 6) & kMask1) << 4);
    aux[0] = (aux[0] & kMask2) | (((tmp >> 0) & kMask1) << 4);
    aux[1] = (aux[1] & kMask2) | (((tmp >> 2) & kMask1) << 4);
    std::memcpy(scales, aux, 16);
}

void dequantize_q3_k(const std::uint8_t* b, float* y) noexcept {
    const std::uint8_t* hm = b;
    const std::uint8_t* q = b + 32;
    std::int8_t scales[16];
    q3_k_scales(b + 96, scales);
    const float d_all = load_f16(b + 108);
    std::uint8_t m = 1;
    int is = 0;
    for (int n = 0; n < kQKK; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * static_cast<float>(scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * static_cast<float>(((q[l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4));
            }
            dl = d_all * static_cast<float>(scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                *y++ = dl * static_cast<float>(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4));
            }
            shift += 2;
            m = static_cast<std::uint8_t>(m << 1);
        }
        q += 32;
    }
}

void quantize_q3_k(const float* x, std::uint8_t* b) noexcept {
    std::int8_t L[kQKK];
    float scales[kQKK / 16];
    std::uint8_t* hmask = b;
    std::uint8_t* qs = b + 32;
    std::uint8_t* y_scales = b + 96;
    float max_scale = 0;
    float amax = 0;
    for (int j = 0; j < kQKK / 16; ++j) {
        scales[j] = make_q3_quants(16, 4, x + 16 * j, L + 16 * j);
        const float scale = std::fabs(scales[j]);
        if (scale > amax) {
            amax = scale;
            max_scale = scales[j];
        }
    }
    std::fill(y_scales, y_scales + 12, std::uint8_t{0});
    if (max_scale != 0.0f) {
        const float iscale = -32.f / max_scale;
        for (int j = 0; j < kQKK / 16; ++j) {
            int l = nearest_int(iscale * scales[j]);
            l = std::max(-32, std::min(31, l)) + 32;
            if (j < 8) {
                y_scales[j] = static_cast<std::uint8_t>(l & 0x0F);
            } else {
                y_scales[j - 8] = static_cast<std::uint8_t>(y_scales[j - 8] | ((l & 0x0F) << 4));
            }
            l >>= 4;
            y_scales[j % 4 + 8] = static_cast<std::uint8_t>(y_scales[j % 4 + 8] | (l << (2 * (j / 4))));
        }
        store_f16(b + 108, 1 / iscale);
    } else {
        store_f16(b + 108, 0.0f);
    }
    const float d_all = load_f16(b + 108);
    for (int j = 0; j < kQKK / 16; ++j) {
        int sc = j < 8 ? y_scales[j] & 0x0F : y_scales[j - 8] >> 4;
        sc = (sc | (((y_scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
        const float d = d_all * static_cast<float>(sc);
        if (d == 0.0f) {
            continue;
        }
        for (int ii = 0; ii < 16; ++ii) {
            const int l = nearest_int(x[16 * j + ii] / d);
            L[16 * j + ii] = static_cast<std::int8_t>(std::max(-4, std::min(3, l)) + 4);
        }
    }
    std::fill(hmask, hmask + kQKK / 8, std::uint8_t{0});
    int m = 0;
    std::uint8_t hm = 1;
    for (int j = 0; j < kQKK; ++j) {
        if (L[j] > 3) {
            hmask[m] = static_cast<std::uint8_t>(hmask[m] | hm);
            L[j] = static_cast<std::int8_t>(L[j] - 4);
        }
        if (++m == kQKK / 8) {
            m = 0;
            hm = static_cast<std::uint8_t>(hm << 1);
        }
    }
    for (int j = 0; j < kQKK; j += 128) {
        for (int l = 0; l < 32; ++l) {
            qs[j / 4 + l] = static_cast<std::uint8_t>(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) |
                                                      (L[j + l + 96] << 6));
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// q4_k: d dmin scales[12] qs[128];  q5_k: d dmin scales[12] qh[32] qs[128]

void dequantize_q4_k(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f16(b);
    const float min = load_f16(b + 2);
    const std::uint8_t* scales = b + 4;
    const std::uint8_t* q = b + 16;
    int is = 0;
    std::uint8_t sc = 0;
    std::uint8_t m = 0;
    for (int j = 0; j < kQKK; j += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>(q[l] & 0x0F) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
        }
        q += 32;
        is += 2;
    }
}

void dequantize_q5_k(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f16(b);
    const float min = load_f16(b + 2);
    const std::uint8_t* scales = b + 4;
    const std::uint8_t* qh = b + 16;
    const std::uint8_t* ql = b + 48;
    int is = 0;
    std::uint8_t sc = 0;
    std::uint8_t m = 0;
    std::uint8_t u1 = 1;
    std::uint8_t u2 = 2;
    for (int j = 0; j < kQKK; j += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>((ql[l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
        }
        ql += 32;
        is += 2;
        u1 = static_cast<std::uint8_t>(u1 << 2);
        u2 = static_cast<std::uint8_t>(u2 << 2);
    }
}

/// Shared part of the q4_k and q5_k quantizers: sub-block scales and minimums, packed scales, and the levels `L`.
void quantize_k4_scales(const float* x, std::uint8_t* b, int nmax, float rmin, int nstep, std::uint8_t* L) noexcept {
    std::uint8_t Laux[32];
    float weights[32];
    float mins[kQKK / 32];
    float scales[kQKK / 32];
    std::uint8_t* y_scales = b + 4;
    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < kQKK / 32; ++j) {
        float sum_x2 = 0;
        for (int l = 0; l < 32; ++l) {
            sum_x2 += x[32 * j + l] * x[32 * j + l];
        }
        const float av_x = std::sqrt(sum_x2 / 32);
        for (int l = 0; l < 32; ++l) {
            weights[l] = av_x + std::fabs(x[32 * j + l]);
        }
        scales[j] =
            make_qkx2_quants(32, nmax, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, rmin, 0.1f, nstep, false);
        max_scale = std::max(max_scale, scales[j]);
        max_min = std::max(max_min, mins[j]);
    }
    const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    std::fill(y_scales, y_scales + 12, std::uint8_t{0});
    for (int j = 0; j < kQKK / 32; ++j) {
        const auto ls = static_cast<std::uint8_t>(std::min(63, nearest_int(inv_scale * scales[j])));
        const auto lm = static_cast<std::uint8_t>(std::min(63, nearest_int(inv_min * mins[j])));
        if (j < 4) {
            y_scales[j] = ls;
            y_scales[j + 4] = lm;
        } else {
            y_scales[j + 4] = static_cast<std::uint8_t>((ls & 0x0F) | ((lm & 0x0F) << 4));
            y_scales[j - 4] = static_cast<std::uint8_t>(y_scales[j - 4] | ((ls >> 4) << 6));
            y_scales[j - 0] = static_cast<std::uint8_t>(y_scales[j - 0] | ((lm >> 4) << 6));
        }
    }
    store_f16(b, max_scale / 63.f);
    store_f16(b + 2, max_min / 63.f);
    const float d_all = load_f16(b);
    const float dmin_all = load_f16(b + 2);
    std::uint8_t sc = 0;
    std::uint8_t m = 0;
    for (int j = 0; j < kQKK / 32; ++j) {
        get_scale_min_k4(j, y_scales, &sc, &m);
        const float d = d_all * sc;
        if (d == 0.0f) {
            continue;
        }
        const float dm = dmin_all * m;
        for (int ii = 0; ii < 32; ++ii) {
            const int l = nearest_int((x[32 * j + ii] + dm) / d);
            L[32 * j + ii] = static_cast<std::uint8_t>(std::max(0, std::min(nmax, l)));
        }
    }
}

void quantize_q4_k(const float* x, std::uint8_t* b) noexcept {
    std::uint8_t L[kQKK];
    quantize_k4_scales(x, b, 15, -1.f, 20, L);
    std::uint8_t* q = b + 16;
    for (int j = 0; j < kQKK; j += 64) {
        for (int l = 0; l < 32; ++l) {
            q[l] = static_cast<std::uint8_t>(L[j + l] | (L[j + l + 32] << 4));
        }
        q += 32;
    }
}

void quantize_q5_k(const float* x, std::uint8_t* b) noexcept {
    std::uint8_t L[kQKK];
    quantize_k4_scales(x, b, 31, -0.5f, 15, L);
    std::uint8_t* qh = b + 16;
    std::uint8_t* ql = b + 48;
    std::fill(qh, qh + kQKK / 8, std::uint8_t{0});
    std::uint8_t m1 = 1;
    std::uint8_t m2 = 2;
    for (int n = 0; n < kQKK; n += 64) {
        for (int j = 0; j < 32; ++j) {
            int l1 = L[n + j];
            if (l1 > 15) {
                l1 -= 16;
                qh[j] = static_cast<std::uint8_t>(qh[j] | m1);
            }
            int l2 = L[n + j + 32];
            if (l2 > 15) {
                l2 -= 16;
                qh[j] = static_cast<std::uint8_t>(qh[j] | m2);
            }
            ql[j] = static_cast<std::uint8_t>(l1 | (l2 << 4));
        }
        m1 = static_cast<std::uint8_t>(m1 << 2);
        m2 = static_cast<std::uint8_t>(m2 << 2);
        ql += 32;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// q6_k: ql[128] qh[64] scales[16] (int8) d

void dequantize_q6_k(const std::uint8_t* b, float* y) noexcept {
    const std::uint8_t* ql = b;
    const std::uint8_t* qh = b + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(b + 192);
    const float d = load_f16(b + 208);
    for (int n = 0; n < kQKK; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = ((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

void quantize_q6_k(const float* x, std::uint8_t* b) noexcept {
    std::int8_t L[kQKK];
    float scales[kQKK / 16];
    float max_scale = 0;
    float max_abs_scale = 0;
    for (int ib = 0; ib < kQKK / 16; ++ib) {
        const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib);
        scales[ib] = scale;
        const float abs_scale = std::fabs(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }
    if (max_abs_scale < kGroupMaxEps) {
        std::fill(b, b + 210, std::uint8_t{0});
        return;
    }
    std::uint8_t* ql = b;
    std::uint8_t* qh = b + 128;
    auto* y_scales = reinterpret_cast<std::int8_t*>(b + 192);
    const float iscale = -128.f / max_scale;
    store_f16(b + 208, 1 / iscale);
    for (int ib = 0; ib < kQKK / 16; ++ib) {
        y_scales[ib] = static_cast<std::int8_t>(std::min(127, nearest_int(iscale * scales[ib])));
    }
    const float d_all = load_f16(b + 208);
    for (int j = 0; j < kQKK / 16; ++j) {
        const float d = d_all * static_cast<float>(y_scales[j]);
        if (d == 0.0f) {
            continue;
        }
        for (int ii = 0; ii < 16; ++ii) {
            const int l = nearest_int(x[16 * j + ii] / d);
            L[16 * j + ii] = static_cast<std::int8_t>(std::max(-32, std::min(31, l)) + 32);
        }
    }
    for (int j = 0; j < kQKK; j += 128) {
        for (int l = 0; l < 32; ++l) {
            const int q1 = L[j + l + 0] & 0x0F;
            const int q2 = L[j + l + 32] & 0x0F;
            const int q3 = L[j + l + 64] & 0x0F;
            const int q4 = L[j + l + 96] & 0x0F;
            ql[l + 0] = static_cast<std::uint8_t>(q1 | (q3 << 4));
            ql[l + 32] = static_cast<std::uint8_t>(q2 | (q4 << 4));
            qh[l] = static_cast<std::uint8_t>((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                                              ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
        }
        ql += 64;
        qh += 32;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// q8_k: d (f32) qs[256] bsums[16] (int16)

void dequantize_q8_k(const std::uint8_t* b, float* y) noexcept {
    const float d = load_f32(b);
    for (int j = 0; j < kQKK; ++j) {
        y[j] = d * static_cast<float>(static_cast<std::int8_t>(b[4 + j]));
    }
}

void quantize_q8_k(const float* x, std::uint8_t* b) noexcept {
    float max = 0;
    float amax = 0;
    for (int j = 0; j < kQKK; ++j) {
        const float ax = std::fabs(x[j]);
        if (ax > amax) {
            amax = ax;
            max = x[j];
        }
    }
    if (amax == 0.0f) {
        std::fill(b, b + 292, std::uint8_t{0});
        return;
    }
    const float iscale = -127.f / max;
    for (int j = 0; j < kQKK; ++j) {
        const int v = nearest_int(iscale * x[j]);
        b[4 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::min(127, v)));
    }
    for (int j = 0; j < kQKK / 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) {
            sum += static_cast<std::int8_t>(b[4 + j * 16 + ii]);
        }
        const auto bsum = static_cast<std::int16_t>(sum);
        std::memcpy(b + 260 + 2 * j, &bsum, 2);
    }
    const float d = 1 / iscale;
    std::memcpy(b, &d, 4);
}

using DequantizeBlock = void (*)(const std::uint8_t*, float*) noexcept;
using QuantizeBlock = void (*)(const float*, std::uint8_t*) noexcept;

struct Codec {
    DequantizeBlock dequantize = nullptr;
    QuantizeBlock quantize = nullptr;
};

Codec codec(DType dtype) noexcept {
    switch (dtype) {
    case DType::kQ4_0:
        return {dequantize_q4_0, quantize_q4_0};
    case DType::kQ4_1:
        return {dequantize_q4_1, quantize_q4_1};
    case DType::kQ5_0:
        return {dequantize_q5_0, quantize_q5_0};
    case DType::kQ5_1:
        return {dequantize_q5_1, quantize_q5_1};
    case DType::kQ8_0:
        return {dequantize_q8_0, quantize_q8_0};
    case DType::kQ8_1:
        return {dequantize_q8_1, quantize_q8_1};
    case DType::kQ2_K:
        return {dequantize_q2_k, quantize_q2_k};
    case DType::kQ3_K:
        return {dequantize_q3_k, quantize_q3_k};
    case DType::kQ4_K:
        return {dequantize_q4_k, quantize_q4_k};
    case DType::kQ5_K:
        return {dequantize_q5_k, quantize_q5_k};
    case DType::kQ6_K:
        return {dequantize_q6_k, quantize_q6_k};
    case DType::kQ8_K:
        return {dequantize_q8_k, quantize_q8_k};
    default:
        return {};
    }
}

}  // namespace

bool can_dequantize(DType dtype) noexcept {
    return codec(dtype).dequantize != nullptr;
}

bool can_quantize(DType dtype) noexcept {
    return codec(dtype).quantize != nullptr;
}

void dequantize(DType dtype, const void* blocks, float* out, std::int64_t count) noexcept {
    const Codec c = codec(dtype);
    const DTypeInfo& info = dtype_info(dtype);
    const auto* in = static_cast<const std::uint8_t*>(blocks);
    for (std::int64_t i = 0; i < count / info.block_size; ++i) {
        c.dequantize(in + i * info.block_bytes, out + i * info.block_size);
    }
}

void quantize(DType dtype, const float* in, void* blocks, std::int64_t count) noexcept {
    const Codec c = codec(dtype);
    const DTypeInfo& info = dtype_info(dtype);
    auto* out = static_cast<std::uint8_t*>(blocks);
    for (std::int64_t i = 0; i < count / info.block_size; ++i) {
        c.quantize(in + i * info.block_size, out + i * info.block_bytes);
    }
}

Status check_row(DType dtype, std::int64_t count) {
    const DTypeInfo& info = dtype_info(dtype);
    if (count % info.block_size != 0) {
        return errors::invalid_argument("rows of {} tensors must be a multiple of {} elements, got {}", info.name,
                                        info.block_size, count);
    }
    return ok_status();
}

}  // namespace ii::quant
