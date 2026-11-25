// SPDX-License-Identifier: MIT
// Row-wise kernels: reductions, softmax, normalizations, attention rows (compiled once per ISA variant).
#include "backends/cpu/isa/isa_kernels.h"
#include "backends/cpu/isa/vmath.h"

#include <limits>

namespace ii::cpu::II_ISA_NS {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();  // evaluated at compile time: no shared code

II_SIMD_INLINE float scalar_nan() {
    float result;
    const std::uint32_t bits = 0x7fc00000u;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}

II_SIMD_INLINE double row_sum(const float* x, std::int64_t n) {
    VD acc0 = vd_zero();
    VD acc1 = vd_zero();
    std::int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        const VF v = vload(x + i);
        acc0 = vd_add(acc0, vd_lo(v));
        acc1 = vd_add(acc1, vd_hi(v));
    }
    double sum = vd_reduce_add(vd_add(acc0, acc1));
    for (; i < n; ++i) {
        sum += x[i];
    }
    return sum;
}

/// Sum of (x - mean)^2 (for the variance) or of x^2 (mean = 0).
II_SIMD_INLINE double row_sum_squares(const float* x, std::int64_t n, double mean) {
    const VF m = vset1(static_cast<float>(mean));
    const double mean_error = mean - static_cast<double>(static_cast<float>(mean));
    VD acc0 = vd_zero();
    VD acc1 = vd_zero();
    std::int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        const VF v = vsub(vload(x + i), m);
        const VD lo = vd_lo(v);
        const VD hi = vd_hi(v);
        acc0 = vd_fma(lo, lo, acc0);
        acc1 = vd_fma(hi, hi, acc1);
    }
    double sum = vd_reduce_add(vd_add(acc0, acc1));
    for (; i < n; ++i) {
        const double d = static_cast<double>(x[i]) - static_cast<double>(static_cast<float>(mean));
        sum += d * d;
    }
    // Correct for using the float-rounded mean: sum((x - m')^2) = sum((x - m)^2) + n * (m - m')^2.
    return sum - static_cast<double>(n) * mean_error * mean_error;
}

II_SIMD_INLINE double sqrt_double(double x) {
    // Newton refinement of the float square root: two steps reach double precision.
    if (!(x > 0)) {
        return x == 0 ? 0.0 : scalar_nan();
    }
    float approx_f;
    vstore_partial(&approx_f, vsqrt(vset1(static_cast<float>(x))), 1);
    double r = approx_f;
    if (!(r > 0) || r > 1e300) {
        return r;
    }
    r = 0.5 * (r + x / r);
    r = 0.5 * (r + x / r);
    return r;
}

void reduce_rows(ReduceKind kind, const float* x, std::int64_t rows, std::int64_t n, float* out) {
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* p = x + row * n;
        switch (kind) {
        case ReduceKind::kSum:
            out[row] = static_cast<float>(row_sum(p, n));
            break;
        case ReduceKind::kMean:
            out[row] = static_cast<float>(row_sum(p, n) / static_cast<double>(n));
            break;
        case ReduceKind::kProd: {
            double prod = 1.0;
            for (std::int64_t i = 0; i < n; ++i) {
                prod *= p[i];
            }
            out[row] = static_cast<float>(prod);
            break;
        }
        case ReduceKind::kMax:
        case ReduceKind::kMin: {
            const bool is_max = kind == ReduceKind::kMax;
            VF acc = vset1(is_max ? -kInf : kInf);
            VM nan = visnan(vzero());
            std::int64_t i = 0;
            for (; i + kLanes <= n; i += kLanes) {
                const VF v = vload(p + i);
                nan = vmor(nan, visnan(v));
                acc = is_max ? vmax(acc, v) : vmin(acc, v);
            }
            float result = is_max ? vreduce_max(acc) : vreduce_min(acc);
            bool has_nan = vreduce_max(vmask_to_float(nan)) != 0.0f;
            for (; i < n; ++i) {
                const float v = p[i];
                has_nan = has_nan || v != v;
                result = is_max ? (v > result ? v : result) : (v < result ? v : result);
            }
            out[row] = has_nan ? scalar_nan() : result;
            break;
        }
        }
    }
}

/// Reduction over the middle dimension of [outer, r, inner] with inner > 1: blocks of columns accumulated on the
/// stack (the inner loops are plain C++ that the compiler vectorizes for this ISA).
void reduce_columns(ReduceKind kind, const float* x, std::int64_t outer, std::int64_t r, std::int64_t inner,
                    float* out) {
    constexpr std::int64_t kBlock = 64;
    for (std::int64_t o = 0; o < outer; ++o) {
        for (std::int64_t i0 = 0; i0 < inner; i0 += kBlock) {
            const std::int64_t count = inner - i0 < kBlock ? inner - i0 : kBlock;
            const float* base = x + o * r * inner + i0;
            float* dst = out + o * inner + i0;
            if (kind == ReduceKind::kMax || kind == ReduceKind::kMin) {
                const bool is_max = kind == ReduceKind::kMax;
                float acc[kBlock];
                bool nan[kBlock];
                for (std::int64_t i = 0; i < count; ++i) {
                    acc[i] = is_max ? -kInf : kInf;
                    nan[i] = false;
                }
                for (std::int64_t j = 0; j < r; ++j) {
                    const float* row = base + j * inner;
                    for (std::int64_t i = 0; i < count; ++i) {
                        const float v = row[i];
                        nan[i] = nan[i] || v != v;
                        acc[i] = is_max ? (v > acc[i] ? v : acc[i]) : (v < acc[i] ? v : acc[i]);
                    }
                }
                for (std::int64_t i = 0; i < count; ++i) {
                    dst[i] = nan[i] ? scalar_nan() : acc[i];
                }
                continue;
            }
            double acc[kBlock];
            const bool prod = kind == ReduceKind::kProd;
            for (std::int64_t i = 0; i < count; ++i) {
                acc[i] = prod ? 1.0 : 0.0;
            }
            for (std::int64_t j = 0; j < r; ++j) {
                const float* row = base + j * inner;
                if (prod) {
                    for (std::int64_t i = 0; i < count; ++i) {
                        acc[i] *= row[i];
                    }
                } else {
                    for (std::int64_t i = 0; i < count; ++i) {
                        acc[i] += row[i];
                    }
                }
            }
            const double divisor = kind == ReduceKind::kMean ? static_cast<double>(r) : 1.0;
            for (std::int64_t i = 0; i < count; ++i) {
                dst[i] = static_cast<float>(acc[i] / divisor);
            }
        }
    }
}

void softmax_rows(const float* x, float* y, std::int64_t rows, std::int64_t n, bool log) {
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* p = x + row * n;
        float* q = y + row * n;
        // Maximum (NaN anywhere makes the whole row NaN through x - max below).
        VF vmax_acc = vset1(-kInf);
        VM nan = visnan(vzero());
        std::int64_t i = 0;
        for (; i + kLanes <= n; i += kLanes) {
            const VF v = vload(p + i);
            nan = vmor(nan, visnan(v));
            vmax_acc = vmax(vmax_acc, v);
        }
        float max = vreduce_max(vmax_acc);
        bool has_nan = vreduce_max(vmask_to_float(nan)) != 0.0f;
        for (; i < n; ++i) {
            has_nan = has_nan || p[i] != p[i];
            max = p[i] > max ? p[i] : max;
        }
        if (has_nan) {
            max = scalar_nan();
        }
        const VF m = vset1(max);
        VF vsum = vzero();
        i = 0;
        for (; i + kLanes <= n; i += kLanes) {
            const VF e = vexp(vsub(vload(p + i), m));
            vsum = vadd(vsum, e);
            if (!log) {
                vstore(q + i, e);
            }
        }
        float sum = vreduce_add(vsum);
        if (i < n) {
            const VF e = vexp(vsub(vload_partial(p + i, n - i), m));
            float tail[kLanes];
            vstore(tail, e);
            for (std::int64_t j = 0; j < n - i; ++j) {
                sum += tail[j];
                if (!log) {
                    q[i + j] = tail[j];
                }
            }
        }
        if (log) {
            float log_sum;
            vstore_partial(&log_sum, vlog(vset1(sum)), 1);
            const VF shift = vadd(m, vset1(log_sum));
            i = 0;
            for (; i + kLanes <= n; i += kLanes) {
                vstore(q + i, vsub(vload(p + i), shift));
            }
            if (i < n) {
                vstore_partial(q + i, vsub(vload_partial(p + i, n - i), shift), n - i);
            }
        } else {
            const VF inv = vset1(1.0f / sum);
            i = 0;
            for (; i + kLanes <= n; i += kLanes) {
                vstore(q + i, vmul(vload(q + i), inv));
            }
            for (; i < n; ++i) {
                q[i] *= 1.0f / sum;
            }
        }
    }
}

/// Softmax along the middle dimension of [outer, n, inner] with inner > 1, vectorized across `inner`.
void softmax_columns(const float* x, float* y, std::int64_t outer, std::int64_t n, std::int64_t inner, bool log) {
    for (std::int64_t o = 0; o < outer; ++o) {
        const float* px = x + o * n * inner;
        float* py = y + o * n * inner;
        for (std::int64_t i0 = 0; i0 < inner; i0 += kLanes) {
            const std::int64_t count = inner - i0 < kLanes ? inner - i0 : kLanes;
            auto load = [&](const float* p) { return count == kLanes ? vload(p) : vload_partial(p, count); };
            auto store = [&](float* p, VF v) {
                if (count == kLanes) {
                    vstore(p, v);
                } else {
                    vstore_partial(p, v, count);
                }
            };
            VF m = vset1(-kInf);
            VM nan = visnan(vzero());
            for (std::int64_t j = 0; j < n; ++j) {
                const VF v = load(px + j * inner + i0);
                nan = vmor(nan, visnan(v));
                m = vmax(m, v);
            }
            m = vselect(nan, vfloat_bits(vseti(0x7fc00000)), m);
            VF sum = vzero();
            for (std::int64_t j = 0; j < n; ++j) {
                const VF e = vexp(vsub(load(px + j * inner + i0), m));
                sum = vadd(sum, e);
                if (!log) {
                    store(py + j * inner + i0, e);
                }
            }
            if (log) {
                const VF shift = vadd(m, vlog(sum));
                for (std::int64_t j = 0; j < n; ++j) {
                    store(py + j * inner + i0, vsub(load(px + j * inner + i0), shift));
                }
            } else {
                const VF inv = vdiv(vset1(1.0f), sum);
                for (std::int64_t j = 0; j < n; ++j) {
                    store(py + j * inner + i0, vmul(load(py + j * inner + i0), inv));
                }
            }
        }
    }
}

II_SIMD_INLINE float dot(const float* a, const float* b, std::int64_t n) {
    VF acc0 = vzero();
    VF acc1 = vzero();
    std::int64_t i = 0;
    for (; i + 2 * kLanes <= n; i += 2 * kLanes) {
        acc0 = vfma(vload(a + i), vload(b + i), acc0);
        acc1 = vfma(vload(a + i + kLanes), vload(b + i + kLanes), acc1);
    }
    for (; i + kLanes <= n; i += kLanes) {
        acc0 = vfma(vload(a + i), vload(b + i), acc0);
    }
    if (i < n) {
        acc1 = vfma(vload_partial(a + i, n - i), vload_partial(b + i, n - i), acc1);
    }
    return vreduce_add(vadd(acc0, acc1));
}

II_SIMD_INLINE void axpy(float alpha, const float* x, float* y, std::int64_t n) {
    const VF a = vset1(alpha);
    std::int64_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        vstore(y + i, vfma(a, vload(x + i), vload(y + i)));
    }
    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

}  // namespace

void reduce(ReduceKind kind, const float* x, std::int64_t outer, std::int64_t r, std::int64_t inner, float* out) {
    if (inner == 1) {
        reduce_rows(kind, x, outer, r, out);
    } else {
        reduce_columns(kind, x, outer, r, inner, out);
    }
}

void softmax(const float* x, float* y, std::int64_t outer, std::int64_t n, std::int64_t inner, bool log) {
    if (inner == 1) {
        softmax_rows(x, y, outer, n, log);
    } else {
        softmax_columns(x, y, outer, n, inner, log);
    }
}

void layer_norm(const float* x, const float* scale, const float* bias, float* y, std::int64_t rows, std::int64_t n,
                float epsilon) {
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* p = x + row * n;
        float* q = y + row * n;
        const double mean = row_sum(p, n) / static_cast<double>(n);
        const double variance = row_sum_squares(p, n, mean) / static_cast<double>(n);
        const double inv = 1.0 / sqrt_double(variance + static_cast<double>(epsilon));
        const VF m = vset1(static_cast<float>(mean));
        const VF s = vset1(static_cast<float>(inv));
        std::int64_t i = 0;
        for (; i + kLanes <= n; i += kLanes) {
            VF v = vmul(vsub(vload(p + i), m), s);
            if (scale != nullptr) {
                v = vmul(v, vload(scale + i));
            }
            if (bias != nullptr) {
                v = vadd(v, vload(bias + i));
            }
            vstore(q + i, v);
        }
        for (; i < n; ++i) {
            double v = (static_cast<double>(p[i]) - mean) * inv;
            if (scale != nullptr) {
                v *= scale[i];
            }
            if (bias != nullptr) {
                v += bias[i];
            }
            q[i] = static_cast<float>(v);
        }
    }
}

void rms_norm(const float* x, const float* scale, float* y, std::int64_t rows, std::int64_t n, float epsilon) {
    for (std::int64_t row = 0; row < rows; ++row) {
        const float* p = x + row * n;
        float* q = y + row * n;
        const double mean_square = row_sum_squares(p, n, 0.0) / static_cast<double>(n);
        const double inv = 1.0 / sqrt_double(mean_square + static_cast<double>(epsilon));
        const VF s = vset1(static_cast<float>(inv));
        std::int64_t i = 0;
        for (; i + kLanes <= n; i += kLanes) {
            VF v = vmul(vload(p + i), s);
            if (scale != nullptr) {
                v = vmul(v, vload(scale + i));
            }
            vstore(q + i, v);
        }
        for (; i < n; ++i) {
            double v = static_cast<double>(p[i]) * inv;
            if (scale != nullptr) {
                v *= scale[i];
            }
            q[i] = static_cast<float>(v);
        }
    }
}

void attention_row(const AttentionRowArgs& args) {
    float* scores = args.scratch;
    float max = -kInf;
    bool has_nan = false;
    for (std::int64_t j = 0; j < args.kv_len; ++j) {
        float s;
        if (j >= args.kv_limit) {
            s = -kInf;
        } else {
            s = dot(args.q, args.k + j * args.k_stride, args.d) * args.scale;
            if (args.mask_kind == MaskKind::kBool) {
                if (static_cast<const std::uint8_t*>(args.mask)[j * args.mask_stride] == 0) {
                    s = -kInf;
                }
            } else if (args.mask_kind == MaskKind::kAdditive) {
                s += static_cast<const float*>(args.mask)[j * args.mask_stride];
            }
        }
        has_nan = has_nan || s != s;
        max = s > max ? s : max;
        scores[j] = s;
    }
    if (has_nan) {
        max = scalar_nan();
    }
    // exp(s - max), vectorized.
    const VF m = vset1(max);
    VF vsum = vzero();
    std::int64_t j = 0;
    for (; j + kLanes <= args.kv_len; j += kLanes) {
        const VF e = vexp(vsub(vload(scores + j), m));
        vsum = vadd(vsum, e);
        vstore(scores + j, e);
    }
    float sum = vreduce_add(vsum);
    if (j < args.kv_len) {
        const VF e = vexp(vsub(vload_partial(scores + j, args.kv_len - j), m));
        float tail[kLanes];
        vstore(tail, e);
        for (std::int64_t t = 0; t < args.kv_len - j; ++t) {
            sum += tail[t];
            scores[j + t] = tail[t];
        }
    }
    for (std::int64_t i = 0; i < args.dv; ++i) {
        args.out[i] = 0.0f;
    }
    for (j = 0; j < args.kv_len; ++j) {
        const float p = scores[j];
        if (p != 0.0f) {  // NaN included
            axpy(p, args.v + j * args.v_stride, args.out, args.dv);
        }
    }
    const float inv = 1.0f / sum;
    for (std::int64_t i = 0; i < args.dv; ++i) {
        args.out[i] *= inv;
    }
}

}  // namespace ii::cpu::II_ISA_NS
