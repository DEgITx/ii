// SPDX-License-Identifier: MIT
/// @file
/// Vectorized elementary functions of ii-cpu, written once on top of simd.h for every ISA.
///
/// Accuracy (float32, measured against double-precision std:: functions by tests/unit/backends/cpu_backend_test.cpp):
///   vexp       <= 2 ulp; results below 2^-126 (x < -87.33) flush to +0, x > 88.72 gives +inf
///   vlog       <= 2 ulp for normal and subnormal inputs; log(0) = -inf, log(x < 0) = NaN
///   vtanh      <= 3 ulp
///   vsigmoid   <= 4 ulp
///   verf       <= 8 ulp (absolute error < 5e-7); |x| >= 4 gives exactly +-1
///   vgelu      <= 8 ulp plus an absolute error below 1.5e-6 (1 + erf(x) cancels for x << 0)
///   vsilu      <= 4 ulp
/// NaN inputs produce NaN. The algorithms follow the Cephes library (range reduction + minimax polynomials).
#pragma once

#include "backends/cpu/isa/simd.h"

namespace ii::cpu::II_ISA_NS {

/// e^x.
II_SIMD_INLINE VF vexp(VF x) {
    const VF log2e = vset1(1.44269504088896341f);
    const VF ln2_hi = vset1(0.693359375f);
    const VF ln2_lo = vset1(-2.12194440e-4f);
    const VF hi_limit = vset1(88.7228391116729996f);
    const VF lo_limit = vset1(-87.3365447505531f);

    const VF clamped = vmin(vmax(x, vset1(-88.0f)), vset1(88.5f));
    const VF n = vround(vmul(clamped, log2e));
    VF r = vsub(clamped, vmul(n, ln2_hi));
    r = vsub(r, vmul(n, ln2_lo));

    VF p = vset1(1.9875691500e-4f);
    p = vfma(p, r, vset1(1.3981999507e-3f));
    p = vfma(p, r, vset1(8.3334519073e-3f));
    p = vfma(p, r, vset1(4.1665795894e-2f));
    p = vfma(p, r, vset1(1.6666665459e-1f));
    p = vfma(p, r, vset1(5.0000001201e-1f));
    p = vfma(p, vmul(r, r), vadd(r, vset1(1.0f)));

    // 2^n built from the exponent bits; n is in [-127, 128): split 2^n = 2^(n/2) * 2^(n - n/2) to stay in range.
    const VI ni = vcvt_round(n);
    const VI n1 = vsrai<1>(ni);
    const VI n2 = vsubi(ni, n1);
    const VF scale1 = vfloat_bits(vshli<23>(vaddi(n1, vseti(127))));
    const VF scale2 = vfloat_bits(vshli<23>(vaddi(n2, vseti(127))));
    VF result = vmul(vmul(p, scale1), scale2);

    result = vselect(vgt(x, hi_limit), vfloat_bits(vseti(0x7f800000)), result);
    result = vselect(vlt(x, lo_limit), vzero(), result);
    return vselect(visnan(x), x, result);
}

/// Natural logarithm.
II_SIMD_INLINE VF vlog(VF x) {
    const VF zero = vzero();
    const VF one = vset1(1.0f);
    // Subnormals: scale by 2^23 and correct the exponent afterwards.
    const VM subnormal = vmand(vlt(x, vset1(1.17549435e-38f)), vgt(x, zero));
    const VF scaled = vselect(subnormal, vmul(x, vset1(8388608.0f)), x);
    const VF exponent_bias = vselect(subnormal, vset1(-23.0f), zero);

    const VI bits = vbits(scaled);
    // m in [0.5, 1), e = unbiased exponent + 1.
    VF e = vadd(vcvt_int(vsubi(vsrli<23>(vandi(bits, vseti(0x7f800000))), vseti(126))), exponent_bias);
    VF m = vfloat_bits(vori(vandi(bits, vseti(0x007fffff)), vseti(0x3f000000)));

    const VM small = vlt(m, vset1(0.707106781186547524f));
    e = vsub(e, vselect(small, one, zero));
    m = vsub(vadd(m, vselect(small, m, zero)), one);

    const VF z = vmul(m, m);
    VF y = vset1(7.0376836292e-2f);
    y = vfma(y, m, vset1(-1.1514610310e-1f));
    y = vfma(y, m, vset1(1.1676998740e-1f));
    y = vfma(y, m, vset1(-1.2420140846e-1f));
    y = vfma(y, m, vset1(1.4249322787e-1f));
    y = vfma(y, m, vset1(-1.6668057665e-1f));
    y = vfma(y, m, vset1(2.0000714765e-1f));
    y = vfma(y, m, vset1(-2.4999993993e-1f));
    y = vfma(y, m, vset1(3.3333331174e-1f));
    y = vmul(vmul(y, m), z);
    y = vfma(e, vset1(-2.12194440e-4f), y);
    y = vfma(z, vset1(-0.5f), y);
    VF result = vadd(m, y);
    result = vfma(e, vset1(0.693359375f), result);

    const VF inf = vfloat_bits(vseti(0x7f800000));
    result = vselect(veq(x, inf), inf, result);
    result = vselect(veq(x, zero), vneg(inf), result);
    const VM invalid = vmor(vlt(x, zero), visnan(x));
    return vselect(invalid, vfloat_bits(vseti(0x7fc00000)), result);
}

/// Hyperbolic tangent.
II_SIMD_INLINE VF vtanh(VF x) {
    const VF ax = vabs(x);
    // Small |x|: odd polynomial.
    const VF z = vmul(x, x);
    VF p = vset1(-5.70498872745e-3f);
    p = vfma(p, z, vset1(2.06390887954e-2f));
    p = vfma(p, z, vset1(-5.37397155531e-2f));
    p = vfma(p, z, vset1(1.33314422036e-1f));
    p = vfma(p, z, vset1(-3.33332819422e-1f));
    const VF small = vfma(vmul(x, z), p, x);
    // Larger |x|: 1 - 2 / (e^(2|x|) + 1), with the sign of x.
    const VF e = vexp(vadd(ax, ax));
    const VF large = vcopysign(vsub(vset1(1.0f), vdiv(vset1(2.0f), vadd(e, vset1(1.0f)))), x);
    const VF result = vselect(vlt(ax, vset1(0.625f)), small, large);
    return vselect(visnan(x), x, result);
}

/// 1 / (1 + e^-x).
II_SIMD_INLINE VF vsigmoid(VF x) {
    return vdiv(vset1(1.0f), vadd(vset1(1.0f), vexp(vneg(x))));
}

/// Error function: a rational minimax approximation on [-4, 4] (as in Eigen / XLA).
II_SIMD_INLINE VF verf(VF x) {
    const VF c = vmin(vmax(x, vset1(-4.0f)), vset1(4.0f));
    const VF x2 = vmul(c, c);
    VF p = vset1(-2.72614225801306e-10f);
    p = vfma(p, x2, vset1(2.77068142495902e-08f));
    p = vfma(p, x2, vset1(-2.10102402082508e-06f));
    p = vfma(p, x2, vset1(-5.69250639462346e-05f));
    p = vfma(p, x2, vset1(-7.34990630326855e-04f));
    p = vfma(p, x2, vset1(-2.95459980854025e-03f));
    p = vfma(p, x2, vset1(-1.60960333262415e-02f));
    p = vmul(p, c);
    VF q = vset1(-1.45660718464996e-05f);
    q = vfma(q, x2, vset1(-2.13374055278905e-04f));
    q = vfma(q, x2, vset1(-1.68282697438203e-03f));
    q = vfma(q, x2, vset1(-7.37332916720468e-03f));
    q = vfma(q, x2, vset1(-1.42647390514189e-02f));
    VF result = vdiv(p, q);
    result = vselect(vge(vabs(x), vset1(4.0f)), vcopysign(vset1(1.0f), x), result);
    return vselect(visnan(x), x, result);
}

/// GELU, exact form: 0.5 * x * (1 + erf(x / sqrt(2))).
II_SIMD_INLINE VF vgelu(VF x) {
    const VF half_x = vmul(x, vset1(0.5f));
    return vfma(half_x, verf(vmul(x, vset1(0.70710678118654752f))), half_x);
}

/// GELU, tanh approximation: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
II_SIMD_INLINE VF vgelu_tanh(VF x) {
    const VF x3 = vmul(vmul(x, x), x);
    const VF inner = vmul(vset1(0.797884560802865356f), vfma(x3, vset1(0.044715f), x));
    const VF half_x = vmul(x, vset1(0.5f));
    return vfma(half_x, vtanh(inner), half_x);
}

/// x * sigmoid(x).
II_SIMD_INLINE VF vsilu(VF x) {
    return vdiv(x, vadd(vset1(1.0f), vexp(vneg(x))));
}

}  // namespace ii::cpu::II_ISA_NS
