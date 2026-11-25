// SPDX-License-Identifier: MIT
// Depthwise convolution of one plane.
#include "backends/cpu/isa/isa_kernels.h"
#include "backends/cpu/isa/simd.h"

#include <cstring>

namespace ii::cpu::II_ISA_NS {

void depthwise(const DepthwiseArgs& a) {
    // The part of the input the window touches is copied into zero-padded buffers, one per column phase of the
    // horizontal stride (phase p holds padded columns p, p + sw, p + 2 sw, ...). Output column ox of tap kx then reads
    // phase (kx * dw) % sw at index ox + (kx * dw) / sw: every tap is a contiguous vector load, for any stride, and
    // the stencil runs without bounds checks.
    const std::int64_t rows = (a.oh - 1) * a.sh + (a.kh - 1) * a.dh + 1;
    const std::int64_t cols = (a.ow - 1) * a.sw + (a.kw - 1) * a.dw + 1;
    const std::int64_t phase_cols = (cols + a.sw - 1) / a.sw;
    const std::int64_t stride = phase_cols + 16;
    const std::int64_t phase_size = rows * stride;
    float* padded = a.scratch;
    for (std::int64_t r = 0; r < rows; ++r) {
        const std::int64_t ih = r - a.ph;
        const bool inside_row = ih >= 0 && ih < a.h;
        const float* src = inside_row ? a.in + ih * a.w : nullptr;
        if (a.sw == 1) {
            float* row = padded + r * stride;
            if (!inside_row) {
                std::memset(row, 0, static_cast<std::size_t>(stride) * sizeof(float));
                continue;
            }
            const std::int64_t lo = a.pw < cols ? a.pw : cols;  // first column inside the input
            std::int64_t hi = a.pw + a.w;
            hi = hi < cols ? hi : cols;
            std::memset(row, 0, static_cast<std::size_t>(lo) * sizeof(float));
            if (hi > lo) {
                std::memcpy(row + lo, src + (lo - a.pw), static_cast<std::size_t>(hi - lo) * sizeof(float));
            }
            const std::int64_t tail = hi > lo ? hi : lo;
            std::memset(row + tail, 0, static_cast<std::size_t>(stride - tail) * sizeof(float));
            continue;
        }
        for (std::int64_t p = 0; p < a.sw; ++p) {
            float* row = padded + p * phase_size + r * stride;
            std::int64_t i = 0;
            for (std::int64_t c = p; c < cols; c += a.sw, ++i) {
                const std::int64_t iw = c - a.pw;
                row[i] = inside_row && iw >= 0 && iw < a.w ? src[iw] : 0.0f;
            }
            for (; i < stride; ++i) {
                row[i] = 0.0f;
            }
        }
    }

    // Filter taps as vectors and their offsets in the padded buffers (relative to the first row of an output row).
    constexpr std::int64_t kMaxTaps = 49;
    const std::int64_t taps = a.kh * a.kw;
    if (taps > kMaxTaps) {
        // Large kernels: scalar fallback.
        for (std::int64_t oy = 0; oy < a.oh; ++oy) {
            for (std::int64_t ox = 0; ox < a.ow; ++ox) {
                float sum = a.bias;
                for (std::int64_t ky = 0; ky < a.kh; ++ky) {
                    for (std::int64_t kx = 0; kx < a.kw; ++kx) {
                        const std::int64_t c0 = kx * a.dw;
                        sum += a.filter[ky * a.kw + kx] *
                               padded[(c0 % a.sw) * phase_size + (oy * a.sh + ky * a.dh) * stride + c0 / a.sw + ox];
                    }
                }
                a.out[oy * a.ow + ox] = sum;
            }
        }
        return;
    }
    VF weights[kMaxTaps];
    std::int64_t offsets[kMaxTaps];
    for (std::int64_t ky = 0; ky < a.kh; ++ky) {
        for (std::int64_t kx = 0; kx < a.kw; ++kx) {
            const std::int64_t t = ky * a.kw + kx;
            const std::int64_t c0 = kx * a.dw;
            weights[t] = vset1(a.filter[t]);
            offsets[t] = (c0 % a.sw) * phase_size + ky * a.dh * stride + c0 / a.sw;
        }
    }
    const VF bias = vset1(a.bias);
    for (std::int64_t oy = 0; oy < a.oh; ++oy) {
        const float* base = padded + oy * a.sh * stride;
        float* out = a.out + oy * a.ow;
        for (std::int64_t ox = 0; ox < a.ow; ox += kLanes) {
            VF acc = bias;
            for (std::int64_t t = 0; t < taps; ++t) {
                acc = vfma(weights[t], vload(base + offsets[t] + ox), acc);
            }
            if (ox + kLanes <= a.ow) {
                vstore(out + ox, acc);
            } else {
                vstore_partial(out + ox, acc, a.ow - ox);
            }
        }
    }
}

}  // namespace ii::cpu::II_ISA_NS
