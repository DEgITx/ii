// SPDX-License-Identifier: MIT
// GEMM packing and register micro-kernels (compiled once per ISA variant).
//
// The micro-kernel keeps an MR x NR block of C in registers (MR rows x NV vectors of kLanes columns) and walks the
// depth: per step it loads NV vectors of a packed B row, broadcasts MR values of a packed A column and issues
// MR * NV fused multiply-adds. B may be f32, f16 or bf16 (widened to f32 when loaded); accumulation is in f32.
#include "backends/cpu/isa/isa_kernels.h"
#include "backends/cpu/isa/simd.h"

#include <utility>  // std::integer_sequence (a type only: no code is emitted from the header)

namespace ii::cpu::II_ISA_NS {

namespace {

// Register blocking per ISA: MR rows x NV vectors (NR = NV * kLanes columns).
#if defined(II_ISA_IS_AVX512)
constexpr int kMR = 14;  // 28 accumulators + 2 B vectors + 1 broadcast of 32 ZMM registers
constexpr int kNV = 2;
#elif defined(II_ISA_IS_AVX2)
constexpr int kMR = 6;  // 12 accumulators + 2 B vectors + 1 broadcast of 16 YMM registers
constexpr int kNV = 2;
#elif defined(II_ISA_IS_NEON)
constexpr int kMR = 8;  // 24 accumulators + 3 B vectors + A values of 32 Q registers
constexpr int kNV = 3;
#else
constexpr int kMR = 4;
constexpr int kNV = 2;
#endif
constexpr int kNR = kNV * kLanes;

template <int kValue>
struct IntC {
    static constexpr int value = kValue;
};

/// Calls f(IntC<0>{}), ..., f(IntC<N-1>{}) — a compile-time unrolled loop with constant indices.
template <int kN, class F>
II_SIMD_INLINE void unroll(F&& f) {
    if constexpr (kN > 0) {
        unroll<kN - 1>(f);
        f(IntC<kN - 1>{});
    }
}

enum BType : int {
    kBF32 = 0,
    kBF16 = 1,
    kBBF16 = 2,
};

template <int kB>
II_SIMD_INLINE VF load_b(const void* base, std::int64_t index) {
    if constexpr (kB == kBF32) {
        return vload(static_cast<const float*>(base) + index);
    } else if constexpr (kB == kBF16) {
        return vload_f16(static_cast<const std::uint16_t*>(base) + index);
    } else {
        return vload_bf16(static_cast<const std::uint16_t*>(base) + index);
    }
}

/// Computes kRows (<= MR) rows of an MR x NR tile; edge tiles with fewer rows (e.g. GEMV, M = 1) do only the work
/// they need.
template <int kB, int kRows>
void micro_kernel(std::int64_t k, const float* a, const void* b, float* c, std::int64_t ldc, std::int64_t n,
                  bool accumulate, const float* bias) {
    VF acc[kRows][kNV];
    unroll<kRows>([&](auto r) { unroll<kNV>([&](auto v) { acc[r.value][v.value] = vzero(); }); });

    // Bytes of packed B consumed per depth step; prefetch a few steps ahead to keep several memory requests in flight
    // (matters for GEMV, which streams B once).
    constexpr std::int64_t kBStep = kNR * (kB == kBF32 ? 4 : 2);
    const auto* b_bytes = static_cast<const std::uint8_t*>(b);
    for (std::int64_t kk = 0; kk < k; ++kk) {
        if constexpr (kRows <= 2) {
            vprefetch(b_bytes + (kk + 16) * kBStep);
            if constexpr (kBStep > 64) {
                vprefetch(b_bytes + (kk + 16) * kBStep + 64);
            }
        }
        VF bv[kNV];
        unroll<kNV>([&](auto v) { bv[v.value] = load_b<kB>(b, kk * kNR + v.value * kLanes); });
        const float* ak = a + kk * kMR;
        unroll<kRows>([&](auto r) {
            const VF av = vset1(ak[r.value]);
            unroll<kNV>([&](auto v) { acc[r.value][v.value] = vfma(av, bv[v.value], acc[r.value][v.value]); });
        });
    }

    if (n == kNR) {
        unroll<kRows>([&](auto r) {
            float* row = c + r.value * ldc;
            unroll<kNV>([&](auto v) {
                VF x = acc[r.value][v.value];
                if (accumulate) {
                    x = vadd(x, vload(row + v.value * kLanes));
                }
                if (bias != nullptr) {
                    x = vadd(x, vload(bias + v.value * kLanes));
                }
                vstore(row + v.value * kLanes, x);
            });
        });
        return;
    }
    // Edge tile: spill the accumulators and copy the valid part.
    alignas(64) float tile[kRows][kNR];
    unroll<kRows>(
        [&](auto r) { unroll<kNV>([&](auto v) { vstore(&tile[r.value][v.value * kLanes], acc[r.value][v.value]); }); });
    for (std::int64_t r = 0; r < kRows; ++r) {
        float* row = c + r * ldc;
        for (std::int64_t j = 0; j < n; ++j) {
            float x = tile[r][j];
            if (accumulate) {
                x += row[j];
            }
            if (bias != nullptr) {
                x += bias[j];
            }
            row[j] = x;
        }
    }
}

using MicroKernelFn = void (*)(std::int64_t, const float*, const void*, float*, std::int64_t, std::int64_t, bool,
                               const float*);

/// micro_kernel<kB, 1..MR>, indexed by the number of rows minus one.
template <int kB, int... kRows>
constexpr MicroKernelFn kMicroKernels[] = {&micro_kernel<kB, kRows + 1>...};

template <int kB, int... kI>
constexpr const MicroKernelFn* micro_kernels(std::integer_sequence<int, kI...>) {
    return kMicroKernels<kB, kI...>;
}

template <int kB>
void macro_kernel(const GemmMacroArgs& args) {
    const MicroKernelFn* kernels = micro_kernels<kB>(std::make_integer_sequence<int, kMR>{});
    for (std::int64_t jr = 0; jr < args.n; jr += kNR) {
        const std::int64_t nr = args.n - jr < kNR ? args.n - jr : kNR;
        const void* b_panel =
            static_cast<const std::uint8_t*>(args.b) + (jr / kNR) * args.b_panel_stride * (kB == kBF32 ? 4 : 2);
        const float* bias = args.bias != nullptr ? args.bias + jr : nullptr;
        for (std::int64_t ir = 0; ir < args.m; ir += kMR) {
            const std::int64_t mr = args.m - ir < kMR ? args.m - ir : kMR;
            kernels[mr - 1](args.k, args.a + (ir / kMR) * args.k * kMR, b_panel, args.c + ir * args.ldc + jr, args.ldc,
                            nr, args.accumulate, bias);
        }
    }
}

/// Reads element i of a strided vector of `elem` as f32 (scalar path for packing; conversion of 16-bit values
/// goes through a vector load of a small buffer).
II_SIMD_INLINE void gather_f32(const void* base, Elem elem, std::int64_t stride, std::int64_t count, float* out) {
    if (elem == Elem::kF32) {
        const auto* p = static_cast<const float*>(base);
        for (std::int64_t i = 0; i < count; ++i) {
            out[i] = p[i * stride];
        }
        return;
    }
    const auto* p = static_cast<const std::uint16_t*>(base);
    std::int64_t i = 0;
    for (; i < count; i += kLanes) {
        std::uint16_t raw[kLanes] = {};
        const std::int64_t chunk = count - i < kLanes ? count - i : kLanes;
        for (std::int64_t j = 0; j < chunk; ++j) {
            raw[j] = p[(i + j) * stride];
        }
        float converted[kLanes];
        vstore(converted, elem == Elem::kF16 ? vload_f16(raw) : vload_bf16(raw));
        for (std::int64_t j = 0; j < chunk; ++j) {
            out[i + j] = converted[j];
        }
    }
}

}  // namespace

int gemm_mr() noexcept {
    return kMR;
}

int gemm_nr() noexcept {
    return kNR;
}

void gemm_pack_a(const void* a, Elem elem, std::int64_t rs, std::int64_t cs, std::int64_t m, std::int64_t k,
                 float* out) {
    const std::int64_t size = elem == Elem::kF32 ? 4 : 2;
    const auto* base = static_cast<const std::uint8_t*>(a);
    constexpr std::int64_t kChunk = 256;
    alignas(64) float row[kChunk];
    for (std::int64_t r0 = 0; r0 < m; r0 += kMR) {
        const std::int64_t mr = m - r0 < kMR ? m - r0 : kMR;
        float* panel = out + (r0 / kMR) * k * kMR;
        if (rs == 1 && elem == Elem::kF32) {
            // Column-contiguous A (transposed): each depth step copies mr contiguous values.
            for (std::int64_t kk = 0; kk < k; ++kk) {
                const auto* src = reinterpret_cast<const float*>(base) + r0 + kk * cs;
                for (std::int64_t r = 0; r < kMR; ++r) {
                    panel[kk * kMR + r] = r < mr ? src[r] : 0.0f;
                }
            }
            continue;
        }
        for (std::int64_t r = 0; r < kMR; ++r) {
            if (r >= mr) {
                for (std::int64_t kk = 0; kk < k; ++kk) {
                    panel[kk * kMR + r] = 0.0f;
                }
                continue;
            }
            const std::uint8_t* src_row = base + (r0 + r) * rs * size;
            for (std::int64_t k0 = 0; k0 < k; k0 += kChunk) {
                const std::int64_t count = k - k0 < kChunk ? k - k0 : kChunk;
                const float* values;
                if (elem == Elem::kF32 && cs == 1) {
                    values = reinterpret_cast<const float*>(src_row) + k0;
                } else {
                    gather_f32(src_row + k0 * cs * size, elem, cs, count, row);
                    values = row;
                }
                for (std::int64_t kk = 0; kk < count; ++kk) {
                    panel[(k0 + kk) * kMR + r] = values[kk];
                }
            }
        }
    }
}

void gemm_pack_b(const void* b, Elem elem, std::int64_t rs, std::int64_t cs, std::int64_t k, std::int64_t n,
                 void* out) {
    const std::int64_t size = elem == Elem::kF32 ? 4 : 2;
    const auto* src = static_cast<const std::uint8_t*>(b);
    auto* dst = static_cast<std::uint8_t*>(out);
    for (std::int64_t c0 = 0; c0 < n; c0 += kNR) {
        const std::int64_t nr = n - c0 < kNR ? n - c0 : kNR;
        std::uint8_t* panel = dst + (c0 / kNR) * k * kNR * size;
        if (cs == 1) {
            // Row-contiguous B: each depth step copies nr contiguous values.
            for (std::int64_t kk = 0; kk < k; ++kk) {
                const std::uint8_t* from = src + (kk * rs + c0) * size;
                std::uint8_t* to = panel + kk * kNR * size;
                std::memcpy(to, from, static_cast<std::size_t>(nr * size));
                if (nr < kNR) {
                    std::memset(to + nr * size, 0, static_cast<std::size_t>((kNR - nr) * size));
                }
            }
            continue;
        }
        // General strides (e.g. transposed weights [n, k]): walk each column along the depth.
        for (std::int64_t j = 0; j < kNR; ++j) {
            if (j >= nr) {
                for (std::int64_t kk = 0; kk < k; ++kk) {
                    std::memset(panel + (kk * kNR + j) * size, 0, static_cast<std::size_t>(size));
                }
                continue;
            }
            const std::uint8_t* column = src + (c0 + j) * cs * size;
            if (size == 4) {
                const auto* from = reinterpret_cast<const float*>(column);
                auto* to = reinterpret_cast<float*>(panel) + j;
                for (std::int64_t kk = 0; kk < k; ++kk) {
                    to[kk * kNR] = from[kk * rs];
                }
            } else {
                const auto* from = reinterpret_cast<const std::uint16_t*>(column);
                auto* to = reinterpret_cast<std::uint16_t*>(panel) + j;
                for (std::int64_t kk = 0; kk < k; ++kk) {
                    to[kk * kNR] = from[kk * rs];
                }
            }
        }
    }
}

void gemm_macro(const GemmMacroArgs& args) {
    switch (args.b_elem) {
    case Elem::kF16:
        macro_kernel<kBF16>(args);
        break;
    case Elem::kBF16:
        macro_kernel<kBBF16>(args);
        break;
    default:
        macro_kernel<kBF32>(args);
        break;
    }
}

}  // namespace ii::cpu::II_ISA_NS
