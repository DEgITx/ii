// SPDX-License-Identifier: MIT
// Elementwise program interpreter and storage conversions (compiled once per ISA variant).
#include "backends/cpu/isa/isa_kernels.h"
#include "backends/cpu/isa/vmath.h"

namespace ii::cpu::II_ISA_NS {

namespace {

/// Elements per tile: small enough that all registers of a program stay in L1, large enough to amortize dispatch.
constexpr std::int64_t kTile = 128;
static_assert(kTile % kLanes == 0);

II_SIMD_INLINE std::int64_t round_up_lanes(std::int64_t n) {
    return (n + kLanes - 1) / kLanes * kLanes;
}

/// Converts `n` elements at `src` into f32 at `dst`, zero-padding to a multiple of kLanes.
void load_f32(const void* src, Elem elem, std::int64_t n, float* dst) {
    const std::int64_t full = n / kLanes * kLanes;
    switch (elem) {
    case Elem::kF32: {
        const auto* p = static_cast<const float*>(src);
        for (std::int64_t i = 0; i < full; i += kLanes) {
            vstore(dst + i, vload(p + i));
        }
        if (full < n) {
            vstore(dst + full, vload_partial(p + full, n - full));
        }
        break;
    }
    case Elem::kF16:
    case Elem::kBF16: {
        const auto* p = static_cast<const std::uint16_t*>(src);
        const bool half = elem == Elem::kF16;
        for (std::int64_t i = 0; i < full; i += kLanes) {
            vstore(dst + i, half ? vload_f16(p + i) : vload_bf16(p + i));
        }
        if (full < n) {
            std::uint16_t tail[kLanes] = {};
            for (std::int64_t i = full; i < n; ++i) {
                tail[i - full] = p[i];
            }
            vstore(dst + full, half ? vload_f16(tail) : vload_bf16(tail));
        }
        break;
    }
    case Elem::kBool: {
        const auto* p = static_cast<const std::uint8_t*>(src);
        for (std::int64_t i = 0; i < n; ++i) {
            dst[i] = p[i] != 0 ? 1.0f : 0.0f;
        }
        for (std::int64_t i = n; i < round_up_lanes(n); ++i) {
            dst[i] = 0.0f;
        }
        break;
    }
    }
}

/// Stores `n` f32 values from `src` (readable up to a multiple of kLanes) as `elem` at `dst`.
void store_f32(const float* src, Elem elem, std::int64_t n, void* dst) {
    const std::int64_t full = n / kLanes * kLanes;
    switch (elem) {
    case Elem::kF32: {
        auto* p = static_cast<float*>(dst);
        for (std::int64_t i = 0; i < full; i += kLanes) {
            vstore(p + i, vload(src + i));
        }
        if (full < n) {
            vstore_partial(p + full, vload(src + full), n - full);
        }
        break;
    }
    case Elem::kF16:
    case Elem::kBF16: {
        auto* p = static_cast<std::uint16_t*>(dst);
        const bool half = elem == Elem::kF16;
        for (std::int64_t i = 0; i < full; i += kLanes) {
            if (half) {
                vstore_f16(p + i, vload(src + i));
            } else {
                vstore_bf16(p + i, vload(src + i));
            }
        }
        if (full < n) {
            std::uint16_t tail[kLanes];
            if (half) {
                vstore_f16(tail, vload(src + full));
            } else {
                vstore_bf16(tail, vload(src + full));
            }
            for (std::int64_t i = full; i < n; ++i) {
                p[i] = tail[i - full];
            }
        }
        break;
    }
    case Elem::kBool: {
        auto* p = static_cast<std::uint8_t*>(dst);
        for (std::int64_t i = 0; i < n; ++i) {
            p[i] = src[i] != 0.0f || src[i] != src[i] ? 1 : 0;  // NaN is true
        }
        break;
    }
    }
}

std::int64_t elem_size(Elem elem) {
    switch (elem) {
    case Elem::kF32:
        return 4;
    case Elem::kF16:
    case Elem::kBF16:
        return 2;
    case Elem::kBool:
        return 1;
    }
    return 1;
}

template <class Fn>
II_SIMD_INLINE void map1(std::int64_t nv, const float* a, float* d, Fn fn) {
    for (std::int64_t i = 0; i < nv; i += kLanes) {
        vstore(d + i, fn(vload(a + i)));
    }
}

template <class Fn>
II_SIMD_INLINE void map2(std::int64_t nv, const float* a, const float* b, float* d, Fn fn) {
    for (std::int64_t i = 0; i < nv; i += kLanes) {
        vstore(d + i, fn(vload(a + i), vload(b + i)));
    }
}

template <class Fn>
II_SIMD_INLINE void scalar1(std::int64_t nv, const float* a, float* d, Fn fn) {
    for (std::int64_t i = 0; i < nv; ++i) {
        d[i] = fn(a[i]);
    }
}

template <class Fn>
II_SIMD_INLINE void scalar2(std::int64_t nv, const float* a, const float* b, float* d, Fn fn) {
    for (std::int64_t i = 0; i < nv; ++i) {
        d[i] = fn(a[i], b[i]);
    }
}

/// Rounds through a 16-bit float format and back.
template <bool kHalf>
II_SIMD_INLINE VF round16(VF x) {
    std::uint16_t tmp[kLanes];
    if constexpr (kHalf) {
        vstore_f16(tmp, x);
        return vload_f16(tmp);
    } else {
        vstore_bf16(tmp, x);
        return vload_bf16(tmp);
    }
}

void run_tile(const EwProgram& program, const EwInput* inputs, std::int64_t offset, std::int64_t n,
              float (*regs)[kTile]) {
    const std::int64_t nv = round_up_lanes(n);
    for (int pc = 0; pc < program.size; ++pc) {
        const EwInstr& ins = program.code[pc];
        float* d = regs[ins.dst];
        const float* a = regs[ins.a];
        const float* b = regs[ins.b];
        const float* c = regs[ins.c];
        switch (ins.op) {
        case EwOp::kLoad: {
            const EwInput& input = inputs[ins.a];
            if (input.broadcast) {
                alignas(64) float value[kLanes] = {};
                load_f32(input.data, input.elem, 1, value);
                const VF splat = vset1(value[0]);
                for (std::int64_t i = 0; i < nv; i += kLanes) {
                    vstore(d + i, splat);
                }
            } else {
                load_f32(static_cast<const std::uint8_t*>(input.data) + offset * elem_size(input.elem), input.elem, n,
                         d);
            }
            break;
        }
        case EwOp::kNeg:
            map1(nv, a, d, [](VF x) { return vneg(x); });
            break;
        case EwOp::kAbs:
            map1(nv, a, d, [](VF x) { return vabs(x); });
            break;
        case EwOp::kSign:
            map1(nv, a, d, [](VF x) {
                const VF s = vselect(vgt(x, vzero()), vset1(1.0f), vselect(vlt(x, vzero()), vset1(-1.0f), vzero()));
                return vselect(visnan(x), x, s);
            });
            break;
        case EwOp::kExp:
            map1(nv, a, d, [](VF x) { return vexp(x); });
            break;
        case EwOp::kLog:
            map1(nv, a, d, [](VF x) { return vlog(x); });
            break;
        case EwOp::kSqrt:
            map1(nv, a, d, [](VF x) { return vsqrt(x); });
            break;
        case EwOp::kRsqrt:
            map1(nv, a, d, [](VF x) { return vdiv(vset1(1.0f), vsqrt(x)); });
            break;
        case EwOp::kRecip:
            map1(nv, a, d, [](VF x) { return vdiv(vset1(1.0f), x); });
            break;
        case EwOp::kSin:
            scalar1(nv, a, d, [](float x) { return scalar_sin(x); });
            break;
        case EwOp::kCos:
            scalar1(nv, a, d, [](float x) { return scalar_cos(x); });
            break;
        case EwOp::kTanh:
            map1(nv, a, d, [](VF x) { return vtanh(x); });
            break;
        case EwOp::kSigmoid:
            map1(nv, a, d, [](VF x) { return vsigmoid(x); });
            break;
        case EwOp::kErf:
            map1(nv, a, d, [](VF x) { return verf(x); });
            break;
        case EwOp::kRelu:
            map1(nv, a, d, [](VF x) { return vselect(vgt(x, vzero()), x, vselect(visnan(x), x, vzero())); });
            break;
        case EwOp::kFloor:
            map1(nv, a, d, [](VF x) { return vfloor(x); });
            break;
        case EwOp::kCeil:
            map1(nv, a, d, [](VF x) { return vceil(x); });
            break;
        case EwOp::kRound:
            map1(nv, a, d, [](VF x) { return vround(x); });
            break;
        case EwOp::kNot:
            map1(nv, a, d, [](VF x) { return vmask_to_float(vmnot(vtruthy(x))); });
            break;
        case EwOp::kGelu:
            map1(nv, a, d, [](VF x) { return vgelu(x); });
            break;
        case EwOp::kGeluTanh:
            map1(nv, a, d, [](VF x) { return vgelu_tanh(x); });
            break;
        case EwOp::kSilu:
            map1(nv, a, d, [](VF x) { return vsilu(x); });
            break;
        case EwOp::kToBool:
            map1(nv, a, d, [](VF x) { return vmask_to_float(vtruthy(x)); });
            break;
        case EwOp::kRoundF16:
            map1(nv, a, d, [](VF x) { return round16<true>(x); });
            break;
        case EwOp::kRoundBF16:
            map1(nv, a, d, [](VF x) { return round16<false>(x); });
            break;
        case EwOp::kClamp: {
            const VF lo = vset1(ins.imm0);
            const VF hi = vset1(ins.imm1);
            map1(nv, a, d, [lo, hi](VF x) { return vselect(visnan(x), x, vmin(vmax(x, lo), hi)); });
            break;
        }
        case EwOp::kAdd:
            map2(nv, a, b, d, [](VF x, VF y) { return vadd(x, y); });
            break;
        case EwOp::kSub:
            map2(nv, a, b, d, [](VF x, VF y) { return vsub(x, y); });
            break;
        case EwOp::kMul:
            map2(nv, a, b, d, [](VF x, VF y) { return vmul(x, y); });
            break;
        case EwOp::kDiv:
            map2(nv, a, b, d, [](VF x, VF y) { return vdiv(x, y); });
            break;
        case EwOp::kPow:
            scalar2(nv, a, b, d, [](float x, float y) { return scalar_pow(x, y); });
            break;
        case EwOp::kMax:
            map2(nv, a, b, d, [](VF x, VF y) { return vmax_nan(x, y); });
            break;
        case EwOp::kMin:
            map2(nv, a, b, d, [](VF x, VF y) { return vmin_nan(x, y); });
            break;
        case EwOp::kMod:
            scalar2(nv, a, b, d, [](float x, float y) { return scalar_mod(x, y); });
            break;
        case EwOp::kFmod:
            scalar2(nv, a, b, d, [](float x, float y) { return scalar_fmod(x, y); });
            break;
        case EwOp::kEq:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(veq(x, y)); });
            break;
        case EwOp::kNe:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(vne(x, y)); });
            break;
        case EwOp::kLt:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(vlt(x, y)); });
            break;
        case EwOp::kLe:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(vle(x, y)); });
            break;
        case EwOp::kGt:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(vgt(x, y)); });
            break;
        case EwOp::kGe:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(vge(x, y)); });
            break;
        case EwOp::kAnd:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(vmand(vtruthy(x), vtruthy(y))); });
            break;
        case EwOp::kOr:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(vmor(vtruthy(x), vtruthy(y))); });
            break;
        case EwOp::kXor:
            map2(nv, a, b, d, [](VF x, VF y) { return vmask_to_float(vmxor(vtruthy(x), vtruthy(y))); });
            break;
        case EwOp::kWhere:
            for (std::int64_t i = 0; i < nv; i += kLanes) {
                vstore(d + i, vselect(vtruthy(vload(a + i)), vload(b + i), vload(c + i)));
            }
            break;
        }
    }
}

}  // namespace

void ew_run(const EwProgram& program, const EwInput* inputs, void* out, std::int64_t n) {
    alignas(64) float regs[kEwMaxRegisters][kTile];
    auto* out_bytes = static_cast<std::uint8_t*>(out);
    const std::int64_t out_size = elem_size(program.out_elem);
    for (std::int64_t offset = 0; offset < n; offset += kTile) {
        const std::int64_t count = n - offset < kTile ? n - offset : kTile;
        run_tile(program, inputs, offset, count, regs);
        store_f32(regs[program.result], program.out_elem, count, out_bytes + offset * out_size);
    }
}

void convert(const void* src, Elem src_elem, void* dst, Elem dst_elem, std::int64_t n) {
    alignas(64) float tile[kTile];
    const auto* src_bytes = static_cast<const std::uint8_t*>(src);
    auto* dst_bytes = static_cast<std::uint8_t*>(dst);
    const std::int64_t src_size = elem_size(src_elem);
    const std::int64_t dst_size = elem_size(dst_elem);
    for (std::int64_t offset = 0; offset < n; offset += kTile) {
        const std::int64_t count = n - offset < kTile ? n - offset : kTile;
        load_f32(src_bytes + offset * src_size, src_elem, count, tile);
        store_f32(tile, dst_elem, count, dst_bytes + offset * dst_size);
    }
}

}  // namespace ii::cpu::II_ISA_NS
