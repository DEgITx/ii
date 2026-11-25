// SPDX-License-Identifier: MIT
/// @file
/// The boundary between ii-cpu drivers and instruction-set-specific kernels.
///
/// Every ISA variant (generic, avx2, avx512, neon) compiles the sources in src/backends/cpu/isa/ into its own
/// namespace and exposes them through one `KernelTable`; the backend picks the table of the selected variant at
/// creation. Drivers (shapes, broadcasting, blocking, threading) are ordinary C++ and call only through the table.
///
/// This header is included by ISA-specific translation units, so it must stay free of inline functions, templates,
/// default member initializers and anything else that makes the compiler emit shared code (see the ODR rules in
/// cmake/IiIsa.cmake): only enums, plain structs and function declarations.
#pragma once

#include <cstdint>

namespace ii::cpu {

/// Element storage understood by the kernels. Floats are computed in f32; bool is one byte (0 / non-zero).
enum class Elem : std::uint8_t {
    kF32,
    kF16,
    kBF16,
    kBool,
};

// --------------------------------------------------------------------------------------------------------------------
// Elementwise programs
//
// A program is a list of instructions over registers, each register holding a tile of f32 values. The interpreter
// evaluates the program tile by tile, so a chain of fused elementwise operations reads its inputs and writes its
// output once, with intermediates staying in the L1 cache. Booleans are represented as 0.0f / 1.0f.

enum class EwOp : std::uint8_t {
    kLoad,  ///< dst = input `a` (converted to f32)
    // unary (a)
    kNeg,
    kAbs,
    kSign,
    kExp,
    kLog,
    kSqrt,
    kRsqrt,
    kRecip,
    kSin,
    kCos,
    kTanh,
    kSigmoid,
    kErf,
    kRelu,
    kFloor,
    kCeil,
    kRound,
    kNot,
    kGelu,
    kGeluTanh,
    kSilu,
    kToBool,     ///< 1 if a != 0 (NaN included), else 0
    kRoundF16,   ///< round a to the nearest f16 value (a cast to f16 inside an f32 computation)
    kRoundBF16,  ///< round a to the nearest bf16 value
    kClamp,      ///< min(max(a, imm0), imm1), NaN stays NaN
    // binary (a, b)
    kAdd,
    kSub,
    kMul,
    kDiv,
    kPow,
    kMax,
    kMin,
    kMod,   ///< Python-style modulo (sign of the divisor)
    kFmod,  ///< C fmod (sign of the dividend)
    kEq,
    kNe,
    kLt,
    kLe,
    kGt,
    kGe,
    kAnd,
    kOr,
    kXor,
    // ternary (a, b, c)
    kWhere,  ///< a ? b : c
};

struct EwInstr {
    EwOp op;
    std::uint8_t dst;
    std::uint8_t a;
    std::uint8_t b;
    std::uint8_t c;
    float imm0;
    float imm1;
};

/// Maximum number of registers of a program.
inline constexpr int kEwMaxRegisters = 32;

struct EwProgram {
    const EwInstr* code;
    int size;
    int result;  ///< register holding the result
    Elem out_elem;
};

/// One input of an elementwise evaluation: `n` consecutive elements, or one element repeated (`broadcast`).
struct EwInput {
    const void* data;
    Elem elem;
    bool broadcast;
};

/// Evaluates `program` for `n` consecutive output elements.
using EwRunFn = void (*)(const EwProgram& program, const EwInput* inputs, void* out, std::int64_t n);

/// Converts `n` elements between storage types.
using ConvertFn = void (*)(const void* src, Elem src_elem, void* dst, Elem dst_elem, std::int64_t n);

// --------------------------------------------------------------------------------------------------------------------
// GEMM (BLIS-style)
//
// Packed A: panels of MR rows; panel p holds rows [p*MR, p*MR + MR) as k groups of MR f32 values (zero padded).
// Packed B: panels of NR columns; panel q holds columns [q*NR, q*NR + NR) as K groups of NR values (zero padded),
//           stored in the element type of B (f32, f16 or bf16). A k-block [k0, k0 + kc) of panel q starts at
//           panel_start + k0 * NR.

/// Packs the m x k block of A at `a` (element (i, j) at a[i * rs + j * cs]) into MR-row panels of f32.
using GemmPackAFn = void (*)(const void* a, Elem elem, std::int64_t rs, std::int64_t cs, std::int64_t m, std::int64_t k,
                             float* out);

/// Packs the k x n matrix B (element (i, j) at b[i * rs + j * cs]) into NR-column panels of the same element type.
using GemmPackBFn = void (*)(const void* b, Elem elem, std::int64_t rs, std::int64_t cs, std::int64_t k, std::int64_t n,
                             void* out);

struct GemmMacroArgs {
    std::int64_t m;               ///< rows of the block (<= packed A rows)
    std::int64_t n;               ///< columns of the block
    std::int64_t k;               ///< depth of the block
    const float* a;               ///< packed A block
    const void* b;                ///< first B panel of the block, offset to the k-block start
    std::int64_t b_panel_stride;  ///< elements from one B panel to the next (= K * NR)
    Elem b_elem;
    float* c;  ///< row-major f32 output block
    std::int64_t ldc;
    bool accumulate;    ///< C += A * B instead of C = A * B
    const float* bias;  ///< optional, one value per column of the block, added to C
};

/// C (m x n) (+)= packed A (m x k) * packed B (k x n) [+ bias].
using GemmMacroFn = void (*)(const GemmMacroArgs& args);

// --------------------------------------------------------------------------------------------------------------------
// Reductions, normalizations, attention

enum class ReduceKind : std::uint8_t {
    kSum,
    kMean,
    kMax,
    kMin,
    kProd,
};

/// Reduces x viewed as [outer, r, inner] over the middle dimension into out [outer, inner]. Sums and products are
/// accumulated in double precision; max / min propagate NaN.
using ReduceFn = void (*)(ReduceKind kind, const float* x, std::int64_t outer, std::int64_t r, std::int64_t inner,
                          float* out);

/// Softmax (or log-softmax) of x viewed as [outer, n, inner] along the middle dimension.
using SoftmaxFn = void (*)(const float* x, float* y, std::int64_t outer, std::int64_t n, std::int64_t inner, bool log);

/// Layer normalization of `rows` contiguous rows of length n; scale and bias (length n) are optional.
using LayerNormFn = void (*)(const float* x, const float* scale, const float* bias, float* y, std::int64_t rows,
                             std::int64_t n, float epsilon);

/// RMS normalization of `rows` contiguous rows of length n; scale (length n) is optional.
using RmsNormFn = void (*)(const float* x, const float* scale, float* y, std::int64_t rows, std::int64_t n,
                           float epsilon);

/// Mask of one attention row.
enum class MaskKind : std::uint8_t {
    kNone,
    kBool,      ///< non-zero byte = attend
    kAdditive,  ///< f32 added to the scores
};

struct AttentionRowArgs {
    const float* q;  ///< [d]
    const void* k;   ///< [kv_len, d] with row stride k_stride (elements of kv_elem)
    std::int64_t k_stride;
    const void* v;  ///< [kv_len, dv] with row stride v_stride
    std::int64_t v_stride;
    Elem kv_elem;  ///< f32 or f16 keys and values
    std::int64_t d;
    std::int64_t dv;
    std::int64_t kv_len;
    std::int64_t kv_start;  ///< keys [0, kv_start) are masked (sliding window); 0 otherwise
    std::int64_t kv_limit;  ///< keys [kv_limit, kv_len) are masked (causal attention); kv_len otherwise
    float scale;
    float softcap;  ///< > 0: scores become softcap * tanh(score / softcap) before masking
    MaskKind mask_kind;
    const void* mask;  ///< mask row: kv_len elements with stride mask_stride (0 = broadcast)
    std::int64_t mask_stride;
    float* scratch;  ///< kv_len + dv floats
    float* out;      ///< [dv]
};

/// One query row of scaled dot-product attention: softmax(q k^T * scale + mask) v.
using AttentionRowFn = void (*)(const AttentionRowArgs& args);

// --------------------------------------------------------------------------------------------------------------------

/// The kernels of one ISA variant.
/// One plane of a depthwise convolution (f32): out[oh][ow] = bias + sum filter[ky][kx] *
/// in[oh * sh - ph + ky * dh][ow * sw - pw + kx * dw] (zero outside the input).
struct DepthwiseArgs {
    const float* in;
    std::int64_t h, w;
    const float* filter;
    std::int64_t kh, kw;
    float bias;
    float* out;
    std::int64_t oh, ow;
    std::int64_t sh, sw, dh, dw, ph, pw;
    float* scratch;  ///< depthwise_scratch_floats(...) floats for the zero-padded input
};
using DepthwiseFn = void (*)(const DepthwiseArgs& args);

/// Floats of scratch memory a depthwise plane needs.
inline std::int64_t depthwise_scratch_floats(std::int64_t oh, std::int64_t ow, std::int64_t kh, std::int64_t kw,
                                             std::int64_t sh, std::int64_t sw, std::int64_t dh, std::int64_t dw) {
    const std::int64_t rows = (oh - 1) * sh + (kh - 1) * dh + 1;
    const std::int64_t cols = (ow - 1) * sw + (kw - 1) * dw + 1;
    // One plane per column phase (stride sw), each row padded so that vector loads may run past its last column.
    return sw * rows * ((cols + sw - 1) / sw + 16) + 64;
}

// --------------------------------------------------------------------------------------------------------------------
// Quantized products (block-quantized weights, GGUF layouts)
//
// Activations are quantized to 8 bits per row first, in one of two layouts:
//   kQ8x32: blocks of 32 values: f32 scale d, f32 s = d * sum(q), 32 int8 q   (40 bytes)
//   kQ8K:   blocks of 256 values: f32 scale d, 256 int8 q, 16 int16 sums of groups of 16 (292 bytes, ggml's q8_K)
// and each weight row is multiplied with several activation rows at once, so it is decoded once for all of them.

enum class QType : std::uint8_t { kQ4_0, kQ4_1, kQ5_0, kQ5_1, kQ8_0, kQ2_K, kQ3_K, kQ4_K, kQ5_K, kQ6_K, kCount };

enum class QAct : std::uint8_t { kQ8x32, kQ8K };

/// Quantizes k floats (a multiple of the block size) into activation blocks.
using QuantizeActFn = void (*)(const float* x, void* out, std::int64_t k);

/// out[r * out_stride] = dot(weight row w (k values), activation row r) for r < rows (1 <= rows <= 4); activation
/// rows are `a_stride` bytes apart.
using QDotFn = void (*)(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
                        std::int64_t out_stride);

/// Panels of 8 weight rows (repacked at compile time, see qmatmul.cpp) make prompt processing a small GEMM: four bytes
/// of one activation row are broadcast against 8 weight rows, so every weight byte decoded serves 4 activation rows and
/// every activation byte 8 weight rows. One panel block (one weight block of each of the 8 rows) holds, per type:
///   q4_0: f32 d[8]; 8 steps x 16 bytes of nibbles               (160 bytes per 32 values)
///   q5_0: f32 d[8]; 8 steps x 16 bytes of nibbles; 8 x u32 high bits (192 bytes)
///   q8_0: f32 d[8]; 8 steps x 32 int8                            (288 bytes)
///   q4_K: f32 d[8], dmin[8]; u8 scale[8][8], min[8][8] (sub-block, row); 64 steps x 16 bytes of nibbles (1216 bytes)
///   q6_K: f32 d[8]; i8 scale[16][8]; 16 x (4 steps x 16 bytes of low nibbles, 32 bytes of 2-bit high parts)
///         (1696 bytes per 256 values)
/// A step covers 4 consecutive values of each of the 8 rows: 32 values, ordered row 0 (4 values), row 1, ... For
/// nibbles, byte j of a step holds row j / 4 in its low half and row 4 + j / 4 in its high half (value j % 4); high
/// bits use bit i of a 32-bit word for value i of the step; the q6_K high parts of 4 steps share 32 bytes (bits
/// 2t..2t+1 of byte i for step t, value i).
inline constexpr int kPanelRows = 8;

/// out[r * out_stride + c] = dot(weight row c of the panel, activation row r) for c < 8 and r < rows (1..4).
using QGemm8Fn = void (*)(std::int64_t k, const void* panel, const void* a, std::int64_t a_stride, int rows, float* out,
                          std::int64_t out_stride);

/// y[i] += a * x[i] (f32).
using AxpyFn = void (*)(float* y, const float* x, float a, std::int64_t n);
/// y[i] = max(y[i], x[i]) (f32).
using MaxInplaceFn = void (*)(float* y, const float* x, std::int64_t n);

struct KernelTable {
    const char* isa;
    EwRunFn ew_run;
    ConvertFn convert;
    int gemm_mr;
    int gemm_nr;
    GemmPackAFn gemm_pack_a;
    GemmPackBFn gemm_pack_b;
    GemmMacroFn gemm_macro;
    ReduceFn reduce;
    SoftmaxFn softmax;
    LayerNormFn layer_norm;
    RmsNormFn rms_norm;
    AttentionRowFn attention_row;
    AxpyFn axpy;
    MaxInplaceFn max_inplace;
    DepthwiseFn depthwise;
    QuantizeActFn quantize_q8x32;
    QuantizeActFn quantize_q8k;
    QDotFn qdot[static_cast<int>(QType::kCount)];
    QGemm8Fn qgemm8[static_cast<int>(QType::kCount)];  ///< nullptr: no panel kernel for the type in this variant
};

// Scalar fallbacks for rare functions, compiled once without ISA flags (src/backends/cpu/scalar_math.cpp).
// They compute in double precision and round once, like ii-ref.
float scalar_pow(float a, float b) noexcept;
float scalar_mod(float a, float b) noexcept;
float scalar_fmod(float a, float b) noexcept;
float scalar_sin(float a) noexcept;
float scalar_cos(float a) noexcept;

#define II_DECLARE_CPU_KERNEL_TABLE(isa)                                                                               \
    namespace isa {                                                                                                    \
    const KernelTable& kernel_table() noexcept;                                                                        \
    }

II_DECLARE_CPU_KERNEL_TABLE(generic)
#if defined(II_HAVE_ISA_AVX2)
II_DECLARE_CPU_KERNEL_TABLE(avx2)
#endif
#if defined(II_HAVE_ISA_AVX512)
II_DECLARE_CPU_KERNEL_TABLE(avx512)
#endif
#if defined(II_HAVE_ISA_NEON)
II_DECLARE_CPU_KERNEL_TABLE(neon)
#endif
#if defined(II_HAVE_ISA_AVX_VNNI)
II_DECLARE_CPU_KERNEL_TABLE(avx_vnni)
#endif
#if defined(II_HAVE_ISA_NEON_DOTPROD)
II_DECLARE_CPU_KERNEL_TABLE(neon_dotprod)
#endif

#undef II_DECLARE_CPU_KERNEL_TABLE

}  // namespace ii::cpu
