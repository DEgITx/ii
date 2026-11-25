// SPDX-License-Identifier: MIT
/// @file
/// Declarations of the kernels of one ISA variant (namespace ii::cpu::II_ISA_NS); collected into a `KernelTable` by
/// isa/table.cpp. Included only by ISA-specific translation units.
#pragma once

#include "backends/cpu/kernel_table.h"

#include <cstdint>

#if !defined(II_ISA_NS)
#error "isa_kernels.h is for ISA-specific translation units only"
#endif

namespace ii::cpu::II_ISA_NS {

void ew_run(const EwProgram& program, const EwInput* inputs, void* out, std::int64_t n);
void convert(const void* src, Elem src_elem, void* dst, Elem dst_elem, std::int64_t n);

int gemm_mr() noexcept;
int gemm_nr() noexcept;
void gemm_pack_a(const void* a, Elem elem, std::int64_t rs, std::int64_t cs, std::int64_t m, std::int64_t k,
                 float* out);
void gemm_pack_b(const void* b, Elem elem, std::int64_t rs, std::int64_t cs, std::int64_t k, std::int64_t n, void* out);
void gemm_macro(const GemmMacroArgs& args);

void reduce(ReduceKind kind, const float* x, std::int64_t outer, std::int64_t r, std::int64_t inner, float* out);
void softmax(const float* x, float* y, std::int64_t outer, std::int64_t n, std::int64_t inner, bool log);
void layer_norm(const float* x, const float* scale, const float* bias, float* y, std::int64_t rows, std::int64_t n,
                float epsilon);
void rms_norm(const float* x, const float* scale, float* y, std::int64_t rows, std::int64_t n, float epsilon);
void attention_row(const AttentionRowArgs& args);
void axpy(float* y, const float* x, float a, std::int64_t n);
void max_inplace(float* y, const float* x, std::int64_t n);
void depthwise(const DepthwiseArgs& args);

/// Panel kernels (isa/qgemm.cpp); entries are nullptr where the variant has none.
extern const QGemm8Fn kQGemm8[static_cast<int>(QType::kCount)];

void quantize_q8x32(const float* x, void* out, std::int64_t k);
void quantize_q8k(const float* x, void* out, std::int64_t k);
void qdot_q4_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q4_1(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q5_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q5_1(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q8_0(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q2_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q3_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q4_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q5_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);
void qdot_q6_k(std::int64_t k, const void* w, const void* a, std::int64_t a_stride, int rows, float* out,
               std::int64_t out_stride);

}  // namespace ii::cpu::II_ISA_NS
