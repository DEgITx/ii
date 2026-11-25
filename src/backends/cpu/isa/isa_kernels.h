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

}  // namespace ii::cpu::II_ISA_NS
