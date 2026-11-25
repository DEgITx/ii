// SPDX-License-Identifier: MIT
/// @file
/// Reference codecs of the block-quantized dtypes (GGUF / ggml layouts).
///
/// The block layouts are byte-compatible with ggml: `q4_0` … `q8_1` hold 32 elements per block, the "K-quants"
/// `q2_k` … `q8_k` hold 256 elements in a super-block with 6- or 8-bit sub-block scales. The quantizers follow the
/// reference (`*_ref`) algorithms of ggml without an importance matrix, so `ii quantize` produces the same blocks as
/// `llama-quantize` without `--imatrix` (verified against gguf-py for the types it implements).
///
/// These functions are scalar and meant for loading, conversion, tests and the reference backend; ii-cpu has
/// vectorized dot-product kernels.
#pragma once

#include <ii/dtype.h>
#include <ii/status.h>

#include <cstdint>

namespace ii::quant {

/// True when `dtype` is a block-quantized type with a dequantizer (every block type).
bool can_dequantize(DType dtype) noexcept;

/// True when `quantize` supports `dtype`.
bool can_quantize(DType dtype) noexcept;

/// Dequantizes `count` elements (a multiple of the block size) from `blocks` into `out`.
void dequantize(DType dtype, const void* blocks, float* out, std::int64_t count) noexcept;

/// Quantizes `count` elements (a multiple of the block size) of `in` into `blocks`. `dtype` must satisfy
/// `can_quantize`.
void quantize(DType dtype, const float* in, void* blocks, std::int64_t count) noexcept;

/// Checks that a block-quantized tensor row of `count` elements is whole blocks.
Status check_row(DType dtype, std::int64_t count);

}  // namespace ii::quant
