// SPDX-License-Identifier: MIT
/// @file
/// Text representation of tensor values used by the IR text format: `f32[2, 2] {1, 2, 3, 4.5}`.
///
/// Elements are printed in row-major order with the shortest representation that round-trips; booleans as
/// `true`/`false`; non-finite floats as `inf`, `-inf`, `nan`. Block-quantized tensors are printed as raw bytes in hex:
/// `q4_0[1, 32] 0x0a1b...`.
#pragma once

#include <ii/tensor.h>

#include <string>

namespace ii::ir {

/// Formats a tensor literal. When the tensor has more than `max_elements` elements the values are replaced by
/// `{...}` (the result is then not parsable).
std::string format_tensor_literal(const Tensor& tensor, std::int64_t max_elements = -1);

}  // namespace ii::ir
