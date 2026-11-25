// SPDX-License-Identifier: MIT
/// @file
/// The IR text format, used for debugging, `ii inspect` and pass tests ("input text → expected text").
///
/// @code
/// graph mlp {
///   input %x : f32[N, 4]
///   state %steps : i64[] = i64[] {0}
///   const %w : f32[4, 2] = f32[4, 2] {1, 2, 3, 4, 5, 6, 7, 8}
///   %0 : f32[N, 2] = matmul(%x, %w)
///   %y : f32[N, 2] = relu(%0)
///   %a : f32[N, 1], %b : f32[N, 1] = split(%y) {axis = 1, sizes = [1, 1]}
///   %n : f32[N, 2] = layer_norm(%y, _, _) {axis = -1, epsilon = 1e-05}
///   update %steps = %next
///   output y = %y
/// }
/// @endcode
///
/// - Values: `%name` (identifier characters and dots), `%"any text"`, or `%<number>` for unnamed values.
/// - `_` is an absent optional input.
/// - Attribute values: `true`, `3`, `1.5`, `inf`, `"text"`, `f16` (a dtype), `[1, 2]`, `[N, 3]` (dims), and tensor
///   literals `f32[2] {1, 2}` (block-quantized: `q4_0[32] 0x...`). The symbols `inf` and `nan` are reserved.
/// - A constant printed as `{...}` (elided data) is parsable only with a `ParseOptions::resolve_constant` callback.
/// - Comments start with `#` and run to the end of the line.
#pragma once

#include <ii/status.h>

#include "ir/graph.h"
#include "ir/op_schema.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace ii::ir {

/// Formats a graph (same as `Graph::to_text`).
std::string print_graph(const Graph& graph, const PrintOptions& options = {});

struct ParseOptions {
    const OpRegistry* registry = nullptr;  ///< nullptr: the built-in OpSet
    /// Provides data for constants printed with elided data (`{...}`).
    std::function<Result<Tensor>(std::string_view name, const TensorType& type)> resolve_constant;
    /// Run `Graph::verify()` on the result (also checks that declared types match shape inference).
    bool verify = true;
};

/// Parses the text format. Errors carry line and column numbers.
Result<std::unique_ptr<Graph>> parse_graph(std::string_view text, const ParseOptions& options = {});

}  // namespace ii::ir
