// SPDX-License-Identifier: MIT
/// @file
/// `fused_elementwise`: an internal operation created by the elementwise fusion pass.
///
/// It evaluates a subgraph of elementwise operations in one pass over memory. The subgraph is stored in the
/// `program` attribute in the IR text format, with one graph input per node input (in order) and one output:
///
/// @code
///   %y : f32[N, 8] = fused_elementwise(%x, %b) {dtype = f32, program = "graph fused {
///     input %x0 : f32[N, 8]
///     input %x1 : f32[8]
///     %0 : f32[N, 8] = add(%x0, %x1)
///     %1 : f32[N, 8] = silu(%0)
///     output y = %1
///   }"}
/// @endcode
///
/// The result has dtype `dtype` and the broadcast shape of all inputs. Backends that do not implement the operation
/// get its decomposition: the subgraph inlined back.
#pragma once

#include <ii/status.h>

#include "ir/graph.h"

#include <memory>
#include <string_view>

namespace ii::ops {

inline constexpr std::string_view kFusedElementwise = "fused_elementwise";

/// Parses the `program` attribute of a `fused_elementwise` node and checks it against the node's inputs.
Result<std::unique_ptr<ir::Graph>> parse_fused_program(const ir::Node& node);

}  // namespace ii::ops
