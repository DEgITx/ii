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

/// `fused_conv(x, w, bias?, extras...)`: a convolution followed by an elementwise program over its result (the
/// epilogue, e.g. an activation or a residual addition), created by `compiler::fuse_conv_epilogue`. The program's
/// input 0 is the convolution result, inputs 1... are the extra inputs (each the shape of the result, one value per
/// channel, or a scalar); the result has the shape of the convolution and dtype `dtype`.
inline constexpr std::string_view kFusedConv = "fused_conv";

/// `transfer(x) {from, to}`: copies a value from the memory of device `from` to device `to` (names as printed by
/// `to_string(Device)`, e.g. "cpu", "cuda:0"). Inserted by the compiler when the planner places a value and a node
/// reading it on different devices; executed by the runtime itself, not by a backend.
inline constexpr std::string_view kTransfer = "transfer";

/// Parses the `program` attribute of a `fused_elementwise` node and checks it against the node's inputs.
Result<std::unique_ptr<ir::Graph>> parse_fused_program(const ir::Node& node);

/// Parses the epilogue program of a `fused_conv` node and checks it against the node's extra inputs.
Result<std::unique_ptr<ir::Graph>> parse_epilogue_program(const ir::Node& node);

/// The attributes of the convolution inside a `fused_conv` node.
Attributes conv_attributes(const Attributes& fused);

}  // namespace ii::ops
