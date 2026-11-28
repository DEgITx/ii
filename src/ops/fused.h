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

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

/// `subgraph(inputs...) {backend, program}`: a part of the graph that one backend compiles into one kernel (a graph
/// backend such as TensorRT, see `Backend::compiles_subgraphs`). Created by the compiler after planning
/// (runtime/partition.h); `program` holds the part in the IR text format with one graph input per node input (in
/// order; constants of the model are inputs too) and one graph output per node output; `backend` names the backend
/// that claimed it. Every symbol the program uses appears as a whole dimension of one of its inputs. Backends that do
/// not implement the operation get its decomposition: the part inlined back.
inline constexpr std::string_view kSubgraph = "subgraph";

/// Parses the `program` attribute of a `subgraph` node (cached by text: nodes share the parsed graph) and checks it
/// against the node's inputs and outputs.
Result<std::shared_ptr<const ir::Graph>> parse_subgraph_program(const ir::Node& node);

/// The program of a `subgraph` node as a standalone graph for a backend to compile: inputs of the node that are
/// constants become constants of the graph (sharing their data); `inputs[i]` is the node input that graph input i
/// receives.
struct SubgraphProgram {
    std::unique_ptr<ir::Graph> graph;
    std::vector<int> inputs;
};
Result<SubgraphProgram> subgraph_program(const ir::Node& node);

/// Values of the program's symbols for concrete input shapes (each symbol is a whole dimension of an input).
Result<std::vector<std::pair<std::string, std::int64_t>>> subgraph_symbols(const ir::Graph& program,
                                                                           std::span<const Shape> inputs);

/// Parses the `program` attribute of a `fused_elementwise` node and checks it against the node's inputs.
Result<std::unique_ptr<ir::Graph>> parse_fused_program(const ir::Node& node);

/// Parses the epilogue program of a `fused_conv` node and checks it against the node's extra inputs.
Result<std::unique_ptr<ir::Graph>> parse_epilogue_program(const ir::Node& node);

/// The attributes of the convolution inside a `fused_conv` node.
Attributes conv_attributes(const Attributes& fused);

}  // namespace ii::ops
