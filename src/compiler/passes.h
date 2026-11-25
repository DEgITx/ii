// SPDX-License-Identifier: MIT
/// @file
/// Graph transformation passes. Every pass preserves results (up to floating-point tolerance) — tests check this by
/// executing graphs before and after each pass.
#pragma once

#include <ii/function_ref.h>
#include <ii/status.h>

#include "backend/backend.h"
#include "ir/graph.h"

#include <cstddef>

namespace ii::compiler {

/// Removes nodes whose results are not needed by graph outputs or state updates, and unused constants.
/// Returns the number of removed nodes.
std::size_t eliminate_dead_code(ir::Graph& graph);

struct FoldOptions {
    /// Results larger than this many bytes (and larger than the node's constant inputs together) are not folded,
    /// so that folding never inflates a model (e.g. `expand` of a scalar to a huge tensor).
    std::size_t max_result_bytes = std::size_t{1} << 20;
};

/// Evaluates nodes whose inputs are all constants with `evaluator` and replaces their outputs by constants.
/// Random and creation operations are never folded. Returns the number of folded nodes.
Result<std::size_t> fold_constants(ir::Graph& graph, const backend::Backend& evaluator,
                                   const FoldOptions& options = {});

/// Replaces composite nodes for which `predicate` is true by their decompositions, repeatedly (decompositions may
/// produce further composites). Returns the number of decomposed nodes.
Result<std::size_t> decompose(ir::Graph& graph, FunctionRef<bool(const ir::Node&)> predicate);

/// Decomposes a single composite node.
Status decompose_node(ir::Graph& graph, ir::Node& node);

}  // namespace ii::compiler
