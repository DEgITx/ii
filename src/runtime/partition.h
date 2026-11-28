// SPDX-License-Identifier: MIT
/// @file
/// Partitioning for graph backends: after planning, parts of the graph go to backends that compile whole subgraphs
/// (`Backend::compiles_subgraphs`, e.g. TensorRT).
///
/// For each graph backend, the nodes on its device that it supports (whichever backend of the device the planner
/// chose for them) are grouped into parts: connected through their values and convex (no path leaves a part and
/// comes back, so a part can run as one step). Each part becomes a `subgraph` node (ops/fused.h) for that backend
/// when the backend's estimate for the whole part is lower than the planned estimates of its nodes, or when the
/// planner gave one of its nodes to the graph backend itself. Backends of the same device share its memory, so
/// taking nodes from another backend of the device adds no copies.
#pragma once

#include <ii/status.h>

#include "ir/graph.h"
#include "runtime/planner.h"

#include <span>

namespace ii::runtime {

/// Replaces parts of `graph` by `subgraph` nodes of graph backends and updates `plan` (per-node arrays, device
/// estimates, notes) to match the new graph.
Status claim_subgraphs(ir::Graph& graph, GraphPlan& plan, std::span<const PlanBackend> backends,
                       const PlannerOptions& options);

}  // namespace ii::runtime
