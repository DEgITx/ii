// SPDX-License-Identifier: MIT
/// @file
/// Translation of ii graphs into TensorRT networks.
///
/// A program (the graph of a `subgraph` node, see ops/fused.h) becomes a strongly typed `INetworkDefinition`: graph
/// input i is network input "x<i>" (symbolic dimensions are dynamic, -1), graph output i is network output "y<i>",
/// constants are constant layers (or the weights of convolutions). Composite operations without a TensorRT layer of
/// their own (rms_norm, log_softmax, sdpa, ...) are decomposed by ii first (`prepare_program`). Shapes that depend on
/// symbols (reshape targets, slice sizes) are computed in the network from the shapes of the inputs.
#pragma once

#include <ii/status.h>
#include <ii/tensor.h>

#include "ir/graph.h"

#include <NvInfer.h>

#include <cstddef>
#include <deque>
#include <memory>
#include <vector>

namespace ii::tensorrt {

/// OK if a node (an operation of the OpSet, a fused operation or a `subgraph`) can be translated; otherwise an
/// kUnsupported status explaining why.
Status supports_node(const ir::Node& node);

/// Decomposes the composite operations of `graph` that have no TensorRT translation of their own.
Status prepare_program(ir::Graph& graph);

/// Memory the network refers to (weights made during translation); must live until the engine is built.
struct NetworkStorage {
    std::deque<std::vector<std::byte>> bytes;
    std::deque<Tensor> tensors;
};

/// Adds the program's inputs, layers and outputs to an empty network.
Status translate(const ir::Graph& program, nvinfer1::INetworkDefinition& network, NetworkStorage& storage);

}  // namespace ii::tensorrt
