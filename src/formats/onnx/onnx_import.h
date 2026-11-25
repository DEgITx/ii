// SPDX-License-Identifier: MIT
/// @file
/// Import of ONNX models into the ii IR.
///
/// The importer walks the ONNX graph in order and translates every node through a table of handlers
/// (`onnx_ops.cpp`) into ii operations. ONNX passes many operation parameters as tensors (the target shape of
/// Reshape, the starts of Slice, the axes of reductions, ...) while ii keeps them in attributes, so the importer
/// tracks the *contents* of small integer tensors symbolically: `Shape(x)` of an input with dimensions [N, 3, 224,
/// 224] is known to hold [N, 3, 224, 224], and Gather / Slice / Concat / arithmetic over such values are evaluated
/// while importing. Larger constant subgraphs feeding parameters are evaluated with the reference backend.
#pragma once

#include <ii/dim.h>
#include <ii/status.h>

#include "backend/backend.h"
#include "formats/onnx/onnx_proto.h"
#include "ir/builder.h"
#include "ir/graph.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ii::formats::onnx {

struct ImportOptions {
    /// Directory of the model file, for external data. Empty: external data is not available.
    std::filesystem::path base_dir;
    /// Values for symbolic input dimensions (by ONNX dim_param name), e.g. {"batch", 1}.
    std::vector<std::pair<std::string, std::int64_t>> dims;
    /// Inputs replaced by constants.
    std::vector<std::pair<std::string, Tensor>> constant_inputs;
    /// Evaluates constant subgraphs while importing (the reference backend). Required.
    const backend::Backend* evaluator = nullptr;
};

/// Imports a serialized ONNX model. `keep_alive` (optional) owns `bytes`: tensors may then refer to the bytes without
/// copying; without it all tensor data is copied.
Result<std::unique_ptr<ir::Graph>> import_model(std::span<const std::byte> bytes,
                                                std::shared_ptr<const void> keep_alive, const ImportOptions& options);

/// True when `bytes` look like a serialized ONNX ModelProto (a cheap check of the first fields).
bool looks_like_onnx(std::span<const std::byte> bytes) noexcept;

// ---------------------------------------------------------------------------------------------------------------
// Internals shared by the importer and the operation handlers.

class Importer;

/// Contents of a small integer tensor of rank 0 or 1 known while importing (possibly symbolic).
struct KnownContents {
    Dims values;
    bool scalar = false;
    DType dtype = DType::kI64;
};

/// What the importer knows about one ONNX value.
struct Entry {
    ir::Value* value = nullptr;  ///< the ii value; may be null while only `known` is set
    /// Contents of a small integer (or integer-valued) tensor of rank 0 or 1, possibly symbolic.
    std::optional<Dims> known;
    bool scalar = false;            ///< rank 0 (for `known`)
    DType dtype = DType::kInvalid;  ///< dtype of `known`
};

/// The view of one ONNX node given to its handler.
class NodeContext {
public:
    NodeContext(Importer& importer, const OnnxNode& node, std::int64_t opset)
        : importer_(importer), node_(node), opset_(opset) {}

    const OnnxNode& node() const noexcept { return node_; }
    std::int64_t opset() const noexcept { return opset_; }
    std::string_view op_type() const noexcept { return node_.op_type; }

    // --- Inputs ---

    int num_inputs() const noexcept { return static_cast<int>(node_.inputs.size()); }
    bool has_input(int index) const noexcept {
        return index >= 0 && index < num_inputs() && !node_.inputs[static_cast<std::size_t>(index)].empty();
    }
    /// The ii value of input `index` (materialized if only its contents are known). Fails if absent.
    Result<ir::Value*> input(int index);
    /// The ii value of input `index`, or nullptr if absent.
    Result<ir::Value*> optional_input(int index);
    /// The type of input `index` (without materializing it).
    Result<ir::TensorType> input_type(int index);
    /// The contents of input `index` as dimensions (symbolic allowed); fails if they are not known when importing.
    Result<Dims> dims_input(int index);
    /// The contents of input `index` as constant integers; fails if they are not constant.
    Result<std::vector<std::int64_t>> ints_input(int index);
    /// The contents of input `index` as constant floats (any numeric dtype).
    Result<std::vector<double>> floats_input(int index);
    /// The constant tensor of input `index` (evaluating a constant subgraph if needed).
    Result<Tensor> constant_input(int index);
    /// True if input `index` is a constant (or a constant subgraph) without materializing anything.
    bool is_constant_input(int index);
    /// The contents of input `index` if they are known (tracked symbolically or a small integer constant).
    std::optional<KnownContents> known_input(int index);

    // --- Attributes ---

    const OnnxAttribute* attribute(std::string_view name) const noexcept { return node_.find_attribute(name); }
    bool has_attribute(std::string_view name) const noexcept { return attribute(name) != nullptr; }
    std::int64_t int_attr(std::string_view name, std::int64_t fallback) const noexcept;
    double float_attr(std::string_view name, double fallback) const noexcept;
    std::string string_attr(std::string_view name, std::string_view fallback) const;
    std::optional<std::vector<std::int64_t>> ints_attr(std::string_view name) const;
    std::optional<std::vector<double>> floats_attr(std::string_view name) const;
    Result<Tensor> tensor_attr(std::string_view name) const;

    // --- Building ---

    /// Creates an ii node and returns its first output.
    Result<ir::Value*> op(std::string_view name, std::vector<ir::Value*> inputs, Attributes attrs = {});
    /// Creates an ii node and returns all its outputs.
    Result<std::vector<ir::Value*>> op_outputs(std::string_view name, std::vector<ir::Value*> inputs,
                                               Attributes attrs = {});
    /// A constant.
    ir::Value* constant(Tensor tensor);
    /// A rank-0 constant of `dtype`.
    Result<ir::Value*> scalar(double value, DType dtype);
    /// `cast` to `dtype` unless the value already has it.
    Result<ir::Value*> cast(ir::Value* value, DType dtype);
    /// reshape(value, dims).
    Result<ir::Value*> reshape(ir::Value* value, Dims dims);

    // --- Outputs ---

    int num_outputs() const noexcept { return static_cast<int>(node_.outputs.size()); }
    bool has_output(int index) const noexcept {
        return index >= 0 && index < num_outputs() && !node_.outputs[static_cast<std::size_t>(index)].empty();
    }
    /// Binds output `index` to an ii value.
    Status set_output(int index, ir::Value* value);
    /// Binds output `index` to known contents (rank 0 when `scalar`, otherwise rank 1).
    Status set_known_output(int index, Dims contents, bool scalar, DType dtype);

    /// An "unsupported" error naming this node.
    Status unsupported(std::string_view what) const;

    Importer& importer() noexcept { return importer_; }

private:
    Result<Entry*> entry(int index);

    Importer& importer_;
    const OnnxNode& node_;
    std::int64_t opset_;
};

/// Translates one ONNX node; reads inputs and binds every output through `NodeContext`.
using OpHandler = Status (*)(NodeContext& context);

/// The handler of an ONNX operation of the default domain, or nullptr.
OpHandler find_op_handler(std::string_view op_type) noexcept;

/// Names of the supported ONNX operators (default domain), sorted.
std::vector<std::string_view> supported_onnx_ops();

/// Handlers of operations that compute on known contents (Shape, Gather, Concat, ...): return true when the outputs
/// were bound to known contents, false to fall back to the regular handler.
Result<bool> try_known_contents(NodeContext& context);

class Importer {
public:
    Importer(const OnnxModel& model, std::shared_ptr<const void> keep_alive, const ImportOptions& options);

    Result<std::unique_ptr<ir::Graph>> run();

    // Used by NodeContext and the handlers.
    ir::Graph& graph() noexcept { return *graph_; }
    ir::Builder& builder() noexcept { return builder_; }
    Entry* find(const std::string& name);
    Status bind(const std::string& name, Entry entry);
    Result<ir::Value*> materialize(const std::string& name, Entry& entry);
    /// Evaluates `value` if it is computed from constants only; returns null if it is not.
    Result<ir::Value*> try_fold(ir::Value* value, int depth = 0);
    /// Imports the nodes of a subgraph (an If branch) into the current graph; its outputs are returned in order.
    Result<std::vector<Entry>> inline_subgraph(const OnnxGraph& subgraph);
    Result<Tensor> tensor_of(const OnnxTensor& proto);

private:
    Status import_graph_inputs();
    Status import_nodes(std::span<const OnnxNode> nodes, std::int64_t opset, int depth);
    Status import_node(const OnnxNode& node, std::int64_t opset, int depth);
    Status import_function_call(const OnnxNode& node, const OnnxFunction& function, int depth);
    std::string fresh_symbol();

    const OnnxModel& model_;
    std::shared_ptr<const void> keep_alive_;
    const ImportOptions& options_;
    std::unique_ptr<ir::Graph> graph_;
    ir::Builder builder_;
    ExternalData external_;
    std::unordered_map<std::string, Entry> values_;
    int next_symbol_ = 0;
    int next_inline_ = 0;
};

}  // namespace ii::formats::onnx
