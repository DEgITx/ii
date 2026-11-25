// SPDX-License-Identifier: MIT
/// @file
/// The ii intermediate representation: an SSA graph of operations over tensor values.
///
///   Graph   owns values and nodes; nodes are kept in topological order.
///   Value   a tensor-typed SSA value: a graph input, a constant, a state, or an output of a node.
///   Node    an operation (described by an `OpSchema`) with input values, attributes and output values.
///
/// **State values** model data that persists between runs of a model (KV caches, recurrent state, training
/// parameters): a state is read like an input at the start of a run, and a value bound as its *update* is written
/// back at the end.
#pragma once

#include <ii/attribute.h>
#include <ii/dim.h>
#include <ii/dtype.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ii::ir {

struct OpSchema;
class Graph;
class Node;

/// The type of a tensor value: element type and (possibly symbolic) shape.
struct TensorType {
    DType dtype = DType::kInvalid;
    Dims shape;

    int rank() const noexcept { return static_cast<int>(shape.size()); }

    /// True when every dimension is a constant.
    bool is_static() const noexcept;

    /// The concrete shape if the type is static.
    std::optional<Shape> static_shape() const;

    /// "f32[N, 3]".
    std::string to_string() const;

    friend bool operator==(const TensorType& a, const TensorType& b) noexcept {
        return a.dtype == b.dtype && a.shape == b.shape;
    }
};

/// The type of a concrete tensor.
TensorType type_of(const Tensor& tensor);

/// Where a value comes from.
enum class ValueKind : std::uint8_t {
    kInput,     ///< A graph input, bound by the caller for each run.
    kConstant,  ///< A constant with data (weights); see `Value::data()`.
    kState,     ///< A state value; see the file comment.
    kNode,      ///< An output of a node.
};

std::string_view to_string(ValueKind kind) noexcept;

/// A use of a value: input `index` of `node`.
struct Use {
    Node* node = nullptr;
    int index = 0;
};

/// A tensor-typed SSA value.
class Value {
public:
    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    /// Unique id within the graph (stable, never reused).
    int id() const noexcept { return id_; }

    /// Optional name (graph inputs, states and named intermediates); may be empty.
    const std::string& name() const noexcept { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

    const TensorType& type() const noexcept { return type_; }
    void set_type(TensorType type) { type_ = std::move(type); }
    DType dtype() const noexcept { return type_.dtype; }
    const Dims& shape() const noexcept { return type_.shape; }
    int rank() const noexcept { return type_.rank(); }

    ValueKind kind() const noexcept { return kind_; }

    /// The producing node (kNode values) or nullptr.
    Node* producer() const noexcept { return producer_; }
    /// Index among the outputs of the producer.
    int producer_output() const noexcept { return producer_output_; }

    /// Node inputs reading this value. Graph outputs and state updates are not included.
    const std::vector<Use>& uses() const noexcept { return uses_; }

    /// Constant data (kConstant), or the initial value of a state (kState; empty means zeros).
    const Tensor& data() const noexcept { return data_; }
    void set_data(Tensor data) { data_ = std::move(data); }

private:
    friend class Graph;
    friend class Node;
    Value(int id, ValueKind kind, TensorType type) : id_(id), kind_(kind), type_(std::move(type)) {}

    int id_;
    ValueKind kind_;
    TensorType type_;
    std::string name_;
    Node* producer_ = nullptr;
    int producer_output_ = 0;
    std::vector<Use> uses_;
    Tensor data_;
};

/// An operation in a graph.
class Node {
public:
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    int id() const noexcept { return id_; }
    Graph& graph() const noexcept { return *graph_; }

    const OpSchema& op() const noexcept { return *op_; }
    std::string_view op_name() const noexcept;

    /// Optional name (e.g. the name of the node in an imported model); may be empty.
    const std::string& name() const noexcept { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

    /// Inputs; an absent optional input is nullptr.
    std::span<Value* const> inputs() const noexcept { return inputs_; }
    int num_inputs() const noexcept { return static_cast<int>(inputs_.size()); }
    /// Input `index`, or nullptr when absent or out of range.
    Value* input(int index) const noexcept {
        return index >= 0 && index < num_inputs() ? inputs_[static_cast<std::size_t>(index)] : nullptr;
    }
    /// Replaces input `index` (updating use lists).
    void set_input(int index, Value* value);

    std::span<Value* const> outputs() const noexcept { return outputs_; }
    int num_outputs() const noexcept { return static_cast<int>(outputs_.size()); }
    Value* output(int index = 0) const noexcept { return outputs_[static_cast<std::size_t>(index)]; }

    const Attributes& attrs() const noexcept { return attrs_; }
    Attributes& mutable_attrs() noexcept { return attrs_; }

private:
    friend class Graph;
    Node(Graph* graph, int id, const OpSchema* op) : id_(id), graph_(graph), op_(op) {}

    int id_;
    Graph* graph_;
    const OpSchema* op_;
    std::string name_;
    std::vector<Value*> inputs_;
    std::vector<Value*> outputs_;
    Attributes attrs_;
};

/// A named graph output.
struct GraphOutput {
    std::string name;
    Value* value = nullptr;
};

/// A state value and the value written back to it at the end of a run (nullptr: the state is never updated).
struct StateBinding {
    Value* state = nullptr;
    Value* update = nullptr;
};

/// Options for `Graph::to_text()`.
struct PrintOptions {
    /// Constants with more elements are printed as `{...}` (not parsable). Negative: print everything.
    std::int64_t max_constant_elements = 64;
};

/// An SSA graph of operations.
class Graph {
public:
    Graph();
    ~Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept;
    Graph& operator=(Graph&&) noexcept;

    /// A deep copy (constant data is shared, not copied).
    std::unique_ptr<Graph> clone() const;

    // --- Construction ---

    /// Adds a graph input. Names must be unique among inputs and states.
    Value* add_input(std::string name, TensorType type);

    /// Adds a constant holding `data`.
    Value* add_constant(Tensor data, std::string name = {});

    /// Adds a state. `initial` (optional) is its value before the first run; zeros when empty.
    Value* add_state(std::string name, TensorType type, Tensor initial = {});

    /// Binds the value written back to `state` at the end of each run.
    void set_state_update(Value* state, Value* update);

    /// Adds a named graph output.
    void add_output(std::string name, Value* value);

    /// Creates a node with outputs of the given types. Does not run shape inference or validation — use
    /// `ir::Builder` for that. The node is inserted before `insert_before`, or appended when it is nullptr.
    Node* create_node(const OpSchema& op, std::vector<Value*> inputs, Attributes attrs,
                      std::span<const TensorType> output_types, Node* insert_before = nullptr);

    // --- Queries ---

    std::span<Value* const> inputs() const noexcept { return inputs_; }
    const std::vector<GraphOutput>& outputs() const noexcept { return outputs_; }
    const std::vector<StateBinding>& states() const noexcept { return states_; }

    /// Nodes in topological order.
    const std::vector<std::unique_ptr<Node>>& nodes() const noexcept { return nodes_; }
    std::size_t num_nodes() const noexcept { return nodes_.size(); }

    /// All live values (inputs, constants, states, node outputs) in creation order.
    const std::vector<std::unique_ptr<Value>>& values() const noexcept { return values_; }

    Value* find_input(std::string_view name) const noexcept;
    Value* find_state(std::string_view name) const noexcept;
    const GraphOutput* find_output(std::string_view name) const noexcept;

    /// True when the value is a graph output or a state update.
    bool is_graph_result(const Value* value) const noexcept;

    // --- Mutation ---

    /// Redirects every use of `from` (node inputs, graph outputs, state updates) to `to`.
    void replace_all_uses(Value* from, Value* to);

    /// Replaces the value of graph output `index`.
    void set_output(std::size_t index, Value* value);

    /// Removes nodes for which `predicate(node)` is true. Their outputs must have no remaining uses outside of
    /// the removed set and must not be graph results.
    template <class Predicate>
    void erase_nodes_if(Predicate&& predicate) {
        std::vector<Node*> doomed;
        for (const auto& node : nodes_) {
            if (predicate(*node)) {
                doomed.push_back(node.get());
            }
        }
        erase_nodes(doomed);
    }
    void erase_nodes(std::span<Node* const> nodes);

    /// Removes constants that have no uses and are not graph results.
    void erase_unused_constants();

    /// Moves `node` directly before `position` (or to the end when nullptr).
    void move_before(Node* node, Node* position);

    /// Restores topological order (stable: keeps the current order where possible). Fails on cycles.
    Status topological_sort();

    /// Checks structural invariants: SSA, use lists, topological order, operation arity and attribute kinds,
    /// result types consistent with shape inference, outputs and states well-formed.
    Status verify() const;

    /// The text form of the graph (see `ir/text.h`).
    std::string to_text(const PrintOptions& options = {}) const;

    /// Graph name (informational).
    const std::string& name() const noexcept { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

private:
    friend class Node;
    Value* new_value(ValueKind kind, TensorType type);
    void erase_values(std::span<Value* const> values);

    std::string name_;
    int next_value_id_ = 0;
    int next_node_id_ = 0;
    std::vector<std::unique_ptr<Value>> values_;
    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<Value*> inputs_;
    std::vector<GraphOutput> outputs_;
    std::vector<StateBinding> states_;
};

}  // namespace ii::ir
