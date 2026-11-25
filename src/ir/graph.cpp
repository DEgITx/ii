// SPDX-License-Identifier: MIT
#include "ir/graph.h"

#include "core/assert.h"
#include "ir/op_schema.h"

#include <algorithm>
#include <format>
#include <unordered_map>
#include <unordered_set>

namespace ii::ir {

// ------------------------------------------------------------------------------------------------------------------
// TensorType

bool TensorType::is_static() const noexcept {
    return std::all_of(shape.begin(), shape.end(), [](const Dim& dim) { return dim.is_constant(); });
}

std::optional<Shape> TensorType::static_shape() const {
    if (!is_static() || shape.size() > static_cast<std::size_t>(kMaxRank)) {
        return std::nullopt;
    }
    Shape result;
    for (const Dim& dim : shape) {
        result.push_back(dim.value());
    }
    return result;
}

std::string TensorType::to_string() const {
    return std::string(ii::to_string(dtype)) + ii::to_string(shape);
}

TensorType type_of(const Tensor& tensor) {
    TensorType type;
    type.dtype = tensor.dtype();
    for (const std::int64_t dim : tensor.shape()) {
        type.shape.emplace_back(dim);
    }
    return type;
}

std::string_view to_string(ValueKind kind) noexcept {
    switch (kind) {
    case ValueKind::kInput:
        return "input";
    case ValueKind::kConstant:
        return "constant";
    case ValueKind::kState:
        return "state";
    case ValueKind::kNode:
        return "node";
    }
    return "unknown";
}

// ------------------------------------------------------------------------------------------------------------------
// Node

std::string_view Node::op_name() const noexcept {
    return op_->name;
}

void Node::set_input(int index, Value* value) {
    II_CHECK(index >= 0 && index < num_inputs());
    Value*& slot = inputs_[static_cast<std::size_t>(index)];
    if (slot == value) {
        return;
    }
    if (slot != nullptr) {
        std::erase_if(slot->uses_, [&](const Use& use) { return use.node == this && use.index == index; });
    }
    slot = value;
    if (value != nullptr) {
        value->uses_.push_back(Use{this, index});
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Graph

Graph::Graph() = default;
Graph::~Graph() = default;
Graph::Graph(Graph&&) noexcept = default;
Graph& Graph::operator=(Graph&&) noexcept = default;

Value* Graph::new_value(ValueKind kind, TensorType type) {
    values_.push_back(std::unique_ptr<Value>(new Value(next_value_id_++, kind, std::move(type))));
    return values_.back().get();
}

Value* Graph::add_input(std::string name, TensorType type) {
    Value* value = new_value(ValueKind::kInput, std::move(type));
    value->name_ = std::move(name);
    inputs_.push_back(value);
    return value;
}

Value* Graph::add_constant(Tensor data, std::string name) {
    Value* value = new_value(ValueKind::kConstant, type_of(data));
    value->data_ = std::move(data);
    value->name_ = std::move(name);
    return value;
}

Value* Graph::add_state(std::string name, TensorType type, Tensor initial) {
    Value* value = new_value(ValueKind::kState, std::move(type));
    value->name_ = std::move(name);
    value->data_ = std::move(initial);
    states_.push_back(StateBinding{value, nullptr});
    return value;
}

void Graph::set_state_update(Value* state, Value* update) {
    for (StateBinding& binding : states_) {
        if (binding.state == state) {
            binding.update = update;
            return;
        }
    }
    II_CHECK(false, "set_state_update: value is not a state of this graph");
}

void Graph::add_output(std::string name, Value* value) {
    outputs_.push_back(GraphOutput{std::move(name), value});
}

void Graph::set_output(std::size_t index, Value* value) {
    II_CHECK(index < outputs_.size());
    outputs_[index].value = value;
}

Node* Graph::create_node(const OpSchema& op, std::vector<Value*> inputs, Attributes attrs,
                         std::span<const TensorType> output_types, Node* insert_before) {
    auto node = std::unique_ptr<Node>(new Node(this, next_node_id_++, &op));
    Node* raw = node.get();
    raw->inputs_ = std::move(inputs);
    raw->attrs_ = std::move(attrs);
    for (std::size_t i = 0; i < raw->inputs_.size(); ++i) {
        if (Value* input = raw->inputs_[i]) {
            input->uses_.push_back(Use{raw, static_cast<int>(i)});
        }
    }
    for (std::size_t i = 0; i < output_types.size(); ++i) {
        Value* output = new_value(ValueKind::kNode, output_types[i]);
        output->producer_ = raw;
        output->producer_output_ = static_cast<int>(i);
        raw->outputs_.push_back(output);
    }
    if (insert_before == nullptr) {
        nodes_.push_back(std::move(node));
    } else {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                                     [&](const std::unique_ptr<Node>& n) { return n.get() == insert_before; });
        II_CHECK(it != nodes_.end(), "insertion point is not in this graph");
        nodes_.insert(it, std::move(node));
    }
    return raw;
}

Value* Graph::find_input(std::string_view name) const noexcept {
    for (Value* value : inputs_) {
        if (value->name() == name) {
            return value;
        }
    }
    return nullptr;
}

Value* Graph::find_state(std::string_view name) const noexcept {
    for (const StateBinding& binding : states_) {
        if (binding.state->name() == name) {
            return binding.state;
        }
    }
    return nullptr;
}

const GraphOutput* Graph::find_output(std::string_view name) const noexcept {
    for (const GraphOutput& output : outputs_) {
        if (output.name == name) {
            return &output;
        }
    }
    return nullptr;
}

bool Graph::is_graph_result(const Value* value) const noexcept {
    for (const GraphOutput& output : outputs_) {
        if (output.value == value) {
            return true;
        }
    }
    for (const StateBinding& binding : states_) {
        if (binding.update == value) {
            return true;
        }
    }
    return false;
}

void Graph::replace_all_uses(Value* from, Value* to) {
    if (from == to) {
        return;
    }
    const std::vector<Use> uses = from->uses_;
    for (const Use& use : uses) {
        use.node->set_input(use.index, to);
    }
    for (GraphOutput& output : outputs_) {
        if (output.value == from) {
            output.value = to;
        }
    }
    for (StateBinding& binding : states_) {
        if (binding.update == from) {
            binding.update = to;
        }
    }
}

void Graph::erase_values(std::span<Value* const> values) {
    if (values.empty()) {
        return;
    }
    const std::unordered_set<const Value*> doomed(values.begin(), values.end());
    std::erase_if(values_, [&](const std::unique_ptr<Value>& v) { return doomed.contains(v.get()); });
}

void Graph::erase_nodes(std::span<Node* const> nodes) {
    if (nodes.empty()) {
        return;
    }
    const std::unordered_set<const Node*> doomed(nodes.begin(), nodes.end());
    std::vector<Value*> dead_values;
    for (Node* node : nodes) {
        for (Value* output : node->outputs_) {
            II_CHECK(!is_graph_result(output), "erasing a node whose output is a graph result");
            for (const Use& use : output->uses_) {
                II_CHECK(doomed.contains(use.node), "erasing a node whose output is still used");
            }
            dead_values.push_back(output);
        }
        for (std::size_t i = 0; i < node->inputs_.size(); ++i) {
            if (Value* input = node->inputs_[i]) {
                std::erase_if(input->uses_, [&](const Use& use) { return use.node == node; });
            }
        }
    }
    std::erase_if(nodes_, [&](const std::unique_ptr<Node>& n) { return doomed.contains(n.get()); });
    erase_values(dead_values);
}

void Graph::erase_unused_constants() {
    std::vector<Value*> dead;
    for (const auto& value : values_) {
        if (value->kind() == ValueKind::kConstant && value->uses().empty() && !is_graph_result(value.get())) {
            dead.push_back(value.get());
        }
    }
    erase_values(dead);
}

void Graph::move_before(Node* node, Node* position) {
    const auto it =
        std::find_if(nodes_.begin(), nodes_.end(), [&](const std::unique_ptr<Node>& n) { return n.get() == node; });
    II_CHECK(it != nodes_.end());
    std::unique_ptr<Node> owned = std::move(*it);
    nodes_.erase(it);
    if (position == nullptr) {
        nodes_.push_back(std::move(owned));
        return;
    }
    const auto pos =
        std::find_if(nodes_.begin(), nodes_.end(), [&](const std::unique_ptr<Node>& n) { return n.get() == position; });
    II_CHECK(pos != nodes_.end());
    nodes_.insert(pos, std::move(owned));
}

Status Graph::topological_sort() {
    // Kahn's algorithm, always taking the earliest ready node in the current order (stable).
    const std::size_t n = nodes_.size();
    std::unordered_map<const Node*, std::size_t> position;
    for (std::size_t i = 0; i < n; ++i) {
        position[nodes_[i].get()] = i;
    }
    std::vector<int> pending(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        for (Value* input : nodes_[i]->inputs_) {
            if (input != nullptr && input->producer_ != nullptr) {
                ++pending[i];
            }
        }
    }
    std::vector<std::size_t> ready_heap;  // min-heap of positions
    auto push_ready = [&](std::size_t index) {
        ready_heap.push_back(index);
        std::push_heap(ready_heap.begin(), ready_heap.end(), std::greater<>());
    };
    for (std::size_t i = 0; i < n; ++i) {
        if (pending[i] == 0) {
            push_ready(i);
        }
    }
    std::vector<std::unique_ptr<Node>> sorted;
    sorted.reserve(n);
    while (!ready_heap.empty()) {
        std::pop_heap(ready_heap.begin(), ready_heap.end(), std::greater<>());
        const std::size_t index = ready_heap.back();
        ready_heap.pop_back();
        Node* node = nodes_[index].get();
        for (Value* output : node->outputs_) {
            for (const Use& use : output->uses_) {
                const std::size_t user = position.at(use.node);
                if (--pending[user] == 0) {
                    push_ready(user);
                }
            }
        }
        sorted.push_back(std::move(nodes_[index]));
    }
    if (sorted.size() != n) {
        // Put everything back so the graph stays intact.
        for (auto& node : nodes_) {
            if (node) {
                sorted.push_back(std::move(node));
            }
        }
        nodes_ = std::move(sorted);
        return errors::invalid_argument("the graph contains a cycle");
    }
    nodes_ = std::move(sorted);
    return ok_status();
}

std::unique_ptr<Graph> Graph::clone() const {
    auto copy = std::make_unique<Graph>();
    copy->name_ = name_;
    std::unordered_map<const Value*, Value*> mapping;
    auto map = [&](const Value* value) -> Value* { return value == nullptr ? nullptr : mapping.at(value); };
    for (const auto& value : values_) {
        if (value->kind() == ValueKind::kNode) {
            continue;
        }
        Value* v = copy->new_value(value->kind(), value->type());
        v->name_ = value->name_;
        v->data_ = value->data_;
        mapping[value.get()] = v;
    }
    for (Value* input : inputs_) {
        copy->inputs_.push_back(map(input));
    }
    for (const StateBinding& binding : states_) {
        copy->states_.push_back(StateBinding{map(binding.state), nullptr});
    }
    for (const auto& node : nodes_) {
        std::vector<Value*> inputs;
        for (const Value* input : node->inputs_) {
            inputs.push_back(map(input));
        }
        std::vector<TensorType> types;
        for (const Value* output : node->outputs_) {
            types.push_back(output->type());
        }
        Node* n = copy->create_node(*node->op_, std::move(inputs), node->attrs_, types);
        n->name_ = node->name_;
        for (std::size_t i = 0; i < node->outputs_.size(); ++i) {
            n->outputs_[i]->name_ = node->outputs_[i]->name_;
            mapping[node->outputs_[i]] = n->outputs_[i];
        }
    }
    for (std::size_t i = 0; i < states_.size(); ++i) {
        copy->states_[i].update = map(states_[i].update);
    }
    for (const GraphOutput& output : outputs_) {
        copy->outputs_.push_back(GraphOutput{output.name, map(output.value)});
    }
    copy->next_value_id_ = std::max(copy->next_value_id_, next_value_id_);
    copy->next_node_id_ = std::max(copy->next_node_id_, next_node_id_);
    return copy;
}

namespace {

std::string describe(const Node& node) {
    return std::format("node #{} '{}'{}", node.id(), node.op_name(),
                       node.name().empty() ? "" : " (" + node.name() + ")");
}

}  // namespace

Status Graph::verify() const {
    std::unordered_set<const Value*> owned;
    for (const auto& value : values_) {
        owned.insert(value.get());
    }
    std::unordered_set<const Value*> defined;
    std::unordered_set<std::string_view> names;
    for (const Value* input : inputs_) {
        if (input->kind() != ValueKind::kInput || !owned.contains(input)) {
            return errors::internal("graph input '{}' is not an input value of this graph", input->name());
        }
        if (input->name().empty() || !names.insert(input->name()).second) {
            return errors::internal("graph input name '{}' is empty or not unique", input->name());
        }
    }
    for (const StateBinding& binding : states_) {
        if (binding.state == nullptr || binding.state->kind() != ValueKind::kState || !owned.contains(binding.state)) {
            return errors::internal("invalid state binding");
        }
        if (binding.state->name().empty() || !names.insert(binding.state->name()).second) {
            return errors::internal("state name '{}' is empty or not unique among inputs and states",
                                    binding.state->name());
        }
        const Tensor& initial = binding.state->data();
        if (initial.defined() && !(type_of(initial) == binding.state->type())) {
            return errors::internal("initial value {} of state '{}' does not match its type {}",
                                    type_of(initial).to_string(), binding.state->name(),
                                    binding.state->type().to_string());
        }
    }
    for (const auto& value : values_) {
        if (value->kind() != ValueKind::kNode) {
            defined.insert(value.get());
        }
        if (value->kind() == ValueKind::kConstant) {
            if (!value->data().defined() || !(type_of(value->data()) == value->type())) {
                return errors::internal("constant #{} has no data or its data does not match its type", value->id());
            }
        }
        for (const Use& use : value->uses()) {
            if (use.node == nullptr || use.node->input(use.index) != value.get()) {
                return errors::internal("value #{} has a stale use entry", value->id());
            }
        }
    }

    std::vector<TensorType> inferred;
    for (const auto& node_ptr : nodes_) {
        const Node& node = *node_ptr;
        if (&node.graph() != this) {
            return errors::internal("{} belongs to another graph", describe(node));
        }
        std::vector<const TensorType*> input_types;
        for (int i = 0; i < node.num_inputs(); ++i) {
            const Value* input = node.input(i);
            if (input == nullptr) {
                input_types.push_back(nullptr);
                continue;
            }
            if (!owned.contains(input)) {
                return errors::internal("{} input {} is not a value of this graph", describe(node), i);
            }
            if (!defined.contains(input)) {
                return errors::internal("{} input {} (value #{}) is used before it is defined", describe(node), i,
                                        input->id());
            }
            const auto& uses = input->uses();
            if (std::none_of(uses.begin(), uses.end(),
                             [&](const Use& use) { return use.node == &node && use.index == i; })) {
                return errors::internal("{} input {} is missing from the use list of value #{}", describe(node), i,
                                        input->id());
            }
            input_types.push_back(&input->type());
        }
        // Attributes must already be canonical.
        Attributes canonical = node.attrs();
        if (Status status = canonicalize_attributes(node.op(), canonical); !status.ok()) {
            return status.add_context(describe(node));
        }
        if (!(canonical == node.attrs())) {
            return errors::internal("{} attributes are not canonical: {} vs {}", describe(node),
                                    node.attrs().to_string(), canonical.to_string());
        }
        inferred.clear();
        if (Status status = infer_types(node.op(), input_types, node.attrs(), inferred); !status.ok()) {
            return status.add_context(describe(node));
        }
        if (inferred.size() != node.outputs().size()) {
            return errors::internal("{} has {} outputs, shape inference expects {}", describe(node),
                                    node.outputs().size(), inferred.size());
        }
        for (std::size_t i = 0; i < inferred.size(); ++i) {
            const Value* output = node.outputs()[i];
            if (output->producer() != &node || output->producer_output() != static_cast<int>(i) ||
                output->kind() != ValueKind::kNode || !owned.contains(output)) {
                return errors::internal("{} output {} is inconsistent", describe(node), i);
            }
            if (!(output->type() == inferred[i])) {
                return errors::internal("{} output {} has type {}, but shape inference gives {}", describe(node), i,
                                        output->type().to_string(), inferred[i].to_string());
            }
            defined.insert(output);
        }
    }

    std::unordered_set<std::string_view> output_names;
    for (const GraphOutput& output : outputs_) {
        if (output.name.empty() || !output_names.insert(output.name).second) {
            return errors::internal("graph output name '{}' is empty or not unique", output.name);
        }
        if (output.value == nullptr || !defined.contains(output.value)) {
            return errors::internal("graph output '{}' is not a defined value", output.name);
        }
    }
    for (const StateBinding& binding : states_) {
        if (binding.update == nullptr) {
            continue;
        }
        if (!defined.contains(binding.update)) {
            return errors::internal("update of state '{}' is not a defined value", binding.state->name());
        }
        if (!(binding.update->type() == binding.state->type())) {
            return errors::internal("update of state '{}' has type {}, expected {}", binding.state->name(),
                                    binding.update->type().to_string(), binding.state->type().to_string());
        }
    }
    return ok_status();
}

}  // namespace ii::ir
