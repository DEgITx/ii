// SPDX-License-Identifier: MIT
// Graph optimizations that merge nodes: common subexpression elimination, pattern fusion and elementwise fusion.
#include "compiler/passes.h"

#include "ir/builder.h"
#include "ir/op_schema.h"
#include "ops/fused.h"

#include <algorithm>
#include <format>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ii::compiler {

// ---------------------------------------------------------------------------------------------------------------------
// Common subexpression elimination

std::size_t eliminate_common_subexpressions(ir::Graph& graph) {
    // Nodes are visited in topological order, so the inputs of a node are already canonical when it is looked up.
    std::multimap<std::pair<const ir::OpSchema*, std::vector<const ir::Value*>>, ir::Node*> seen;
    std::vector<ir::Node*> duplicates;
    for (const auto& node : graph.nodes()) {
        if (node->op().has_flag(ir::kOpRandom)) {
            continue;  // every random node draws its own numbers
        }
        std::pair<const ir::OpSchema*, std::vector<const ir::Value*>> key{&node->op(), {}};
        for (const ir::Value* input : node->inputs()) {
            key.second.push_back(input);
        }
        ir::Node* original = nullptr;
        const auto [first, last] = seen.equal_range(key);
        for (auto it = first; it != last; ++it) {
            if (it->second->attrs() == node->attrs() && it->second->num_outputs() == node->num_outputs()) {
                original = it->second;
                break;
            }
        }
        if (original == nullptr) {
            seen.emplace(std::move(key), node.get());
            continue;
        }
        for (int i = 0; i < node->num_outputs(); ++i) {
            graph.replace_all_uses(node->output(i), original->output(i));
        }
        duplicates.push_back(node.get());
    }
    graph.erase_nodes(duplicates);
    return duplicates.size();
}

// ---------------------------------------------------------------------------------------------------------------------
// Pattern fusion

namespace {

bool single_use(const ir::Graph& graph, const ir::Value* value) {
    return value->uses().size() == 1 && !graph.is_graph_result(value);
}

/// add(matmul(x, w), bias) with a 2-D weight and a 1-D bias over the output features => linear(x, w, bias).
Result<bool> try_fuse_linear(ir::Graph& graph, ir::Node& add) {
    for (int side = 0; side < 2; ++side) {
        ir::Value* product = add.input(side);
        ir::Value* bias = add.input(1 - side);
        ir::Node* matmul = product->producer();
        if (matmul == nullptr || matmul->op_name() != "matmul" || !single_use(graph, product)) {
            continue;
        }
        ir::Value* x = matmul->input(0);
        ir::Value* w = matmul->input(1);
        if (matmul->attrs().get_bool("transpose_a") || w->rank() != 2 || !is_float(x->dtype()) ||
            is_block_quantized(w->dtype()) || bias->rank() != 1 || bias->dtype() != product->dtype() ||
            product->rank() == 0 || !(bias->shape()[0] == product->shape().back()) ||
            !(add.output()->type() == product->type())) {
            continue;
        }
        ir::Builder builder(graph);
        builder.set_insertion_point(&add);
        Attributes attrs;
        attrs.set("transpose_w", matmul->attrs().get_bool("transpose_b"));
        II_ASSIGN_OR_RETURN(ir::Node * linear, builder.create_node("linear", {x, w, bias}, std::move(attrs)));
        if (!(linear->output()->type() == add.output()->type())) {
            ir::Node* doomed[] = {linear};
            graph.erase_nodes(doomed);
            continue;
        }
        if (!add.output()->name().empty()) {
            linear->output()->set_name(add.output()->name());
        }
        graph.replace_all_uses(add.output(), linear->output());
        ir::Node* doomed[] = {&add, matmul};
        graph.erase_nodes(doomed);
        return true;
    }
    return false;
}

}  // namespace

Result<std::size_t> fuse_patterns(ir::Graph& graph) {
    std::size_t fused = 0;
    std::vector<ir::Node*> adds;
    for (const auto& node : graph.nodes()) {
        if (node->op_name() == "add") {
            adds.push_back(node.get());
        }
    }
    for (ir::Node* add : adds) {
        II_ASSIGN_OR_RETURN(const bool changed, try_fuse_linear(graph, *add));
        fused += changed ? 1 : 0;
    }
    return fused;
}

// ---------------------------------------------------------------------------------------------------------------------
// Elementwise fusion

namespace {

bool fusable_dtype(DType dtype) {
    return dtype == DType::kF32 || dtype == DType::kF16 || dtype == DType::kBF16 || dtype == DType::kBool;
}

bool fusable(const ir::Node& node) {
    const ir::OpSchema& op = node.op();
    if (!op.has_flag(ir::kOpElementwise) || op.has_flag(ir::kOpInternal) || op.has_flag(ir::kOpRandom) ||
        node.num_outputs() != 1 || !fusable_dtype(node.output()->dtype())) {
        return false;
    }
    for (const ir::Value* input : node.inputs()) {
        if (input == nullptr || !fusable_dtype(input->dtype())) {
            return false;
        }
    }
    return true;
}

}  // namespace

Result<std::size_t> fuse_elementwise(ir::Graph& graph, const ElementwiseFusionOptions& options) {
    std::unordered_map<const ir::Node*, int> order;
    std::vector<ir::Node*> nodes;
    for (const auto& node : graph.nodes()) {
        order.emplace(node.get(), static_cast<int>(nodes.size()));
        nodes.push_back(node.get());
    }
    std::unordered_set<const ir::Node*> assigned;
    std::size_t created = 0;
    const ir::OpSchema& fused_op = ir::builtin_op(ops::kFusedElementwise);

    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        ir::Node* root = *it;
        if (assigned.contains(root) || !fusable(*root)) {
            continue;
        }
        // Grow the group backwards through producers whose results are used only inside the group and have the
        // group's shape (no broadcasting inside the group, so nothing is recomputed).
        std::vector<ir::Node*> group = {root};
        std::unordered_set<const ir::Node*> members = {root};
        const Dims& shape = root->output()->shape();
        auto external_inputs = [&] {
            std::vector<ir::Value*> inputs;
            for (const ir::Node* member : group) {
                for (ir::Value* value : member->inputs()) {
                    if (!members.contains(value->producer()) &&
                        std::find(inputs.begin(), inputs.end(), value) == inputs.end()) {
                        inputs.push_back(value);
                    }
                }
            }
            return inputs;
        };
        for (bool grew = true; grew && static_cast<int>(group.size()) < options.max_nodes;) {
            grew = false;
            for (std::size_t g = 0; g < group.size() && !grew; ++g) {
                for (ir::Value* value : group[g]->inputs()) {
                    ir::Node* producer = value->producer();
                    if (producer == nullptr || members.contains(producer) || assigned.contains(producer) ||
                        !fusable(*producer) || graph.is_graph_result(value) || !(value->shape() == shape)) {
                        continue;
                    }
                    bool all_inside = true;
                    for (const ir::Use& use : value->uses()) {
                        all_inside = all_inside && members.contains(use.node);
                    }
                    if (!all_inside) {
                        continue;
                    }
                    members.insert(producer);
                    group.push_back(producer);
                    if (static_cast<int>(external_inputs().size()) > options.max_inputs) {
                        members.erase(producer);
                        group.pop_back();
                        continue;
                    }
                    grew = true;
                    break;
                }
            }
        }
        if (group.size() < 2) {
            continue;
        }
        std::sort(group.begin(), group.end(),
                  [&](const ir::Node* a, const ir::Node* b) { return order.at(a) < order.at(b); });
        const std::vector<ir::Value*> inputs = external_inputs();

        // The program: a standalone graph with one input per external value.
        ir::Graph program;
        program.set_name("fused");
        std::unordered_map<const ir::Value*, ir::Value*> mapping;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            mapping.emplace(inputs[i], program.add_input(std::format("x{}", i), inputs[i]->type()));
        }
        for (const ir::Node* member : group) {
            std::vector<ir::Value*> member_inputs;
            for (const ir::Value* value : member->inputs()) {
                member_inputs.push_back(mapping.at(value));
            }
            const ir::TensorType type = member->output()->type();
            ir::Node* copy = program.create_node(member->op(), std::move(member_inputs), member->attrs(),
                                                 std::span<const ir::TensorType>(&type, 1));
            mapping.emplace(member->output(), copy->output());
        }
        program.add_output("y", mapping.at(root->output()));

        Attributes attrs;
        attrs.set("program", program.to_text(ir::PrintOptions{-1}));
        attrs.set("dtype", root->output()->dtype());
        const ir::TensorType type = root->output()->type();
        ir::Node* fused =
            graph.create_node(fused_op, inputs, std::move(attrs), std::span<const ir::TensorType>(&type, 1), root);
        if (!root->output()->name().empty()) {
            fused->output()->set_name(root->output()->name());
        }
        if (!root->name().empty()) {
            fused->set_name(root->name());
        }
        graph.replace_all_uses(root->output(), fused->output());
        for (const ir::Node* member : group) {
            assigned.insert(member);
        }
        graph.erase_nodes(group);
        ++created;
    }
    return created;
}

// ---------------------------------------------------------------------------------------------------------------------
// Convolution epilogues

namespace {

/// True when `extra` can be read per output row of a convolution result of `shape` ([N, C, spatial...]): the same
/// shape, one value per channel, or a scalar (after numpy alignment, with batch broadcasting allowed).
bool epilogue_operand(const Dims& shape, const Dims& extra) {
    if (extra.size() > shape.size()) {
        return false;
    }
    const std::size_t offset = shape.size() - extra.size();
    // Per pixel: every spatial dimension present and equal (contiguous pixels); per row: all spatial dimensions 1.
    bool spatial_full = true;
    bool spatial_one = true;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        const bool present = axis >= offset;
        const Dim e = present ? extra[axis - offset] : Dim(1);
        if (!(e == Dim(1)) && !(e == shape[axis])) {
            return false;
        }
        if (axis >= 2) {
            spatial_full = spatial_full && present && e == shape[axis];
            spatial_one = spatial_one && e == Dim(1);
        }
    }
    return spatial_full || spatial_one;
}

Result<bool> try_fuse_conv(ir::Graph& graph, ir::Node& conv) {
    ir::Value* result = conv.output();
    if (!single_use(graph, result) || result->dtype() != DType::kF32) {
        return false;
    }
    ir::Node* user = result->uses().front().node;
    const bool chain = user->op_name() == ops::kFusedElementwise;
    const ir::OpSchema& op = user->op();
    if (!chain && (!op.has_flag(ir::kOpElementwise) || op.has_flag(ir::kOpInternal) || op.has_flag(ir::kOpRandom) ||
                   user->num_outputs() != 1)) {
        return false;
    }
    if (!(user->output()->shape() == result->shape()) || user->output()->dtype() != DType::kF32) {
        return false;
    }
    std::vector<ir::Value*> extras;
    for (ir::Value* input : user->inputs()) {
        if (input == nullptr) {
            return false;
        }
        if (input != result && std::find(extras.begin(), extras.end(), input) == extras.end()) {
            if (!epilogue_operand(result->shape(), input->shape())) {
                return false;
            }
            extras.push_back(input);
        }
    }

    // The epilogue program: input 0 is the convolution result, then the extra inputs.
    ir::Graph program;
    program.set_name("epilogue");
    std::unordered_map<const ir::Value*, ir::Value*> mapping;
    ir::Value* c = program.add_input("c", result->type());
    for (std::size_t i = 0; i < extras.size(); ++i) {
        mapping.emplace(extras[i], program.add_input(std::format("e{}", i), extras[i]->type()));
    }
    ir::Value* program_result = nullptr;
    if (chain) {
        II_ASSIGN_OR_RETURN(const std::unique_ptr<ir::Graph> body, ops::parse_fused_program(*user));
        std::unordered_map<const ir::Value*, ir::Value*> inner;
        for (int i = 0; i < user->num_inputs(); ++i) {
            const ir::Value* outer = user->input(i);
            inner.emplace(body->inputs()[static_cast<std::size_t>(i)], outer == result ? c : mapping.at(outer));
        }
        for (const auto& node : body->nodes()) {
            std::vector<ir::Value*> inputs;
            for (const ir::Value* value : node->inputs()) {
                inputs.push_back(inner.at(value));
            }
            const ir::TensorType type = node->output()->type();
            ir::Node* copy = program.create_node(node->op(), std::move(inputs), node->attrs(),
                                                 std::span<const ir::TensorType>(&type, 1));
            inner.emplace(node->output(), copy->output());
        }
        program_result = inner.at(body->outputs()[0].value);
    } else {
        std::vector<ir::Value*> inputs;
        for (const ir::Value* value : user->inputs()) {
            inputs.push_back(value == result ? c : mapping.at(value));
        }
        const ir::TensorType type = user->output()->type();
        program_result =
            program
                .create_node(user->op(), std::move(inputs), user->attrs(), std::span<const ir::TensorType>(&type, 1))
                ->output();
    }
    program.add_output("y", program_result);

    Attributes attrs = conv.attrs();
    attrs.set("program", program.to_text(ir::PrintOptions{-1}));
    attrs.set("dtype", DType::kF32);
    std::vector<ir::Value*> inputs = {conv.input(0), conv.input(1), conv.input(2)};
    inputs.insert(inputs.end(), extras.begin(), extras.end());
    ir::Builder builder(graph);
    builder.set_insertion_point(user);
    II_ASSIGN_OR_RETURN(ir::Node * fused, builder.create_node(ops::kFusedConv, std::move(inputs), std::move(attrs)));
    if (!(fused->output()->type() == user->output()->type())) {
        ir::Node* doomed[] = {fused};
        graph.erase_nodes(doomed);
        return false;
    }
    fused->set_name(conv.name());
    if (!user->output()->name().empty()) {
        fused->output()->set_name(user->output()->name());
    }
    graph.replace_all_uses(user->output(), fused->output());
    ir::Node* doomed[] = {user, &conv};
    graph.erase_nodes(doomed);
    return true;
}

}  // namespace

Result<std::size_t> fuse_conv_epilogue(ir::Graph& graph) {
    std::vector<ir::Node*> convs;
    for (const auto& node : graph.nodes()) {
        if (node->op_name() == "conv") {
            convs.push_back(node.get());
        }
    }
    std::size_t fused = 0;
    for (ir::Node* conv : convs) {
        II_ASSIGN_OR_RETURN(const bool changed, try_fuse_conv(graph, *conv));
        fused += changed ? 1 : 0;
    }
    return fused;
}

}  // namespace ii::compiler
