// SPDX-License-Identifier: MIT
#include "compiler/passes.h"

#include "core/aligned_buffer.h"
#include "core/inline_executor.h"
#include "ir/builder.h"
#include "ir/op_schema.h"

#include <unordered_set>
#include <vector>

namespace ii::compiler {

std::size_t eliminate_dead_code(ir::Graph& graph) {
    std::unordered_set<const ir::Value*> needed;
    for (const ir::GraphOutput& output : graph.outputs()) {
        needed.insert(output.value);
    }
    for (const ir::StateBinding& binding : graph.states()) {
        if (binding.update != nullptr) {
            needed.insert(binding.update);
        }
    }
    std::unordered_set<const ir::Node*> live;
    const auto& nodes = graph.nodes();
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        const ir::Node& node = **it;
        bool is_live = false;
        for (const ir::Value* output : node.outputs()) {
            is_live = is_live || needed.contains(output);
        }
        if (!is_live) {
            continue;
        }
        live.insert(&node);
        for (const ir::Value* input : node.inputs()) {
            if (input != nullptr) {
                needed.insert(input);
            }
        }
    }
    const std::size_t before = graph.num_nodes();
    graph.erase_nodes_if([&](const ir::Node& node) { return !live.contains(&node); });
    graph.erase_unused_constants();
    return before - graph.num_nodes();
}

namespace {

bool foldable(const ir::Node& node) {
    const ir::OpSchema& op = node.op();
    if (op.has_flag(ir::kOpRandom) || op.has_flag(ir::kOpNoConstantFold) || node.num_inputs() == 0) {
        return false;
    }
    for (const ir::Value* input : node.inputs()) {
        if (input != nullptr && input->kind() != ir::ValueKind::kConstant) {
            return false;
        }
    }
    for (const ir::Value* output : node.outputs()) {
        if (!output->type().is_static()) {
            return false;
        }
    }
    return true;
}

}  // namespace

Result<std::vector<Tensor>> evaluate_node(const ir::Node& node, const backend::Backend& evaluator) {
    II_ASSIGN_OR_RETURN(std::unique_ptr<backend::Kernel> kernel, evaluator.create_kernel(node));
    std::vector<backend::TensorView> inputs;
    std::vector<Shape> input_shapes;
    std::vector<Tensor> contiguous_inputs;
    for (const ir::Value* input : node.inputs()) {
        if (input == nullptr) {
            inputs.emplace_back();
            input_shapes.emplace_back();
            continue;
        }
        II_ASSIGN_OR_RETURN(Tensor data, input->data().contiguous());
        inputs.push_back(backend::TensorView{data.data(), data.dtype(), data.shape()});
        input_shapes.push_back(data.shape());
        contiguous_inputs.push_back(std::move(data));
    }
    std::vector<Tensor> results;
    std::vector<backend::TensorView> outputs;
    std::vector<Shape> output_shapes;
    for (const ir::Value* output : node.outputs()) {
        II_ASSIGN_OR_RETURN(Tensor result, Tensor::empty(output->dtype(), *output->type().static_shape()));
        outputs.push_back(backend::TensorView{result.data(), result.dtype(), result.shape()});
        output_shapes.push_back(result.shape());
        results.push_back(std::move(result));
    }
    II_ASSIGN_OR_RETURN(AlignedBuffer workspace,
                        AlignedBuffer::allocate(kernel->workspace_size(input_shapes, output_shapes)));
    InlineExecutor executor;
    backend::KernelContext context;
    context.inputs = inputs;
    context.outputs = outputs;
    context.workspace = std::span<std::byte>(static_cast<std::byte*>(workspace.data()), workspace.size());
    context.executor = &executor;
    II_TRY(kernel->run(context));
    return results;
}

Result<std::size_t> fold_constants(ir::Graph& graph, const backend::Backend& evaluator, const FoldOptions& options) {
    std::size_t folded = 0;
    std::vector<ir::Node*> doomed;
    for (const auto& node_ptr : graph.nodes()) {
        ir::Node& node = *node_ptr;
        if (!foldable(node) || !evaluator.supports(node).ok()) {
            continue;
        }
        std::size_t input_bytes = 0;
        for (const ir::Value* input : node.inputs()) {
            if (input != nullptr) {
                input_bytes += input->data().nbytes();
            }
        }
        std::size_t output_bytes = 0;
        for (const ir::Value* output : node.outputs()) {
            output_bytes += storage_size(output->dtype(), output->type().static_shape()->numel());
        }
        if (output_bytes > options.max_result_bytes && output_bytes > input_bytes) {
            continue;
        }
        Result<std::vector<Tensor>> results = evaluate_node(node, evaluator);
        if (!results.ok()) {
            // Leave the node in place: the same error will surface (with context) when the graph runs.
            continue;
        }
        for (int i = 0; i < node.num_outputs(); ++i) {
            ir::Value* output = node.output(i);
            ir::Value* constant =
                graph.add_constant(std::move((*results)[static_cast<std::size_t>(i)]), output->name());
            graph.replace_all_uses(output, constant);
        }
        doomed.push_back(&node);
        ++folded;
    }
    graph.erase_nodes(doomed);
    graph.erase_unused_constants();
    return folded;
}

Status decompose_node(ir::Graph& graph, ir::Node& node) {
    const ir::OpSchema& op = node.op();
    if (!op.is_composite()) {
        return errors::invalid_argument("'{}' is not a composite operation", op.name);
    }
    ir::Builder builder(graph);
    builder.set_insertion_point(&node);
    std::vector<ir::Value*> replacements;
    Status status = op.decompose(builder, node, replacements);
    if (!status.ok()) {
        return status.add_context(std::format("decomposing '{}'", op.name));
    }
    if (replacements.size() != static_cast<std::size_t>(node.num_outputs())) {
        return errors::internal("decomposition of '{}' returned {} values for {} outputs", op.name, replacements.size(),
                                node.num_outputs());
    }
    for (int i = 0; i < node.num_outputs(); ++i) {
        ir::Value* output = node.output(i);
        ir::Value* replacement = replacements[static_cast<std::size_t>(i)];
        if (replacement == nullptr || !(replacement->type() == output->type())) {
            return errors::internal("decomposition of '{}' produced {} instead of {}", op.name,
                                    replacement != nullptr ? replacement->type().to_string() : "nothing",
                                    output->type().to_string());
        }
        if (!output->name().empty() && replacement->name().empty()) {
            replacement->set_name(output->name());
        }
        graph.replace_all_uses(output, replacement);
    }
    ir::Node* doomed[] = {&node};
    graph.erase_nodes(doomed);
    return ok_status();
}

Result<std::size_t> decompose(ir::Graph& graph, FunctionRef<bool(const ir::Node&)> predicate) {
    std::size_t total = 0;
    // Each round decomposes the composites present at its start; decompositions are finite, so a small bound on
    // the nesting depth suffices.
    for (int round = 0; round < 16; ++round) {
        std::vector<ir::Node*> targets;
        for (const auto& node : graph.nodes()) {
            if (node->op().is_composite() && predicate(*node)) {
                targets.push_back(node.get());
            }
        }
        if (targets.empty()) {
            return total;
        }
        for (ir::Node* node : targets) {
            II_TRY(decompose_node(graph, *node));
            ++total;
        }
    }
    return errors::internal("decomposition did not converge");
}

}  // namespace ii::compiler
