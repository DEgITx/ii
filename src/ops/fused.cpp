// SPDX-License-Identifier: MIT
// Internal operations created by compiler passes.
#include "ops/fused.h"

#include "ir/builder.h"
#include "ir/text.h"
#include "ops/infer_util.h"
#include "ops/opset.h"

#include <format>
#include <unordered_map>

namespace ii::ops {

namespace {

using ir::AttrSpec;
using ir::InferContext;
using ir::InputSpec;
using ir::OpSchema;
using ir::OutputSpec;
using ir::TensorType;

Status infer_fused(const InferContext& context, std::vector<TensorType>& outputs) {
    if (context.attrs.get_string("program").empty()) {
        return errors::invalid_argument("'{}' requires a program", context.op.name);
    }
    const DType dtype = context.attrs.get_dtype("dtype");
    if (!is_plain(dtype)) {
        return errors::invalid_argument("'{}' has an invalid result dtype", context.op.name);
    }
    Dims shape;
    for (int i = 0; i < context.num_inputs(); ++i) {
        const TensorType* input = context.input(i);
        if (input == nullptr || !is_plain(input->dtype)) {
            return errors::invalid_argument("'{}' input {} must be a plain tensor", context.op.name, i);
        }
        II_ASSIGN_OR_RETURN(shape, broadcast(shape, input->shape));
    }
    outputs.push_back(TensorType{dtype, std::move(shape)});
    return ok_status();
}

Status decompose_fused(ir::Builder& b, const ir::Node& node, std::vector<ir::Value*>& outputs) {
    II_ASSIGN_OR_RETURN(const std::unique_ptr<ir::Graph> program, parse_fused_program(node));
    std::unordered_map<const ir::Value*, ir::Value*> mapping;
    for (std::size_t i = 0; i < program->inputs().size(); ++i) {
        mapping.emplace(program->inputs()[i], node.input(static_cast<int>(i)));
    }
    for (const auto& inner : program->nodes()) {
        std::vector<ir::Value*> inputs;
        for (ir::Value* value : inner->inputs()) {
            inputs.push_back(value == nullptr ? nullptr : mapping.at(value));
        }
        II_ASSIGN_OR_RETURN(ir::Node * created, b.create_node(inner->op(), std::move(inputs), inner->attrs()));
        for (int o = 0; o < inner->num_outputs(); ++o) {
            mapping.emplace(inner->output(o), created->output(o));
        }
    }
    outputs.push_back(mapping.at(program->outputs()[0].value));
    return ok_status();
}

}  // namespace

Result<std::unique_ptr<ir::Graph>> parse_fused_program(const ir::Node& node) {
    ir::ParseOptions options;
    options.verify = true;
    Result<std::unique_ptr<ir::Graph>> program = ir::parse_graph(node.attrs().get_string("program"), options);
    if (!program.ok()) {
        Status status = std::move(program).status();
        return status.add_context(std::format("program of '{}'", node.op_name()));
    }
    const ir::Graph& graph = **program;
    if (graph.inputs().size() != static_cast<std::size_t>(node.num_inputs()) || graph.outputs().size() != 1 ||
        !graph.states().empty()) {
        return errors::invalid_argument("the program of '{}' must have {} inputs, one output and no states",
                                        node.op_name(), node.num_inputs());
    }
    for (int i = 0; i < node.num_inputs(); ++i) {
        const ir::Value* inner = graph.inputs()[static_cast<std::size_t>(i)];
        if (node.input(i) == nullptr || inner->dtype() != node.input(i)->dtype() ||
            inner->rank() != node.input(i)->rank()) {
            return errors::invalid_argument("input {} of '{}' does not match its program", i, node.op_name());
        }
    }
    for (const auto& inner : graph.nodes()) {
        if (!inner->op().has_flag(ir::kOpElementwise) || inner->op().has_flag(ir::kOpInternal) ||
            inner->num_outputs() != 1) {
            return errors::invalid_argument("the program of '{}' contains the non-elementwise operation '{}'",
                                            node.op_name(), inner->op_name());
        }
    }
    if (graph.outputs()[0].value->dtype() != node.output()->dtype()) {
        return errors::invalid_argument("the program of '{}' produces {}, but the node declares {}", node.op_name(),
                                        to_string(graph.outputs()[0].value->dtype()),
                                        to_string(node.output()->dtype()));
    }
    return program;
}

void register_internal_ops(ir::OpRegistry& registry) {
    registry.add(OpSchema{
        .name = kFusedElementwise,
        .category = "internal",
        .doc = "A subgraph of elementwise operations evaluated in one pass (created by the fusion pass); program "
               "holds the subgraph in the IR text format.",
        .inputs = {InputSpec{"inputs", false, true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"program", AttrKind::kString, std::nullopt},
                  AttrSpec{"dtype", AttrKind::kDType, std::nullopt}},
        .infer = infer_fused,
        .decompose = decompose_fused,
        .flags = ir::kOpElementwise | ir::kOpInternal | ir::kOpNoConstantFold,
    });
}

}  // namespace ii::ops
