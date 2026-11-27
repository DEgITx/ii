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

Status infer_fused_conv(const InferContext& context, std::vector<TensorType>& outputs) {
    if (context.num_inputs() < 2) {
        return errors::invalid_argument("'fused_conv' needs x and w");
    }
    const TensorType* conv_inputs[] = {context.input(0), context.input(1), context.input(2)};
    std::vector<TensorType> conv_outputs;
    II_TRY(ir::infer_types(
        ir::builtin_op("conv"),
        std::span<const TensorType* const>(conv_inputs, static_cast<std::size_t>(std::min(context.num_inputs(), 3))),
        conv_attributes(context.attrs), conv_outputs));
    const TensorType& result = conv_outputs.front();
    for (int i = 3; i < context.num_inputs(); ++i) {
        const TensorType* extra = context.input(i);
        if (extra == nullptr || !is_plain(extra->dtype)) {
            return errors::invalid_argument("'fused_conv' extra input {} must be a plain tensor", i);
        }
        II_ASSIGN_OR_RETURN(const Dims shape, broadcast(result.shape, extra->shape));
        if (!(shape == result.shape)) {
            return errors::invalid_argument("'fused_conv' extra input {} {} would enlarge the result {}", i,
                                            extra->to_string(), result.to_string());
        }
    }
    if (context.attrs.get_string("program").empty()) {
        return errors::invalid_argument("'fused_conv' requires a program");
    }
    outputs.push_back(TensorType{context.attrs.get_dtype("dtype"), result.shape});
    return ok_status();
}

Status decompose_fused_conv(ir::Builder& b, const ir::Node& node, std::vector<ir::Value*>& outputs) {
    II_ASSIGN_OR_RETURN(const std::unique_ptr<ir::Graph> program, parse_epilogue_program(node));
    std::vector<ir::Value*> conv_inputs = {node.input(0), node.input(1)};
    if (node.input(2) != nullptr) {
        conv_inputs.push_back(node.input(2));
    }
    II_ASSIGN_OR_RETURN(ir::Node * conv, b.create_node("conv", std::move(conv_inputs), conv_attributes(node.attrs())));
    std::unordered_map<const ir::Value*, ir::Value*> mapping;
    mapping.emplace(program->inputs()[0], conv->output());
    for (std::size_t i = 1; i < program->inputs().size(); ++i) {
        mapping.emplace(program->inputs()[i], node.input(static_cast<int>(i + 2)));
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

Attributes conv_attributes(const Attributes& fused) {
    Attributes attrs;
    for (const std::string_view name : {"strides", "dilations", "pads", "auto_pad", "groups"}) {
        if (const AttrValue* value = fused.find(name)) {
            attrs.set(name, *value);
        }
    }
    return attrs;
}

Result<std::unique_ptr<ir::Graph>> parse_epilogue_program(const ir::Node& node) {
    ir::ParseOptions options;
    options.verify = true;
    Result<std::unique_ptr<ir::Graph>> program = ir::parse_graph(node.attrs().get_string("program"), options);
    if (!program.ok()) {
        Status status = std::move(program).status();
        return status.add_context("program of 'fused_conv'");
    }
    const ir::Graph& graph = **program;
    const int extras = std::max(node.num_inputs() - 3, 0);
    if (graph.inputs().size() != static_cast<std::size_t>(1 + extras) || graph.outputs().size() != 1 ||
        !graph.states().empty()) {
        return errors::invalid_argument("the program of 'fused_conv' must have {} inputs and one output", 1 + extras);
    }
    for (int i = 0; i < extras; ++i) {
        const ir::Value* inner = graph.inputs()[static_cast<std::size_t>(i + 1)];
        const ir::Value* outer = node.input(i + 3);
        if (outer == nullptr || inner->dtype() != outer->dtype()) {
            return errors::invalid_argument("extra input {} of 'fused_conv' does not match its program", i);
        }
    }
    for (const auto& inner : graph.nodes()) {
        if (!inner->op().has_flag(ir::kOpElementwise) || inner->op().has_flag(ir::kOpInternal) ||
            inner->num_outputs() != 1) {
            return errors::invalid_argument("the program of 'fused_conv' contains the non-elementwise operation '{}'",
                                            inner->op_name());
        }
    }
    return program;
}

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

namespace {

Status infer_transfer(const InferContext& context, std::vector<TensorType>& outputs) {
    outputs.push_back(*context.input(0));
    return ok_status();
}

}  // namespace

void register_internal_ops(ir::OpRegistry& registry) {
    registry.add(OpSchema{
        .name = kTransfer,
        .category = "internal",
        .doc = "Copies a value from the memory of device `from` to device `to` (inserted by the planner between "
               "nodes on different devices; executed by the runtime).",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"from", AttrKind::kString, std::nullopt}, AttrSpec{"to", AttrKind::kString, std::nullopt}},
        .infer = infer_transfer,
        .flags = ir::kOpDataMovement | ir::kOpInternal | ir::kOpNoConstantFold,
    });
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
    registry.add(OpSchema{
        .name = kFusedConv,
        .category = "internal",
        .doc = "A convolution with an elementwise epilogue (created by the conv epilogue fusion pass); program holds "
               "the epilogue in the IR text format.",
        .inputs = {InputSpec{"x"}, InputSpec{"w"}, InputSpec{"bias", true}, InputSpec{"extras", true, true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"strides", AttrKind::kInts, std::vector<std::int64_t>{}},
                  AttrSpec{"dilations", AttrKind::kInts, std::vector<std::int64_t>{}},
                  AttrSpec{"pads", AttrKind::kInts, std::vector<std::int64_t>{}},
                  AttrSpec{"auto_pad", AttrKind::kString, std::string("none")},
                  AttrSpec{"groups", AttrKind::kInt, std::int64_t{1}},
                  AttrSpec{"program", AttrKind::kString, std::nullopt},
                  AttrSpec{"dtype", AttrKind::kDType, std::nullopt}},
        .infer = infer_fused_conv,
        .decompose = decompose_fused_conv,
        .flags = ir::kOpInternal | ir::kOpNoConstantFold,
    });
}

}  // namespace ii::ops
