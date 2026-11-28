// SPDX-License-Identifier: MIT
// Internal operations created by compiler passes.
#include "ops/fused.h"

#include "ir/builder.h"
#include "ir/text.h"
#include "ops/infer_util.h"
#include "ops/opset.h"

#include <algorithm>
#include <format>
#include <mutex>
#include <optional>
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

// --- subgraph ---

/// Parsed programs by text. Shape inference and cost estimates of a `subgraph` node parse its program; nodes and
/// repeated bindings share one parse. Never destroyed (graphs refer to the operation registry).
Result<std::shared_ptr<const ir::Graph>> cached_program(std::string_view text) {
    static auto* const mutex = new std::mutex();
    static auto* const cache = new std::unordered_map<std::string, std::shared_ptr<const ir::Graph>>();
    {
        const std::lock_guard lock(*mutex);
        if (const auto it = cache->find(std::string(text)); it != cache->end()) {
            return it->second;
        }
    }
    ir::ParseOptions options;
    options.verify = true;
    Result<std::unique_ptr<ir::Graph>> parsed = ir::parse_graph(text, options);
    if (!parsed.ok()) {
        Status status = std::move(parsed).status();
        return status.add_context("program of 'subgraph'");
    }
    std::shared_ptr<const ir::Graph> program = std::move(*parsed);
    if (!program->states().empty()) {
        return errors::invalid_argument("the program of 'subgraph' must not have states");
    }
    const std::lock_guard lock(*mutex);
    if (cache->size() >= 64) {
        cache->clear();
    }
    cache->emplace(std::string(text), program);
    return program;
}

/// Matches the program's inputs with the node's input types: each symbol of the program that is a whole dimension
/// of an input gets the node's dimension there.
Result<std::unordered_map<std::string, Dim>> bind_symbols(const ir::Graph& program,
                                                          std::span<const TensorType* const> inputs) {
    if (program.inputs().size() != inputs.size()) {
        return errors::invalid_argument("'subgraph' has {} inputs, its program {}", inputs.size(),
                                        program.inputs().size());
    }
    std::unordered_map<std::string, Dim> symbols;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const ir::TensorType& inner = program.inputs()[i]->type();
        const TensorType* outer = inputs[i];
        if (outer == nullptr || outer->dtype != inner.dtype || outer->rank() != inner.rank()) {
            return errors::invalid_argument("input {} of 'subgraph' ({}) does not match its program ({})", i,
                                            outer == nullptr ? std::string("absent") : outer->to_string(),
                                            inner.to_string());
        }
        for (std::size_t d = 0; d < inner.shape.size(); ++d) {
            const Dim& dim = inner.shape[d];
            const Dim& actual = outer->shape[d];
            if (const std::string_view name = dim.symbol_name(); !name.empty()) {
                const auto [it, inserted] = symbols.emplace(std::string(name), actual);
                if (!inserted && !(it->second == actual)) {
                    return errors::invalid_argument("'subgraph' binds {} to both {} and {}", name,
                                                    it->second.to_string(), actual.to_string());
                }
            } else if (dim.is_constant() && actual.is_constant() && dim.value() != actual.value()) {
                return errors::invalid_argument("input {} of 'subgraph' is {}, its program expects {}", i,
                                                outer->to_string(), inner.to_string());
            }
        }
    }
    return symbols;
}

Status infer_subgraph(const InferContext& context, std::vector<TensorType>& outputs) {
    II_ASSIGN_OR_RETURN(const std::shared_ptr<const ir::Graph> program,
                        cached_program(context.attrs.get_string("program")));
    II_ASSIGN_OR_RETURN(const auto symbols, bind_symbols(*program, context.inputs));
    for (const ir::GraphOutput& output : program->outputs()) {
        TensorType type{output.value->dtype(), {}};
        for (const Dim& dim : output.value->shape()) {
            bool unknown = false;
            Dim bound = dim.substitute([&](std::string_view name) -> std::optional<Dim> {
                if (const auto it = symbols.find(std::string(name)); it != symbols.end()) {
                    return it->second;
                }
                unknown = true;
                return std::nullopt;
            });
            if (unknown) {
                return errors::invalid_argument("output {} of 'subgraph' uses symbols its inputs do not bind",
                                                output.value->type().to_string());
            }
            type.shape.push_back(std::move(bound));
        }
        outputs.push_back(std::move(type));
    }
    return ok_status();
}

ir::Cost cost_subgraph(const ir::CostContext& context) {
    Result<std::shared_ptr<const ir::Graph>> program = cached_program(context.node.attrs().get_string("program"));
    if (!program.ok()) {
        return ir::Cost{};
    }
    const ir::Graph& graph = **program;
    std::vector<std::pair<std::string, std::int64_t>> symbols;
    if (Result<std::vector<std::pair<std::string, std::int64_t>>> bound = subgraph_symbols(graph, context.inputs);
        bound.ok()) {
        symbols = std::move(*bound);
    }
    auto shape_of = [&](const ir::Value* value) {
        Shape shape;
        for (const Dim& dim : value->shape()) {
            const std::optional<std::int64_t> v =
                dim.evaluate([&](std::string_view name) -> std::optional<std::int64_t> {
                    for (const auto& [symbol, number] : symbols) {
                        if (symbol == name) {
                            return number;
                        }
                    }
                    return 1;
                });
            shape.push_back(std::max<std::int64_t>(v.value_or(1), 0));
        }
        return shape;
    };
    ir::Cost total;
    std::vector<Shape> in;
    std::vector<Shape> out;
    for (const auto& inner : graph.nodes()) {
        in.clear();
        out.clear();
        for (const ir::Value* value : inner->inputs()) {
            in.push_back(value == nullptr ? Shape() : shape_of(value));
        }
        for (const ir::Value* value : inner->outputs()) {
            out.push_back(shape_of(value));
        }
        const ir::Cost cost = ir::estimate_cost(ir::CostContext{*inner, in, out});
        total.flops += cost.flops;
        total.bytes += cost.bytes;
    }
    return total;
}

Status decompose_subgraph(ir::Builder& b, const ir::Node& node, std::vector<ir::Value*>& outputs) {
    II_ASSIGN_OR_RETURN(const std::shared_ptr<const ir::Graph> program, parse_subgraph_program(node));
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
    for (const ir::GraphOutput& output : program->outputs()) {
        outputs.push_back(mapping.at(output.value));
    }
    return ok_status();
}

}  // namespace

Result<std::shared_ptr<const ir::Graph>> parse_subgraph_program(const ir::Node& node) {
    II_ASSIGN_OR_RETURN(std::shared_ptr<const ir::Graph> program, cached_program(node.attrs().get_string("program")));
    if (program->inputs().size() != static_cast<std::size_t>(node.num_inputs()) ||
        program->outputs().size() != static_cast<std::size_t>(node.num_outputs())) {
        return errors::invalid_argument("the program of 'subgraph' must have {} inputs and {} outputs",
                                        node.num_inputs(), node.num_outputs());
    }
    for (int i = 0; i < node.num_inputs(); ++i) {
        const ir::Value* outer = node.input(i);
        if (outer == nullptr || outer->dtype() != program->inputs()[static_cast<std::size_t>(i)]->dtype()) {
            return errors::invalid_argument("input {} of 'subgraph' does not match its program", i);
        }
    }
    return program;
}

Result<SubgraphProgram> subgraph_program(const ir::Node& node) {
    II_ASSIGN_OR_RETURN(const std::shared_ptr<const ir::Graph> program, parse_subgraph_program(node));
    SubgraphProgram out;
    out.graph = std::make_unique<ir::Graph>();
    ir::Graph& graph = *out.graph;
    graph.set_name(program->name());
    std::unordered_map<const ir::Value*, ir::Value*> mapping;
    for (std::size_t i = 0; i < program->inputs().size(); ++i) {
        const ir::Value* inner = program->inputs()[i];
        const ir::Value* outer = node.input(static_cast<int>(i));
        ir::Value* copy = nullptr;
        if (outer->kind() == ir::ValueKind::kConstant) {
            copy = graph.add_constant(outer->data(), inner->name());
        } else {
            copy = graph.add_input(inner->name(), inner->type());
            out.inputs.push_back(static_cast<int>(i));
        }
        mapping.emplace(inner, copy);
    }
    for (const auto& inner : program->nodes()) {
        std::vector<ir::Value*> inputs;
        for (ir::Value* value : inner->inputs()) {
            inputs.push_back(value == nullptr ? nullptr : mapping.at(value));
        }
        std::vector<ir::TensorType> types;
        for (const ir::Value* output : inner->outputs()) {
            types.push_back(output->type());
        }
        ir::Node* created = graph.create_node(inner->op(), std::move(inputs), inner->attrs(), types);
        created->set_name(inner->name());
        for (int o = 0; o < inner->num_outputs(); ++o) {
            mapping.emplace(inner->output(o), created->output(o));
        }
    }
    for (const ir::GraphOutput& output : program->outputs()) {
        graph.add_output(output.name, mapping.at(output.value));
    }
    return out;
}

Result<std::vector<std::pair<std::string, std::int64_t>>> subgraph_symbols(const ir::Graph& program,
                                                                           std::span<const Shape> inputs) {
    std::vector<std::pair<std::string, std::int64_t>> symbols;
    for (std::size_t i = 0; i < program.inputs().size() && i < inputs.size(); ++i) {
        const ir::Value* inner = program.inputs()[i];
        if (inner->rank() != inputs[i].rank()) {
            return errors::invalid_argument("input {} of 'subgraph' has rank {}, its program expects {}", i,
                                            inputs[i].rank(), inner->rank());
        }
        for (int d = 0; d < inner->rank(); ++d) {
            const std::string_view name = inner->shape()[static_cast<std::size_t>(d)].symbol_name();
            if (name.empty()) {
                continue;
            }
            const std::int64_t value = inputs[i][d];
            bool known = false;
            for (const auto& [symbol, number] : symbols) {
                if (symbol == name) {
                    known = true;
                    if (number != value) {
                        return errors::invalid_argument("'subgraph' binds {} to both {} and {}", name, number, value);
                    }
                }
            }
            if (!known) {
                symbols.emplace_back(std::string(name), value);
            }
        }
    }
    return symbols;
}

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
    registry.add(OpSchema{
        .name = kSubgraph,
        .category = "internal",
        .doc = "A part of the graph compiled into one kernel by a graph backend (created by the compiler after "
               "planning); program holds the part in the IR text format, backend names the backend.",
        .inputs = {InputSpec{"inputs", true, true}},
        .outputs = {OutputSpec{"outputs", true}},
        .attrs = {AttrSpec{"program", AttrKind::kString, std::nullopt},
                  AttrSpec{"backend", AttrKind::kString, std::string()}},
        .infer = infer_subgraph,
        .decompose = decompose_subgraph,
        .cost = cost_subgraph,
        .flags = ir::kOpInternal | ir::kOpNoConstantFold,
    });
}

}  // namespace ii::ops
