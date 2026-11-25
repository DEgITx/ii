// SPDX-License-Identifier: MIT
// Model and Program: compilation of a model into an executable program.
#include "api/builder_impl.h"
#include "compiler/passes.h"
#include "ir/op_schema.h"
#include "ir/text.h"
#include "runtime/runtime.h"

#include <format>
#include <unordered_map>

namespace ii {

namespace {

TensorInfo info_of(std::string name, const ir::TensorType& type) {
    return TensorInfo{std::move(name), type.dtype, type.shape};
}

std::vector<TensorInfo> graph_inputs(const ir::Graph& graph) {
    std::vector<TensorInfo> result;
    for (const ir::Value* input : graph.inputs()) {
        result.push_back(info_of(input->name(), input->type()));
    }
    return result;
}

std::vector<TensorInfo> graph_outputs(const ir::Graph& graph) {
    std::vector<TensorInfo> result;
    for (const ir::GraphOutput& output : graph.outputs()) {
        result.push_back(info_of(output.name, output.value->type()));
    }
    return result;
}

bool has_symbolic_dims(const Attributes& attrs) {
    for (const auto& [name, value] : attrs) {
        if (const Dims* dims = std::get_if<Dims>(&value)) {
            for (const Dim& dim : *dims) {
                if (!dim.is_constant()) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// Assigns every node to the first allowed backend that supports it. Composite nodes that no backend supports are
/// decomposed and the assignment is repeated.
Result<std::unordered_map<const ir::Node*, const backend::Backend*>>
assign_backends(ir::Graph& graph, const std::vector<const backend::Backend*>& backends) {
    for (int round = 0; round < 16; ++round) {
        std::unordered_map<const ir::Node*, const backend::Backend*> assignment;
        std::vector<ir::Node*> to_decompose;
        for (const auto& node : graph.nodes()) {
            const backend::Backend* chosen = nullptr;
            std::string reasons;
            for (const backend::Backend* backend : backends) {
                Status status = backend->supports(*node);
                if (status.ok()) {
                    chosen = backend;
                    break;
                }
                reasons += std::format("\n  {}", status.message());
            }
            if (chosen != nullptr) {
                assignment.emplace(node.get(), chosen);
            } else if (node->op().is_composite()) {
                to_decompose.push_back(node.get());
            } else {
                return errors::unsupported("no backend can execute '{}' (value {}):{}", node->op_name(),
                                           node->output()->type().to_string(), reasons);
            }
        }
        if (to_decompose.empty()) {
            return assignment;
        }
        for (ir::Node* node : to_decompose) {
            II_TRY(compiler::decompose_node(graph, *node));
        }
    }
    return errors::internal("backend assignment did not converge");
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Model

Result<Model> Model::from_builder(const Context& context, GraphBuilder&& builder) {
    detail::GraphBuilderImpl& impl = builder.impl();
    if (!impl.status.ok()) {
        Status status = impl.status;
        return status.add_context("model builder");
    }
    if (Status status = impl.graph->verify(); !status.ok()) {
        return status.add_context("model builder produced an invalid graph");
    }
    auto model = std::make_shared<detail::ModelImpl>(context);
    model->graph = std::move(impl.graph);
    impl.graph = std::make_unique<ir::Graph>();
    impl.values.clear();
    return Model(std::move(model));
}

Result<Model> Model::from_text(const Context& context, std::string_view text) {
    II_ASSIGN_OR_RETURN(std::unique_ptr<ir::Graph> graph, ir::parse_graph(text));
    auto model = std::make_shared<detail::ModelImpl>(context);
    model->graph = std::move(graph);
    return Model(std::move(model));
}

std::vector<TensorInfo> Model::inputs() const {
    return graph_inputs(*impl_->graph);
}

std::vector<TensorInfo> Model::outputs() const {
    return graph_outputs(*impl_->graph);
}

std::vector<TensorInfo> Model::states() const {
    std::vector<TensorInfo> result;
    for (const ir::StateBinding& binding : impl_->graph->states()) {
        result.push_back(info_of(binding.state->name(), binding.state->type()));
    }
    return result;
}

std::string Model::to_text(std::int64_t max_constant_elements) const {
    return impl_->graph->to_text(ir::PrintOptions{max_constant_elements});
}

const Context& Model::context() const noexcept {
    return impl_->context;
}

Result<Program> Model::compile(const CompileOptions& options) const {
    const detail::ContextImpl& context = *impl_->context.impl();
    std::vector<const backend::Backend*> backends;
    if (options.backends.empty()) {
        for (const auto& backend : context.backends) {
            backends.push_back(backend.get());
        }
    } else {
        for (const std::string& name : options.backends) {
            const backend::Backend* backend = context.find_backend(name);
            if (backend == nullptr) {
                return errors::not_found("unknown backend '{}'", name);
            }
            backends.push_back(backend);
        }
    }

    auto program = std::make_shared<detail::ProgramImpl>(impl_->context);
    program->graph = impl_->graph->clone();
    ir::Graph& graph = *program->graph;

    // --- Passes ---
    auto check = [&](std::string_view pass) -> Status {
#if !defined(NDEBUG)
        if (Status status = graph.verify(); !status.ok()) {
            return status.add_context(std::format("after pass '{}'", pass));
        }
#else
        (void)pass;
#endif
        return ok_status();
    };
    compiler::eliminate_dead_code(graph);
    II_TRY(check("dead code elimination"));
    if (options.decompose_composites) {
        II_TRY(compiler::decompose(graph, [](const ir::Node&) { return true; }));
        II_TRY(check("decomposition"));
    }
    if (options.fold_constants) {
        II_TRY(compiler::fold_constants(graph, context.reference()));
        compiler::eliminate_dead_code(graph);
        II_TRY(check("constant folding"));
    }
    II_ASSIGN_OR_RETURN(const auto assignment, assign_backends(graph, backends));
    compiler::eliminate_dead_code(graph);
    II_TRY(check("backend assignment"));

    // --- Value table ---
    std::unordered_map<const ir::Value*, int> index;
    for (const auto& value : graph.values()) {
        detail::ValueInfo info;
        info.kind = value->kind();
        info.type = value->type();
        info.name = value->name();
        if (value->kind() == ir::ValueKind::kConstant) {
            II_ASSIGN_OR_RETURN(info.constant, value->data().contiguous());
        }
        index.emplace(value.get(), static_cast<int>(program->values.size()));
        program->values.push_back(std::move(info));
    }
    for (const ir::Value* input : graph.inputs()) {
        const int v = index.at(input);
        program->values[static_cast<std::size_t>(v)].input_index = static_cast<int>(program->inputs.size());
        program->inputs.push_back(v);
        program->input_names.push_back(input->name());
    }
    for (const ir::StateBinding& binding : graph.states()) {
        detail::StateInfo state;
        state.name = binding.state->name();
        state.value = index.at(binding.state);
        state.update = binding.update != nullptr ? index.at(binding.update) : -1;
        state.initial = binding.state->data();
        program->values[static_cast<std::size_t>(state.value)].state_index = static_cast<int>(program->states.size());
        program->states.push_back(std::move(state));
    }

    // --- Steps ---
    for (const auto& node : graph.nodes()) {
        detail::Step step;
        step.node = node.get();
        step.backend = assignment.at(node.get());
        II_ASSIGN_OR_RETURN(step.kernel, step.backend->create_kernel(*node));
        for (const ir::Value* input : node->inputs()) {
            step.inputs.push_back(input == nullptr ? -1 : index.at(input));
        }
        for (const ir::Value* output : node->outputs()) {
            const int v = index.at(output);
            program->values[static_cast<std::size_t>(v)].producer = static_cast<int>(program->steps.size());
            step.outputs.push_back(v);
        }
        step.symbolic_attrs = has_symbolic_dims(node->attrs());
        const int step_index = static_cast<int>(program->steps.size());
        for (const int v : step.inputs) {
            if (v >= 0) {
                program->values[static_cast<std::size_t>(v)].last_use = step_index;
            }
        }
        program->steps.push_back(std::move(step));
    }
    const int end = static_cast<int>(program->steps.size());
    for (const ir::GraphOutput& output : graph.outputs()) {
        const int v = index.at(output.value);
        program->outputs.push_back(v);
        program->output_names.push_back(output.name);
        program->values[static_cast<std::size_t>(v)].last_use = end;
    }
    for (const detail::StateInfo& state : program->states) {
        if (state.update >= 0) {
            program->values[static_cast<std::size_t>(state.update)].last_use = end;
        }
    }
    return Program(std::move(program));
}

// ---------------------------------------------------------------------------------------------------------------
// Program

namespace detail {

int ProgramImpl::find_input(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < input_names.size(); ++i) {
        if (input_names[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int ProgramImpl::find_output(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < output_names.size(); ++i) {
        if (output_names[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int ProgramImpl::find_state(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (states[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

}  // namespace detail

std::vector<TensorInfo> Program::inputs() const {
    return graph_inputs(*impl_->graph);
}

std::vector<TensorInfo> Program::outputs() const {
    return graph_outputs(*impl_->graph);
}

std::string Program::to_text(std::int64_t max_constant_elements) const {
    return impl_->graph->to_text(ir::PrintOptions{max_constant_elements});
}

std::string Program::describe() const {
    std::string text = std::format("{} steps\n", impl_->steps.size());
    for (std::size_t i = 0; i < impl_->steps.size(); ++i) {
        const detail::Step& step = impl_->steps[i];
        std::string outputs;
        for (const int v : step.outputs) {
            if (!outputs.empty()) {
                outputs += ", ";
            }
            outputs += impl_->values[static_cast<std::size_t>(v)].type.to_string();
        }
        text += std::format("  {:>4}  {:<16} {:<24} {}\n", i, step.node->op_name(), outputs, step.backend->name());
    }
    return text;
}

}  // namespace ii
