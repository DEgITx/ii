// SPDX-License-Identifier: MIT
// Model and Program: compilation of a model into an executable program.
#include "api/builder_impl.h"
#include "compiler/passes.h"
#include "core/assert.h"
#include "core/platform/mapped_file.h"
#include "formats/onnx/onnx_import.h"
#include "ir/op_schema.h"
#include "ir/text.h"
#include "ops/fused.h"
#include "runtime/runtime.h"

#include <algorithm>
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

/// Makes sure some allowed backend supports every node: composite nodes that none supports are decomposed (repeatedly,
/// a decomposition may produce composites too).
Status ensure_supported(ir::Graph& graph, const std::vector<const backend::Backend*>& backends) {
    for (int round = 0; round < 16; ++round) {
        std::vector<ir::Node*> to_decompose;
        for (const auto& node : graph.nodes()) {
            bool supported = false;
            std::string reasons;
            for (const backend::Backend* backend : backends) {
                Status status = backend->supports(*node);
                if (status.ok()) {
                    supported = true;
                    break;
                }
                reasons += std::format("\n  {}", status.message());
            }
            if (supported) {
                continue;
            }
            if (!node->op().is_composite()) {
                return errors::unsupported("no backend can execute '{}' (value {}):{}", node->op_name(),
                                           node->output()->type().to_string(), reasons);
            }
            to_decompose.push_back(node.get());
        }
        if (to_decompose.empty()) {
            return ok_status();
        }
        for (ir::Node* node : to_decompose) {
            II_TRY(compiler::decompose_node(graph, *node));
        }
    }
    return errors::internal("backend assignment did not converge");
}

/// Composite nodes the plan leaves on another device than all of their neighbours (producers and readers), where no
/// backend implements them but some backend runs. Decomposed, their primitives can run next to the neighbours
/// instead of costing copies there and back (a backend only has to implement primitives, P3).
std::vector<ir::Node*> stranded_composites(const ir::Graph& graph, const runtime::GraphPlan& plan,
                                           const std::vector<runtime::PlanBackend>& backends) {
    std::unordered_map<const ir::Node*, int> device_of;
    for (std::size_t n = 0; n < graph.nodes().size(); ++n) {
        device_of.emplace(graph.nodes()[n].get(), backends[static_cast<std::size_t>(plan.node_backend[n])].device);
    }
    std::vector<ir::Node*> stranded;
    for (const auto& node : graph.nodes()) {
        if (!node->op().is_composite()) {
            continue;
        }
        int neighbours = -1;  // the common device of the neighbours; -2: they differ
        auto see = [&](int device) { neighbours = neighbours == -1 || neighbours == device ? device : -2; };
        for (const ir::Value* input : node->inputs()) {
            if (input == nullptr || input->kind() == ir::ValueKind::kConstant ||
                input->kind() == ir::ValueKind::kState) {
                continue;
            }
            see(input->producer() != nullptr ? device_of.at(input->producer()) : 0);
        }
        for (const ir::Value* output : node->outputs()) {
            for (const ir::Use& use : output->uses()) {
                see(device_of.at(use.node));
            }
            if (graph.is_graph_result(output)) {
                see(0);
            }
        }
        if (neighbours < 0 || neighbours == device_of.at(node.get())) {
            continue;
        }
        bool reachable = false;  // a backend runs on the neighbours' device: the primitives could move there
        bool native = false;
        for (const runtime::PlanBackend& backend : backends) {
            reachable = reachable || backend.device == neighbours;
            native = native || (backend.device == neighbours && backend.backend->supports(*node).ok());
        }
        if (reachable && !native) {
            stranded.push_back(node.get());
        }
    }
    return stranded;
}

/// What the planner chose for a node.
struct NodeChoice {
    const backend::Backend* backend = nullptr;
    int device = 0;
    std::unique_ptr<backend::Kernel> kernel;
    double estimate_us = 0;
};

/// Inserts `transfer` nodes wherever a value is read on another device than the one holding it, and copies the
/// constants read by device kernels to the device (once per constant and device).
class TransferInserter {
public:
    TransferInserter(ir::Graph& graph, const std::vector<detail::ProgramDevice>& devices,
                     std::unordered_map<const ir::Value*, int>& value_device,
                     std::unordered_map<const ir::Node*, int>& transfer_source)
        : graph_(graph), devices_(devices), value_device_(value_device), transfer_source_(transfer_source) {}

    /// `value` on device `to`, copied before `before` (nullptr: at the end) if it lives elsewhere.
    Result<ir::Value*> on_device(ir::Value* value, int to, ir::Node* before) {
        if (value->kind() == ir::ValueKind::kConstant) {
            if (!value->data().device().is_host()) {
                return value;  // already a device copy
            }
            return to == 0 ? Result<ir::Value*>(value) : upload(value, to);
        }
        const int from = value_device_.at(value);
        if (from == to) {
            return value;
        }
        if (const auto it = copies_.find({value, to}); it != copies_.end()) {
            return it->second;
        }
        ir::Value* source = value;
        int source_device = from;
        if (from != 0 && to != 0) {  // between two devices: through the host
            II_ASSIGN_OR_RETURN(source, on_device(value, 0, before));
            source_device = 0;
        }
        Attributes attrs;
        attrs.set("from", to_string(devices_[static_cast<std::size_t>(source_device)].info.device));
        attrs.set("to", to_string(devices_[static_cast<std::size_t>(to)].info.device));
        const ir::TensorType type = value->type();
        ir::Node* node = graph_.create_node(ir::builtin_op(ops::kTransfer), {source}, std::move(attrs),
                                            std::span<const ir::TensorType>(&type, 1), before);
        ir::Value* copy = node->output();
        value_device_[copy] = to;
        transfer_source_[node] = source_device;
        copies_.emplace(std::make_pair(static_cast<const ir::Value*>(value), to), copy);
        return copy;
    }

private:
    Result<ir::Value*> upload(ir::Value* constant, int to) {
        if (const auto it = copies_.find({constant, to}); it != copies_.end()) {
            return it->second;
        }
        backend::DeviceApi* api = devices_[static_cast<std::size_t>(to)].api;
        II_CHECK(api != nullptr);
        std::unique_ptr<backend::Queue>& queue = queues_[to];
        if (!queue) {
            II_ASSIGN_OR_RETURN(queue, api->create_queue());
        }
        II_ASSIGN_OR_RETURN(const Tensor host, constant->data().contiguous());
        II_ASSIGN_OR_RETURN(std::shared_ptr<Buffer> buffer, api->allocate(std::max<std::size_t>(host.nbytes(), 1)));
        II_TRY(queue->copy_to_device(buffer->data(), host.data(), host.nbytes()));
        II_TRY(queue->synchronize());
        II_ASSIGN_OR_RETURN(
            Tensor data, Tensor::view(std::move(buffer), host.dtype(), host.shape(), contiguous_strides(host.shape())));
        ir::Value* copy = graph_.add_constant(std::move(data), constant->name());
        value_device_[copy] = to;
        copies_.emplace(std::make_pair(static_cast<const ir::Value*>(constant), to), copy);
        return copy;
    }

    struct PairHash {
        std::size_t operator()(const std::pair<const ir::Value*, int>& key) const noexcept {
            return std::hash<const void*>()(key.first) * 31 + static_cast<std::size_t>(key.second);
        }
    };

    ir::Graph& graph_;
    const std::vector<detail::ProgramDevice>& devices_;
    std::unordered_map<const ir::Value*, int>& value_device_;
    std::unordered_map<const ir::Node*, int>& transfer_source_;
    std::unordered_map<std::pair<const ir::Value*, int>, ir::Value*, PairHash> copies_;
    std::unordered_map<int, std::unique_ptr<backend::Queue>> queues_;
};

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

namespace {

std::string utf8_path(const std::filesystem::path& path) {
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
}

}  // namespace

Result<Model> Model::from_onnx(const Context& context, std::span<const std::byte> data, const LoadOptions& options) {
    formats::onnx::ImportOptions import;
    import.dims = options.dims;
    import.constant_inputs = options.constant_inputs;
    import.evaluator = &context.impl()->reference();
    Result<std::unique_ptr<ir::Graph>> graph = formats::onnx::import_model(data, nullptr, import);
    if (!graph.ok()) {
        Status status = graph.status();
        return status.add_context("ONNX import");
    }
    auto model = std::make_shared<detail::ModelImpl>(context);
    model->graph = std::move(*graph);
    return Model(std::move(model));
}

Result<Model> Model::load(const Context& context, const std::filesystem::path& path, const LoadOptions& options) {
    Result<platform::MappedFile> mapped = platform::MappedFile::open(path);
    if (!mapped.ok()) {
        return mapped.status();
    }
    auto file = std::make_shared<platform::MappedFile>(std::move(*mapped));
    const std::span<const std::byte> bytes = file->bytes();
    std::string extension = utf8_path(path.extension());
    for (char& c : extension) {
        c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    }
    const bool is_onnx = extension == ".onnx" || (extension != ".txt" && formats::onnx::looks_like_onnx(bytes));
    if (is_onnx) {
        formats::onnx::ImportOptions import;
        import.base_dir = path.parent_path();
        if (import.base_dir.empty()) {
            import.base_dir = ".";
        }
        import.dims = options.dims;
        import.constant_inputs = options.constant_inputs;
        import.evaluator = &context.impl()->reference();
        Result<std::unique_ptr<ir::Graph>> graph =
            formats::onnx::import_model(bytes, std::static_pointer_cast<const void>(file), import);
        if (!graph.ok()) {
            Status status = graph.status();
            return status.add_context(std::format("loading '{}'", utf8_path(path)));
        }
        auto model = std::make_shared<detail::ModelImpl>(context);
        model->graph = std::move(*graph);
        return Model(std::move(model));
    }
    const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    Result<Model> model = from_text(context, text);
    if (!model.ok()) {
        Status status = model.status();
        return status.add_context(std::format("loading '{}'", utf8_path(path)));
    }
    return model;
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
    if (options.fuse) {
        compiler::eliminate_common_subexpressions(graph);
        II_TRY(check("common subexpression elimination"));
    }
    if (options.fold_constants) {
        II_TRY(compiler::fold_constants(graph, context.reference()));
        compiler::eliminate_dead_code(graph);
        II_TRY(check("constant folding"));
    }
    if (options.fuse && !options.decompose_composites) {
        II_TRY(compiler::fuse_patterns(graph));
        II_TRY(check("pattern fusion"));
        II_TRY(compiler::fuse_elementwise(graph));
        II_TRY(check("elementwise fusion"));
        II_TRY(compiler::fuse_conv_epilogue(graph));
        II_TRY(check("convolution epilogue fusion"));
    }
    II_TRY(ensure_supported(graph, backends));
    compiler::eliminate_dead_code(graph);
    II_TRY(check("decomposition of unsupported operations"));

    // --- Devices: the host, then every other device of the allowed backends ---
    std::vector<runtime::PlanDevice> plan_devices;
    std::vector<runtime::PlanBackend> plan_backends;
    backend::DeviceInfo host = backend::host_device_info(context.executor->concurrency(), 8);
    for (const backend::Backend* backend : backends) {
        if (backend->device().is_host()) {
            host = backend->device_info();
            break;
        }
    }
    program->devices.push_back(detail::ProgramDevice{host, nullptr});
    plan_devices.push_back(runtime::PlanDevice{host, 0});
    for (const backend::Backend* backend : backends) {
        const Device device = backend->device();
        int index = -1;
        for (std::size_t d = 0; d < program->devices.size(); ++d) {
            if (program->devices[d].info.device == device) {
                index = static_cast<int>(d);
            }
        }
        if (index < 0) {
            if (backend->device_api() == nullptr) {
                return errors::internal("backend '{}' runs on {} but has no device API", backend->name(),
                                        to_string(device));
            }
            index = static_cast<int>(program->devices.size());
            program->devices.push_back(detail::ProgramDevice{backend->device_info(), backend->device_api()});
            std::uint64_t limit = backend->device_info().memory_bytes;
            for (const auto& [name, bytes] : options.device_memory) {
                if (name == to_string(device)) {
                    limit = bytes;
                }
            }
            plan_devices.push_back(runtime::PlanDevice{backend->device_info(), limit});
        }
        plan_backends.push_back(runtime::PlanBackend{backend, index});
    }
    for (const auto& [name, bytes] : options.device_memory) {
        bool known = false;
        for (const detail::ProgramDevice& device : program->devices) {
            known = known || to_string(device.info.device) == name;
        }
        if (!known) {
            return errors::not_found("device_memory names '{}', which no allowed backend runs on", name);
        }
    }

    // --- Plan ---
    runtime::PlannerOptions planner_options;
    planner_options.dims = options.plan_dims;
    for (int round = 0;; ++round) {
        II_ASSIGN_OR_RETURN(program->plan, runtime::plan_graph(graph, plan_backends, plan_devices, planner_options));
        if (plan_devices.size() == 1 || round == 4) {
            break;
        }
        const std::vector<ir::Node*> stranded = stranded_composites(graph, program->plan, plan_backends);
        if (stranded.empty()) {
            break;
        }
        for (ir::Node* node : stranded) {
            II_TRY(compiler::decompose_node(graph, *node));
        }
        compiler::eliminate_dead_code(graph);
        II_TRY(ensure_supported(graph, backends));
        II_TRY(check("decomposition of stranded composites"));
    }

    // --- Kernels, created before transfers are inserted so that they see the constants they may capture ---
    std::unordered_map<const ir::Node*, NodeChoice> choices;
    std::vector<ir::Node*> planned_nodes;
    for (std::size_t n = 0; n < graph.nodes().size(); ++n) {
        ir::Node* node = graph.nodes()[n].get();
        const runtime::PlanBackend& chosen = plan_backends[static_cast<std::size_t>(program->plan.node_backend[n])];
        NodeChoice choice;
        choice.backend = chosen.backend;
        choice.device = chosen.device;
        choice.estimate_us = program->plan.node_us[n];
        II_ASSIGN_OR_RETURN(choice.kernel, chosen.backend->create_kernel(*node));
        choices.emplace(node, std::move(choice));
        planned_nodes.push_back(node);
    }

    // --- Copies between devices ---
    std::unordered_map<const ir::Value*, int> value_device;
    std::unordered_map<const ir::Node*, int> transfer_source;
    for (const ir::Value* input : graph.inputs()) {
        value_device[input] = 0;
    }
    for (std::size_t s = 0; s < graph.states().size(); ++s) {
        value_device[graph.states()[s].state] = program->plan.state_device[s];
    }
    for (ir::Node* node : planned_nodes) {
        for (const ir::Value* output : node->outputs()) {
            value_device[output] = choices.at(node).device;
        }
    }
    if (program->devices.size() > 1) {
        TransferInserter inserter(graph, program->devices, value_device, transfer_source);
        for (ir::Node* node : planned_nodes) {
            const NodeChoice& choice = choices.at(node);
            for (int i = 0; i < node->num_inputs(); ++i) {
                ir::Value* input = node->input(i);
                if (input == nullptr ||
                    (input->kind() == ir::ValueKind::kConstant && choice.kernel->captures_input(i))) {
                    continue;
                }
                II_ASSIGN_OR_RETURN(ir::Value * local, inserter.on_device(input, choice.device, node));
                if (local != input) {
                    node->set_input(i, local);
                }
            }
        }
        for (std::size_t o = 0; o < graph.outputs().size(); ++o) {
            II_ASSIGN_OR_RETURN(ir::Value * on_host, inserter.on_device(graph.outputs()[o].value, 0, nullptr));
            graph.set_output(o, on_host);
        }
        for (std::size_t s = 0; s < graph.states().size(); ++s) {
            const ir::StateBinding binding = graph.states()[s];
            if (binding.update != nullptr) {
                II_ASSIGN_OR_RETURN(ir::Value * update,
                                    inserter.on_device(binding.update, program->plan.state_device[s], nullptr));
                if (update != binding.update) {
                    graph.set_state_update(binding.state, update);
                }
            }
        }
        graph.erase_unused_constants();
        II_TRY(check("insertion of transfers"));
    }

    // --- Value table ---
    std::unordered_map<const ir::Value*, int> index;
    for (const auto& value : graph.values()) {
        detail::ValueInfo info;
        info.kind = value->kind();
        info.type = value->type();
        info.name = value->name();
        if (value->kind() == ir::ValueKind::kConstant) {
            II_ASSIGN_OR_RETURN(info.constant, value->data().contiguous());
            const auto it = value_device.find(value.get());
            info.device = it != value_device.end() ? it->second : 0;
        } else {
            info.device = value_device.at(value.get());
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
    for (std::size_t s = 0; s < graph.states().size(); ++s) {
        const ir::StateBinding& binding = graph.states()[s];
        detail::StateInfo state;
        state.name = binding.state->name();
        state.value = index.at(binding.state);
        state.update = binding.update != nullptr ? index.at(binding.update) : -1;
        state.initial = binding.state->data();
        state.device = program->plan.state_device[s];
        program->values[static_cast<std::size_t>(state.value)].state_index = static_cast<int>(program->states.size());
        program->states.push_back(std::move(state));
    }

    // --- Steps ---
    for (const auto& node : graph.nodes()) {
        detail::Step step;
        step.node = node.get();
        if (const auto source = transfer_source.find(node.get()); source != transfer_source.end()) {
            step.transfer = true;
            step.transfer_from = source->second;
            step.device = value_device.at(node->output());
        } else {
            NodeChoice& choice = choices.at(node.get());
            step.backend = choice.backend;
            step.kernel = std::move(choice.kernel);
            step.device = choice.device;
            step.estimate_us = choice.estimate_us;
        }
        for (int i = 0; i < node->num_inputs(); ++i) {
            const ir::Value* input = node->input(i);
            const int v = input == nullptr ? -1 : index.at(input);
            step.inputs.push_back(v);
            if (v >= 0 && !step.transfer) {
                const detail::ValueInfo& info = program->values[static_cast<std::size_t>(v)];
                const bool captured = info.kind == ir::ValueKind::kConstant && step.kernel->captures_input(i);
                II_CHECK(captured || info.device == step.device, "input {} of '{}' is on device {}, the step on {}", i,
                         node->op_name(), info.device, step.device);
            }
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

namespace {

std::string format_bytes(std::uint64_t bytes) {
    if (bytes >= (std::uint64_t{1} << 30)) {
        return std::format("{:.2f} GiB", static_cast<double>(bytes) / (1 << 30));
    }
    return std::format("{:.1f} MiB", static_cast<double>(bytes) / (1 << 20));
}

std::string format_us(double us) {
    return us >= 1000 ? std::format("{:.2f} ms", us / 1000) : std::format("{:.1f} us", us);
}

/// "ii-cpu" for kernels, "transfer cpu -> test:0" for copies.
std::string step_executor(const detail::ProgramImpl& program, const detail::Step& step) {
    if (step.is_transfer()) {
        return std::format("transfer {} -> {}",
                           to_string(program.devices[static_cast<std::size_t>(step.transfer_from)].info.device),
                           to_string(program.devices[static_cast<std::size_t>(step.device)].info.device));
    }
    return std::string(step.backend->name());
}

}  // namespace

std::string Program::describe() const {
    const detail::ProgramImpl& p = *impl_;
    std::string text = std::format("{} steps\n", p.steps.size());
    for (std::size_t i = 0; i < p.steps.size(); ++i) {
        const detail::Step& step = p.steps[i];
        std::string outputs;
        for (const int v : step.outputs) {
            if (!outputs.empty()) {
                outputs += ", ";
            }
            outputs += p.values[static_cast<std::size_t>(v)].type.to_string();
        }
        std::string executor = step_executor(p, step);
        if (!step.is_transfer() && p.devices.size() > 1) {
            executor += " @ " + to_string(p.devices[static_cast<std::size_t>(step.device)].info.device);
        }
        text += std::format("  {:>4}  {:<16} {:<24} {}\n", i, step.node->op_name(), outputs, executor);
    }
    return text;
}

std::string Program::describe_plan() const {
    const detail::ProgramImpl& p = *impl_;
    const runtime::GraphPlan& plan = p.plan;
    std::string text = "Devices:\n";
    for (std::size_t d = 0; d < p.devices.size(); ++d) {
        const backend::DeviceInfo& info = p.devices[d].info;
        text += std::format("  {:<8} {}", to_string(info.device), info.name);
        if (info.memory_bytes != 0) {
            text += std::format(", {}", format_bytes(info.memory_bytes));
        }
        text += std::format(" (estimates: {:.0f} GFLOP/s, {:.0f} GB/s", info.gflops, info.bandwidth_gbs);
        if (d != 0) {
            text += std::format(", link {:.0f} GB/s", info.link_gbs);
        }
        text += ")\n";
    }

    text += "Partitions:\n";
    text += std::format("  {:>4}  {:<11} {:<26} {:>5}  {}\n", "#", "steps", "executor", "ops", "estimate");
    int partition = 0;
    for (std::size_t begin = 0; begin < p.steps.size();) {
        const std::string executor = step_executor(p, p.steps[begin]);
        std::size_t end = begin + 1;
        double estimate = p.steps[begin].estimate_us;
        while (end < p.steps.size() && !p.steps[begin].is_transfer() && step_executor(p, p.steps[end]) == executor) {
            estimate += p.steps[end].estimate_us;
            ++end;
        }
        const std::string range = end - begin == 1 ? std::to_string(begin) : std::format("{}-{}", begin, end - 1);
        std::string label = executor;
        if (!p.steps[begin].is_transfer()) {
            label += " @ " + to_string(p.devices[static_cast<std::size_t>(p.steps[begin].device)].info.device);
        }
        text += std::format("  {:>4}  {:<11} {:<26} {:>5}  {}\n", partition++, range, label, end - begin,
                            p.steps[begin].is_transfer() ? std::string("-") : format_us(estimate));
        begin = end;
    }
    if (plan.transfers > 0) {
        text += std::format("Copies between devices: {} ({}), estimated {}\n", plan.transfers,
                            format_bytes(plan.transfer_bytes), format_us(plan.transfer_us));
    }
    text += std::format("Estimated time: {}", format_us(plan.total_us()));
    if (p.devices.size() > 1) {
        text += " (";
        for (std::size_t d = 0; d < plan.device_us.size(); ++d) {
            text += std::format("{}{}: {}", d == 0 ? "" : ", ", to_string(p.devices[d].info.device),
                                format_us(plan.device_us[d]));
        }
        text += std::format(", copies: {})", format_us(plan.transfer_us));
    }
    text += "\n";
    if (p.devices.size() > 1) {
        text += "Memory (estimates):\n";
        for (std::size_t d = 1; d < p.devices.size() && d < plan.usage.size(); ++d) {
            const runtime::DeviceUsage& usage = plan.usage[d];
            text +=
                std::format("  {:<8} weights {}, states {}, activations {}\n", to_string(p.devices[d].info.device),
                            format_bytes(usage.weights), format_bytes(usage.states), format_bytes(usage.activations));
        }
    }
    for (const std::string& note : plan.notes) {
        text += "Note: " + note + "\n";
    }
    if (!plan.declines.empty()) {
        text += "Declined:\n";
        for (const runtime::Decline& decline : plan.declines) {
            text += std::format("  {} ({} node{})\n", decline.reason, decline.nodes, decline.nodes == 1 ? "" : "s");
        }
    }
    return text;
}

}  // namespace ii
