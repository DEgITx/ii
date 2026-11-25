// SPDX-License-Identifier: MIT
// Session: binding of inputs, shape specialization, memory planning and execution.
#include "core/assert.h"
#include "ir/op_schema.h"
#include "runtime/memory_planner.h"
#include "runtime/runtime.h"

#include <chrono>
#include <cstring>
#include <format>

namespace ii {

namespace detail {

namespace {

using SymbolValues = std::vector<std::pair<std::string, std::int64_t>>;

std::optional<std::int64_t> lookup(const SymbolValues& symbols, std::string_view name) {
    for (const auto& [symbol, value] : symbols) {
        if (symbol == name) {
            return value;
        }
    }
    return std::nullopt;
}

std::optional<std::int64_t> evaluate(const Dim& dim, const SymbolValues& symbols) {
    return dim.evaluate([&](std::string_view name) { return lookup(symbols, name); });
}

/// Tries to determine the single unknown symbol of `dim` from `value` for affine expressions (`2*N + 1`).
std::optional<std::pair<std::string, std::int64_t>> solve(const Dim& dim, std::int64_t value,
                                                          const SymbolValues& symbols) {
    std::vector<std::string> unknown;
    for (std::string& symbol : dim.symbols()) {
        if (!lookup(symbols, symbol)) {
            unknown.push_back(std::move(symbol));
        }
    }
    if (unknown.size() != 1) {
        return std::nullopt;
    }
    auto at = [&](std::int64_t x) {
        SymbolValues trial = symbols;
        trial.emplace_back(unknown[0], x);
        return evaluate(dim, trial);
    };
    const std::optional<std::int64_t> e0 = at(0);
    const std::optional<std::int64_t> e1 = at(1);
    if (!e0 || !e1 || *e1 == *e0 || (value - *e0) % (*e1 - *e0) != 0) {
        return std::nullopt;
    }
    const std::int64_t x = (value - *e0) / (*e1 - *e0);
    if (at(x) != value) {
        return std::nullopt;  // not affine after all
    }
    return std::make_pair(unknown[0], x);
}

std::string shape_text(const Shape& shape) {
    return shape.to_string();
}

}  // namespace

Status SessionImpl::allocate_states() {
    const ProgramImpl& p = *program;
    Allocator& allocator = p.context.allocator();
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        const ValueInfo& info = p.values[static_cast<std::size_t>(p.states[s].value)];
        Shape shape;
        if (info.type.is_static()) {
            shape = *info.type.static_shape();
        } else if (planned) {
            shape = plan.shapes[static_cast<std::size_t>(p.states[s].value)];
        } else {
            continue;  // allocated once the symbols are bound
        }
        if (states[s].defined() && state_shapes[s] == shape) {
            continue;
        }
        II_ASSIGN_OR_RETURN(states[s], Tensor::empty(info.type.dtype, shape, allocator));
        state_shapes[s] = shape;
        const Tensor& initial = p.states[s].initial;
        if (initial.defined() && initial.shape() == shape) {
            II_TRY(states[s].copy_from(initial));
        } else {
            std::memset(states[s].data(), 0, states[s].nbytes());
        }
    }
    return ok_status();
}

Status SessionImpl::reset_states() {
    const ProgramImpl& p = *program;
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        if (!states[s].defined()) {
            continue;
        }
        const Tensor& initial = p.states[s].initial;
        if (initial.defined() && initial.shape() == states[s].shape()) {
            II_TRY(states[s].copy_from(initial));
        } else {
            std::memset(states[s].data(), 0, states[s].nbytes());
        }
    }
    return ok_status();
}

Status SessionImpl::prepare() {
    const ProgramImpl& p = *program;
    // Fast path: same input shapes as the current plan.
    if (planned) {
        bool same = true;
        for (std::size_t i = 0; i < p.inputs.size() && same; ++i) {
            same = inputs[i].shape() == plan.input_shapes[i];
        }
        if (same) {
            return ok_status();
        }
    }

    ExecutionPlan next;
    SymbolValues& symbols = next.symbols;

    // --- Bind symbols from the input shapes ---
    struct Deferred {
        std::size_t input;
        std::size_t dim;
    };
    std::vector<Deferred> deferred;
    for (std::size_t i = 0; i < p.inputs.size(); ++i) {
        const ValueInfo& info = p.values[static_cast<std::size_t>(p.inputs[i])];
        const Tensor& tensor = inputs[i];
        const std::string& name = p.input_names[i];
        if (tensor.rank() != info.type.rank()) {
            return errors::invalid_argument("input '{}' has shape {}, expected rank {} ({})", name,
                                            shape_text(tensor.shape()), info.type.rank(), info.type.to_string());
        }
        for (std::size_t d = 0; d < info.type.shape.size(); ++d) {
            const Dim& dim = info.type.shape[d];
            const std::int64_t actual = tensor.shape()[static_cast<int>(d)];
            if (dim.is_constant()) {
                if (dim.value() != actual) {
                    return errors::invalid_argument("input '{}' has shape {}, expected {}", name,
                                                    shape_text(tensor.shape()), info.type.to_string());
                }
            } else if (const std::string_view symbol = dim.symbol_name(); !symbol.empty()) {
                if (const std::optional<std::int64_t> bound = lookup(symbols, symbol)) {
                    if (*bound != actual) {
                        return errors::invalid_argument(
                            "input '{}' has shape {}: dimension {} is {}, but {} = {} from another input", name,
                            shape_text(tensor.shape()), d, actual, symbol, *bound);
                    }
                } else {
                    symbols.emplace_back(std::string(symbol), actual);
                }
            } else {
                deferred.push_back(Deferred{i, d});
            }
        }
    }
    for (const Deferred& item : deferred) {
        const Dim& dim = p.values[static_cast<std::size_t>(p.inputs[item.input])].type.shape[item.dim];
        const std::int64_t actual = inputs[item.input].shape()[static_cast<int>(item.dim)];
        if (!evaluate(dim, symbols)) {
            if (const auto solved = solve(dim, actual, symbols)) {
                symbols.push_back(*solved);
            }
        }
        const std::optional<std::int64_t> expected = evaluate(dim, symbols);
        if (!expected) {
            return errors::invalid_argument("cannot determine the symbols of dimension {} ({}) of input '{}'", item.dim,
                                            dim.to_string(), p.input_names[item.input]);
        }
        if (*expected != actual) {
            return errors::invalid_argument("input '{}' dimension {} is {}, expected {} = {}",
                                            p.input_names[item.input], item.dim, actual, dim.to_string(), *expected);
        }
    }

    // --- Concrete shapes of every value ---
    next.shapes.assign(p.values.size(), Shape());
    auto evaluate_type = [&](const ir::TensorType& type, std::string_view what) -> Result<Shape> {
        Shape shape;
        for (const Dim& dim : type.shape) {
            const std::optional<std::int64_t> value = evaluate(dim, symbols);
            if (!value) {
                return errors::invalid_argument("the inputs do not determine the symbols of {} ({})", what,
                                                type.to_string());
            }
            if (*value < 0) {
                return errors::invalid_argument("{} would have the negative dimension {} ({})", what, *value,
                                                type.to_string());
            }
            shape.push_back(*value);
        }
        return shape;
    };
    for (std::size_t v = 0; v < p.values.size(); ++v) {
        const ValueInfo& info = p.values[v];
        switch (info.kind) {
        case ir::ValueKind::kInput:
            next.shapes[v] = inputs[static_cast<std::size_t>(info.input_index)].shape();
            break;
        case ir::ValueKind::kConstant:
            next.shapes[v] = info.constant.shape();
            break;
        case ir::ValueKind::kState: {
            II_ASSIGN_OR_RETURN(next.shapes[v], evaluate_type(info.type, std::format("state '{}'", info.name)));
            break;
        }
        case ir::ValueKind::kNode:
            break;
        }
    }
    std::vector<ir::TensorType> input_types;
    std::vector<const ir::TensorType*> input_type_ptrs;
    std::vector<ir::TensorType> output_types;
    for (std::size_t s = 0; s < p.steps.size(); ++s) {
        const Step& step = p.steps[s];
        const ir::Node& node = *step.node;
        input_types.assign(step.inputs.size(), ir::TensorType{});
        input_type_ptrs.assign(step.inputs.size(), nullptr);
        for (std::size_t i = 0; i < step.inputs.size(); ++i) {
            const int v = step.inputs[i];
            if (v < 0) {
                continue;
            }
            const Shape& shape = next.shapes[static_cast<std::size_t>(v)];
            input_types[i].dtype = p.values[static_cast<std::size_t>(v)].type.dtype;
            input_types[i].shape.assign(shape.begin(), shape.end());
            input_type_ptrs[i] = &input_types[i];
        }
        Attributes attrs = node.attrs();
        if (step.symbolic_attrs) {
            for (const auto& [name, value] : node.attrs()) {
                if (const Dims* dims = std::get_if<Dims>(&value)) {
                    Dims concrete;
                    for (const Dim& dim : *dims) {
                        const std::optional<std::int64_t> resolved = evaluate(dim, symbols);
                        if (!resolved) {
                            return errors::invalid_argument("attribute '{}' of '{}' uses symbols the inputs do not "
                                                            "determine ({})",
                                                            name, node.op_name(), to_string(*dims));
                        }
                        concrete.emplace_back(*resolved);
                    }
                    attrs.set(name, std::move(concrete));
                }
            }
        }
        output_types.clear();
        if (Status status = ir::infer_types(node.op(), input_type_ptrs, attrs, output_types); !status.ok()) {
            return status.add_context(std::format("step {} ('{}')", s, node.op_name()));
        }
        for (std::size_t o = 0; o < step.outputs.size(); ++o) {
            const ValueInfo& info = p.values[static_cast<std::size_t>(step.outputs[o])];
            const std::optional<Shape> shape = output_types[o].static_shape();
            II_CHECK(shape.has_value());
            // The concrete result must agree with the symbolic type the program was compiled for.
            for (std::size_t d = 0; d < info.type.shape.size(); ++d) {
                const std::optional<std::int64_t> expected = evaluate(info.type.shape[d], symbols);
                if (expected && *expected != (*shape)[static_cast<int>(d)]) {
                    return errors::invalid_argument(
                        "step {} ('{}') produces {} for these inputs, but the model declares {} = {}", s,
                        node.op_name(), shape_text(*shape), info.type.to_string(), info.type.shape[d].to_string());
                }
            }
            next.shapes[static_cast<std::size_t>(step.outputs[o])] = *shape;
        }
    }

    // --- Buffer sharing ---
    // Values are grouped into buffers: a view (reshape) shares its input's buffer, and an operation that can run in
    // place writes into the buffer of an input that dies at that step. A buffer lives from its first producer to the
    // last use of any of its values.
    const int end_step = static_cast<int>(p.steps.size());
    std::vector<int> root(p.values.size(), -1);         // buffer root per kNode value (-1: external memory)
    std::vector<int> buffer_last(p.values.size(), -1);  // per root: last step using the buffer
    std::vector<std::size_t> buffer_size(p.values.size(), 0);
    next.alias_steps.assign(p.steps.size(), 0);
    auto last_use_of = [&](int v) {
        const ValueInfo& info = p.values[static_cast<std::size_t>(v)];
        return std::max(info.last_use, info.producer);
    };
    auto nbytes_of = [&](int v) {
        return storage_size(p.values[static_cast<std::size_t>(v)].type.dtype,
                            next.shapes[static_cast<std::size_t>(v)].numel());
    };
    auto is_result = [&](int v) { return p.values[static_cast<std::size_t>(v)].last_use == end_step; };
    for (std::size_t s = 0; s < p.steps.size(); ++s) {
        const Step& step = p.steps[s];
        const int step_index = static_cast<int>(s);
        for (const int v : step.outputs) {
            root[static_cast<std::size_t>(v)] = v;
            buffer_last[static_cast<std::size_t>(v)] = last_use_of(v);
            buffer_size[static_cast<std::size_t>(v)] = nbytes_of(v);
        }
        if (step.outputs.size() != 1 || step.inputs.empty() || step.inputs[0] < 0) {
            continue;
        }
        const int out = step.outputs[0];
        if (step.node->op().has_flag(ir::kOpView)) {
            const int in = step.inputs[0];
            const ir::ValueKind kind = p.values[static_cast<std::size_t>(in)].kind;
            const int r = kind == ir::ValueKind::kNode ? root[static_cast<std::size_t>(in)] : -1;
            if (r >= 0) {
                root[static_cast<std::size_t>(out)] = r;
                buffer_last[static_cast<std::size_t>(r)] =
                    std::max(buffer_last[static_cast<std::size_t>(r)], last_use_of(out));
                next.alias_steps[s] = 1;
            } else if (!is_result(out) && kind != ir::ValueKind::kState) {
                // A view of external memory (an input, a constant, or a view of one): valid while the steps run.
                root[static_cast<std::size_t>(out)] = -1;
                next.alias_steps[s] = 1;
            }
            continue;
        }
        for (std::size_t i = 0; i < step.inputs.size(); ++i) {
            const int in = step.inputs[i];
            if (in < 0 || p.values[static_cast<std::size_t>(in)].kind != ir::ValueKind::kNode ||
                !step.kernel->can_run_in_place(static_cast<int>(i))) {
                continue;
            }
            const int r = root[static_cast<std::size_t>(in)];
            if (r < 0 || buffer_last[static_cast<std::size_t>(r)] != step_index ||
                next.shapes[static_cast<std::size_t>(in)].numel() !=
                    next.shapes[static_cast<std::size_t>(out)].numel() ||
                nbytes_of(in) != nbytes_of(out) || buffer_size[static_cast<std::size_t>(r)] < nbytes_of(out)) {
                continue;
            }
            root[static_cast<std::size_t>(out)] = r;
            buffer_last[static_cast<std::size_t>(r)] = std::max(step_index, last_use_of(out));
            break;
        }
    }

    // --- Memory plan ---
    std::vector<runtime::MemoryRequest> requests;
    std::vector<int> request_value;  // value index, or -(step + 1) for a workspace
    for (std::size_t v = 0; v < p.values.size(); ++v) {
        const ValueInfo& info = p.values[v];
        if (info.kind != ir::ValueKind::kNode || root[v] != static_cast<int>(v)) {
            continue;
        }
        requests.push_back(runtime::MemoryRequest{buffer_size[v], info.producer, buffer_last[v]});
        request_value.push_back(static_cast<int>(v));
    }
    next.workspace_sizes.assign(p.steps.size(), 0);
    next.workspace_offsets.assign(p.steps.size(), 0);
    std::vector<Shape> in_shapes;
    std::vector<Shape> out_shapes;
    for (std::size_t s = 0; s < p.steps.size(); ++s) {
        const Step& step = p.steps[s];
        in_shapes.clear();
        out_shapes.clear();
        for (const int v : step.inputs) {
            in_shapes.push_back(v < 0 ? Shape() : next.shapes[static_cast<std::size_t>(v)]);
        }
        for (const int v : step.outputs) {
            out_shapes.push_back(next.shapes[static_cast<std::size_t>(v)]);
        }
        const std::size_t size = step.kernel->workspace_size(in_shapes, out_shapes);
        next.workspace_sizes[s] = size;
        if (size > 0) {
            requests.push_back(runtime::MemoryRequest{size, static_cast<int>(s), static_cast<int>(s)});
            request_value.push_back(-static_cast<int>(s) - 1);
        }
    }
    // Staging for state updates whose source is another state (e.g. swapping two states).
    const int end = static_cast<int>(p.steps.size());
    std::vector<std::size_t> staging_request(p.states.size(), static_cast<std::size_t>(-1));
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        const int update = p.states[s].update;
        if (update >= 0 && p.values[static_cast<std::size_t>(update)].kind == ir::ValueKind::kState &&
            update != p.states[s].value) {
            staging_request[s] = requests.size();
            requests.push_back(
                runtime::MemoryRequest{storage_size(p.values[static_cast<std::size_t>(update)].type.dtype,
                                                    next.shapes[static_cast<std::size_t>(update)].numel()),
                                       end, end});
            request_value.push_back(std::numeric_limits<int>::min());
        }
    }
    const runtime::MemoryPlan memory = runtime::plan_memory(requests);
    next.arena_size = memory.total;
    next.offsets.assign(p.values.size(), ExecutionPlan::kNotInArena);
    for (std::size_t r = 0; r < requests.size(); ++r) {
        const int target = request_value[r];
        if (target >= 0) {
            next.offsets[static_cast<std::size_t>(target)] = memory.offsets[r];
        } else if (target != std::numeric_limits<int>::min()) {
            next.workspace_offsets[static_cast<std::size_t>(-target - 1)] = memory.offsets[r];
        }
    }
    for (std::size_t v = 0; v < p.values.size(); ++v) {
        if (p.values[v].kind == ir::ValueKind::kNode && root[v] != static_cast<int>(v)) {
            next.offsets[v] =
                root[v] >= 0 ? next.offsets[static_cast<std::size_t>(root[v])] : ExecutionPlan::kNotInArena;
        }
    }
    next.state_staging_offsets.assign(p.states.size(), ExecutionPlan::kNotInArena);
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        if (staging_request[s] != static_cast<std::size_t>(-1)) {
            next.state_staging_offsets[s] = memory.offsets[staging_request[s]];
        }
    }

    // --- Pre-built kernel arguments ---
    for (const Step& step : p.steps) {
        next.view_begin.push_back(next.views.size());
        for (const int v : step.inputs) {
            backend::TensorView view;
            if (v >= 0) {
                view.dtype = p.values[static_cast<std::size_t>(v)].type.dtype;
                view.shape = next.shapes[static_cast<std::size_t>(v)];
            }
            next.views.push_back(view);
        }
        for (const int v : step.outputs) {
            next.views.push_back(backend::TensorView{nullptr, p.values[static_cast<std::size_t>(v)].type.dtype,
                                                     next.shapes[static_cast<std::size_t>(v)]});
        }
    }
    next.view_begin.push_back(next.views.size());
    for (std::size_t i = 0; i < p.inputs.size(); ++i) {
        next.input_shapes.push_back(inputs[i].shape());
    }

    // --- Commit: arena and states ---
    if (!arena || arena->size() < next.arena_size) {
        II_ASSIGN_OR_RETURN(arena, Buffer::allocate(next.arena_size, p.context.allocator()));
    }
    plan = std::move(next);
    planned = true;
    return allocate_states();
}

Status SessionImpl::run() {
    const ProgramImpl& p = *program;
    for (std::size_t i = 0; i < p.inputs.size(); ++i) {
        if (!inputs[i].defined()) {
            return errors::invalid_argument("input '{}' is not set", p.input_names[i]);
        }
    }
    II_TRY(prepare());

    auto* base = static_cast<std::byte*>(arena->data());
    for (std::size_t v = 0; v < p.values.size(); ++v) {
        const ValueInfo& info = p.values[v];
        switch (info.kind) {
        case ir::ValueKind::kInput:
            value_data[v] = inputs[static_cast<std::size_t>(info.input_index)].data();
            break;
        case ir::ValueKind::kConstant:
            value_data[v] = info.constant.data();
            break;
        case ir::ValueKind::kState:
            value_data[v] = states[static_cast<std::size_t>(info.state_index)].data();
            break;
        case ir::ValueKind::kNode:
            value_data[v] = plan.offsets[v] != ExecutionPlan::kNotInArena ? base + plan.offsets[v] : nullptr;
            break;
        }
    }
    // Bound outputs produced by a node are written in place.
    for (std::size_t o = 0; o < p.outputs.size(); ++o) {
        const Tensor& bound = bound_outputs[o];
        const int v = p.outputs[o];
        if (!bound.defined()) {
            continue;
        }
        if (!(bound.shape() == plan.shapes[static_cast<std::size_t>(v)])) {
            return errors::invalid_argument("output '{}' is bound to a tensor of shape {}, but it has shape {}",
                                            p.output_names[o], shape_text(bound.shape()),
                                            shape_text(plan.shapes[static_cast<std::size_t>(v)]));
        }
        if (p.values[static_cast<std::size_t>(v)].kind == ir::ValueKind::kNode) {
            value_data[static_cast<std::size_t>(v)] = bound.data();
        }
    }

    backend::KernelContext context;
    context.executor = p.context.impl()->executor;
    using Clock = std::chrono::steady_clock;
    const Clock::time_point run_start = Clock::now();
    auto elapsed_ns = [&] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - run_start).count();
    };
    profiled = false;
    for (std::size_t s = 0; s < p.steps.size(); ++s) {
        const Step& step = p.steps[s];
        if (profiling) {
            step_times[2 * s] = elapsed_ns();
            step_times[2 * s + 1] = step_times[2 * s];
        }
        if (plan.alias_steps[s] != 0) {
            value_data[static_cast<std::size_t>(step.outputs[0])] =
                value_data[static_cast<std::size_t>(step.inputs[0])];
            continue;
        }
        const std::size_t begin = plan.view_begin[s];
        const std::size_t middle = begin + step.inputs.size();
        const std::size_t end = plan.view_begin[s + 1];
        for (std::size_t i = 0; i < step.inputs.size(); ++i) {
            const int v = step.inputs[i];
            plan.views[begin + i].data = v < 0 ? nullptr : value_data[static_cast<std::size_t>(v)];
        }
        for (std::size_t o = 0; o < step.outputs.size(); ++o) {
            plan.views[middle + o].data = value_data[static_cast<std::size_t>(step.outputs[o])];
        }
        context.inputs = std::span<const backend::TensorView>(plan.views.data() + begin, middle - begin);
        context.outputs = std::span<const backend::TensorView>(plan.views.data() + middle, end - middle);
        context.workspace = std::span<std::byte>(base + plan.workspace_offsets[s], plan.workspace_sizes[s]);
        if (Status status = step.kernel->run(context); !status.ok()) {
            const std::string& name = step.node->name();
            return status.add_context(std::format("step {} ('{}'{}, {})", s, step.node->op_name(),
                                                  name.empty() ? "" : " " + name, step.backend->name()));
        }
        if (profiling) {
            step_times[2 * s + 1] = elapsed_ns();
        }
    }

    // Bound outputs that were not written in place (inputs, constants, states, or the same value bound twice).
    for (std::size_t o = 0; o < p.outputs.size(); ++o) {
        const Tensor& bound = bound_outputs[o];
        const auto v = static_cast<std::size_t>(p.outputs[o]);
        if (bound.defined() && bound.data() != value_data[v]) {
            std::memcpy(bound.data(), value_data[v], bound.nbytes());
        }
    }

    // State updates: stage updates read from other states first, then write all states.
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        if (plan.state_staging_offsets[s] != ExecutionPlan::kNotInArena) {
            const auto update = static_cast<std::size_t>(p.states[s].update);
            std::memcpy(base + plan.state_staging_offsets[s], value_data[update],
                        storage_size(p.values[update].type.dtype, plan.shapes[update].numel()));
        }
    }
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        const int update = p.states[s].update;
        if (update < 0) {
            continue;
        }
        const void* source = plan.state_staging_offsets[s] != ExecutionPlan::kNotInArena
                                 ? base + plan.state_staging_offsets[s]
                                 : value_data[static_cast<std::size_t>(update)];
        if (source != states[s].data()) {
            std::memmove(states[s].data(), source, states[s].nbytes());
        }
    }
    has_run = true;
    if (profiling) {
        run_time_ns = elapsed_ns();
        profiled = true;
    }
    return ok_status();
}

Result<Tensor> SessionImpl::value_tensor(int value) const {
    const ProgramImpl& p = *program;
    const ValueInfo& info = p.values[static_cast<std::size_t>(value)];
    switch (info.kind) {
    case ir::ValueKind::kInput:
        return inputs[static_cast<std::size_t>(info.input_index)];
    case ir::ValueKind::kConstant:
        return info.constant;
    case ir::ValueKind::kState:
        return states[static_cast<std::size_t>(info.state_index)];
    case ir::ValueKind::kNode:
        break;
    }
    const Shape& shape = plan.shapes[static_cast<std::size_t>(value)];
    return Tensor::view(arena, info.type.dtype, shape, contiguous_strides(shape),
                        plan.offsets[static_cast<std::size_t>(value)]);
}

}  // namespace detail

// ---------------------------------------------------------------------------------------------------------------
// Public API

Result<Session> Program::new_session() const {
    auto impl = std::make_unique<detail::SessionImpl>();
    impl->program = impl_;
    impl->inputs.resize(impl_->inputs.size());
    impl->bound_outputs.resize(impl_->outputs.size());
    impl->states.resize(impl_->states.size());
    impl->state_shapes.resize(impl_->states.size());
    impl->value_data.resize(impl_->values.size());
    II_TRY(impl->allocate_states());
    return Session(std::move(impl));
}

Session::Session(std::unique_ptr<detail::SessionImpl> impl) noexcept : impl_(std::move(impl)) {}
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;
Session::~Session() = default;

Status Session::set_input(std::string_view name, const Tensor& tensor) {
    const detail::ProgramImpl& p = *impl_->program;
    const int index = p.find_input(name);
    if (index < 0) {
        return errors::not_found("the model has no input '{}'", name);
    }
    if (!tensor.defined()) {
        return errors::invalid_argument("input '{}': empty tensor", name);
    }
    const ir::TensorType& type = p.values[static_cast<std::size_t>(p.inputs[static_cast<std::size_t>(index)])].type;
    if (tensor.dtype() != type.dtype) {
        return errors::invalid_argument("input '{}' must be {}, got {}", name, to_string(type.dtype),
                                        to_string(tensor.dtype()));
    }
    if (tensor.device().kind != DeviceKind::kCpu) {
        return errors::unsupported("input '{}': only host tensors are supported", name);
    }
    II_ASSIGN_OR_RETURN(impl_->inputs[static_cast<std::size_t>(index)], tensor.contiguous(p.context.allocator()));
    return ok_status();
}

Status Session::set_output(std::string_view name, const Tensor& tensor) {
    const detail::ProgramImpl& p = *impl_->program;
    const int index = p.find_output(name);
    if (index < 0) {
        return errors::not_found("the model has no output '{}'", name);
    }
    const ir::TensorType& type = p.values[static_cast<std::size_t>(p.outputs[static_cast<std::size_t>(index)])].type;
    if (!tensor.defined() || tensor.dtype() != type.dtype || !tensor.is_contiguous() ||
        tensor.device().kind != DeviceKind::kCpu) {
        return errors::invalid_argument("output '{}' must be bound to a contiguous host {} tensor", name,
                                        to_string(type.dtype));
    }
    impl_->bound_outputs[static_cast<std::size_t>(index)] = tensor;
    return ok_status();
}

Status Session::clear_output(std::string_view name) {
    const int index = impl_->program->find_output(name);
    if (index < 0) {
        return errors::not_found("the model has no output '{}'", name);
    }
    impl_->bound_outputs[static_cast<std::size_t>(index)] = Tensor();
    return ok_status();
}

Status Session::run() {
    return impl_->run();
}

Result<Tensor> Session::output(std::string_view name) const {
    const detail::ProgramImpl& p = *impl_->program;
    const int index = p.find_output(name);
    if (index < 0) {
        return errors::not_found("the model has no output '{}'", name);
    }
    if (!impl_->has_run) {
        return errors::invalid_argument("output '{}' is not available before run()", name);
    }
    if (const Tensor& bound = impl_->bound_outputs[static_cast<std::size_t>(index)]; bound.defined()) {
        return bound;
    }
    return impl_->value_tensor(p.outputs[static_cast<std::size_t>(index)]);
}

Result<std::vector<Tensor>> Session::run(std::initializer_list<std::pair<std::string_view, Tensor>> inputs) {
    for (const auto& [name, tensor] : inputs) {
        II_TRY(set_input(name, tensor));
    }
    II_TRY(run());
    std::vector<Tensor> outputs;
    for (const std::string& name : impl_->program->output_names) {
        II_ASSIGN_OR_RETURN(const Tensor output, this->output(name));
        II_ASSIGN_OR_RETURN(Tensor copy, output.clone(impl_->program->context.allocator()));
        outputs.push_back(std::move(copy));
    }
    return outputs;
}

Result<Tensor> Session::state(std::string_view name) const {
    const int index = impl_->program->find_state(name);
    if (index < 0) {
        return errors::not_found("the model has no state '{}'", name);
    }
    const Tensor& state = impl_->states[static_cast<std::size_t>(index)];
    if (!state.defined()) {
        return errors::invalid_argument("state '{}' has a symbolic shape and is allocated by the first run()", name);
    }
    return state;
}

Status Session::reset_states() {
    return impl_->reset_states();
}

void Session::set_profiling(bool enabled) {
    impl_->profiling = enabled;
    impl_->step_times.assign(2 * impl_->program->steps.size(), 0);
}

Result<Profile> Session::profile() const {
    const detail::SessionImpl& s = *impl_;
    if (!s.profiled) {
        return errors::invalid_argument("the last run was not profiled (call set_profiling(true) before run())");
    }
    const detail::ProgramImpl& p = *s.program;
    Profile profile;
    profile.total_us = static_cast<double>(s.run_time_ns) / 1000.0;
    for (std::size_t i = 0; i < p.steps.size(); ++i) {
        const detail::Step& step = p.steps[i];
        StepProfile item;
        item.op = std::string(step.node->op_name());
        item.name = step.node->name();
        item.backend = s.plan.alias_steps[i] != 0 ? "view" : std::string(step.backend->name());
        if (!step.outputs.empty()) {
            const auto v = static_cast<std::size_t>(step.outputs[0]);
            item.output = std::string(to_string(p.values[v].type.dtype)) + s.plan.shapes[v].to_string();
        }
        item.start_us = static_cast<double>(s.step_times[2 * i]) / 1000.0;
        item.duration_us = static_cast<double>(s.step_times[2 * i + 1] - s.step_times[2 * i]) / 1000.0;
        profile.steps.push_back(std::move(item));
    }
    return profile;
}

namespace {

std::string json_escape(std::string_view text) {
    std::string out;
    for (const char c : text) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += std::format("\\u{:04x}", static_cast<unsigned>(c));
            } else {
                out += c;
            }
        }
    }
    return out;
}

}  // namespace

std::string to_chrome_trace(std::span<const Profile> runs) {
    std::string json = "{\"traceEvents\": [\n";
    double offset = 0;
    bool first = true;
    auto event = [&](std::string_view name, std::string_view category, double start, double duration,
                     const std::string& args, int thread) {
        json += std::format(
            "{}  {{\"name\": \"{}\", \"cat\": \"{}\", \"ph\": \"X\", \"ts\": {:.3f}, \"dur\": {:.3f}, \"pid\": 1, "
            "\"tid\": {}, \"args\": {{{}}}}}",
            first ? "" : ",\n", json_escape(name), json_escape(category), start, duration, thread, args);
        first = false;
    };
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const Profile& run = runs[r];
        event(std::format("run {}", r), "run", offset, run.total_us, std::string(), 1);
        for (const StepProfile& step : run.steps) {
            const std::string args =
                std::format("\"backend\": \"{}\", \"output\": \"{}\", \"node\": \"{}\"", json_escape(step.backend),
                            json_escape(step.output), json_escape(step.name));
            event(step.op, step.backend, offset + step.start_us, step.duration_us, args, 2);
        }
        offset += run.total_us + 1.0;
    }
    json += "\n], \"displayTimeUnit\": \"ms\"}\n";
    return json;
}

}  // namespace ii
