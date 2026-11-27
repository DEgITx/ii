// SPDX-License-Identifier: MIT
// Session: binding of inputs, shape specialization, memory planning and execution.
#include "core/assert.h"
#include "ir/op_schema.h"
#include "runtime/memory_planner.h"
#include "runtime/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>

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

Status SessionImpl::initialize_state(std::size_t s) {
    const ProgramImpl& p = *program;
    const Tensor& state = states[s];
    const Tensor& initial = p.states[s].initial;
    const bool has_initial = initial.defined() && initial.shape() == state.shape();
    const int device = p.states[s].device;
    if (device == 0) {
        if (has_initial) {
            return state.copy_from(initial);
        }
        std::memset(state.data(), 0, state.nbytes());
        return ok_status();
    }
    backend::Queue& queue = *queues[static_cast<std::size_t>(device)];
    if (has_initial) {
        II_ASSIGN_OR_RETURN(const Tensor dense, initial.contiguous(p.context.allocator()));
        return queue.copy_to_device(state.data(), dense.data(), state.nbytes());
    }
    return queue.fill_zero(state.data(), state.nbytes());
}

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
        const int device = p.states[s].device;
        if (device == 0) {
            II_ASSIGN_OR_RETURN(states[s], Tensor::empty(info.type.dtype, shape, allocator));
        } else {
            const std::size_t nbytes = storage_size(info.type.dtype, shape.numel());
            backend::DeviceApi& api = *p.devices[static_cast<std::size_t>(device)].api;
            II_ASSIGN_OR_RETURN(std::shared_ptr<Buffer> buffer, api.allocate(std::max<std::size_t>(nbytes, 1)));
            II_ASSIGN_OR_RETURN(states[s],
                                Tensor::view(std::move(buffer), info.type.dtype, shape, contiguous_strides(shape)));
        }
        state_shapes[s] = shape;
        II_TRY(initialize_state(s));
    }
    return ok_status();
}

Status SessionImpl::reset_states() {
    for (std::size_t s = 0; s < program->states.size(); ++s) {
        if (states[s].defined()) {
            II_TRY(initialize_state(s));
        }
    }
    return synchronize();
}

Status SessionImpl::synchronize() {
    Status first = ok_status();
    for (const std::unique_ptr<backend::Queue>& queue : queues) {
        if (queue) {
            Status status = queue->synchronize();
            if (first.ok() && !status.ok()) {
                first = std::move(status);
            }
        }
    }
    return first;
}

namespace {

/// A permute that keeps the element order (see backend::keeps_element_order) runs as a view.
bool order_preserving_permute(const Step& step, const std::vector<Shape>& shapes) {
    return !step.is_transfer() && step.node->op_name() == "permute" && !step.inputs.empty() && step.inputs[0] >= 0 &&
           backend::keeps_element_order(step.node->attrs().get_ints("perm"),
                                        shapes[static_cast<std::size_t>(step.inputs[0])]);
}

}  // namespace

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
        for (std::size_t c = 0; c < plan_cache.size(); ++c) {
            bool match = true;
            for (std::size_t i = 0; i < p.inputs.size() && match; ++i) {
                match = inputs[i].shape() == plan_cache[c].input_shapes[i];
            }
            if (match) {
                // The arena only grows, so a cached plan still fits.
                std::swap(plan, plan_cache[c]);
                return allocate_states();
            }
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
    next.state_alias.assign(p.values.size(), -1);
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
        if (step.is_transfer() || step.outputs.size() != 1 || step.inputs.empty() || step.inputs[0] < 0) {
            continue;
        }
        const int out = step.outputs[0];
        if (step.node->op().has_flag(ir::kOpView) || order_preserving_permute(step, next.shapes)) {
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
        // An in-place step that turns a state into its own update (a KV cache write) works directly in the state's
        // memory when nothing reads the old state value afterwards: no arena buffer and no copy at the end of run().
        bool into_state = false;
        for (std::size_t i = 0; i < step.inputs.size() && !into_state; ++i) {
            const int in = step.inputs[i];
            if (in < 0 || p.values[static_cast<std::size_t>(in)].kind != ir::ValueKind::kState ||
                !step.kernel->can_run_in_place(static_cast<int>(i))) {
                continue;
            }
            const int state = p.values[static_cast<std::size_t>(in)].state_index;
            if (p.states[static_cast<std::size_t>(state)].update != out ||
                p.values[static_cast<std::size_t>(in)].last_use != step_index || nbytes_of(in) != nbytes_of(out)) {
                continue;
            }
            root[static_cast<std::size_t>(out)] = -1;
            next.state_alias[static_cast<std::size_t>(out)] = state;
            into_state = true;
        }
        if (into_state) {
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

    // --- Memory plan: one arena per device ---
    const std::size_t num_devices = p.devices.size();
    std::vector<std::vector<runtime::MemoryRequest>> requests(num_devices);
    // Per request: the value index, -(step + 1) for a workspace, or kStaging - state for a state staging buffer.
    std::vector<std::vector<int>> request_value(num_devices);
    constexpr int kStaging = std::numeric_limits<int>::min() / 2;
    for (std::size_t v = 0; v < p.values.size(); ++v) {
        const ValueInfo& info = p.values[v];
        if (info.kind != ir::ValueKind::kNode || root[v] != static_cast<int>(v)) {
            continue;
        }
        const auto d = static_cast<std::size_t>(info.device);
        requests[d].push_back(runtime::MemoryRequest{buffer_size[v], info.producer, buffer_last[v]});
        request_value[d].push_back(static_cast<int>(v));
    }
    next.workspace_sizes.assign(p.steps.size(), 0);
    next.workspace_offsets.assign(p.steps.size(), 0);
    std::vector<Shape> in_shapes;
    std::vector<Shape> out_shapes;
    for (std::size_t s = 0; s < p.steps.size(); ++s) {
        const Step& step = p.steps[s];
        if (step.is_transfer()) {
            continue;
        }
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
            const auto d = static_cast<std::size_t>(step.device);
            requests[d].push_back(runtime::MemoryRequest{size, static_cast<int>(s), static_cast<int>(s)});
            request_value[d].push_back(-static_cast<int>(s) - 1);
        }
    }
    // Staging for state updates whose source is another state (e.g. swapping two states).
    const int end = static_cast<int>(p.steps.size());
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        const int update = p.states[s].update;
        if (update >= 0 && p.values[static_cast<std::size_t>(update)].kind == ir::ValueKind::kState &&
            update != p.states[s].value) {
            const auto d = static_cast<std::size_t>(p.states[s].device);
            requests[d].push_back(
                runtime::MemoryRequest{storage_size(p.values[static_cast<std::size_t>(update)].type.dtype,
                                                    next.shapes[static_cast<std::size_t>(update)].numel()),
                                       end, end});
            request_value[d].push_back(kStaging - static_cast<int>(s));
        }
    }
    next.offsets.assign(p.values.size(), ExecutionPlan::kNotInArena);
    next.state_staging_offsets.assign(p.states.size(), ExecutionPlan::kNotInArena);
    next.arena_sizes.assign(num_devices, 0);
    for (std::size_t d = 0; d < num_devices; ++d) {
        const runtime::MemoryPlan memory = runtime::plan_memory(requests[d], d == 0 ? 64 : 256);
        next.arena_sizes[d] = memory.total;
        for (std::size_t r = 0; r < requests[d].size(); ++r) {
            const int target = request_value[d][r];
            if (target >= 0) {
                next.offsets[static_cast<std::size_t>(target)] = memory.offsets[r];
            } else if (target <= kStaging) {
                next.state_staging_offsets[static_cast<std::size_t>(kStaging - target)] = memory.offsets[r];
            } else {
                next.workspace_offsets[static_cast<std::size_t>(-target - 1)] = memory.offsets[r];
            }
        }
    }
    for (std::size_t v = 0; v < p.values.size(); ++v) {
        if (p.values[v].kind == ir::ValueKind::kNode && root[v] != static_cast<int>(v)) {
            next.offsets[v] =
                root[v] >= 0 ? next.offsets[static_cast<std::size_t>(root[v])] : ExecutionPlan::kNotInArena;
        }
    }

    // --- Segments that can be recorded: runs of recordable kernels on one device ---
    constexpr int kMinSegmentKernels = 3;
    next.segment_at.assign(p.steps.size(), -1);
    for (std::size_t s = 0; s < p.steps.size();) {
        const Step& first = p.steps[s];
        if (first.is_transfer() || first.device == 0 || next.alias_steps[s] != 0 || !first.kernel->can_record()) {
            ++s;
            continue;
        }
        std::size_t e = s;
        int kernels = 0;
        while (e < p.steps.size()) {
            const Step& step = p.steps[e];
            if (next.alias_steps[e] == 0) {
                if (step.is_transfer() || step.device != first.device || !step.kernel->can_record()) {
                    break;
                }
                ++kernels;
            }
            ++e;
        }
        if (kernels >= kMinSegmentKernels) {
            next.segment_at[s] = static_cast<int>(next.segments.size());
            ExecutionPlan::Segment segment;
            segment.begin = s;
            segment.end = e;
            segment.device = first.device;
            next.segments.push_back(std::move(segment));
        }
        s = e;
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

    // --- Commit: arenas and states ---
    for (std::size_t d = 0; d < num_devices; ++d) {
        std::shared_ptr<Buffer>& arena = arenas[d];
        const std::size_t size = std::max<std::size_t>(next.arena_sizes[d], 1);
        if (arena && arena->size() >= size) {
            continue;
        }
        if (d == 0) {
            II_ASSIGN_OR_RETURN(arena, Buffer::allocate(size, p.context.allocator()));
        } else {
            arena.reset();  // free the old arena first
            II_ASSIGN_OR_RETURN(arena, p.devices[d].api->allocate(size));
        }
    }
    if (planned) {
        if (plan_cache.size() == kPlanCacheSize) {
            plan_cache.erase(plan_cache.begin());
        }
        plan_cache.push_back(std::move(plan));
    }
    plan = std::move(next);
    planned = true;
    return allocate_states();
}

Status SessionImpl::run() {
    Status status = execute();
    // Device work still in flight must finish before the caller reuses inputs or reads results, also after errors.
    Status synchronized = synchronize();
    if (!status.ok()) {
        return status;
    }
    II_TRY(synchronized);
    has_run = true;
    return ok_status();
}

Status SessionImpl::execute() {
    const ProgramImpl& p = *program;
    for (std::size_t i = 0; i < p.inputs.size(); ++i) {
        if (!inputs[i].defined()) {
            return errors::invalid_argument("input '{}' is not set", p.input_names[i]);
        }
    }
    II_TRY(prepare());

    for (std::size_t d = 0; d < arenas.size(); ++d) {
        bases[d] = static_cast<std::byte*>(arenas[d]->data());
    }
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
            if (plan.state_alias[v] >= 0) {
                value_data[v] = states[static_cast<std::size_t>(plan.state_alias[v])].data();
            } else {
                value_data[v] = plan.offsets[v] != ExecutionPlan::kNotInArena
                                    ? bases[static_cast<std::size_t>(info.device)] + plan.offsets[v]
                                    : nullptr;
            }
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
        if (!profiling && plan.segment_at[s] >= 0) {
            ExecutionPlan::Segment& segment = plan.segments[static_cast<std::size_t>(plan.segment_at[s])];
            II_ASSIGN_OR_RETURN(const bool replayed, run_segment(segment, context));
            if (replayed) {
                s = segment.end - 1;
                continue;
            }
        }
        if (profiling) {
            step_times[2 * s] = elapsed_ns();
            step_times[2 * s + 1] = step_times[2 * s];
        }
        if (plan.alias_steps[s] != 0) {
            value_data[static_cast<std::size_t>(step.outputs[0])] =
                value_data[static_cast<std::size_t>(step.inputs[0])];
            continue;
        }
        Status status = run_step(s, context);
        if (profiling && status.ok()) {
            // Device work is asynchronous: wait for it, so that a step's time is its own.
            const int device = step.is_transfer() ? (step.transfer_from == 0 ? step.device : 0) : step.device;
            if (device != 0) {
                status = queues[static_cast<std::size_t>(device)]->synchronize();
            }
        }
        if (!status.ok()) {
            const std::string& name = step.node->name();
            return status.add_context(std::format("step {} ('{}'{}, {})", s, step.node->op_name(),
                                                  name.empty() ? "" : " " + name,
                                                  step.backend != nullptr ? step.backend->name() : "transfer"));
        }
        if (profiling) {
            step_times[2 * s + 1] = elapsed_ns();
        }
    }

    // Bound outputs that were not written in place (inputs, constants, states, or the same value bound twice).
    // Outputs are always in host memory (the planner adds transfers).
    for (std::size_t o = 0; o < p.outputs.size(); ++o) {
        const Tensor& bound = bound_outputs[o];
        const auto v = static_cast<std::size_t>(p.outputs[o]);
        if (bound.defined() && bound.data() != value_data[v]) {
            std::memcpy(bound.data(), value_data[v], bound.nbytes());
        }
    }

    // State updates: stage updates read from other states first, then write all states.
    auto copy_on = [&](int device, void* destination, const void* source, std::size_t nbytes) -> Status {
        if (device == 0) {
            std::memmove(destination, source, nbytes);
            return ok_status();
        }
        return queues[static_cast<std::size_t>(device)]->copy(destination, source, nbytes);
    };
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        if (plan.state_staging_offsets[s] != ExecutionPlan::kNotInArena) {
            const auto update = static_cast<std::size_t>(p.states[s].update);
            const int device = p.states[s].device;
            II_TRY(copy_on(device, bases[static_cast<std::size_t>(device)] + plan.state_staging_offsets[s],
                           value_data[update], storage_size(p.values[update].type.dtype, plan.shapes[update].numel())));
        }
    }
    for (std::size_t s = 0; s < p.states.size(); ++s) {
        const int update = p.states[s].update;
        if (update < 0) {
            continue;
        }
        const int device = p.states[s].device;
        const void* source = plan.state_staging_offsets[s] != ExecutionPlan::kNotInArena
                                 ? bases[static_cast<std::size_t>(device)] + plan.state_staging_offsets[s]
                                 : value_data[static_cast<std::size_t>(update)];
        if (source != states[s].data()) {
            II_TRY(copy_on(device, states[s].data(), source, states[s].nbytes()));
        }
    }
    if (profiling) {
        run_time_ns = elapsed_ns();
        profiled = true;
    }
    return ok_status();
}

Status SessionImpl::run_step(std::size_t s, backend::KernelContext& context) {
    const ProgramImpl& p = *program;
    const Step& step = p.steps[s];
    if (plan.alias_steps[s] != 0) {
        value_data[static_cast<std::size_t>(step.outputs[0])] = value_data[static_cast<std::size_t>(step.inputs[0])];
        return ok_status();
    }
    Status status;
    if (step.is_transfer()) {
        const auto in = static_cast<std::size_t>(step.inputs[0]);
        void* destination = value_data[static_cast<std::size_t>(step.outputs[0])];
        const std::size_t nbytes = storage_size(p.values[in].type.dtype, plan.shapes[in].numel());
        if (step.transfer_from == 0) {
            status = queues[static_cast<std::size_t>(step.device)]->copy_to_device(destination, value_data[in], nbytes);
        } else {
            backend::Queue& queue = *queues[static_cast<std::size_t>(step.transfer_from)];
            status = queue.copy_to_host(destination, value_data[in], nbytes);
            if (status.ok()) {
                status = queue.synchronize();  // the host steps after this one read the copy
            }
        }
    } else {
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
        const auto device = static_cast<std::size_t>(step.device);
        context.inputs = std::span<const backend::TensorView>(plan.views.data() + begin, middle - begin);
        context.outputs = std::span<const backend::TensorView>(plan.views.data() + middle, end - middle);
        context.workspace = std::span<std::byte>(bases[device] + plan.workspace_offsets[s], plan.workspace_sizes[s]);
        context.queue = queues[device].get();
        status = step.kernel->run(context);
    }
    return status;
}

Result<bool> SessionImpl::run_segment(ExecutionPlan::Segment& segment, backend::KernelContext& context) {
    if (segment.disabled) {
        return false;
    }
    // The kernel arguments of this run (views resolved as the steps would resolve them).
    const ProgramImpl& p = *program;
    std::vector<const void*>& pointers = segment_pointers;
    pointers.clear();
    for (std::size_t s = segment.begin; s < segment.end; ++s) {
        const Step& step = p.steps[s];
        if (plan.alias_steps[s] != 0) {
            value_data[static_cast<std::size_t>(step.outputs[0])] =
                value_data[static_cast<std::size_t>(step.inputs[0])];
            continue;
        }
        for (const int v : step.inputs) {
            pointers.push_back(v < 0 ? nullptr : value_data[static_cast<std::size_t>(v)]);
        }
        for (const int v : step.outputs) {
            pointers.push_back(value_data[static_cast<std::size_t>(v)]);
        }
        pointers.push_back(bases[static_cast<std::size_t>(step.device)] + plan.workspace_offsets[s]);
    }
    backend::Queue& queue = *queues[static_cast<std::size_t>(segment.device)];
    if (pointers != segment.pointers) {
        segment.pointers = pointers;
        segment.repeats = 0;
        segment.recording.reset();
    }
    if (segment.recording) {
        II_TRY(queue.replay(*segment.recording));
        return true;
    }
    // The first run with these arguments calls the kernels (they may still set themselves up, e.g. choose cuBLASLt
    // algorithms); the second records them and replays the recording.
    if (++segment.repeats < 2) {
        return false;
    }
    if (!queue.begin_recording().ok()) {
        segment.disabled = true;
        return false;
    }
    Status status;
    for (std::size_t s = segment.begin; s < segment.end && status.ok(); ++s) {
        status = run_step(s, context);
    }
    Result<std::unique_ptr<backend::Recording>> recording = queue.end_recording();
    if (!status.ok() || !recording.ok()) {
        // Nothing ran: the steps run one by one (and report their own errors).
        segment.disabled = true;
        return false;
    }
    segment.recording = std::move(*recording);
    II_TRY(queue.replay(*segment.recording));
    return true;
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
    if (const int state = plan.state_alias[static_cast<std::size_t>(value)]; state >= 0) {
        return states[static_cast<std::size_t>(state)];
    }
    const Shape& shape = plan.shapes[static_cast<std::size_t>(value)];
    return Tensor::view(arenas[static_cast<std::size_t>(info.device)], info.type.dtype, shape,
                        contiguous_strides(shape), plan.offsets[static_cast<std::size_t>(value)]);
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
    impl->arenas.resize(impl_->devices.size());
    impl->bases.resize(impl_->devices.size());
    impl->queues.resize(impl_->devices.size());
    for (std::size_t d = 1; d < impl_->devices.size(); ++d) {
        II_ASSIGN_OR_RETURN(impl->queues[d], impl_->devices[d].api->create_queue());
    }
    II_TRY(impl->allocate_states());
    II_TRY(impl->synchronize());
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
    const int device = impl_->program->states[static_cast<std::size_t>(index)].device;
    if (device == 0) {
        return state;
    }
    // A state in device memory: a host copy.
    II_ASSIGN_OR_RETURN(Tensor copy, Tensor::empty(state.dtype(), state.shape(), impl_->program->context.allocator()));
    backend::Queue& queue = *impl_->queues[static_cast<std::size_t>(device)];
    II_TRY(queue.copy_to_host(copy.data(), state.data(), state.nbytes()));
    II_TRY(queue.synchronize());
    return copy;
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
        item.backend = s.plan.alias_steps[i] != 0 ? "view"
                       : step.is_transfer()       ? "transfer"
                                                  : std::string(step.backend->name());
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
