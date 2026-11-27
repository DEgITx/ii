// SPDX-License-Identifier: MIT
#include "runtime/planner.h"

#include "core/assert.h"
#include "ir/op_schema.h"
#include "runtime/memory_planner.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <unordered_map>

namespace ii::runtime {

double GraphPlan::total_us() const noexcept {
    double total = transfer_us;
    for (const double t : device_us) {
        total += t;
    }
    return total;
}

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

std::string format_bytes(std::uint64_t bytes) {
    if (bytes >= (std::uint64_t{1} << 30)) {
        return std::format("{:.2f} GiB", static_cast<double>(bytes) / (1 << 30));
    }
    return std::format("{:.1f} MiB", static_cast<double>(bytes) / (1 << 20));
}

class Planner {
public:
    Planner(const ir::Graph& graph, std::span<const PlanBackend> backends, std::span<const PlanDevice> devices,
            const PlannerOptions& options)
        : graph_(graph), backends_(backends), devices_(devices), options_(options) {}

    Result<GraphPlan> run() {
        II_CHECK(!devices_.empty() && devices_[0].info.device.is_host(), "device 0 of a plan must be the host");
        II_CHECK(devices_.size() <= 64, "too many devices");
        index_graph();
        II_TRY(estimate_nodes());
        greedy();
        II_TRY(apply_memory_limits());
        bool multi_device = false;
        for (const PlanBackend& backend : backends_) {
            multi_device = multi_device || backend.device != 0;
        }
        if (multi_device) {
            refine();
        }
        return result();
    }

private:
    struct NodeRec {
        const ir::Node* node = nullptr;
        std::vector<int> inputs;  ///< value indices; -1 for absent inputs
        std::vector<int> outputs;
        std::vector<double> time;  ///< per backend; infinity when unsupported
        std::vector<int> candidates;
    };
    struct ValueRec {
        ir::ValueKind kind = ir::ValueKind::kNode;
        std::uint64_t bytes = 0;
        int producer = -1;           ///< node index (kNode)
        int state = -1;              ///< state index (kState)
        std::vector<int> consumers;  ///< node indices, ascending, unique
        bool output = false;         ///< a graph output: needed on the host
        std::vector<int> updates;    ///< states this value is written back to
    };

    // --- Model of the graph ---

    std::optional<std::int64_t> hint(std::string_view symbol) const {
        for (const auto& [name, value] : options_.dims) {
            if (name == symbol) {
                return value;
            }
        }
        return 1;
    }

    Shape shape_of(const ir::TensorType& type) const {
        Shape shape;
        for (const Dim& dim : type.shape) {
            const std::optional<std::int64_t> value = dim.evaluate([&](std::string_view s) { return hint(s); });
            shape.push_back(std::max<std::int64_t>(value.value_or(1), 0));
        }
        return shape;
    }

    void index_graph() {
        std::unordered_map<const ir::Value*, int> value_index;
        for (const auto& value : graph_.values()) {
            ValueRec rec;
            rec.kind = value->kind();
            const Shape shape = shape_of(value->type());
            shapes_.push_back(shape);
            rec.bytes = static_cast<std::uint64_t>(storage_size(value->dtype(), shape.numel()));
            value_index.emplace(value.get(), static_cast<int>(values_.size()));
            values_.push_back(std::move(rec));
        }
        for (std::size_t s = 0; s < graph_.states().size(); ++s) {
            const ir::StateBinding& binding = graph_.states()[s];
            values_[static_cast<std::size_t>(value_index.at(binding.state))].state = static_cast<int>(s);
            if (binding.update != nullptr) {
                values_[static_cast<std::size_t>(value_index.at(binding.update))].updates.push_back(
                    static_cast<int>(s));
            }
            state_values_.push_back(value_index.at(binding.state));
            state_updates_.push_back(binding.update != nullptr ? value_index.at(binding.update) : -1);
        }
        for (const ir::GraphOutput& output : graph_.outputs()) {
            values_[static_cast<std::size_t>(value_index.at(output.value))].output = true;
        }
        for (const auto& node : graph_.nodes()) {
            NodeRec rec;
            rec.node = node.get();
            const int n = static_cast<int>(nodes_.size());
            for (const ir::Value* input : node->inputs()) {
                const int v = input == nullptr ? -1 : value_index.at(input);
                rec.inputs.push_back(v);
                if (v >= 0) {
                    std::vector<int>& consumers = values_[static_cast<std::size_t>(v)].consumers;
                    if (consumers.empty() || consumers.back() != n) {
                        consumers.push_back(n);
                    }
                }
            }
            for (const ir::Value* output : node->outputs()) {
                const int v = value_index.at(output);
                values_[static_cast<std::size_t>(v)].producer = n;
                rec.outputs.push_back(v);
            }
            nodes_.push_back(std::move(rec));
        }
    }

    Status estimate_nodes() {
        std::map<std::string, int> declines;
        std::vector<Shape> in_shapes;
        std::vector<Shape> out_shapes;
        for (NodeRec& rec : nodes_) {
            in_shapes.clear();
            out_shapes.clear();
            for (const int v : rec.inputs) {
                in_shapes.push_back(v < 0 ? Shape() : shapes_[static_cast<std::size_t>(v)]);
            }
            for (const int v : rec.outputs) {
                out_shapes.push_back(shapes_[static_cast<std::size_t>(v)]);
            }
            const ir::CostContext context{*rec.node, in_shapes, out_shapes};
            rec.time.assign(backends_.size(), kInf);
            for (std::size_t b = 0; b < backends_.size(); ++b) {
                const backend::Backend& backend = *backends_[b].backend;
                if (Status status = backend.supports(*rec.node); !status.ok()) {
                    ++declines[std::string(status.message())];
                    continue;
                }
                rec.time[b] = std::max(backend.estimate_us(context), 0.0);
                rec.candidates.push_back(static_cast<int>(b));
            }
            if (rec.candidates.empty()) {
                return errors::unsupported("no backend can execute '{}'", rec.node->op_name());
            }
        }
        for (auto& [reason, count] : declines) {
            declines_.push_back(Decline{reason, count});
        }
        return ok_status();
    }

    // --- Costs ---

    int device_of_backend(int b) const { return backends_[static_cast<std::size_t>(b)].device; }
    int device_of_node(int n) const { return device_of_backend(assign_[static_cast<std::size_t>(n)]); }

    /// The device a value lives on; -1 for constants (copied anywhere for free) and states not placed yet.
    int home(int v) const {
        const ValueRec& rec = values_[static_cast<std::size_t>(v)];
        switch (rec.kind) {
        case ir::ValueKind::kInput:
            return 0;
        case ir::ValueKind::kConstant:
            return -1;
        case ir::ValueKind::kState:
            return state_device_[static_cast<std::size_t>(rec.state)];
        case ir::ValueKind::kNode:
            return assign_[static_cast<std::size_t>(rec.producer)] < 0 ? -1 : device_of_node(rec.producer);
        }
        return 0;
    }

    double link_us(int device, std::uint64_t bytes) const {
        const backend::DeviceInfo& info = devices_[static_cast<std::size_t>(device)].info;
        return info.link_latency_us + static_cast<double>(bytes) / (std::max(info.link_gbs, 1e-3) * 1e3);
    }

    double transfer_us(std::uint64_t bytes, int from, int to) const {
        if (from == to) {
            return 0;
        }
        return (from != 0 ? link_us(from, bytes) : 0.0) + (to != 0 ? link_us(to, bytes) : 0.0);
    }

    /// Devices (bit mask) other than the value's own that need a copy of it.
    std::uint64_t needed_devices(int v) const {
        const ValueRec& rec = values_[static_cast<std::size_t>(v)];
        std::uint64_t mask = 0;
        for (const int c : rec.consumers) {
            if (assign_[static_cast<std::size_t>(c)] >= 0) {
                mask |= std::uint64_t{1} << device_of_node(c);
            }
        }
        if (rec.output) {
            mask |= 1;
        }
        for (const int s : rec.updates) {
            if (const int d = state_device_[static_cast<std::size_t>(s)]; d >= 0) {
                mask |= std::uint64_t{1} << d;
            }
        }
        const int h = home(v);
        if (h >= 0) {
            mask &= ~(std::uint64_t{1} << h);
        }
        return mask;
    }

    /// Cost of copying a value to the devices that need it.
    double value_cost(int v) const {
        const ValueRec& rec = values_[static_cast<std::size_t>(v)];
        const int h = home(v);
        if (rec.kind == ir::ValueKind::kConstant || h < 0) {
            return 0;
        }
        double cost = 0;
        for (std::uint64_t mask = needed_devices(v); mask != 0; mask &= mask - 1) {
            cost += transfer_us(rec.bytes, h, std::countr_zero(mask));
        }
        return cost;
    }

    double total_cost() const {
        double cost = 0;
        for (std::size_t n = 0; n < nodes_.size(); ++n) {
            cost += nodes_[n].time[static_cast<std::size_t>(assign_[n])];
        }
        for (std::size_t v = 0; v < values_.size(); ++v) {
            cost += value_cost(static_cast<int>(v));
        }
        return cost;
    }

    /// Values whose copy cost depends on where node `n` runs.
    void affected_values(int n, std::vector<int>& out) const {
        out.clear();
        const NodeRec& rec = nodes_[static_cast<std::size_t>(n)];
        for (const int v : rec.inputs) {
            if (v >= 0 && values_[static_cast<std::size_t>(v)].kind != ir::ValueKind::kConstant &&
                std::find(out.begin(), out.end(), v) == out.end()) {
                out.push_back(v);
            }
        }
        for (const int v : rec.outputs) {
            out.push_back(v);
        }
    }

    // --- Memory ---

    DeviceUsage usage(int d) const {
        DeviceUsage result;
        const std::size_t end = nodes_.size();
        std::vector<char> counted(values_.size(), 0);
        std::vector<MemoryRequest> requests;
        for (std::size_t n = 0; n < nodes_.size(); ++n) {
            if (device_of_node(static_cast<int>(n)) != d) {
                continue;
            }
            for (const int v : nodes_[n].inputs) {
                if (v < 0 || counted[static_cast<std::size_t>(v)] != 0) {
                    continue;
                }
                const ValueRec& rec = values_[static_cast<std::size_t>(v)];
                if (rec.kind == ir::ValueKind::kConstant) {
                    result.weights += rec.bytes;
                    counted[static_cast<std::size_t>(v)] = 1;
                } else if (home(v) != d && rec.kind != ir::ValueKind::kState) {
                    // A copy of a value from another device, alive between its first and last reader here.
                    int first = static_cast<int>(n);
                    int last = first;
                    for (const int c : rec.consumers) {
                        if (device_of_node(c) == d) {
                            first = std::min(first, c);
                            last = std::max(last, c);
                        }
                    }
                    requests.push_back(MemoryRequest{rec.bytes, first, last});
                    counted[static_cast<std::size_t>(v)] = 1;
                }
            }
            for (const int v : nodes_[n].outputs) {
                const ValueRec& rec = values_[static_cast<std::size_t>(v)];
                int last = static_cast<int>(n);
                if (!rec.consumers.empty()) {
                    last = std::max(last, rec.consumers.back());
                }
                if (rec.output || !rec.updates.empty()) {
                    last = static_cast<int>(end);
                }
                requests.push_back(MemoryRequest{rec.bytes, static_cast<int>(n), last});
            }
        }
        for (std::size_t s = 0; s < state_device_.size(); ++s) {
            if (state_device_[s] == d) {
                result.states += values_[static_cast<std::size_t>(state_values_[s])].bytes;
            }
        }
        result.activations = plan_memory(requests).total;
        return result;
    }

    std::uint64_t budget(int d) const {
        const std::uint64_t limit = devices_[static_cast<std::size_t>(d)].memory_limit;
        return static_cast<std::uint64_t>(static_cast<double>(limit) * (1.0 - options_.memory_reserve));
    }

    bool fits(int d) const {
        return devices_[static_cast<std::size_t>(d)].memory_limit == 0 || usage(d).total() <= budget(d);
    }

    // --- Assignment ---

    /// The device of a state: where its update is produced (an in-place write such as a KV cache update), else where
    /// it is first read, else the host.
    int natural_state_device(std::size_t s) const {
        const int update = state_updates_[s];
        if (update >= 0 && values_[static_cast<std::size_t>(update)].kind == ir::ValueKind::kNode) {
            return home(update);
        }
        const ValueRec& rec = values_[static_cast<std::size_t>(state_values_[s])];
        if (!rec.consumers.empty()) {
            return device_of_node(rec.consumers.front());
        }
        return 0;
    }

    void greedy() {
        assign_.assign(nodes_.size(), -1);
        state_device_.assign(state_values_.size(), -1);
        std::vector<std::uint64_t> available(values_.size(), 0);  // devices holding the value or a copy
        auto available_on = [&](int v) {
            const int h = home(v);
            return available[static_cast<std::size_t>(v)] | (h >= 0 ? std::uint64_t{1} << h : 0);
        };
        for (std::size_t n = 0; n < nodes_.size(); ++n) {
            const NodeRec& rec = nodes_[n];
            double best = kInf;
            int chosen = rec.candidates.front();
            for (const int b : rec.candidates) {
                const int d = device_of_backend(b);
                double t = rec.time[static_cast<std::size_t>(b)];
                for (const int v : rec.inputs) {
                    if (v < 0 || values_[static_cast<std::size_t>(v)].kind == ir::ValueKind::kConstant) {
                        continue;
                    }
                    const int h = home(v);
                    if (h >= 0 && (available_on(v) & (std::uint64_t{1} << d)) == 0) {
                        t += transfer_us(values_[static_cast<std::size_t>(v)].bytes, h, d);
                    }
                }
                if (t < best * (1 - 1e-12)) {
                    best = t;
                    chosen = b;
                }
            }
            assign_[n] = chosen;
            const int d = device_of_backend(chosen);
            for (const int v : rec.inputs) {
                if (v < 0) {
                    continue;
                }
                const ValueRec& value = values_[static_cast<std::size_t>(v)];
                if (value.kind == ir::ValueKind::kState && state_device_[static_cast<std::size_t>(value.state)] < 0) {
                    state_device_[static_cast<std::size_t>(value.state)] = d;
                }
                available[static_cast<std::size_t>(v)] |= std::uint64_t{1} << d;
            }
        }
        for (std::size_t s = 0; s < state_device_.size(); ++s) {
            state_device_[s] = natural_state_device(s);
        }
    }

    /// Moves nodes from the front of the graph off devices that exceed their memory limit.
    Status apply_memory_limits() {
        for (std::size_t d = 1; d < devices_.size(); ++d) {
            const int device = static_cast<int>(d);
            if (fits(device)) {
                continue;
            }
            const DeviceUsage before = usage(device);
            int moved = 0;
            for (std::size_t n = 0; n < nodes_.size() && !fits(device); ++n) {
                if (device_of_node(static_cast<int>(n)) != device) {
                    continue;
                }
                const NodeRec& rec = nodes_[n];
                int best = -1;
                for (const int b : rec.candidates) {
                    if (device_of_backend(b) != device && (best < 0 || rec.time[static_cast<std::size_t>(b)] <
                                                                           rec.time[static_cast<std::size_t>(best)])) {
                        best = b;
                    }
                }
                if (best < 0) {
                    continue;  // only this device can run it
                }
                assign_[n] = best;
                ++moved;
                for (std::size_t s = 0; s < state_device_.size(); ++s) {
                    state_device_[s] = natural_state_device(s);
                }
            }
            const DeviceUsage after = usage(device);
            const std::string name = to_string(devices_[d].info.device);
            if (after.total() > budget(device)) {
                return errors::out_of_memory(
                    "the model does not fit into the memory of {}: it needs {} (weights {}, states {}, activations {}) "
                    "even with every movable operation on other devices; the limit is {} minus {:.0f}% reserve",
                    name, format_bytes(after.total()), format_bytes(after.weights), format_bytes(after.states),
                    format_bytes(after.activations), format_bytes(devices_[d].memory_limit),
                    options_.memory_reserve * 100);
            }
            notes_.push_back(std::format("{}: {} operations moved to other devices to fit the memory limit of {} "
                                         "(needed {}, now {})",
                                         name, moved, format_bytes(devices_[d].memory_limit),
                                         format_bytes(before.total()), format_bytes(after.total())));
        }
        return ok_status();
    }

    /// Local search over single nodes, islands and states.
    void refine() {
        double cost = total_cost();
        std::vector<int> affected;
        auto accept = [&](double candidate, int device) {
            return candidate < cost - 1e-9 * std::max(1.0, std::abs(cost)) && (device < 0 || fits(device));
        };
        for (int round = 0; round < 8; ++round) {
            bool improved = false;
            // Single nodes.
            for (std::size_t n = 0; n < nodes_.size(); ++n) {
                const NodeRec& rec = nodes_[n];
                if (rec.candidates.size() < 2) {
                    continue;
                }
                affected_values(static_cast<int>(n), affected);
                const int current = assign_[n];
                double before = rec.time[static_cast<std::size_t>(current)];
                for (const int v : affected) {
                    before += value_cost(v);
                }
                for (const int b : rec.candidates) {
                    if (b == current) {
                        continue;
                    }
                    assign_[n] = b;
                    double after = rec.time[static_cast<std::size_t>(b)];
                    for (const int v : affected) {
                        after += value_cost(v);
                    }
                    const int target = device_of_backend(b);
                    if (accept(cost - before + after, target == device_of_backend(current) ? -1 : target)) {
                        cost = cost - before + after;
                        before = after;
                        improved = true;
                        break;
                    }
                    assign_[n] = current;
                }
            }
            // Islands: runs of consecutive nodes on one device move together to another device.
            for (std::size_t begin = 0; begin < nodes_.size();) {
                const int device = device_of_node(static_cast<int>(begin));
                std::size_t end = begin + 1;
                while (end < nodes_.size() && device_of_node(static_cast<int>(end)) == device) {
                    ++end;
                }
                if (end - begin > 1) {
                    for (std::size_t target = 0; target < devices_.size(); ++target) {
                        if (static_cast<int>(target) == device) {
                            continue;
                        }
                        std::vector<int> saved(assign_.begin() + static_cast<std::ptrdiff_t>(begin),
                                               assign_.begin() + static_cast<std::ptrdiff_t>(end));
                        bool movable = true;
                        for (std::size_t n = begin; n < end && movable; ++n) {
                            int best = -1;
                            for (const int b : nodes_[n].candidates) {
                                if (device_of_backend(b) == static_cast<int>(target) &&
                                    (best < 0 || nodes_[n].time[static_cast<std::size_t>(b)] <
                                                     nodes_[n].time[static_cast<std::size_t>(best)])) {
                                    best = b;
                                }
                            }
                            movable = best >= 0;
                            assign_[n] = best;
                        }
                        if (movable) {
                            const double candidate = total_cost();
                            if (accept(candidate, static_cast<int>(target))) {
                                cost = candidate;
                                improved = true;
                                break;
                            }
                        }
                        std::copy(saved.begin(), saved.end(), assign_.begin() + static_cast<std::ptrdiff_t>(begin));
                    }
                }
                begin = end;
            }
            // States.
            for (std::size_t s = 0; s < state_device_.size(); ++s) {
                const int current = state_device_[s];
                for (std::size_t target = 0; target < devices_.size(); ++target) {
                    if (static_cast<int>(target) == current) {
                        continue;
                    }
                    state_device_[s] = static_cast<int>(target);
                    const double candidate = total_cost();
                    if (accept(candidate, static_cast<int>(target))) {
                        cost = candidate;
                        improved = true;
                        break;
                    }
                    state_device_[s] = current;
                }
            }
            if (!improved) {
                break;
            }
        }
    }

    GraphPlan result() const {
        GraphPlan plan;
        plan.node_backend = assign_;
        plan.state_device = state_device_;
        plan.device_us.assign(devices_.size(), 0.0);
        for (std::size_t n = 0; n < nodes_.size(); ++n) {
            const double t = nodes_[n].time[static_cast<std::size_t>(assign_[n])];
            plan.node_us.push_back(t);
            plan.device_us[static_cast<std::size_t>(device_of_node(static_cast<int>(n)))] += t;
        }
        for (std::size_t v = 0; v < values_.size(); ++v) {
            const int h = home(static_cast<int>(v));
            if (values_[v].kind == ir::ValueKind::kConstant || h < 0) {
                continue;
            }
            for (std::uint64_t mask = needed_devices(static_cast<int>(v)); mask != 0; mask &= mask - 1) {
                const int to = std::countr_zero(mask);
                ++plan.transfers;
                plan.transfer_bytes += values_[v].bytes;
                plan.transfer_us += transfer_us(values_[v].bytes, h, to);
            }
        }
        for (std::size_t d = 0; d < devices_.size(); ++d) {
            plan.usage.push_back(usage(static_cast<int>(d)));
        }
        plan.notes = notes_;
        plan.declines = declines_;
        return plan;
    }

    const ir::Graph& graph_;
    std::span<const PlanBackend> backends_;
    std::span<const PlanDevice> devices_;
    const PlannerOptions& options_;

    std::vector<NodeRec> nodes_;
    std::vector<ValueRec> values_;
    std::vector<Shape> shapes_;
    std::vector<int> state_values_;   ///< per state: its value
    std::vector<int> state_updates_;  ///< per state: the update value or -1
    std::vector<int> assign_;         ///< per node: backend index
    std::vector<int> state_device_;   ///< per state
    std::vector<std::string> notes_;
    std::vector<Decline> declines_;
};

}  // namespace

Result<GraphPlan> plan_graph(const ir::Graph& graph, std::span<const PlanBackend> backends,
                             std::span<const PlanDevice> devices, const PlannerOptions& options) {
    return Planner(graph, backends, devices, options).run();
}

}  // namespace ii::runtime
