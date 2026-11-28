// SPDX-License-Identifier: MIT
#include "runtime/partition.h"

#include "ir/op_schema.h"
#include "ops/fused.h"

#include <algorithm>
#include <array>
#include <format>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ii::runtime {

namespace {

/// A part being cut out of the graph: node indices in graph (topological) order.
using Part = std::vector<int>;

class Partitioner {
public:
    Partitioner(ir::Graph& graph, GraphPlan& plan, std::span<const PlanBackend> backends, const PlannerOptions& options)
        : graph_(graph), plan_(plan), backends_(backends), options_(options) {}

    Status run() {
        for (std::size_t g = 0; g < backends_.size(); ++g) {
            if (backends_[g].backend->compiles_subgraphs()) {
                II_TRY(claim_for(static_cast<int>(g)));
            }
        }
        return ok_status();
    }

private:
    // --- Graph structure ---

    void index_nodes() {
        nodes_.clear();
        index_.clear();
        for (const auto& node : graph_.nodes()) {
            index_.emplace(node.get(), static_cast<int>(nodes_.size()));
            nodes_.push_back(node.get());
        }
    }

    /// Index of a node of the graph as it was when claiming started; -1 for `subgraph` nodes created since.
    int index_of(const ir::Node* node) const {
        const auto it = index_.find(node);
        return it == index_.end() ? -1 : it->second;
    }

    /// Readers of node `n`'s outputs (node indices, possibly repeated).
    std::vector<int> successors(int n) const {
        std::vector<int> out;
        for (const ir::Value* output : nodes_[static_cast<std::size_t>(n)]->outputs()) {
            for (const ir::Use& use : output->uses()) {
                out.push_back(index_.at(use.node));
            }
        }
        return out;
    }

    /// Connected components of `part` (edges between its nodes), each in graph order.
    std::vector<Part> components(const Part& part) const {
        std::unordered_map<int, int> parent;
        for (const int n : part) {
            parent.emplace(n, n);
        }
        auto find = [&](int n) {
            while (parent.at(n) != n) {
                parent[n] = parent.at(parent.at(n));
                n = parent.at(n);
            }
            return n;
        };
        for (const int n : part) {
            for (const int s : successors(n)) {
                if (parent.contains(s)) {
                    const int a = find(n);
                    const int b = find(s);
                    if (a != b) {
                        parent[std::max(a, b)] = std::min(a, b);
                    }
                }
            }
        }
        std::map<int, Part> groups;
        for (const int n : part) {
            groups[find(n)].push_back(n);
        }
        std::vector<Part> out;
        for (auto& [root, nodes] : groups) {
            out.push_back(std::move(nodes));
        }
        return out;
    }

    /// A node outside `part` on a path that leaves the part and comes back (the earliest such node), or -1 when the
    /// part is convex.
    int convexity_breaker(const Part& part) const {
        const std::unordered_set<int> inside(part.begin(), part.end());
        const int last = part.back();
        std::vector<int> stack;
        std::unordered_set<int> seen;
        for (const int n : part) {
            for (const int s : successors(n)) {
                if (!inside.contains(s) && s < last && seen.insert(s).second) {
                    stack.push_back(s);
                }
            }
        }
        int breaker = -1;
        while (!stack.empty()) {
            const int x = stack.back();
            stack.pop_back();
            for (const int s : successors(x)) {
                if (inside.contains(s)) {
                    breaker = breaker < 0 ? x : std::min(breaker, x);
                } else if (s < last && seen.insert(s).second) {
                    stack.push_back(s);
                }
            }
        }
        return breaker;
    }

    /// Splits `part` into connected, convex parts.
    void split(const Part& part, std::vector<Part>& out) const {
        for (Part& component : components(part)) {
            const int breaker = convexity_breaker(component);
            if (breaker < 0) {
                out.push_back(std::move(component));
                continue;
            }
            // Nodes before the breaker cannot depend on it; nodes after it are cut apart from them.
            Part before;
            Part after;
            for (const int n : component) {
                (n < breaker ? before : after).push_back(n);
            }
            split(before, out);
            split(after, out);
        }
    }

    // --- Claiming ---

    std::optional<std::int64_t> hint(std::string_view symbol) const {
        for (const auto& [name, value] : options_.dims) {
            if (name == symbol) {
                return value;
            }
        }
        return 1;
    }

    Shape assumed_shape(const ir::TensorType& type) const {
        Shape shape;
        for (const Dim& dim : type.shape) {
            const std::optional<std::int64_t> value = dim.evaluate([&](std::string_view s) { return hint(s); });
            shape.push_back(std::max<std::int64_t>(value.value_or(1), 0));
        }
        return shape;
    }

    /// The `subgraph` node replacing `part` (appended to the graph; its outputs are not used yet), or nullopt if the
    /// part cannot be one: it produces nothing used outside, or its program would use symbols its inputs do not bind.
    std::optional<ir::Node*> make_node(const Part& part, const PlanBackend& owner, std::vector<ir::Value*>& replaced) {
        const std::unordered_set<int> inside(part.begin(), part.end());
        auto produced_inside = [&](const ir::Value* value) {
            return value->producer() != nullptr && inside.contains(index_of(value->producer()));
        };
        ir::Graph program;
        program.set_name("subgraph");
        std::unordered_map<const ir::Value*, ir::Value*> mapping;
        std::vector<ir::Value*> inputs;
        std::set<std::string> bound;
        for (const int n : part) {
            for (ir::Value* value : nodes_[static_cast<std::size_t>(n)]->inputs()) {
                if (value == nullptr || produced_inside(value) || mapping.contains(value)) {
                    continue;
                }
                ir::Value* inner = program.add_input(std::format("x{}", inputs.size()), value->type());
                mapping.emplace(value, inner);
                inputs.push_back(value);
                for (const Dim& dim : value->shape()) {
                    if (const std::string_view name = dim.symbol_name(); !name.empty()) {
                        bound.emplace(name);
                    }
                }
            }
        }
        auto binds = [&](const Dims& dims) {
            for (const Dim& dim : dims) {
                for (const std::string& symbol : dim.symbols()) {
                    if (!bound.contains(symbol)) {
                        return false;
                    }
                }
            }
            return true;
        };
        std::vector<ir::TensorType> output_types;
        replaced.clear();
        for (const int n : part) {
            const ir::Node* node = nodes_[static_cast<std::size_t>(n)];
            std::vector<ir::Value*> node_inputs;
            for (ir::Value* value : node->inputs()) {
                node_inputs.push_back(value == nullptr ? nullptr : mapping.at(value));
            }
            std::vector<ir::TensorType> types;
            for (const ir::Value* output : node->outputs()) {
                types.push_back(output->type());
                if (!binds(output->shape())) {
                    return std::nullopt;
                }
            }
            for (const auto& [name, attr] : node->attrs()) {
                if (const Dims* dims = std::get_if<Dims>(&attr); dims != nullptr && !binds(*dims)) {
                    return std::nullopt;
                }
            }
            ir::Node* copy = program.create_node(node->op(), std::move(node_inputs), node->attrs(), types);
            copy->set_name(node->name());
            for (int o = 0; o < node->num_outputs(); ++o) {
                ir::Value* output = node->output(o);
                mapping.emplace(output, copy->output(o));
                bool outside = graph_.is_graph_result(output);
                for (const ir::Use& use : output->uses()) {
                    outside = outside || !inside.contains(index_of(use.node));
                }
                if (outside) {
                    program.add_output(std::format("y{}", replaced.size()), copy->output(o));
                    replaced.push_back(output);
                    output_types.push_back(output->type());
                }
            }
        }
        if (replaced.empty()) {
            return std::nullopt;
        }
        Attributes attrs;
        attrs.set("program", program.to_text());
        attrs.set("backend", std::string(owner.backend->name()));
        return graph_.create_node(ir::builtin_op(ops::kSubgraph), std::move(inputs), std::move(attrs), output_types);
    }

    double estimate_of(const backend::Backend& backend, const ir::Node& node) const {
        std::vector<Shape> in_shapes;
        std::vector<Shape> out_shapes;
        for (const ir::Value* input : node.inputs()) {
            in_shapes.push_back(input == nullptr ? Shape() : assumed_shape(input->type()));
        }
        for (const ir::Value* output : node.outputs()) {
            out_shapes.push_back(assumed_shape(output->type()));
        }
        return std::max(backend.estimate_us(ir::CostContext{node, in_shapes, out_shapes}), 0.0);
    }

    /// The nodes of `part` that the plan gave graph backend `g`, which cannot compile the part, go to the fastest other
    /// backend that runs them (on another device the copies are inserted later like any others).
    Status reassign(const Part& part, int g, std::unordered_map<const ir::Node*, std::pair<int, double>>& planned,
                    int& moved) {
        for (const int n : part) {
            if (plan_.node_backend[static_cast<std::size_t>(n)] != g) {
                continue;
            }
            const ir::Node& node = *nodes_[static_cast<std::size_t>(n)];
            int best = -1;
            double best_us = 0;
            for (std::size_t b = 0; b < backends_.size(); ++b) {
                const backend::Backend& candidate = *backends_[b].backend;
                if (candidate.compiles_subgraphs() || !candidate.supports(node).ok()) {
                    continue;
                }
                const double us = estimate_of(candidate, node);
                if (best < 0 || us < best_us) {
                    best = static_cast<int>(b);
                    best_us = us;
                }
            }
            if (best < 0) {
                return errors::unsupported("{} cannot compile '{}' (value {}) in a part, and no other backend runs it",
                                           backends_[static_cast<std::size_t>(g)].backend->name(), node.op_name(),
                                           node.output()->type().to_string());
            }
            plan_.node_backend[static_cast<std::size_t>(n)] = best;
            plan_.node_us[static_cast<std::size_t>(n)] = best_us;
            planned[&node] = std::make_pair(best, best_us);
            ++moved;
        }
        return ok_status();
    }

    Status claim_for(int g) {
        const PlanBackend& owner = backends_[static_cast<std::size_t>(g)];
        index_nodes();
        Part candidates;
        for (std::size_t n = 0; n < nodes_.size(); ++n) {
            const ir::Node& node = *nodes_[n];
            const int planned = plan_.node_backend[n];
            if (backends_[static_cast<std::size_t>(planned)].device != owner.device ||
                node.op_name() == ops::kSubgraph) {
                continue;
            }
            if (planned == g || owner.backend->supports(node).ok()) {
                candidates.push_back(static_cast<int>(n));
            }
        }
        std::vector<Part> parts;
        split(candidates, parts);

        std::unordered_map<const ir::Node*, std::pair<int, double>> planned;  // node -> backend, estimate
        for (std::size_t n = 0; n < nodes_.size(); ++n) {
            planned.emplace(nodes_[n], std::make_pair(plan_.node_backend[n], plan_.node_us[n]));
        }
        std::vector<ir::Node*> doomed;
        int moved = 0;
        int claimed = 0;
        int claimed_nodes = 0;
        double saved = 0;
        std::vector<ir::Value*> replaced;
        for (const Part& part : parts) {
            bool forced = false;
            double current = 0;
            for (const int n : part) {
                forced = forced || plan_.node_backend[static_cast<std::size_t>(n)] == g;
                current += plan_.node_us[static_cast<std::size_t>(n)];
            }
            const std::optional<ir::Node*> made = make_node(part, owner, replaced);
            if (!made) {
                if (forced) {
                    II_TRY(reassign(part, g, planned, moved));
                }
                continue;
            }
            ir::Node* node = *made;
            const Status supported = owner.backend->supports(*node);
            double estimate = 0;
            if (supported.ok()) {
                estimate = estimate_of(*owner.backend, *node);
            }
            if (!supported.ok() || (!forced && estimate >= current)) {
                const std::array<ir::Node*, 1> unused{node};
                graph_.erase_nodes(unused);
                if (forced) {
                    II_TRY(reassign(part, g, planned, moved));
                }
                continue;
            }
            for (std::size_t o = 0; o < replaced.size(); ++o) {
                graph_.replace_all_uses(replaced[o], node->output(static_cast<int>(o)));
            }
            for (const int n : part) {
                doomed.push_back(nodes_[static_cast<std::size_t>(n)]);
            }
            planned.emplace(node, std::make_pair(g, estimate));
            ++claimed;
            claimed_nodes += static_cast<int>(part.size());
            saved += current - estimate;
        }
        if (moved > 0) {
            plan_.notes.push_back(std::format("{} cannot compile {} operation{} planned for it (in parts whose shapes "
                                              "their inputs do not determine, or that it declines as a whole); other "
                                              "backends run them",
                                              owner.backend->name(), moved, moved == 1 ? "" : "s"));
        }
        if (claimed == 0) {
            return ok_status();
        }
        graph_.erase_nodes(doomed);
        II_TRY(graph_.topological_sort());
        plan_.node_backend.clear();
        plan_.node_us.clear();
        for (const auto& node : graph_.nodes()) {
            const auto& [backend, us] = planned.at(node.get());
            plan_.node_backend.push_back(backend);
            plan_.node_us.push_back(us);
        }
        plan_.device_us[static_cast<std::size_t>(owner.device)] -= saved;
        plan_.notes.push_back(std::format("{} compiles {} part{} of the graph ({} operations)", owner.backend->name(),
                                          claimed, claimed == 1 ? "" : "s", claimed_nodes));
        return ok_status();
    }

    ir::Graph& graph_;
    GraphPlan& plan_;
    std::span<const PlanBackend> backends_;
    const PlannerOptions& options_;
    std::vector<ir::Node*> nodes_;
    std::unordered_map<const ir::Node*, int> index_;
};

}  // namespace

Status claim_subgraphs(ir::Graph& graph, GraphPlan& plan, std::span<const PlanBackend> backends,
                       const PlannerOptions& options) {
    return Partitioner(graph, plan, backends, options).run();
}

}  // namespace ii::runtime
