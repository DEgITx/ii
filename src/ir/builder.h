// SPDX-License-Identifier: MIT
/// @file
/// `ir::Builder` creates validated nodes: attributes are canonicalized, output types inferred, and the node inserted
/// at the builder's insertion point.
#pragma once

#include <ii/attribute.h>
#include <ii/status.h>

#include "ir/graph.h"
#include "ir/op_schema.h"

#include <initializer_list>
#include <string_view>
#include <vector>

namespace ii::ir {

class Builder {
public:
    explicit Builder(Graph& graph, const OpRegistry& registry = default_op_registry())
        : graph_(&graph), registry_(&registry) {}

    Graph& graph() const noexcept { return *graph_; }
    const OpRegistry& registry() const noexcept { return *registry_; }

    /// New nodes are inserted before `node`; nullptr appends at the end (the default).
    void set_insertion_point(Node* node) noexcept { insertion_point_ = node; }
    Node* insertion_point() const noexcept { return insertion_point_; }

    /// Creates a node. Absent optional inputs are nullptr.
    Result<Node*> create_node(const OpSchema& op, std::vector<Value*> inputs, Attributes attrs = {});
    Result<Node*> create_node(std::string_view op, std::vector<Value*> inputs, Attributes attrs = {});

    // --- "Sticky error" interface ---
    //
    // Convenient for multi-step constructions (decompositions, tests): an operation that fails records the first
    // error and returns nullptr; once an error is recorded, further calls do nothing and return nullptr. Check
    // `status()` at the end.

    /// Creates a node and returns its first output.
    Value* op(std::string_view name, std::initializer_list<Value*> inputs, Attributes attrs = {});
    Value* op(std::string_view name, std::vector<Value*> inputs, Attributes attrs = {});

    /// Creates a node and returns all its outputs.
    std::vector<Value*> op_outputs(std::string_view name, std::vector<Value*> inputs, Attributes attrs = {});

    /// A constant holding `data`.
    Value* constant(Tensor data);

    /// A rank-0 constant of `dtype` holding `value` (converted).
    Value* scalar(double value, DType dtype);

    /// The first error, or OK.
    const Status& status() const noexcept { return status_; }
    bool ok() const noexcept { return status_.ok(); }

    /// Records an error (if none is recorded yet).
    void fail(Status status);

private:
    Graph* graph_;
    const OpRegistry* registry_;
    Node* insertion_point_ = nullptr;
    Status status_;
};

/// Creates a rank-0 tensor of `dtype` holding `value` converted to that type.
Result<Tensor> make_scalar_tensor(double value, DType dtype);

}  // namespace ii::ir
