// SPDX-License-Identifier: MIT
/// @file
/// `OpSchema` — the single source of truth about an operation: signature, attributes with defaults, shape and type
/// inference, decomposition into simpler operations (composite ops), cost model and documentation.
///
/// Adding an operation = a schema (src/ops) + a reference kernel (src/backends/ref) + conformance tests.
#pragma once

#include <ii/attribute.h>
#include <ii/status.h>

#include "ir/graph.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ii::ir {

class Builder;

/// A formal input of an operation.
struct InputSpec {
    std::string_view name;
    bool optional = false;  ///< May be absent (nullptr). Optional inputs follow all required ones.
    bool variadic = false;  ///< The last input may repeat (`concat`); at least one occurrence unless optional.
};

/// A formal output of an operation.
struct OutputSpec {
    std::string_view name;
    bool variadic = false;  ///< The last output may repeat (`split`); the count is decided by shape inference.
};

/// A declared attribute. `default_value` empty means the attribute is required.
struct AttrSpec {
    std::string_view name;
    AttrKind kind;
    std::optional<AttrValue> default_value;
};

/// Inputs of shape inference.
struct InferContext {
    const OpSchema& op;
    std::span<const TensorType* const> inputs;  ///< nullptr for absent optional inputs
    const Attributes& attrs;                    ///< canonical: all declared attributes present

    const TensorType* input(int index) const noexcept {
        return index >= 0 && static_cast<std::size_t>(index) < inputs.size() ? inputs[static_cast<std::size_t>(index)]
                                                                             : nullptr;
    }
    int num_inputs() const noexcept { return static_cast<int>(inputs.size()); }
};

/// Infers output types from input types and attributes; validates dtypes, ranks and attribute values.
using InferFn = Status (*)(const InferContext& context, std::vector<TensorType>& outputs);

/// Builds the replacement of a composite node from simpler operations (inserted by `builder` before the node) and
/// returns the values replacing the node's outputs.
using DecomposeFn = Status (*)(Builder& builder, const Node& node, std::vector<Value*>& outputs);

/// Estimated work of a node.
struct Cost {
    double flops = 0;  ///< Arithmetic operations.
    double bytes = 0;  ///< Bytes read and written (inputs + outputs, ignoring caches).
};

/// Concrete shapes for cost estimation.
struct CostContext {
    const Node& node;
    std::span<const Shape> inputs;  ///< empty Shape for absent inputs
    std::span<const Shape> outputs;
};

using CostFn = Cost (*)(const CostContext& context);

/// Builds the backward computation of a node (training, stage 7).
struct GradContext;
using GradFn = Status (*)(GradContext& context);

/// Operation properties.
enum OpFlags : std::uint32_t {
    kOpNone = 0,
    kOpElementwise = 1u << 0,     ///< Output element i depends only on input elements i (after broadcasting).
    kOpCommutative = 1u << 1,     ///< Binary op with op(a, b) == op(b, a).
    kOpRandom = 1u << 2,          ///< Produces pseudo-random values (never constant-folded or deduplicated).
    kOpDataMovement = 1u << 3,    ///< Only moves/copies elements (reshape, permute, slice, ...).
    kOpNoConstantFold = 1u << 4,  ///< Never constant-folded (e.g. results would be very large).
    kOpInternal = 1u << 5,        ///< Created by compiler passes only (fusion); not part of the public OpSet.
    kOpView = 1u << 6,            ///< Output 0 is input 0 reinterpreted (same bytes): the runtime may alias them.
};

struct OpSchema {
    std::string_view name;
    std::string_view category;  ///< "unary", "binary", "reduction", "composite", ...
    std::string_view doc;
    int since_version = 1;
    std::vector<InputSpec> inputs;
    std::vector<OutputSpec> outputs;
    std::vector<AttrSpec> attrs;
    InferFn infer = nullptr;
    DecomposeFn decompose = nullptr;  ///< Non-null for composite operations.
    CostFn cost = nullptr;            ///< nullptr: elementwise-style estimate from tensor sizes.
    GradFn gradient = nullptr;        ///< Backward rule; filled in with training support (stage 7).
    std::uint32_t flags = kOpNone;

    bool is_composite() const noexcept { return decompose != nullptr; }
    bool has_flag(OpFlags flag) const noexcept { return (flags & flag) != 0; }

    /// The declared attribute or nullptr.
    const AttrSpec* find_attr(std::string_view attr_name) const noexcept;

    int min_inputs() const noexcept;
    /// -1 when the last input is variadic.
    int max_inputs() const noexcept;
};

/// Validates `attrs` against the schema: rejects unknown attributes and wrong kinds, converts compatible kinds
/// (int → float, ints → dims, all-constant dims → ints), and adds defaults for missing optional attributes.
Status canonicalize_attributes(const OpSchema& op, Attributes& attrs);

/// Checks the number of inputs (and which ones are present) against the schema.
Status check_arity(const OpSchema& op, std::span<const TensorType* const> inputs);

/// Runs arity checks and shape inference for a node-to-be.
Status infer_types(const OpSchema& op, std::span<const TensorType* const> inputs, const Attributes& attrs,
                   std::vector<TensorType>& outputs);

/// Estimated cost of a node with concrete shapes.
Cost estimate_cost(const CostContext& context);

/// A set of operation schemas, looked up by name.
class OpRegistry {
public:
    OpRegistry() = default;
    OpRegistry(const OpRegistry&) = delete;
    OpRegistry& operator=(const OpRegistry&) = delete;

    /// Adds a schema; the name must be new.
    void add(OpSchema schema);

    const OpSchema* find(std::string_view name) const noexcept;
    const std::vector<std::unique_ptr<OpSchema>>& all() const noexcept { return schemas_; }

private:
    std::vector<std::unique_ptr<OpSchema>> schemas_;
    std::unordered_map<std::string_view, const OpSchema*> by_name_;
};

/// The built-in operation set (OpSet v1). Built on first use and immutable afterwards.
const OpRegistry& default_op_registry();

/// Looks up a built-in operation; terminates if it does not exist (for use with literal names in ii code).
const OpSchema& builtin_op(std::string_view name);

}  // namespace ii::ir
