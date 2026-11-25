// SPDX-License-Identifier: MIT
#include "ir/op_schema.h"

#include "core/assert.h"
#include "ops/opset.h"

namespace ii::ir {

const AttrSpec* OpSchema::find_attr(std::string_view attr_name) const noexcept {
    for (const AttrSpec& spec : attrs) {
        if (spec.name == attr_name) {
            return &spec;
        }
    }
    return nullptr;
}

int OpSchema::min_inputs() const noexcept {
    int count = 0;
    for (const InputSpec& input : inputs) {
        if (!input.optional) {
            ++count;
        }
    }
    return count;
}

int OpSchema::max_inputs() const noexcept {
    if (!inputs.empty() && inputs.back().variadic) {
        return -1;
    }
    return static_cast<int>(inputs.size());
}

namespace {

/// Converts `value` to `kind` if the conversion is lossless and meaningful.
std::optional<AttrValue> convert(const AttrValue& value, AttrKind kind) {
    const AttrKind from = kind_of(value);
    if (from == kind) {
        return value;
    }
    switch (kind) {
    case AttrKind::kFloat:
        if (from == AttrKind::kInt) {
            return static_cast<double>(std::get<std::int64_t>(value));
        }
        break;
    case AttrKind::kFloats:
        if (from == AttrKind::kInts) {
            const auto& ints = std::get<std::vector<std::int64_t>>(value);
            std::vector<double> floats;
            floats.reserve(ints.size());
            for (const std::int64_t i : ints) {
                floats.push_back(static_cast<double>(i));
            }
            return floats;
        }
        break;
    case AttrKind::kDims:
        if (from == AttrKind::kInts) {
            const auto& ints = std::get<std::vector<std::int64_t>>(value);
            return Dims(ints.begin(), ints.end());
        }
        break;
    case AttrKind::kInts:
        if (from == AttrKind::kDims) {
            std::vector<std::int64_t> ints;
            for (const Dim& dim : std::get<Dims>(value)) {
                if (!dim.is_constant()) {
                    return std::nullopt;
                }
                ints.push_back(dim.value());
            }
            return ints;
        }
        if (from == AttrKind::kInt) {
            return std::vector<std::int64_t>{std::get<std::int64_t>(value)};
        }
        break;
    case AttrKind::kBool:
        if (from == AttrKind::kInt) {
            const std::int64_t i = std::get<std::int64_t>(value);
            if (i == 0 || i == 1) {
                return i == 1;
            }
        }
        break;
    case AttrKind::kStrings:
        if (from == AttrKind::kInts && std::get<std::vector<std::int64_t>>(value).empty()) {
            return std::vector<std::string>{};
        }
        break;
    default:
        break;
    }
    // An empty list "[]" parses as ints; accept it for every list kind.
    if (from == AttrKind::kInts && std::get<std::vector<std::int64_t>>(value).empty()) {
        if (kind == AttrKind::kFloats) {
            return std::vector<double>{};
        }
        if (kind == AttrKind::kDims) {
            return Dims{};
        }
    }
    return std::nullopt;
}

}  // namespace

Status canonicalize_attributes(const OpSchema& op, Attributes& attrs) {
    for (const auto& [name, value] : attrs) {
        if (op.find_attr(name) == nullptr) {
            return errors::invalid_argument("'{}' has no attribute '{}'", op.name, name);
        }
    }
    for (const AttrSpec& spec : op.attrs) {
        const AttrValue* value = attrs.find(spec.name);
        if (value == nullptr) {
            if (!spec.default_value) {
                return errors::invalid_argument("'{}' requires attribute '{}'", op.name, spec.name);
            }
            attrs.set(spec.name, *spec.default_value);
            continue;
        }
        if (kind_of(*value) != spec.kind) {
            std::optional<AttrValue> converted = convert(*value, spec.kind);
            if (!converted) {
                return errors::invalid_argument("attribute '{}' of '{}' must be {}, got {} ({})", spec.name, op.name,
                                                to_string(spec.kind), to_string(kind_of(*value)), to_string(*value));
            }
            attrs.set(spec.name, std::move(*converted));
        }
    }
    return ok_status();
}

Status check_arity(const OpSchema& op, std::span<const TensorType* const> inputs) {
    const int count = static_cast<int>(inputs.size());
    const int min_count = op.min_inputs();
    const int max_count = op.max_inputs();
    if (count < min_count || (max_count >= 0 && count > max_count)) {
        if (max_count < 0) {
            return errors::invalid_argument("'{}' expects at least {} inputs, got {}", op.name, min_count, count);
        }
        if (min_count == max_count) {
            return errors::invalid_argument("'{}' expects {} inputs, got {}", op.name, min_count, count);
        }
        return errors::invalid_argument("'{}' expects {} to {} inputs, got {}", op.name, min_count, max_count, count);
    }
    for (int i = 0; i < count; ++i) {
        const std::size_t spec_index = std::min(static_cast<std::size_t>(i), op.inputs.size() - 1);
        const InputSpec& spec = op.inputs[spec_index];
        if (inputs[static_cast<std::size_t>(i)] == nullptr && !spec.optional) {
            return errors::invalid_argument("input {} ('{}') of '{}' is required", i, spec.name, op.name);
        }
        if (const TensorType* type = inputs[static_cast<std::size_t>(i)]; type != nullptr) {
            if (type->dtype == DType::kInvalid || type->dtype == DType::kCount || type->rank() > kMaxRank) {
                return errors::invalid_argument("input {} of '{}' has an invalid type {}", i, op.name,
                                                type->to_string());
            }
            // Constant dimensions are bounded so that shape arithmetic in inference cannot overflow.
            for (const Dim& dim : type->shape) {
                if (dim.is_constant() && (dim.value() < 0 || dim.value() > (std::int64_t{1} << 48))) {
                    return errors::invalid_argument("input {} of '{}' has the invalid dimension {}", i, op.name,
                                                    dim.value());
                }
            }
        }
    }
    return ok_status();
}

Status infer_types(const OpSchema& op, std::span<const TensorType* const> inputs, const Attributes& attrs,
                   std::vector<TensorType>& outputs) {
    II_TRY(check_arity(op, inputs));
    outputs.clear();
    II_CHECK(op.infer != nullptr, "operation '{}' has no shape inference", op.name);
    II_TRY(op.infer(InferContext{op, inputs, attrs}, outputs));
    for (const TensorType& output : outputs) {
        if (output.rank() > kMaxRank) {
            return errors::invalid_argument("'{}' would produce a tensor of rank {} (maximum {})", op.name,
                                            output.rank(), kMaxRank);
        }
    }
    return ok_status();
}

Cost estimate_cost(const CostContext& context) {
    const OpSchema& op = context.node.op();
    if (op.cost != nullptr) {
        return op.cost(context);
    }
    // Default: one operation per output element; every input and output read/written once.
    Cost cost;
    for (std::size_t i = 0; i < context.inputs.size(); ++i) {
        if (const Value* input = context.node.input(static_cast<int>(i))) {
            cost.bytes += static_cast<double>(storage_size(input->dtype(), context.inputs[i].numel()));
        }
    }
    for (std::size_t i = 0; i < context.outputs.size(); ++i) {
        const std::int64_t numel = context.outputs[i].numel();
        cost.flops += static_cast<double>(numel);
        cost.bytes += static_cast<double>(storage_size(context.node.output(static_cast<int>(i))->dtype(), numel));
    }
    return cost;
}

void OpRegistry::add(OpSchema schema) {
    II_CHECK(!by_name_.contains(schema.name), "operation '{}' is registered twice", schema.name);
    II_CHECK(schema.infer != nullptr, "operation '{}' has no shape inference", schema.name);
    schemas_.push_back(std::make_unique<OpSchema>(std::move(schema)));
    const OpSchema* added = schemas_.back().get();
    by_name_.emplace(added->name, added);
}

const OpSchema* OpRegistry::find(std::string_view name) const noexcept {
    const auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : it->second;
}

const OpRegistry& default_op_registry() {
    // Immutable after construction; function-local statics are initialized thread-safely.
    static const OpRegistry* const instance = [] {
        // Intentionally never destroyed: schemas may still be referenced during static destruction.
        auto* registry = new OpRegistry();
        ops::register_opset(*registry);
        return registry;
    }();
    return *instance;
}

const OpSchema& builtin_op(std::string_view name) {
    const OpSchema* op = default_op_registry().find(name);
    II_CHECK(op != nullptr, "unknown built-in operation '{}'", name);
    return *op;
}

}  // namespace ii::ir
