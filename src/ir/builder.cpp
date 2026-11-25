// SPDX-License-Identifier: MIT
#include "ir/builder.h"

#include <cmath>
#include <limits>

namespace ii::ir {

Result<Node*> Builder::create_node(const OpSchema& op, std::vector<Value*> inputs, Attributes attrs) {
    II_TRY(canonicalize_attributes(op, attrs));
    std::vector<const TensorType*> input_types;
    input_types.reserve(inputs.size());
    for (const Value* input : inputs) {
        input_types.push_back(input != nullptr ? &input->type() : nullptr);
    }
    std::vector<TensorType> output_types;
    II_TRY(infer_types(op, input_types, attrs, output_types));
    return graph_->create_node(op, std::move(inputs), std::move(attrs), output_types, insertion_point_);
}

Result<Node*> Builder::create_node(std::string_view op, std::vector<Value*> inputs, Attributes attrs) {
    const OpSchema* schema = registry_->find(op);
    if (schema == nullptr) {
        return errors::not_found("unknown operation '{}'", op);
    }
    return create_node(*schema, std::move(inputs), std::move(attrs));
}

void Builder::fail(Status status) {
    if (status_.ok()) {
        status_ = std::move(status);
    }
}

Value* Builder::op(std::string_view name, std::initializer_list<Value*> inputs, Attributes attrs) {
    return op(name, std::vector<Value*>(inputs), std::move(attrs));
}

Value* Builder::op(std::string_view name, std::vector<Value*> inputs, Attributes attrs) {
    std::vector<Value*> outputs = op_outputs(name, std::move(inputs), std::move(attrs));
    return outputs.empty() ? nullptr : outputs.front();
}

std::vector<Value*> Builder::op_outputs(std::string_view name, std::vector<Value*> inputs, Attributes attrs) {
    if (!status_.ok()) {
        return {};
    }
    Result<Node*> node = create_node(name, std::move(inputs), std::move(attrs));
    if (!node.ok()) {
        fail(std::move(node).status());
        return {};
    }
    const auto outputs = (*node)->outputs();
    return {outputs.begin(), outputs.end()};
}

Value* Builder::constant(Tensor data) {
    if (!status_.ok()) {
        return nullptr;
    }
    if (!data.defined()) {
        fail(errors::invalid_argument("constant without data"));
        return nullptr;
    }
    return graph_->add_constant(std::move(data));
}

Value* Builder::scalar(double value, DType dtype) {
    if (!status_.ok()) {
        return nullptr;
    }
    Result<Tensor> tensor = make_scalar_tensor(value, dtype);
    if (!tensor.ok()) {
        fail(std::move(tensor).status());
        return nullptr;
    }
    return graph_->add_constant(std::move(*tensor));
}

namespace {

template <class T>
Result<Tensor> integer_scalar(double value) {
    if (std::isnan(value)) {
        return errors::invalid_argument("NaN cannot be converted to an integer scalar");
    }
    const double low = static_cast<double>(std::numeric_limits<T>::lowest());
    const double high = static_cast<double>(std::numeric_limits<T>::max());
    const T converted = value <= low    ? std::numeric_limits<T>::lowest()
                        : value >= high ? std::numeric_limits<T>::max()
                                        : static_cast<T>(value);
    return Tensor::scalar(converted);
}

}  // namespace

Result<Tensor> make_scalar_tensor(double value, DType dtype) {
    switch (dtype) {
    case DType::kF32:
        return Tensor::scalar(static_cast<float>(value));
    case DType::kF64:
        return Tensor::scalar(value);
    case DType::kF16:
        return Tensor::scalar(f16(static_cast<float>(value)));
    case DType::kBF16:
        return Tensor::scalar(bf16(static_cast<float>(value)));
    case DType::kI8:
        return integer_scalar<std::int8_t>(value);
    case DType::kU8:
        return integer_scalar<std::uint8_t>(value);
    case DType::kI16:
        return integer_scalar<std::int16_t>(value);
    case DType::kU16:
        return integer_scalar<std::uint16_t>(value);
    case DType::kI32:
        return integer_scalar<std::int32_t>(value);
    case DType::kU32:
        return integer_scalar<std::uint32_t>(value);
    case DType::kI64:
        return integer_scalar<std::int64_t>(value);
    case DType::kU64:
        return integer_scalar<std::uint64_t>(value);
    case DType::kBool:
        return Tensor::scalar(value != 0.0);
    default:
        return errors::invalid_argument("cannot create a scalar of type {}", to_string(dtype));
    }
}

}  // namespace ii::ir
