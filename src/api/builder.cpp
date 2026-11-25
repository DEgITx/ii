// SPDX-License-Identifier: MIT
// GraphBuilder, Var and the operation functions of the public builder API.
#include "api/builder_impl.h"
#include "ir/op_schema.h"

#include <format>

namespace ii {

// ---------------------------------------------------------------------------------------------------------------
// Var

namespace {

const ir::Value* value_of(const Var& var) {
    if (!var.valid()) {
        return nullptr;
    }
    const auto& values = var.builder()->impl().values;
    return static_cast<std::size_t>(var.id()) < values.size() ? values[static_cast<std::size_t>(var.id())] : nullptr;
}

}  // namespace

DType Var::dtype() const {
    const ir::Value* value = value_of(*this);
    return value != nullptr ? value->dtype() : DType::kInvalid;
}

Dims Var::shape() const {
    const ir::Value* value = value_of(*this);
    return value != nullptr ? value->shape() : Dims{};
}

int Var::rank() const {
    const ir::Value* value = value_of(*this);
    return value != nullptr ? value->rank() : 0;
}

// ---------------------------------------------------------------------------------------------------------------
// GraphBuilder

GraphBuilder::GraphBuilder(std::string name) : impl_(std::make_unique<detail::GraphBuilderImpl>()) {
    impl_->graph->set_name(std::move(name));
}

GraphBuilder::~GraphBuilder() = default;
GraphBuilder::GraphBuilder(GraphBuilder&& other) noexcept = default;
GraphBuilder& GraphBuilder::operator=(GraphBuilder&& other) noexcept = default;

const Status& GraphBuilder::status() const noexcept {
    return impl_->status;
}

void GraphBuilder::fail(Status status) {
    if (impl_->status.ok()) {
        impl_->status = std::move(status);
    }
}

Var GraphBuilder::input(std::string_view name, DType dtype, Dims shape) {
    if (!impl_->status.ok()) {
        return {};
    }
    if (name.empty() || impl_->graph->find_input(name) != nullptr || impl_->graph->find_state(name) != nullptr) {
        fail(errors::invalid_argument("input name '{}' is empty or already used", name));
        return {};
    }
    if (dtype == DType::kInvalid || dtype == DType::kCount || shape.size() > static_cast<std::size_t>(kMaxRank)) {
        fail(errors::invalid_argument("input '{}' has an invalid type", name));
        return {};
    }
    ir::Value* value = impl_->graph->add_input(std::string(name), ir::TensorType{dtype, std::move(shape)});
    return Var(this, impl_->add(value));
}

Var GraphBuilder::constant(const Tensor& tensor) {
    if (!impl_->status.ok()) {
        return {};
    }
    if (!tensor.defined()) {
        fail(errors::invalid_argument("constant: empty tensor"));
        return {};
    }
    if (tensor.device().kind != DeviceKind::kCpu) {
        fail(errors::unsupported("constant: only host tensors are supported"));
        return {};
    }
    return Var(this, impl_->add(impl_->graph->add_constant(tensor)));
}

Var GraphBuilder::scalar(double value, DType dtype) {
    if (!impl_->status.ok()) {
        return {};
    }
    Result<Tensor> tensor = ir::make_scalar_tensor(value, dtype);
    if (!tensor.ok()) {
        fail(std::move(tensor).status());
        return {};
    }
    return constant(*tensor);
}

Var GraphBuilder::state(std::string_view name, DType dtype, Dims shape, const Tensor& initial) {
    if (!impl_->status.ok()) {
        return {};
    }
    if (name.empty() || impl_->graph->find_input(name) != nullptr || impl_->graph->find_state(name) != nullptr) {
        fail(errors::invalid_argument("state name '{}' is empty or already used", name));
        return {};
    }
    ir::TensorType type{dtype, std::move(shape)};
    if (initial.defined() && !(ir::type_of(initial) == type)) {
        fail(errors::invalid_argument("initial value {} of state '{}' does not match {}",
                                      ir::type_of(initial).to_string(), name, type.to_string()));
        return {};
    }
    ir::Value* value = impl_->graph->add_state(std::string(name), std::move(type), initial);
    return Var(this, impl_->add(value));
}

void GraphBuilder::update_state(const Var& state, const Var& value) {
    if (!impl_->status.ok()) {
        return;
    }
    const ir::Value* s = value_of(state);
    const ir::Value* v = value_of(value);
    if (state.builder() != this || value.builder() != this || s == nullptr || v == nullptr ||
        s->kind() != ir::ValueKind::kState) {
        fail(errors::invalid_argument("update_state needs a state and a value of this builder"));
        return;
    }
    if (!(s->type() == v->type())) {
        fail(errors::invalid_argument("state '{}' is {}, but the update is {}", s->name(), s->type().to_string(),
                                      v->type().to_string()));
        return;
    }
    impl_->graph->set_state_update(const_cast<ir::Value*>(s), const_cast<ir::Value*>(v));
}

void GraphBuilder::output(std::string_view name, const Var& value) {
    if (!impl_->status.ok()) {
        return;
    }
    const ir::Value* v = value_of(value);
    if (value.builder() != this || v == nullptr) {
        fail(errors::invalid_argument("output '{}': invalid value", name));
        return;
    }
    if (name.empty() || impl_->graph->find_output(name) != nullptr) {
        fail(errors::invalid_argument("output name '{}' is empty or already used", name));
        return;
    }
    impl_->graph->add_output(std::string(name), const_cast<ir::Value*>(v));
}

std::vector<Var> GraphBuilder::op(std::string_view name, std::span<const Var> inputs, const Attributes& attrs) {
    if (!impl_->status.ok()) {
        return {};
    }
    std::vector<ir::Value*> values;
    for (const Var& input : inputs) {
        if (!input.valid()) {
            values.push_back(nullptr);  // absent optional input
            continue;
        }
        if (input.builder() != this) {
            fail(errors::invalid_argument("'{}': operand belongs to another builder", name));
            return {};
        }
        values.push_back(const_cast<ir::Value*>(value_of(input)));
    }
    ir::Builder builder(*impl_->graph);
    Result<ir::Node*> node = builder.create_node(name, std::move(values), attrs);
    if (!node.ok()) {
        fail(std::move(node).status());
        return {};
    }
    std::vector<Var> outputs;
    for (ir::Value* output : (*node)->outputs()) {
        outputs.push_back(Var(this, impl_->add(output)));
    }
    return outputs;
}

std::vector<Var> GraphBuilder::op(std::string_view name, std::initializer_list<Var> inputs, const Attributes& attrs) {
    return op(name, std::span<const Var>(inputs.begin(), inputs.size()), attrs);
}

Var GraphBuilder::op1(std::string_view name, std::initializer_list<Var> inputs, const Attributes& attrs) {
    std::vector<Var> outputs = op(name, inputs, attrs);
    return outputs.empty() ? Var() : outputs.front();
}

std::string GraphBuilder::to_text() const {
    return impl_->graph->to_text();
}

// ---------------------------------------------------------------------------------------------------------------
// Operations

namespace {

Var apply(std::string_view op, std::initializer_list<Var> inputs, const Attributes& attrs = {}) {
    for (const Var& input : inputs) {
        if (input.valid()) {
            return input.builder()->op1(op, inputs, attrs);
        }
    }
    return {};  // no valid operand: an earlier error was already recorded
}

Var scalar_like(const Var& like, double value) {
    return like.valid() ? like.builder()->scalar(value, like.dtype()) : Var();
}

Attributes window_attrs(const WindowOptions& window) {
    return {{"strides", window.strides},
            {"dilations", window.dilations},
            {"pads", window.pads},
            {"auto_pad", window.auto_pad}};
}

}  // namespace

#define II_UNARY(function, op)                                                                                         \
    Var function(const Var& x) {                                                                                       \
        return apply(op, {x});                                                                                         \
    }
II_UNARY(neg, "neg")
II_UNARY(abs, "abs")
II_UNARY(sign, "sign")
II_UNARY(exp, "exp")
II_UNARY(log, "log")
II_UNARY(sqrt, "sqrt")
II_UNARY(rsqrt, "rsqrt")
II_UNARY(recip, "recip")
II_UNARY(sin, "sin")
II_UNARY(cos, "cos")
II_UNARY(tanh, "tanh")
II_UNARY(sigmoid, "sigmoid")
II_UNARY(erf, "erf")
II_UNARY(relu, "relu")
II_UNARY(floor, "floor")
II_UNARY(ceil, "ceil")
II_UNARY(round, "round")
II_UNARY(logical_not, "not")
II_UNARY(silu, "silu")
#undef II_UNARY

#define II_BINARY(function, op)                                                                                        \
    Var function(const Var& a, const Var& b) {                                                                         \
        return apply(op, {a, b});                                                                                      \
    }
II_BINARY(add, "add")
II_BINARY(sub, "sub")
II_BINARY(mul, "mul")
II_BINARY(div, "div")
II_BINARY(pow, "pow")
II_BINARY(maximum, "max")
II_BINARY(minimum, "min")
II_BINARY(equal, "eq")
II_BINARY(not_equal, "ne")
II_BINARY(less, "lt")
II_BINARY(less_equal, "le")
II_BINARY(greater, "gt")
II_BINARY(greater_equal, "ge")
II_BINARY(logical_and, "and")
II_BINARY(logical_or, "or")
II_BINARY(logical_xor, "xor")
#undef II_BINARY

Var mod(const Var& a, const Var& b, bool fmod) {
    return apply("mod", {a, b}, {{"fmod", fmod}});
}

Var operator+(const Var& a, const Var& b) {
    return add(a, b);
}
Var operator-(const Var& a, const Var& b) {
    return sub(a, b);
}
Var operator*(const Var& a, const Var& b) {
    return mul(a, b);
}
Var operator/(const Var& a, const Var& b) {
    return div(a, b);
}
Var operator-(const Var& a) {
    return neg(a);
}
Var operator+(const Var& a, double b) {
    return add(a, scalar_like(a, b));
}
Var operator-(const Var& a, double b) {
    return sub(a, scalar_like(a, b));
}
Var operator*(const Var& a, double b) {
    return mul(a, scalar_like(a, b));
}
Var operator/(const Var& a, double b) {
    return div(a, scalar_like(a, b));
}
Var operator+(double a, const Var& b) {
    return add(scalar_like(b, a), b);
}
Var operator-(double a, const Var& b) {
    return sub(scalar_like(b, a), b);
}
Var operator*(double a, const Var& b) {
    return mul(scalar_like(b, a), b);
}
Var operator/(double a, const Var& b) {
    return div(scalar_like(b, a), b);
}

Var where(const Var& cond, const Var& a, const Var& b) {
    return apply("where", {cond, a, b});
}
Var cast(const Var& x, DType to) {
    return apply("cast", {x}, {{"to", to}});
}
Var clamp(const Var& x, double min, double max) {
    return apply("clamp", {x}, {{"min", min}, {"max", max}});
}

#define II_REDUCE(function, op)                                                                                        \
    Var function(const Var& x, std::vector<std::int64_t> axes, bool keepdims) {                                        \
        return apply(op, {x}, {{"axes", std::move(axes)}, {"keepdims", keepdims}});                                    \
    }
II_REDUCE(reduce_sum, "reduce_sum")
II_REDUCE(reduce_mean, "reduce_mean")
II_REDUCE(reduce_max, "reduce_max")
II_REDUCE(reduce_min, "reduce_min")
II_REDUCE(reduce_prod, "reduce_prod")
#undef II_REDUCE

Var argmax(const Var& x, std::int64_t axis, bool keepdims) {
    return apply("argmax", {x}, {{"axis", axis}, {"keepdims", keepdims}});
}
Var argmin(const Var& x, std::int64_t axis, bool keepdims) {
    return apply("argmin", {x}, {{"axis", axis}, {"keepdims", keepdims}});
}

Var matmul(const Var& a, const Var& b, bool transpose_a, bool transpose_b) {
    return apply("matmul", {a, b}, {{"transpose_a", transpose_a}, {"transpose_b", transpose_b}});
}

Var conv(const Var& x, const Var& w, const Var& bias, const WindowOptions& window, std::int64_t groups) {
    Attributes attrs = window_attrs(window);
    attrs.set("groups", groups);
    return apply("conv", {x, w, bias}, attrs);
}

Var conv_transpose(const Var& x, const Var& w, const Var& bias, const WindowOptions& window, std::int64_t groups,
                   std::vector<std::int64_t> output_padding) {
    Attributes attrs = window_attrs(window);
    attrs.set("groups", groups);
    attrs.set("output_padding", std::move(output_padding));
    return apply("conv_transpose", {x, w, bias}, attrs);
}

Var max_pool(const Var& x, std::vector<std::int64_t> kernel, const WindowOptions& window, bool ceil_mode) {
    Attributes attrs = window_attrs(window);
    attrs.set("kernel", std::move(kernel));
    attrs.set("ceil_mode", ceil_mode);
    return apply("max_pool", {x}, attrs);
}

Var avg_pool(const Var& x, std::vector<std::int64_t> kernel, const WindowOptions& window, bool ceil_mode,
             bool count_include_pad) {
    Attributes attrs = window_attrs(window);
    attrs.set("kernel", std::move(kernel));
    attrs.set("ceil_mode", ceil_mode);
    attrs.set("count_include_pad", count_include_pad);
    return apply("avg_pool", {x}, attrs);
}

Var reshape(const Var& x, Dims shape) {
    return apply("reshape", {x}, {{"shape", std::move(shape)}});
}
Var permute(const Var& x, std::vector<std::int64_t> perm) {
    return apply("permute", {x}, {{"perm", std::move(perm)}});
}
Var slice(const Var& x, std::vector<std::int64_t> axes, std::vector<std::int64_t> starts,
          std::vector<std::int64_t> ends, std::vector<std::int64_t> steps) {
    return apply("slice", {x},
                 {{"axes", std::move(axes)},
                  {"starts", std::move(starts)},
                  {"ends", std::move(ends)},
                  {"steps", std::move(steps)}});
}
Var expand(const Var& x, Dims shape) {
    return apply("expand", {x}, {{"shape", std::move(shape)}});
}

Var concat(std::span<const Var> inputs, std::int64_t axis) {
    for (const Var& input : inputs) {
        if (input.valid()) {
            std::vector<Var> outputs = input.builder()->op("concat", inputs, {{"axis", axis}});
            return outputs.empty() ? Var() : outputs.front();
        }
    }
    return {};
}
Var concat(std::initializer_list<Var> inputs, std::int64_t axis) {
    return concat(std::span<const Var>(inputs.begin(), inputs.size()), axis);
}

std::vector<Var> split(const Var& x, std::vector<std::int64_t> sizes, std::int64_t axis) {
    if (!x.valid()) {
        return {};
    }
    return x.builder()->op("split", {x}, {{"sizes", std::move(sizes)}, {"axis", axis}});
}

Var pad(const Var& x, std::vector<std::int64_t> pads, std::string_view mode, double value) {
    return apply("pad", {x}, {{"pads", std::move(pads)}, {"mode", std::string(mode)}, {"value", value}});
}
Var gather(const Var& data, const Var& indices, std::int64_t axis) {
    return apply("gather", {data, indices}, {{"axis", axis}});
}
Var scatter(const Var& data, const Var& indices, const Var& updates, std::int64_t axis, std::string_view reduction) {
    return apply("scatter", {data, indices, updates}, {{"axis", axis}, {"reduction", std::string(reduction)}});
}
Var tile(const Var& x, std::vector<std::int64_t> repeats) {
    return apply("tile", {x}, {{"repeats", std::move(repeats)}});
}
Var flip(const Var& x, std::vector<std::int64_t> axes) {
    return apply("flip", {x}, {{"axes", std::move(axes)}});
}
Var cumsum(const Var& x, std::int64_t axis, bool exclusive, bool reverse) {
    return apply("cumsum", {x}, {{"axis", axis}, {"exclusive", exclusive}, {"reverse", reverse}});
}

Var iota(GraphBuilder& builder, Dims shape, DType dtype, std::int64_t axis, double start, double step) {
    return builder.op1(
        "iota", {}, {{"shape", std::move(shape)}, {"dtype", dtype}, {"axis", axis}, {"start", start}, {"step", step}});
}
Var fill(GraphBuilder& builder, Dims shape, DType dtype, double value) {
    return builder.op1("fill", {}, {{"shape", std::move(shape)}, {"dtype", dtype}, {"value", value}});
}
Var random_uniform(GraphBuilder& builder, Dims shape, DType dtype, double low, double high, std::int64_t seed) {
    return builder.op1("random_uniform", {},
                       {{"shape", std::move(shape)}, {"dtype", dtype}, {"low", low}, {"high", high}, {"seed", seed}});
}
Var random_normal(GraphBuilder& builder, Dims shape, DType dtype, double mean, double stddev, std::int64_t seed) {
    return builder.op1(
        "random_normal", {},
        {{"shape", std::move(shape)}, {"dtype", dtype}, {"mean", mean}, {"stddev", stddev}, {"seed", seed}});
}

Var quantize(const Var& x, const Var& scale, const Var& zero_point, DType dtype, std::int64_t axis) {
    return apply("quantize", {x, scale, zero_point}, {{"dtype", dtype}, {"axis", axis}});
}
Var dequantize(const Var& x, const Var& scale, const Var& zero_point, std::int64_t axis) {
    return apply("dequantize", {x, scale, zero_point}, {{"axis", axis}});
}

Var gelu(const Var& x, bool tanh_approximation) {
    return apply("gelu", {x}, {{"approximate", std::string(tanh_approximation ? "tanh" : "none")}});
}
Var softmax(const Var& x, std::int64_t axis) {
    return apply("softmax", {x}, {{"axis", axis}});
}
Var log_softmax(const Var& x, std::int64_t axis) {
    return apply("log_softmax", {x}, {{"axis", axis}});
}
Var layer_norm(const Var& x, const Var& scale, const Var& bias, std::int64_t axis, double epsilon) {
    return apply("layer_norm", {x, scale, bias}, {{"axis", axis}, {"epsilon", epsilon}});
}
Var rms_norm(const Var& x, const Var& scale, std::int64_t axis, double epsilon) {
    return apply("rms_norm", {x, scale}, {{"axis", axis}, {"epsilon", epsilon}});
}
Var linear(const Var& x, const Var& w, const Var& bias, bool transpose_w) {
    return apply("linear", {x, w, bias}, {{"transpose_w", transpose_w}});
}
Var sdpa(const Var& q, const Var& k, const Var& v, const Var& mask, bool causal, double scale) {
    return apply("sdpa", {q, k, v, mask}, {{"causal", causal}, {"scale", scale}});
}

}  // namespace ii
