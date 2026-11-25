// SPDX-License-Identifier: MIT
// ONNX operator handlers: the mapping of ONNX operators (default domain, opsets 1-23) to ii operations.
#include "formats/onnx/onnx_import.h"
#include "ops/infer_util.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <unordered_map>

namespace ii::formats::onnx {

namespace {

constexpr std::int64_t kInt64Max = std::numeric_limits<std::int64_t>::max();

std::string lower(std::string_view text) {
    std::string result(text);
    for (char& c : result) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return result;
}

/// Shorthand: propagate an error, otherwise bind the value to output 0.
Status output(NodeContext& c, const Result<ir::Value*>& value) {
    if (!value.ok()) {
        return value.status();
    }
    return c.set_output(0, *value);
}

Result<int> normalize_axis(std::int64_t axis, int rank, bool allow_end = false) {
    return ops::normalize_axis(axis, rank, allow_end);
}

// ---------------------------------------------------------------------------------------------------------------
// Known contents (shape arithmetic evaluated while importing)

bool all_constant(const Dims& dims) {
    return std::ranges::all_of(dims, [](const Dim& d) { return d.is_constant(); });
}

/// Elementwise operation over two known operands with broadcasting of length-1 operands.
template <class Fn>
std::optional<KnownContents> known_binary(const KnownContents& a, const KnownContents& b, Fn&& fn) {
    const std::size_t n = std::max(a.values.size(), b.values.size());
    if ((a.values.size() != n && a.values.size() != 1) || (b.values.size() != n && b.values.size() != 1)) {
        return std::nullopt;
    }
    KnownContents result;
    result.scalar = a.scalar && b.scalar;
    result.dtype = a.dtype;
    for (std::size_t i = 0; i < n; ++i) {
        const Dim& x = a.values[a.values.size() == 1 ? 0 : i];
        const Dim& y = b.values[b.values.size() == 1 ? 0 : i];
        std::optional<Dim> value = fn(x, y);
        if (!value) {
            return std::nullopt;
        }
        result.values.push_back(std::move(*value));
    }
    return result;
}

/// Axes of Squeeze / Unsqueeze / Reduce*: an attribute before `input_since`, an (optional) input after.
Result<std::optional<std::vector<std::int64_t>>> axes_of(NodeContext& c, std::int64_t input_since, int input_index) {
    if (c.opset() < input_since) {
        return c.ints_attr("axes");
    }
    if (c.has_input(input_index)) {
        II_ASSIGN_OR_RETURN(std::vector<std::int64_t> axes, c.ints_input(input_index));
        return std::optional<std::vector<std::int64_t>>(std::move(axes));
    }
    return std::optional<std::vector<std::int64_t>>();
}

/// Slice parameters (attributes before opset 10, inputs after) as dims; axes / steps as integers.
struct SliceParams {
    Dims starts;
    Dims ends;
    std::vector<std::int64_t> axes;
    std::vector<std::int64_t> steps;
};

Result<SliceParams> slice_params(NodeContext& c, int rank) {
    SliceParams p;
    if (c.opset() < 10) {
        for (const std::int64_t v : c.ints_attr("starts").value_or(std::vector<std::int64_t>{})) {
            p.starts.emplace_back(v);
        }
        for (const std::int64_t v : c.ints_attr("ends").value_or(std::vector<std::int64_t>{})) {
            p.ends.emplace_back(v);
        }
        p.axes = c.ints_attr("axes").value_or(std::vector<std::int64_t>{});
    } else {
        II_ASSIGN_OR_RETURN(p.starts, c.dims_input(1));
        II_ASSIGN_OR_RETURN(p.ends, c.dims_input(2));
        if (c.has_input(3)) {
            II_ASSIGN_OR_RETURN(p.axes, c.ints_input(3));
        }
        if (c.has_input(4)) {
            II_ASSIGN_OR_RETURN(p.steps, c.ints_input(4));
        }
    }
    if (p.starts.size() != p.ends.size()) {
        return errors::invalid_argument("Slice: {} starts and {} ends", p.starts.size(), p.ends.size());
    }
    if (p.axes.empty()) {
        for (std::size_t i = 0; i < p.starts.size(); ++i) {
            p.axes.push_back(static_cast<std::int64_t>(i));
        }
    }
    if (p.steps.empty()) {
        p.steps.assign(p.starts.size(), 1);
    }
    if (p.axes.size() != p.starts.size() || p.steps.size() != p.starts.size()) {
        return errors::invalid_argument("Slice: starts, ends, axes and steps differ in length");
    }
    for (std::int64_t& axis : p.axes) {
        II_ASSIGN_OR_RETURN(const int normalized, normalize_axis(axis, rank));
        axis = normalized;
    }
    return p;
}

Result<bool> known_shape_op(NodeContext& c) {
    const std::string_view op = c.op_type();
    if (op == "Shape") {
        II_ASSIGN_OR_RETURN(const ir::TensorType type, c.input_type(0));
        const auto rank = static_cast<std::int64_t>(type.rank());
        std::int64_t start = c.int_attr("start", 0);
        std::int64_t end = c.int_attr("end", rank);
        start = std::clamp<std::int64_t>(start < 0 ? start + rank : start, 0, rank);
        end = std::clamp<std::int64_t>(end < 0 ? end + rank : end, 0, rank);
        Dims dims;
        for (std::int64_t d = start; d < end; ++d) {
            dims.push_back(type.shape[static_cast<std::size_t>(d)]);
        }
        II_TRY(c.set_known_output(0, std::move(dims), false, DType::kI64));
        return true;
    }
    if (op == "Size") {
        II_ASSIGN_OR_RETURN(const ir::TensorType type, c.input_type(0));
        II_TRY(c.set_known_output(0, Dims{ops::product(type.shape)}, true, DType::kI64));
        return true;
    }
    if (op == "Identity") {
        std::optional<KnownContents> x = c.known_input(0);
        if (!x) {
            return false;
        }
        II_TRY(c.set_known_output(0, std::move(x->values), x->scalar, x->dtype));
        return true;
    }
    if (op == "Gather") {
        std::optional<KnownContents> data = c.known_input(0);
        if (!data || data->scalar || c.int_attr("axis", 0) != 0 || !c.is_constant_input(1)) {
            return false;
        }
        std::optional<KnownContents> indices = c.known_input(1);
        if (!indices || !all_constant(indices->values)) {
            return false;
        }
        Dims result;
        const auto size = static_cast<std::int64_t>(data->values.size());
        for (const Dim& index : indices->values) {
            std::int64_t i = index.value();
            if (i < 0) {
                i += size;
            }
            if (i < 0 || i >= size) {
                return errors::out_of_range("Gather index {} is out of range for {} elements", index.value(), size);
            }
            result.push_back(data->values[static_cast<std::size_t>(i)]);
        }
        II_TRY(c.set_known_output(0, std::move(result), indices->scalar, data->dtype));
        return true;
    }
    if (op == "Slice") {
        std::optional<KnownContents> data = c.known_input(0);
        if (!data || data->scalar) {
            return false;
        }
        II_ASSIGN_OR_RETURN(const SliceParams p, slice_params(c, 1));
        if (p.starts.size() != 1 || !p.starts[0].is_constant() || !p.ends[0].is_constant()) {
            return false;
        }
        const ops::SliceRange range = ops::resolve_slice(static_cast<std::int64_t>(data->values.size()),
                                                         p.starts[0].value(), p.ends[0].value(), p.steps[0]);
        Dims result;
        for (std::int64_t k = 0; k < range.count; ++k) {
            result.push_back(data->values[static_cast<std::size_t>(range.start + k * range.step)]);
        }
        II_TRY(c.set_known_output(0, std::move(result), false, data->dtype));
        return true;
    }
    if (op == "Concat") {
        Dims result;
        DType dtype = DType::kInvalid;
        for (int i = 0; i < c.num_inputs(); ++i) {
            if (!c.has_input(i)) {
                continue;
            }
            std::optional<KnownContents> part = c.known_input(i);
            if (!part || part->scalar) {
                return false;
            }
            dtype = part->dtype;
            result.insert(result.end(), part->values.begin(), part->values.end());
        }
        II_TRY(c.set_known_output(0, std::move(result), false, dtype));
        return true;
    }
    if (op == "Unsqueeze" || op == "Squeeze" || op == "Reshape" || op == "Flatten") {
        std::optional<KnownContents> x = c.known_input(0);
        if (!x) {
            return false;
        }
        bool scalar = x->scalar;
        if (op == "Unsqueeze") {
            II_ASSIGN_OR_RETURN(const auto axes, axes_of(c, 13, 1));
            if (!x->scalar || !axes || axes->size() != 1 || ((*axes)[0] != 0 && (*axes)[0] != -1)) {
                return false;
            }
            scalar = false;
        } else if (op == "Squeeze") {
            II_ASSIGN_OR_RETURN(const auto axes, axes_of(c, 13, 1));
            if (x->scalar || x->values.size() != 1 || (axes && axes->size() > 1)) {
                return false;
            }
            scalar = true;
        } else if (op == "Flatten") {
            return false;
        } else {
            std::optional<KnownContents> shape = c.known_input(1);
            if (!shape || !all_constant(shape->values) || shape->values.size() > 1) {
                return false;
            }
            if (shape->values.empty()) {
                if (x->values.size() != 1) {
                    return false;
                }
                scalar = true;
            } else {
                const std::int64_t n = shape->values[0].value();
                if (n != -1 && n != static_cast<std::int64_t>(x->values.size()) && !(n == 0 && x->scalar)) {
                    return false;
                }
                scalar = false;
            }
        }
        II_TRY(c.set_known_output(0, std::move(x->values), scalar, x->dtype));
        return true;
    }
    if (op == "Cast") {
        std::optional<KnownContents> x = c.known_input(0);
        const DType to = dtype_from_onnx(static_cast<std::int32_t>(c.int_attr("to", 0)));
        if (!x || (to != DType::kI64 && to != DType::kI32 && to != DType::kF32 && to != DType::kF64)) {
            return false;
        }
        II_TRY(c.set_known_output(0, std::move(x->values), x->scalar, to));
        return true;
    }
    if (op == "Neg") {
        std::optional<KnownContents> x = c.known_input(0);
        if (!x) {
            return false;
        }
        for (Dim& d : x->values) {
            d = -d;
        }
        II_TRY(c.set_known_output(0, std::move(x->values), x->scalar, x->dtype));
        return true;
    }
    if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" || op == "Max" || op == "Min") {
        if (c.num_inputs() != 2) {
            return false;
        }
        std::optional<KnownContents> a = c.known_input(0);
        std::optional<KnownContents> b = c.known_input(1);
        if (!a || !b) {
            return false;
        }
        const bool is_float = a->dtype == DType::kF32 || a->dtype == DType::kF64;
        std::optional<KnownContents> result;
        if (op == "Add") {
            result = known_binary(*a, *b, [](const Dim& x, const Dim& y) { return std::optional<Dim>(x + y); });
        } else if (op == "Sub") {
            result = known_binary(*a, *b, [](const Dim& x, const Dim& y) { return std::optional<Dim>(x - y); });
        } else if (op == "Mul") {
            result = known_binary(*a, *b, [](const Dim& x, const Dim& y) { return std::optional<Dim>(x * y); });
        } else if (op == "Div") {
            result = known_binary(*a, *b, [is_float](const Dim& x, const Dim& y) -> std::optional<Dim> {
                if (!y.is_constant() || y.value() == 0) {
                    return std::nullopt;
                }
                if (x.is_constant()) {
                    if (is_float && x.value() % y.value() != 0) {
                        return std::nullopt;
                    }
                    return Dim(x.value() / y.value());  // truncation, like ONNX integer Div
                }
                if (std::optional<Dim> exact = x.exact_div(y.value())) {
                    return exact;
                }
                if (is_float || y.value() < 0) {
                    return std::nullopt;
                }
                return floor_div(x, y.value());  // dimensions are non-negative: floor == truncation
            });
        } else {
            const bool max = op == "Max";
            result = known_binary(*a, *b, [max](const Dim& x, const Dim& y) -> std::optional<Dim> {
                if (x == y) {
                    return x;
                }
                if (!x.is_constant() || !y.is_constant()) {
                    return std::nullopt;
                }
                return Dim(max ? std::max(x.value(), y.value()) : std::min(x.value(), y.value()));
            });
        }
        if (!result) {
            return false;
        }
        II_TRY(c.set_known_output(0, std::move(result->values), result->scalar, result->dtype));
        return true;
    }
    if (op == "Equal" || op == "Less" || op == "Greater") {
        std::optional<KnownContents> a = c.known_input(0);
        std::optional<KnownContents> b = c.known_input(1);
        if (!a || !b) {
            return false;
        }
        // Dimensions are non-negative, so a symbolic dimension compares with a negative constant (the -1 of torch's
        // "expand" idiom) as larger.
        const auto compare = [op](const Dim& x, const Dim& y) -> std::optional<Dim> {
            std::optional<int> order;  // sign of x - y
            if (x == y) {
                order = 0;
            } else if (x.is_constant() && y.is_constant()) {
                order = x.value() < y.value() ? -1 : 1;
            } else if (y.is_constant() && y.value() < 0) {
                order = 1;
            } else if (x.is_constant() && x.value() < 0) {
                order = -1;
            }
            if (!order) {
                return std::nullopt;
            }
            const bool result = op == "Equal" ? *order == 0 : op == "Less" ? *order < 0 : *order > 0;
            return Dim(result ? 1 : 0);
        };
        std::optional<KnownContents> result = known_binary(*a, *b, compare);
        if (!result) {
            return false;
        }
        II_TRY(c.set_known_output(0, std::move(result->values), result->scalar, DType::kBool));
        return true;
    }
    if (op == "Where") {
        std::optional<KnownContents> cond = c.known_input(0);
        std::optional<KnownContents> a = c.known_input(1);
        std::optional<KnownContents> b = c.known_input(2);
        if (!cond || !a || !b || !all_constant(cond->values)) {
            return false;
        }
        std::optional<KnownContents> chosen =
            known_binary(*a, *b, [](const Dim& x, const Dim&) { return std::optional<Dim>(x); });
        if (!chosen) {
            return false;
        }
        const std::size_t n = std::max({cond->values.size(), a->values.size(), b->values.size()});
        Dims result;
        for (std::size_t i = 0; i < n; ++i) {
            const auto pick = [&](const KnownContents& k) { return k.values[k.values.size() == 1 ? 0 : i]; };
            if (cond->values.size() != 1 && cond->values.size() != n) {
                return false;
            }
            result.push_back(pick(*cond).value() != 0 ? pick(*a) : pick(*b));
        }
        II_TRY(c.set_known_output(0, std::move(result), cond->scalar && a->scalar && b->scalar, a->dtype));
        return true;
    }
    if (op == "ReduceProd" || op == "ReduceSum") {
        std::optional<KnownContents> x = c.known_input(0);
        if (!x || x->scalar) {
            return false;
        }
        const bool keepdims = c.int_attr("keepdims", 1) != 0;
        Dim result = op == "ReduceProd" ? Dim(1) : Dim(0);
        for (const Dim& d : x->values) {
            result = op == "ReduceProd" ? result * d : result + d;
        }
        II_TRY(c.set_known_output(0, Dims{result}, !keepdims, x->dtype));
        return true;
    }
    if (op == "Range") {
        std::optional<KnownContents> start = c.known_input(0);
        std::optional<KnownContents> limit = c.known_input(1);
        std::optional<KnownContents> delta = c.known_input(2);
        if (!start || !limit || !delta || !all_constant(start->values) || !all_constant(limit->values) ||
            !all_constant(delta->values) || start->values.size() != 1 || limit->values.size() != 1 ||
            delta->values.size() != 1 || delta->values[0].value() == 0) {
            return false;
        }
        const std::int64_t s = start->values[0].value();
        const std::int64_t l = limit->values[0].value();
        const std::int64_t d = delta->values[0].value();
        const std::int64_t count = std::max<std::int64_t>(0, (l - s + d + (d > 0 ? -1 : 1)) / d);
        if (count > 64) {
            return false;
        }
        Dims result;
        for (std::int64_t i = 0; i < count; ++i) {
            result.emplace_back(s + i * d);
        }
        II_TRY(c.set_known_output(0, std::move(result), false, start->dtype));
        return true;
    }
    if (op == "ConstantOfShape") {
        std::optional<KnownContents> shape = c.known_input(0);
        if (!shape || shape->values.size() != 1 || !all_constant(shape->values) || shape->values[0].value() < 0 ||
            shape->values[0].value() > 64 || !c.has_attribute("value")) {
            return false;
        }
        II_ASSIGN_OR_RETURN(const Tensor value, c.tensor_attr("value"));
        if (value.dtype() != DType::kI64 || value.numel() != 1) {
            return false;
        }
        const std::int64_t fill = value.values<std::int64_t>()[0];
        II_TRY(c.set_known_output(0, Dims(static_cast<std::size_t>(shape->values[0].value()), Dim(fill)), false,
                                  DType::kI64));
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------------------------
// Elementwise

const std::unordered_map<std::string_view, std::string_view>& unary_ops() {
    static const std::unordered_map<std::string_view, std::string_view> table = {
        {"Abs", "abs"},          {"Neg", "neg"},   {"Exp", "exp"},     {"Log", "log"},   {"Sqrt", "sqrt"},
        {"Reciprocal", "recip"}, {"Sin", "sin"},   {"Cos", "cos"},     {"Tanh", "tanh"}, {"Sigmoid", "sigmoid"},
        {"Erf", "erf"},          {"Relu", "relu"}, {"Floor", "floor"}, {"Ceil", "ceil"}, {"Round", "round"},
        {"Sign", "sign"},        {"Not", "not"},
    };
    return table;
}

const std::unordered_map<std::string_view, std::string_view>& binary_ops() {
    static const std::unordered_map<std::string_view, std::string_view> table = {
        {"Add", "add"},           {"Sub", "sub"}, {"Mul", "mul"},        {"Div", "div"},
        {"Equal", "eq"},          {"Less", "lt"}, {"LessOrEqual", "le"}, {"Greater", "gt"},
        {"GreaterOrEqual", "ge"}, {"And", "and"}, {"Or", "or"},          {"Xor", "xor"},
    };
    return table;
}

Status handle_unary(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    return output(c, c.op(unary_ops().at(c.op_type()), {x}));
}

Status check_legacy_broadcast(NodeContext& c) {
    if (c.opset() < 7 && c.int_attr("broadcast", 0) != 0 && c.has_attribute("axis")) {
        return c.unsupported("legacy broadcasting with an axis (opset < 7)");
    }
    return ok_status();
}

Status handle_binary(NodeContext& c) {
    II_TRY(check_legacy_broadcast(c));
    II_ASSIGN_OR_RETURN(ir::Value * a, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * b, c.input(1));
    return output(c, c.op(binary_ops().at(c.op_type()), {a, b}));
}

Status handle_pow(NodeContext& c) {
    II_TRY(check_legacy_broadcast(c));
    II_ASSIGN_OR_RETURN(ir::Value * a, c.input(0));
    if (c.is_constant_input(1)) {
        II_ASSIGN_OR_RETURN(const std::vector<double> exponent, c.floats_input(1));
        II_ASSIGN_OR_RETURN(const ir::TensorType type, c.input_type(1));
        if (exponent.size() == 1 && type.rank() <= a->rank() && is_float(a->dtype())) {
            if (exponent[0] == 1) {
                return c.set_output(0, a);
            }
            if (exponent[0] == 2) {
                return output(c, c.op("mul", {a, a}));
            }
            if (exponent[0] == 0.5) {
                return output(c, c.op("sqrt", {a}));
            }
        }
    }
    II_ASSIGN_OR_RETURN(ir::Value * b, c.input(1));
    II_ASSIGN_OR_RETURN(b, c.cast(b, a->dtype()));
    return output(c, c.op("pow", {a, b}));
}

Status handle_mod(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * a, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * b, c.input(1));
    return output(c, c.op("mod", {a, b}, {{"fmod", c.int_attr("fmod", 0) != 0}}));
}

Status handle_variadic(NodeContext& c) {
    const std::string_view op = c.op_type();
    const std::string_view ii_op = op == "Max" ? "max" : op == "Min" ? "min" : "add";
    ir::Value* result = nullptr;
    int count = 0;
    for (int i = 0; i < c.num_inputs(); ++i) {
        if (!c.has_input(i)) {
            continue;
        }
        II_ASSIGN_OR_RETURN(ir::Value * x, c.input(i));
        ++count;
        if (result == nullptr) {
            result = x;
        } else {
            II_ASSIGN_OR_RETURN(result, c.op(ii_op, {result, x}));
        }
    }
    if (result == nullptr) {
        return c.unsupported("a variadic operation without inputs");
    }
    if (op == "Mean" && count > 1) {
        II_ASSIGN_OR_RETURN(ir::Value * n, c.scalar(count, result->dtype()));
        II_ASSIGN_OR_RETURN(result, c.op("div", {result, n}));
    }
    return c.set_output(0, result);
}

Status handle_where(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * cond, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * a, c.input(1));
    II_ASSIGN_OR_RETURN(ir::Value * b, c.input(2));
    return output(c, c.op("where", {cond, a, b}));
}

Status handle_cast(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    DType to = DType::kInvalid;
    if (c.op_type() == "CastLike") {
        II_ASSIGN_OR_RETURN(const ir::TensorType target, c.input_type(1));
        to = target.dtype;
    } else {
        const auto onnx_type = static_cast<std::int32_t>(c.int_attr("to", 0));
        to = dtype_from_onnx(onnx_type);
        if (to == DType::kInvalid) {
            return c.unsupported(std::format("Cast to {}", onnx_type_name(onnx_type)));
        }
    }
    return output(c, c.cast(x, to));
}

Status handle_identity(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_TRY(c.set_output(0, x));
    if (c.op_type() == "Dropout" && c.has_output(1)) {
        // Inference: nothing is dropped, the mask is all true.
        II_ASSIGN_OR_RETURN(ir::Value * mask,
                            c.op("fill", {}, {{"shape", x->shape()}, {"dtype", DType::kBool}, {"value", 1.0}}));
        II_TRY(c.set_output(1, mask));
    }
    return ok_status();
}

Status handle_clip(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    if (c.opset() < 11) {
        return output(c, c.op("clamp", {x},
                              {{"min", c.float_attr("min", -std::numeric_limits<double>::infinity())},
                               {"max", c.float_attr("max", std::numeric_limits<double>::infinity())}}));
    }
    double low = -std::numeric_limits<double>::infinity();
    double high = std::numeric_limits<double>::infinity();
    ir::Value* result = x;
    for (int i = 1; i <= 2; ++i) {
        if (!c.has_input(i)) {
            continue;
        }
        if (c.is_constant_input(i)) {
            II_ASSIGN_OR_RETURN(const std::vector<double> bound, c.floats_input(i));
            if (bound.size() == 1) {
                (i == 1 ? low : high) = bound[0];
                continue;
            }
        }
        II_ASSIGN_OR_RETURN(ir::Value * bound, c.input(i));
        II_ASSIGN_OR_RETURN(result, c.op(i == 1 ? "max" : "min", {result, bound}));
    }
    if (std::isinf(low) && std::isinf(high)) {
        return c.set_output(0, result);
    }
    return output(c, c.op("clamp", {result}, {{"min", low}, {"max", high}}));
}

/// x * alpha (alpha a scalar of x's dtype).
Result<ir::Value*> scaled(NodeContext& c, ir::Value* x, double alpha) {
    II_ASSIGN_OR_RETURN(ir::Value * a, c.scalar(alpha, x->dtype()));
    return c.op("mul", {x, a});
}

Status handle_activation(NodeContext& c) {
    const std::string_view op = c.op_type();
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    const DType dtype = x->dtype();
    II_ASSIGN_OR_RETURN(ir::Value * zero, c.scalar(0, dtype));
    II_ASSIGN_OR_RETURN(ir::Value * one, c.scalar(1, dtype));
    if (op == "LeakyRelu" || op == "PRelu") {
        ir::Value* slope_x = nullptr;
        if (op == "LeakyRelu") {
            II_ASSIGN_OR_RETURN(slope_x, scaled(c, x, c.float_attr("alpha", 0.01)));
        } else {
            II_ASSIGN_OR_RETURN(ir::Value * slope, c.input(1));
            II_ASSIGN_OR_RETURN(slope_x, c.op("mul", {x, slope}));
        }
        II_ASSIGN_OR_RETURN(ir::Value * negative, c.op("lt", {x, zero}));
        return output(c, c.op("where", {negative, slope_x, x}));
    }
    if (op == "Elu" || op == "Selu" || op == "Celu") {
        const double alpha = c.float_attr("alpha", op == "Selu" ? 1.67326319217681884765625 : 1.0);
        ir::Value* argument = x;
        if (op == "Celu") {
            II_ASSIGN_OR_RETURN(argument, scaled(c, x, 1.0 / alpha));
        }
        II_ASSIGN_OR_RETURN(ir::Value * e, c.op("exp", {argument}));
        II_ASSIGN_OR_RETURN(ir::Value * em1, c.op("sub", {e, one}));
        II_ASSIGN_OR_RETURN(ir::Value * negative_part, scaled(c, em1, alpha));
        II_ASSIGN_OR_RETURN(ir::Value * positive, c.op("gt", {x, zero}));
        II_ASSIGN_OR_RETURN(ir::Value * y, c.op("where", {positive, x, negative_part}));
        if (op == "Selu") {
            II_ASSIGN_OR_RETURN(y, scaled(c, y, c.float_attr("gamma", 1.05070102214813232421875)));
        }
        return c.set_output(0, y);
    }
    if (op == "HardSigmoid" || op == "HardSwish") {
        const double alpha = op == "HardSwish" ? 1.0 / 6 : c.float_attr("alpha", 0.2);
        const double beta = op == "HardSwish" ? 0.5 : c.float_attr("beta", 0.5);
        II_ASSIGN_OR_RETURN(ir::Value * ax, scaled(c, x, alpha));
        II_ASSIGN_OR_RETURN(ir::Value * b, c.scalar(beta, dtype));
        II_ASSIGN_OR_RETURN(ir::Value * sum, c.op("add", {ax, b}));
        II_ASSIGN_OR_RETURN(ir::Value * hard, c.op("clamp", {sum}, {{"min", 0.0}, {"max", 1.0}}));
        if (op == "HardSigmoid") {
            return c.set_output(0, hard);
        }
        return output(c, c.op("mul", {x, hard}));
    }
    if (op == "Softplus" || op == "Mish") {
        // log(1 + e^x) = max(x, 0) + log(1 + e^-|x|), stable for large |x|.
        II_ASSIGN_OR_RETURN(ir::Value * relu, c.op("relu", {x}));
        II_ASSIGN_OR_RETURN(ir::Value * abs, c.op("abs", {x}));
        II_ASSIGN_OR_RETURN(ir::Value * neg, c.op("neg", {abs}));
        II_ASSIGN_OR_RETURN(ir::Value * e, c.op("exp", {neg}));
        II_ASSIGN_OR_RETURN(ir::Value * e1, c.op("add", {e, one}));
        II_ASSIGN_OR_RETURN(ir::Value * log, c.op("log", {e1}));
        II_ASSIGN_OR_RETURN(ir::Value * softplus, c.op("add", {relu, log}));
        if (op == "Softplus") {
            return c.set_output(0, softplus);
        }
        II_ASSIGN_OR_RETURN(ir::Value * t, c.op("tanh", {softplus}));
        return output(c, c.op("mul", {x, t}));
    }
    if (op == "Softsign") {
        II_ASSIGN_OR_RETURN(ir::Value * abs, c.op("abs", {x}));
        II_ASSIGN_OR_RETURN(ir::Value * d, c.op("add", {abs, one}));
        return output(c, c.op("div", {x, d}));
    }
    if (op == "ThresholdedRelu") {
        II_ASSIGN_OR_RETURN(ir::Value * alpha, c.scalar(c.float_attr("alpha", 1.0), dtype));
        II_ASSIGN_OR_RETURN(ir::Value * above, c.op("gt", {x, alpha}));
        return output(c, c.op("where", {above, x, zero}));
    }
    if (op == "Swish") {
        II_ASSIGN_OR_RETURN(ir::Value * ax, scaled(c, x, c.float_attr("alpha", 1.0)));
        II_ASSIGN_OR_RETURN(ir::Value * s, c.op("sigmoid", {ax}));
        return output(c, c.op("mul", {x, s}));
    }
    if (op == "Gelu") {
        return output(c, c.op("gelu", {x}, {{"approximate", c.string_attr("approximate", "none")}}));
    }
    if (op == "IsNaN") {
        return output(c, c.op("ne", {x, x}));
    }
    if (op == "IsInf") {
        II_ASSIGN_OR_RETURN(ir::Value * inf, c.scalar(std::numeric_limits<double>::infinity(), dtype));
        II_ASSIGN_OR_RETURN(ir::Value * minus_inf, c.scalar(-std::numeric_limits<double>::infinity(), dtype));
        const bool positive = c.int_attr("detect_positive", 1) != 0;
        const bool negative = c.int_attr("detect_negative", 1) != 0;
        II_ASSIGN_OR_RETURN(ir::Value * pos, c.op("eq", {x, inf}));
        II_ASSIGN_OR_RETURN(ir::Value * neg, c.op("eq", {x, minus_inf}));
        if (positive && negative) {
            return output(c, c.op("or", {pos, neg}));
        }
        if (positive) {
            return c.set_output(0, pos);
        }
        if (negative) {
            return c.set_output(0, neg);
        }
        return output(c, c.op("and", {pos, neg}));
    }
    if (op == "Tan") {
        II_ASSIGN_OR_RETURN(ir::Value * s, c.op("sin", {x}));
        II_ASSIGN_OR_RETURN(ir::Value * co, c.op("cos", {x}));
        return output(c, c.op("div", {s, co}));
    }
    if (op == "Sinh" || op == "Cosh") {
        II_ASSIGN_OR_RETURN(ir::Value * e, c.op("exp", {x}));
        II_ASSIGN_OR_RETURN(ir::Value * nx, c.op("neg", {x}));
        II_ASSIGN_OR_RETURN(ir::Value * en, c.op("exp", {nx}));
        II_ASSIGN_OR_RETURN(ir::Value * combined, c.op(op == "Sinh" ? "sub" : "add", {e, en}));
        return output(c, scaled(c, combined, 0.5));
    }
    return c.unsupported(std::format("activation {}", op));
}

// ---------------------------------------------------------------------------------------------------------------
// Linear algebra, convolution, pooling, normalization

Status handle_matmul(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * a, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * b, c.input(1));
    return output(c, c.op("matmul", {a, b}));
}

Status handle_gemm(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * a, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * b, c.input(1));
    II_ASSIGN_OR_RETURN(ir::Value * bias, c.optional_input(2));
    const double alpha = c.float_attr("alpha", 1.0);
    const double beta = c.float_attr("beta", 1.0);
    II_ASSIGN_OR_RETURN(ir::Value * y, c.op("matmul", {a, b},
                                            {{"transpose_a", c.int_attr("transA", 0) != 0},
                                             {"transpose_b", c.int_attr("transB", 0) != 0}}));
    if (alpha != 1) {
        II_ASSIGN_OR_RETURN(y, scaled(c, y, alpha));
    }
    if (bias != nullptr && beta != 0) {
        if (beta != 1) {
            II_ASSIGN_OR_RETURN(bias, scaled(c, bias, beta));
        }
        II_ASSIGN_OR_RETURN(y, c.op("add", {y, bias}));
    }
    return c.set_output(0, y);
}

/// strides, dilations, pads and auto_pad of Conv / ConvTranspose / pooling.
Result<Attributes> window_attrs(NodeContext& c) {
    Attributes attrs;
    if (auto strides = c.ints_attr("strides")) {
        attrs.set("strides", std::move(*strides));
    }
    if (auto dilations = c.ints_attr("dilations")) {
        attrs.set("dilations", std::move(*dilations));
    }
    if (auto pads = c.ints_attr("pads")) {
        attrs.set("pads", std::move(*pads));
    }
    const std::string auto_pad = lower(c.string_attr("auto_pad", "NOTSET"));
    if (auto_pad != "notset" && auto_pad != "valid" && auto_pad != "same_upper" && auto_pad != "same_lower") {
        return errors::invalid_argument("unknown auto_pad '{}'", auto_pad);
    }
    attrs.set("auto_pad", auto_pad == "notset" ? std::string("none") : auto_pad);
    return attrs;
}

Status handle_conv(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * w, c.input(1));
    II_ASSIGN_OR_RETURN(ir::Value * bias, c.optional_input(2));
    II_ASSIGN_OR_RETURN(Attributes attrs, window_attrs(c));
    attrs.set("groups", c.int_attr("group", 1));
    std::vector<ir::Value*> inputs = {x, w};
    if (bias != nullptr) {
        inputs.push_back(bias);
    }
    if (c.op_type() == "Conv") {
        return output(c, c.op("conv", std::move(inputs), std::move(attrs)));
    }
    if (auto output_padding = c.ints_attr("output_padding")) {
        attrs.set("output_padding", std::move(*output_padding));
    }
    if (auto output_shape = c.ints_attr("output_shape")) {
        // Explicit output size: derive the padding (ONNX ConvTranspose rules).
        const int spatial = x->rank() - 2;
        if (static_cast<int>(output_shape->size()) != spatial || w->rank() != x->rank()) {
            return c.unsupported("ConvTranspose output_shape of unexpected length");
        }
        const std::vector<std::int64_t> strides = attrs.get_ints("strides");
        const std::vector<std::int64_t> dilations = attrs.get_ints("dilations");
        std::vector<std::int64_t> output_padding = attrs.get_ints("output_padding");
        std::vector<std::int64_t> pads(static_cast<std::size_t>(2 * spatial), 0);
        const bool upper = attrs.get_string("auto_pad") == "same_upper";
        for (int d = 0; d < spatial; ++d) {
            const Dim& in = x->shape()[static_cast<std::size_t>(d + 2)];
            const Dim& k = w->shape()[static_cast<std::size_t>(d + 2)];
            if (!in.is_constant() || !k.is_constant()) {
                return c.unsupported("ConvTranspose output_shape with symbolic sizes");
            }
            const auto at = [](const std::vector<std::int64_t>& v, int i) {
                return v.empty() ? std::int64_t{1} : v[static_cast<std::size_t>(i)];
            };
            const std::int64_t op = output_padding.empty() ? 0 : output_padding[static_cast<std::size_t>(d)];
            const std::int64_t total = at(strides, d) * (in.value() - 1) + op +
                                       ((k.value() - 1) * at(dilations, d) + 1) -
                                       (*output_shape)[static_cast<std::size_t>(d)];
            if (total < 0) {
                // The requested output is larger than the natural one: the extra goes to output_padding.
                output_padding.resize(static_cast<std::size_t>(spatial), 0);
                output_padding[static_cast<std::size_t>(d)] = op - total;
                continue;
            }
            const std::int64_t half = total / 2;
            pads[static_cast<std::size_t>(d)] = upper ? half : total - half;
            pads[static_cast<std::size_t>(d + spatial)] = upper ? total - half : half;
        }
        attrs.set("pads", std::move(pads));
        if (!output_padding.empty()) {
            attrs.set("output_padding", std::move(output_padding));
        }
        attrs.set("auto_pad", std::string("none"));
    }
    return output(c, c.op("conv_transpose", std::move(inputs), std::move(attrs)));
}

Status handle_pool(NodeContext& c) {
    if (c.op_type() == "MaxPool" && c.has_output(1)) {
        return c.unsupported("MaxPool with the Indices output");
    }
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(Attributes attrs, window_attrs(c));
    const auto kernel = c.ints_attr("kernel_shape");
    if (!kernel) {
        return errors::invalid_argument("{} requires kernel_shape", c.op_type());
    }
    attrs.set("kernel", *kernel);
    attrs.set("ceil_mode", c.int_attr("ceil_mode", 0) != 0);
    if (c.op_type() == "AveragePool") {
        attrs.set("count_include_pad", c.int_attr("count_include_pad", 0) != 0);
        return output(c, c.op("avg_pool", {x}, std::move(attrs)));
    }
    return output(c, c.op("max_pool", {x}, std::move(attrs)));
}

Status handle_global_pool(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    std::vector<std::int64_t> axes;
    for (int d = 2; d < x->rank(); ++d) {
        axes.push_back(d);
    }
    return output(c, c.op(c.op_type() == "GlobalMaxPool" ? "reduce_max" : "reduce_mean", {x},
                          {{"axes", std::move(axes)}, {"keepdims", true}}));
}

/// Reshapes a per-channel vector [C] to [C, 1, ..., 1] for broadcasting against [N, C, spatial...].
Result<ir::Value*> per_channel(NodeContext& c, ir::Value* v, int rank, DType dtype) {
    II_ASSIGN_OR_RETURN(v, c.cast(v, dtype));
    if (rank <= 2) {
        return v;
    }
    Dims dims = {v->shape().empty() ? Dim(1) : v->shape()[0]};
    for (int d = 2; d < rank; ++d) {
        dims.emplace_back(1);
    }
    return c.reshape(v, std::move(dims));
}

Status handle_batch_norm(NodeContext& c) {
    if (c.int_attr("training_mode", 0) != 0 || c.has_output(1) || c.has_output(2)) {
        return c.unsupported("BatchNormalization in training mode");
    }
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    const DType dtype = x->dtype();
    std::vector<ir::Value*> params;
    for (int i = 1; i <= 4; ++i) {
        II_ASSIGN_OR_RETURN(ir::Value * p, c.input(i));
        II_ASSIGN_OR_RETURN(p, c.cast(p, dtype));
        params.push_back(p);
    }
    // y = x * scale / sqrt(var + eps) + (bias - mean * scale / sqrt(var + eps)); constant parameters fold.
    II_ASSIGN_OR_RETURN(ir::Value * eps, c.scalar(c.float_attr("epsilon", 1e-5), dtype));
    II_ASSIGN_OR_RETURN(ir::Value * var_eps, c.op("add", {params[3], eps}));
    II_ASSIGN_OR_RETURN(ir::Value * inv_std, c.op("rsqrt", {var_eps}));
    II_ASSIGN_OR_RETURN(ir::Value * mul, c.op("mul", {params[0], inv_std}));
    II_ASSIGN_OR_RETURN(ir::Value * mean_mul, c.op("mul", {params[2], mul}));
    II_ASSIGN_OR_RETURN(ir::Value * shift, c.op("sub", {params[1], mean_mul}));
    II_ASSIGN_OR_RETURN(mul, per_channel(c, mul, x->rank(), dtype));
    II_ASSIGN_OR_RETURN(shift, per_channel(c, shift, x->rank(), dtype));
    for (ir::Value** v : {&mul, &shift}) {
        II_ASSIGN_OR_RETURN(ir::Value * folded, c.importer().try_fold(*v));
        if (folded != nullptr) {
            *v = folded;
        }
    }
    II_ASSIGN_OR_RETURN(ir::Value * scaled_x, c.op("mul", {x, mul}));
    return output(c, c.op("add", {scaled_x, shift}));
}

/// LayerNormalization with the optional Mean / InvStdDev outputs, from primitives (computed in f32 or wider).
Status layer_norm_with_statistics(NodeContext& c, ir::Value* x, ir::Value* scale, ir::Value* bias, int axis) {
    std::vector<std::int64_t> axes;
    for (int d = axis; d < x->rank(); ++d) {
        axes.push_back(d);
    }
    const Attributes reduce = {{"axes", axes}, {"keepdims", true}};
    const DType dtype = x->dtype();
    const DType stash = x->dtype() == DType::kF64 ? DType::kF64 : DType::kF32;
    II_ASSIGN_OR_RETURN(ir::Value * xs, c.cast(x, stash));
    II_ASSIGN_OR_RETURN(ir::Value * mean, c.op("reduce_mean", {xs}, reduce));
    II_ASSIGN_OR_RETURN(ir::Value * deviation, c.op("sub", {xs, mean}));
    II_ASSIGN_OR_RETURN(ir::Value * square, c.op("mul", {deviation, deviation}));
    II_ASSIGN_OR_RETURN(ir::Value * var, c.op("reduce_mean", {square}, reduce));
    II_ASSIGN_OR_RETURN(ir::Value * eps, c.scalar(c.float_attr("epsilon", 1e-5), stash));
    II_ASSIGN_OR_RETURN(ir::Value * var_eps, c.op("add", {var, eps}));
    II_ASSIGN_OR_RETURN(ir::Value * inv_std, c.op("rsqrt", {var_eps}));
    II_ASSIGN_OR_RETURN(ir::Value * normalized, c.op("mul", {deviation, inv_std}));
    II_ASSIGN_OR_RETURN(normalized, c.cast(normalized, dtype));
    II_ASSIGN_OR_RETURN(ir::Value * y, c.op("mul", {normalized, scale}));
    if (bias != nullptr) {
        II_ASSIGN_OR_RETURN(y, c.op("add", {y, bias}));
    }
    II_TRY(c.set_output(0, y));
    II_TRY(c.set_output(1, mean));
    return c.set_output(2, inv_std);
}

Status handle_layer_norm(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * scale, c.input(1));
    II_ASSIGN_OR_RETURN(ir::Value * bias, c.optional_input(2));
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(c.int_attr("axis", -1), x->rank()));
    if (c.op_type() == "LayerNormalization" && (c.has_output(1) || c.has_output(2))) {
        return layer_norm_with_statistics(c, x, scale, bias, axis);
    }
    // ONNX allows scale / bias broadcastable to the normalized shape; ii wants exactly that shape.
    const Dims normalized(x->shape().begin() + axis, x->shape().end());
    for (ir::Value** v : {&scale, &bias}) {
        if (*v != nullptr && (*v)->shape() != normalized) {
            II_ASSIGN_OR_RETURN(*v, c.op("expand", {*v}, {{"shape", normalized}}));
        }
    }
    std::vector<ir::Value*> inputs = {x, scale};
    if (bias != nullptr) {
        inputs.push_back(bias);
    }
    const std::string_view op = c.op_type() == "RMSNormalization" ? "rms_norm" : "layer_norm";
    if (op == "rms_norm" && bias != nullptr) {
        return c.unsupported("RMSNormalization with a bias");
    }
    return output(c, c.op(op, std::move(inputs),
                          {{"axis", static_cast<std::int64_t>(axis)}, {"epsilon", c.float_attr("epsilon", 1e-5)}}));
}

Status handle_instance_norm(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * scale, c.input(1));
    II_ASSIGN_OR_RETURN(ir::Value * bias, c.input(2));
    II_ASSIGN_OR_RETURN(
        ir::Value * normalized,
        c.op("layer_norm", {x}, {{"axis", std::int64_t{2}}, {"epsilon", c.float_attr("epsilon", 1e-5)}}));
    II_ASSIGN_OR_RETURN(scale, per_channel(c, scale, x->rank(), x->dtype()));
    II_ASSIGN_OR_RETURN(bias, per_channel(c, bias, x->rank(), x->dtype()));
    II_ASSIGN_OR_RETURN(ir::Value * y, c.op("mul", {normalized, scale}));
    return output(c, c.op("add", {y, bias}));
}

Status handle_group_norm(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * scale, c.input(1));
    II_ASSIGN_OR_RETURN(ir::Value * bias, c.input(2));
    const std::int64_t groups = c.int_attr("num_groups", 0);
    if (x->rank() < 3 || groups <= 0) {
        return errors::invalid_argument("GroupNormalization needs rank >= 3 and num_groups > 0");
    }
    const Dim& channels = x->shape()[1];
    if (!channels.is_constant() || channels.value() % groups != 0) {
        return c.unsupported("GroupNormalization with a symbolic or indivisible channel count");
    }
    II_ASSIGN_OR_RETURN(ir::Value * grouped, c.reshape(x, Dims{x->shape()[0], groups, -1}));
    II_ASSIGN_OR_RETURN(
        ir::Value * normalized,
        c.op("layer_norm", {grouped}, {{"axis", std::int64_t{2}}, {"epsilon", c.float_attr("epsilon", 1e-5)}}));
    II_ASSIGN_OR_RETURN(normalized, c.reshape(normalized, x->shape()));
    for (ir::Value** v : {&scale, &bias}) {
        if ((*v)->shape() == Dims{Dim(groups)} && groups != channels.value()) {
            // Opset 18: one value per group.
            II_ASSIGN_OR_RETURN(ir::Value * column, c.reshape(*v, Dims{groups, 1}));
            II_ASSIGN_OR_RETURN(ir::Value * repeated,
                                c.op("expand", {column}, {{"shape", Dims{groups, channels.value() / groups}}}));
            II_ASSIGN_OR_RETURN(*v, c.reshape(repeated, Dims{channels}));
        }
        II_ASSIGN_OR_RETURN(*v, per_channel(c, *v, x->rank(), x->dtype()));
    }
    II_ASSIGN_OR_RETURN(ir::Value * y, c.op("mul", {normalized, scale}));
    return output(c, c.op("add", {y, bias}));
}

/// Softmax / LogSoftmax: opset >= 13 along one axis; before, over the flattened dimensions from `axis`.
Status handle_softmax(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    const std::string_view op = c.op_type() == "Softmax" ? "softmax" : "log_softmax";
    const bool legacy = c.opset() < 13;
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(c.int_attr("axis", legacy ? 1 : -1), std::max(x->rank(), 1)));
    if (!legacy || axis == x->rank() - 1) {
        return output(c, c.op(op, {x}, {{"axis", static_cast<std::int64_t>(axis)}}));
    }
    const Dims& shape = x->shape();
    const Dim outer = ops::product(std::span<const Dim>(shape.data(), static_cast<std::size_t>(axis)));
    const Dim inner =
        ops::product(std::span<const Dim>(shape.data() + axis, shape.size() - static_cast<std::size_t>(axis)));
    II_ASSIGN_OR_RETURN(ir::Value * flat, c.reshape(x, Dims{outer, inner}));
    II_ASSIGN_OR_RETURN(ir::Value * y, c.op(op, {flat}, {{"axis", std::int64_t{-1}}}));
    return output(c, c.reshape(y, shape));
}

Status handle_lp_norm(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    const std::int64_t p = c.int_attr("p", 2);
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(c.int_attr("axis", -1), x->rank()));
    const Attributes reduce = {{"axes", std::vector<std::int64_t>{axis}}, {"keepdims", true}};
    ir::Value* norm = nullptr;
    if (p == 1) {
        II_ASSIGN_OR_RETURN(ir::Value * abs, c.op("abs", {x}));
        II_ASSIGN_OR_RETURN(norm, c.op("reduce_sum", {abs}, reduce));
    } else if (p == 2) {
        II_ASSIGN_OR_RETURN(ir::Value * sq, c.op("mul", {x, x}));
        II_ASSIGN_OR_RETURN(ir::Value * sum, c.op("reduce_sum", {sq}, reduce));
        II_ASSIGN_OR_RETURN(norm, c.op("sqrt", {sum}));
    } else {
        return c.unsupported(std::format("LpNormalization with p = {}", p));
    }
    // A zero norm yields zeros (x is zero there): divide by 1 instead.
    II_ASSIGN_OR_RETURN(ir::Value * zero, c.scalar(0, x->dtype()));
    II_ASSIGN_OR_RETURN(ir::Value * one, c.scalar(1, x->dtype()));
    II_ASSIGN_OR_RETURN(ir::Value * is_zero, c.op("eq", {norm, zero}));
    II_ASSIGN_OR_RETURN(ir::Value * divisor, c.op("where", {is_zero, one, norm}));
    return output(c, c.op("div", {x, divisor}));
}

// ---------------------------------------------------------------------------------------------------------------
// Reductions

Status handle_reduce(NodeContext& c) {
    const std::string_view op = c.op_type();
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(const auto axes_opt, axes_of(c, op == "ReduceSum" ? 13 : 18, 1));
    std::vector<std::int64_t> axes = axes_opt.value_or(std::vector<std::int64_t>{});
    const bool keepdims = c.int_attr("keepdims", 1) != 0;
    if (axes.empty() && c.int_attr("noop_with_empty_axes", 0) != 0) {
        return c.set_output(0, x);
    }
    const Attributes attrs = {{"axes", axes}, {"keepdims", keepdims}};
    if (x->dtype() == DType::kBool && (op == "ReduceMax" || op == "ReduceMin")) {
        // any / all
        II_ASSIGN_OR_RETURN(ir::Value * bytes, c.cast(x, DType::kU8));
        II_ASSIGN_OR_RETURN(ir::Value * reduced, c.op(op == "ReduceMax" ? "reduce_max" : "reduce_min", {bytes}, attrs));
        return output(c, c.cast(reduced, DType::kBool));
    }
    if (op == "ReduceSum") {
        return output(c, c.op("reduce_sum", {x}, attrs));
    }
    if (op == "ReduceMean") {
        return output(c, c.op("reduce_mean", {x}, attrs));
    }
    if (op == "ReduceMax") {
        return output(c, c.op("reduce_max", {x}, attrs));
    }
    if (op == "ReduceMin") {
        return output(c, c.op("reduce_min", {x}, attrs));
    }
    if (op == "ReduceProd") {
        return output(c, c.op("reduce_prod", {x}, attrs));
    }
    if (op == "ReduceL1") {
        II_ASSIGN_OR_RETURN(ir::Value * abs, c.op("abs", {x}));
        return output(c, c.op("reduce_sum", {abs}, attrs));
    }
    if (op == "ReduceL2" || op == "ReduceSumSquare") {
        II_ASSIGN_OR_RETURN(ir::Value * sq, c.op("mul", {x, x}));
        II_ASSIGN_OR_RETURN(ir::Value * sum, c.op("reduce_sum", {sq}, attrs));
        if (op == "ReduceSumSquare") {
            return c.set_output(0, sum);
        }
        return output(c, c.op("sqrt", {sum}));
    }
    if (op == "ReduceLogSum") {
        II_ASSIGN_OR_RETURN(ir::Value * sum, c.op("reduce_sum", {x}, attrs));
        return output(c, c.op("log", {sum}));
    }
    if (op == "ReduceLogSumExp") {
        const Attributes kept = {{"axes", axes}, {"keepdims", true}};
        II_ASSIGN_OR_RETURN(ir::Value * max, c.op("reduce_max", {x}, kept));
        II_ASSIGN_OR_RETURN(ir::Value * shifted, c.op("sub", {x, max}));
        II_ASSIGN_OR_RETURN(ir::Value * e, c.op("exp", {shifted}));
        II_ASSIGN_OR_RETURN(ir::Value * sum, c.op("reduce_sum", {e}, attrs));
        II_ASSIGN_OR_RETURN(ir::Value * log, c.op("log", {sum}));
        if (!keepdims) {
            II_ASSIGN_OR_RETURN(max, c.reshape(max, log->shape()));
        }
        return output(c, c.op("add", {log, max}));
    }
    return c.unsupported(std::string(op));
}

Status handle_arg(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    return output(c, c.op(c.op_type() == "ArgMax" ? "argmax" : "argmin", {x},
                          {{"axis", c.int_attr("axis", 0)},
                           {"keepdims", c.int_attr("keepdims", 1) != 0},
                           {"select_last_index", c.int_attr("select_last_index", 0) != 0}}));
}

Status handle_topk(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    std::int64_t k = c.int_attr("k", -1);
    if (c.opset() >= 10) {
        II_ASSIGN_OR_RETURN(const std::vector<std::int64_t> ks, c.ints_input(1));
        if (ks.size() != 1) {
            return errors::invalid_argument("TopK: K must have one element");
        }
        k = ks[0];
    }
    II_ASSIGN_OR_RETURN(
        std::vector<ir::Value*> outputs,
        c.op_outputs("topk", {x},
                     {{"k", k}, {"axis", c.int_attr("axis", -1)}, {"largest", c.int_attr("largest", 1) != 0}}));
    II_TRY(c.set_output(0, outputs[0]));
    return c.set_output(1, outputs[1]);
}

Status handle_cumsum(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(const std::vector<std::int64_t> axis, c.ints_input(1));
    if (axis.size() != 1) {
        return errors::invalid_argument("CumSum: axis must have one element");
    }
    return output(c, c.op("cumsum", {x},
                          {{"axis", axis[0]},
                           {"exclusive", c.int_attr("exclusive", 0) != 0},
                           {"reverse", c.int_attr("reverse", 0) != 0}}));
}

// ---------------------------------------------------------------------------------------------------------------
// Data movement

Status handle_reshape(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    Dims shape;
    if (c.opset() < 5) {
        for (const std::int64_t d : c.ints_attr("shape").value_or(std::vector<std::int64_t>{})) {
            shape.emplace_back(d);
        }
    } else {
        II_ASSIGN_OR_RETURN(shape, c.dims_input(1));
    }
    const bool allow_zero = c.int_attr("allowzero", 0) != 0;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (!allow_zero && shape[i] == Dim(0)) {
            if (i >= x->shape().size()) {
                return errors::invalid_argument("Reshape: 0 at position {} beyond the input rank", i);
            }
            shape[i] = x->shape()[i];
        }
    }
    return output(c, c.reshape(x, std::move(shape)));
}

Status handle_flatten(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    std::int64_t axis = c.int_attr("axis", 1);
    if (axis < 0) {
        axis += x->rank();
    }
    if (axis < 0 || axis > x->rank()) {
        return errors::invalid_argument("Flatten: axis {} is out of range for rank {}", c.int_attr("axis", 1),
                                        x->rank());
    }
    const Dims& shape = x->shape();
    const Dim outer = ops::product(std::span<const Dim>(shape.data(), static_cast<std::size_t>(axis)));
    const Dim inner =
        ops::product(std::span<const Dim>(shape.data() + axis, shape.size() - static_cast<std::size_t>(axis)));
    return output(c, c.reshape(x, Dims{outer, inner}));
}

Status handle_transpose(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    std::vector<std::int64_t> perm = c.ints_attr("perm").value_or(std::vector<std::int64_t>{});
    if (perm.empty()) {
        for (int d = x->rank() - 1; d >= 0; --d) {
            perm.push_back(d);
        }
    }
    bool identity = true;
    for (std::size_t i = 0; i < perm.size(); ++i) {
        identity = identity && perm[i] == static_cast<std::int64_t>(i);
    }
    if (identity && static_cast<int>(perm.size()) == x->rank()) {
        return c.set_output(0, x);
    }
    return output(c, c.op("permute", {x}, {{"perm", std::move(perm)}}));
}

Status handle_squeeze(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(const auto axes, axes_of(c, 13, 1));
    const int rank = x->rank();
    std::vector<bool> drop(static_cast<std::size_t>(rank), false);
    if (axes && !axes->empty()) {
        for (const std::int64_t axis : *axes) {
            II_ASSIGN_OR_RETURN(const int a, normalize_axis(axis, rank));
            drop[static_cast<std::size_t>(a)] = true;
        }
    } else {
        for (int d = 0; d < rank; ++d) {
            drop[static_cast<std::size_t>(d)] = x->shape()[static_cast<std::size_t>(d)] == Dim(1);
        }
    }
    Dims shape;
    for (int d = 0; d < rank; ++d) {
        if (!drop[static_cast<std::size_t>(d)]) {
            shape.push_back(x->shape()[static_cast<std::size_t>(d)]);
        }
    }
    return output(c, c.reshape(x, std::move(shape)));
}

Status handle_unsqueeze(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(const auto axes, axes_of(c, 13, 1));
    if (!axes) {
        return errors::invalid_argument("Unsqueeze requires axes");
    }
    const int rank = x->rank() + static_cast<int>(axes->size());
    std::vector<bool> inserted(static_cast<std::size_t>(rank), false);
    for (const std::int64_t axis : *axes) {
        II_ASSIGN_OR_RETURN(const int a, normalize_axis(axis, rank));
        inserted[static_cast<std::size_t>(a)] = true;
    }
    Dims shape;
    std::size_t next = 0;
    for (int d = 0; d < rank; ++d) {
        if (inserted[static_cast<std::size_t>(d)]) {
            shape.emplace_back(1);
        } else {
            if (next >= x->shape().size()) {
                return errors::invalid_argument("Unsqueeze: duplicate axes");
            }
            shape.push_back(x->shape()[next++]);
        }
    }
    return output(c, c.reshape(x, std::move(shape)));
}

Status handle_concat(NodeContext& c) {
    std::vector<ir::Value*> inputs;
    for (int i = 0; i < c.num_inputs(); ++i) {
        if (c.has_input(i)) {
            II_ASSIGN_OR_RETURN(ir::Value * x, c.input(i));
            inputs.push_back(x);
        }
    }
    if (inputs.size() == 1) {
        return c.set_output(0, inputs[0]);
    }
    return output(c, c.op("concat", std::move(inputs), {{"axis", c.int_attr("axis", 0)}}));
}

Status handle_split(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(c.int_attr("axis", 0), x->rank()));
    std::vector<std::int64_t> sizes;
    if (c.opset() >= 13 && c.has_input(1)) {
        II_ASSIGN_OR_RETURN(sizes, c.ints_input(1));
    } else if (c.opset() < 13 && c.has_attribute("split")) {
        sizes = *c.ints_attr("split");
    } else {
        const Dim& dim = x->shape()[static_cast<std::size_t>(axis)];
        const std::int64_t parts = c.has_attribute("num_outputs") ? c.int_attr("num_outputs", 1) : c.num_outputs();
        if (!dim.is_constant() || parts <= 0) {
            return c.unsupported("Split of a symbolic dimension into equal parts");
        }
        const std::int64_t chunk = (dim.value() + parts - 1) / parts;
        std::int64_t remaining = dim.value();
        for (std::int64_t i = 0; i < parts; ++i) {
            sizes.push_back(std::min(chunk, remaining));
            remaining -= sizes.back();
        }
    }
    II_ASSIGN_OR_RETURN(std::vector<ir::Value*> outputs,
                        c.op_outputs("split", {x}, {{"axis", static_cast<std::int64_t>(axis)}, {"sizes", sizes}}));
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        II_TRY(c.set_output(static_cast<int>(i), outputs[i]));
    }
    return ok_status();
}

Status handle_slice(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(SliceParams p, slice_params(c, x->rank()));
    std::vector<std::int64_t> starts;
    std::vector<std::int64_t> ends;
    bool identity = true;
    for (std::size_t i = 0; i < p.starts.size(); ++i) {
        const Dim& size = x->shape()[static_cast<std::size_t>(p.axes[i])];
        // x[start : end] with a constant start, a symbolic end (not the dimension itself) and a positive step, e.g.
        // position_ids[:, :seq]: gather the indices start, start + step, ... (out-of-range indices fail at run time).
        if (p.starts[i].is_constant() && p.starts[i].value() >= 0 && !p.ends[i].is_constant() && !(p.ends[i] == size) &&
            p.steps[i] > 0) {
            const Dim count = ceil_div(p.ends[i] - p.starts[i], p.steps[i]);
            II_ASSIGN_OR_RETURN(ir::Value * indices, c.op("iota", {},
                                                          {{"shape", Dims{count}},
                                                           {"start", static_cast<double>(p.starts[i].value())},
                                                           {"step", static_cast<double>(p.steps[i])}}));
            II_ASSIGN_OR_RETURN(x, c.op("gather", {x, indices}, {{"axis", p.axes[i]}}));
            starts.push_back(0);
            ends.push_back(kInt64Max);
            p.steps[i] = 1;
            continue;
        }
        identity = false;
        // A symbolic bound equal to the dimension itself is "the end".
        const auto resolve = [&](const Dim& bound, bool is_end) -> Result<std::int64_t> {
            if (bound.is_constant()) {
                return bound.value();
            }
            if (bound == size) {
                return p.steps[i] > 0 || !is_end ? kInt64Max : std::int64_t{-1};
            }
            return errors::unsupported("Slice bound {} along a dimension of size {} is not known when importing; "
                                       "fix the model's input dimensions (e.g. --dims) to import it",
                                       bound.to_string(), size.to_string());
        };
        II_ASSIGN_OR_RETURN(const std::int64_t start, resolve(p.starts[i], false));
        II_ASSIGN_OR_RETURN(const std::int64_t end, resolve(p.ends[i], true));
        starts.push_back(start);
        ends.push_back(end);
    }
    if (identity) {
        return c.set_output(0, x);
    }
    return output(c, c.op("slice", {x}, {{"axes", p.axes}, {"starts", starts}, {"ends", ends}, {"steps", p.steps}}));
}

Status handle_gather(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * data, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * indices, c.input(1));
    if (indices->dtype() != DType::kI32 && indices->dtype() != DType::kI64) {
        II_ASSIGN_OR_RETURN(indices, c.cast(indices, DType::kI64));
    }
    return output(c, c.op("gather", {data, indices}, {{"axis", c.int_attr("axis", 0)}}));
}

Status handle_scatter(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * data, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * indices, c.input(1));
    II_ASSIGN_OR_RETURN(ir::Value * updates, c.input(2));
    return output(c, c.op("scatter", {data, indices, updates},
                          {{"axis", c.int_attr("axis", 0)}, {"reduction", c.string_attr("reduction", "none")}}));
}

Status handle_expand(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(Dims shape, c.dims_input(1));
    return output(c, c.op("expand", {x}, {{"shape", std::move(shape)}}));
}

Status handle_tile(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(std::vector<std::int64_t> repeats, c.ints_input(1));
    return output(c, c.op("tile", {x}, {{"repeats", std::move(repeats)}}));
}

Status handle_pad(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    const std::string mode = lower(c.string_attr("mode", "constant"));
    if (mode != "constant" && mode != "reflect" && mode != "edge") {
        return c.unsupported(std::format("Pad mode '{}'", mode));
    }
    std::vector<std::int64_t> pads;
    double value = 0;
    if (c.opset() < 11) {
        pads = c.ints_attr("pads").value_or(std::vector<std::int64_t>{});
        value = c.float_attr("value", 0.0);
    } else {
        II_ASSIGN_OR_RETURN(pads, c.ints_input(1));
        if (c.has_input(2)) {
            II_ASSIGN_OR_RETURN(const std::vector<double> v, c.floats_input(2));
            if (v.size() != 1) {
                return errors::invalid_argument("Pad: constant_value must have one element");
            }
            value = v[0];
        }
        if (c.has_input(3)) {
            II_ASSIGN_OR_RETURN(const std::vector<std::int64_t> axes, c.ints_input(3));
            if (pads.size() != 2 * axes.size()) {
                return errors::invalid_argument("Pad: {} pads for {} axes", pads.size(), axes.size());
            }
            std::vector<std::int64_t> full(static_cast<std::size_t>(2 * x->rank()), 0);
            for (std::size_t i = 0; i < axes.size(); ++i) {
                II_ASSIGN_OR_RETURN(const int a, normalize_axis(axes[i], x->rank()));
                full[static_cast<std::size_t>(a)] = pads[i];
                full[static_cast<std::size_t>(a + x->rank())] = pads[i + axes.size()];
            }
            pads = std::move(full);
        }
    }
    return output(c, c.op("pad", {x}, {{"pads", std::move(pads)}, {"mode", mode}, {"value", value}}));
}

Status handle_constant(NodeContext& c) {
    Tensor tensor;
    if (c.has_attribute("value")) {
        II_ASSIGN_OR_RETURN(tensor, c.tensor_attr("value"));
    } else if (const OnnxAttribute* f = c.attribute("value_float")) {
        II_ASSIGN_OR_RETURN(tensor, Tensor::scalar(f->f));
    } else if (const OnnxAttribute* fs = c.attribute("value_floats")) {
        II_ASSIGN_OR_RETURN(tensor, Tensor::from(std::span<const float>(fs->floats),
                                                 Shape{static_cast<std::int64_t>(fs->floats.size())}));
    } else if (const OnnxAttribute* i = c.attribute("value_int")) {
        II_ASSIGN_OR_RETURN(tensor, Tensor::scalar(i->i));
    } else if (const OnnxAttribute* is = c.attribute("value_ints")) {
        II_ASSIGN_OR_RETURN(tensor, Tensor::from(std::span<const std::int64_t>(is->ints),
                                                 Shape{static_cast<std::int64_t>(is->ints.size())}));
    } else {
        return c.unsupported("Constant without a numeric value (sparse or string constants)");
    }
    ir::Value* value = c.importer().graph().add_constant(
        std::move(tensor), c.node().outputs.empty() ? std::string() : c.node().outputs[0]);
    return c.set_output(0, value);
}

Status handle_constant_of_shape(NodeContext& c) {
    II_ASSIGN_OR_RETURN(Dims shape, c.dims_input(0));
    DType dtype = DType::kF32;
    double value = 0;
    if (c.has_attribute("value")) {
        II_ASSIGN_OR_RETURN(const Tensor t, c.tensor_attr("value"));
        if (t.numel() != 1) {
            return errors::invalid_argument("ConstantOfShape: value must have one element");
        }
        dtype = t.dtype();
        switch (dtype) {
        case DType::kF32:
            value = t.data_as<float>()[0];
            break;
        case DType::kF64:
            value = t.data_as<double>()[0];
            break;
        case DType::kF16:
            value = t.data_as<f16>()[0].to_float();
            break;
        case DType::kBF16:
            value = t.data_as<bf16>()[0].to_float();
            break;
        case DType::kI64:
            value = static_cast<double>(t.data_as<std::int64_t>()[0]);
            break;
        case DType::kI32:
            value = t.data_as<std::int32_t>()[0];
            break;
        case DType::kBool:
        case DType::kU8:
            value = *static_cast<const std::uint8_t*>(t.data());
            break;
        case DType::kI8:
            value = *static_cast<const std::int8_t*>(t.data());
            break;
        default:
            return c.unsupported(std::format("ConstantOfShape of {}", to_string(dtype)));
        }
    }
    return output(c, c.op("fill", {}, {{"shape", std::move(shape)}, {"dtype", dtype}, {"value", value}}));
}

Status handle_range(NodeContext& c) {
    II_ASSIGN_OR_RETURN(const ir::TensorType type, c.input_type(0));
    II_ASSIGN_OR_RETURN(const std::vector<double> start, c.floats_input(0));
    II_ASSIGN_OR_RETURN(const std::vector<double> delta, c.floats_input(2));
    if (start.size() != 1 || delta.size() != 1 || delta[0] == 0) {
        return errors::invalid_argument("Range: start and delta must be non-zero scalars");
    }
    Dim count;
    if (c.is_constant_input(1)) {
        II_ASSIGN_OR_RETURN(const std::vector<double> limit, c.floats_input(1));
        if (limit.size() != 1) {
            return errors::invalid_argument("Range: limit must be a scalar");
        }
        count = Dim(std::max<std::int64_t>(0, static_cast<std::int64_t>(std::ceil((limit[0] - start[0]) / delta[0]))));
    } else {
        II_ASSIGN_OR_RETURN(const Dims limit, c.dims_input(1));
        const double s = start[0];
        const double d = delta[0];
        if (limit.size() != 1 || s != std::floor(s) || d != std::floor(d) || d <= 0) {
            return c.unsupported("Range with a symbolic limit and non-integer or negative parameters");
        }
        count = ceil_div(limit[0] - Dim(static_cast<std::int64_t>(s)), static_cast<std::int64_t>(d));
    }
    return output(
        c, c.op("iota", {}, {{"shape", Dims{count}}, {"dtype", type.dtype}, {"start", start[0]}, {"step", delta[0]}}));
}

Status handle_resize(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    const int rank = x->rank();
    const bool upsample = c.op_type() == "Upsample";
    std::string mode = lower(c.string_attr("mode", "nearest"));
    std::string coordinate_mode = "asymmetric";
    if (!upsample && c.opset() >= 11) {
        coordinate_mode = lower(c.string_attr("coordinate_transformation_mode", "half_pixel"));
    }
    if (mode == "bilinear" || mode == "trilinear") {
        mode = "linear";
    }
    if (coordinate_mode == "tf_crop_and_resize" || coordinate_mode == "tf_half_pixel_for_nn") {
        return c.unsupported(std::format("Resize coordinate_transformation_mode '{}'", coordinate_mode));
    }
    if (c.int_attr("antialias", 0) != 0) {
        return c.unsupported("Resize with antialias");
    }
    if (lower(c.string_attr("keep_aspect_ratio_policy", "stretch")) != "stretch") {
        return c.unsupported("Resize keep_aspect_ratio_policy other than 'stretch'");
    }
    // Scales and sizes, possibly for a subset of axes (opset 18).
    std::vector<std::int64_t> axes = c.ints_attr("axes").value_or(std::vector<std::int64_t>{});
    std::vector<double> scales;
    Dims sizes;
    if (upsample) {
        if (c.opset() < 9) {
            scales = c.floats_attr("scales").value_or(std::vector<double>{});
        } else {
            II_ASSIGN_OR_RETURN(scales, c.floats_input(1));
        }
    } else if (c.opset() < 11) {
        II_ASSIGN_OR_RETURN(scales, c.floats_input(1));
    } else {
        if (c.has_input(3)) {
            II_ASSIGN_OR_RETURN(sizes, c.dims_input(3));
        }
        if (sizes.empty() && c.has_input(2)) {
            II_ASSIGN_OR_RETURN(scales, c.floats_input(2));
        }
    }
    if (axes.empty()) {
        for (int d = 0; d < rank; ++d) {
            axes.push_back(d);
        }
    }
    const std::size_t count = sizes.empty() ? scales.size() : sizes.size();
    if (count != axes.size()) {
        return errors::invalid_argument("Resize: {} scales / sizes for {} axes", count, axes.size());
    }
    Dims shape = x->shape();
    std::vector<double> full_scales(static_cast<std::size_t>(rank), 1.0);
    for (std::size_t i = 0; i < axes.size(); ++i) {
        II_ASSIGN_OR_RETURN(const int axis, normalize_axis(axes[i], rank));
        const Dim& in = x->shape()[static_cast<std::size_t>(axis)];
        if (!sizes.empty()) {
            shape[static_cast<std::size_t>(axis)] = sizes[i];
            full_scales[static_cast<std::size_t>(axis)] = 0;  // derived from the sizes
            continue;
        }
        const double scale = scales[i];
        if (!(scale > 0)) {
            return errors::invalid_argument("Resize: invalid scale {}", scale);
        }
        full_scales[static_cast<std::size_t>(axis)] = scale;
        if (in.is_constant()) {
            shape[static_cast<std::size_t>(axis)] =
                Dim(static_cast<std::int64_t>(std::floor(static_cast<double>(in.value()) * scale)));
        } else if (scale == std::floor(scale)) {
            shape[static_cast<std::size_t>(axis)] = in * Dim(static_cast<std::int64_t>(scale));
        } else {
            return c.unsupported("Resize of a symbolic dimension by a non-integer scale");
        }
    }
    std::string nearest_mode = lower(c.string_attr("nearest_mode", "round_prefer_floor"));
    if (upsample || c.opset() < 11) {
        // Upsample and Resize-10 pick the nearest input pixel by truncation (ceil when downsampling).
        const bool down = std::ranges::any_of(full_scales, [](double s) { return s > 0 && s < 1; });
        nearest_mode = down ? "ceil" : "floor";
    }
    Attributes attrs = {{"shape", std::move(shape)},
                        {"mode", mode},
                        {"coordinate_mode", coordinate_mode},
                        {"nearest_mode", nearest_mode},
                        {"cubic_coeff_a", c.float_attr("cubic_coeff_a", -0.75)},
                        {"exclude_outside", c.int_attr("exclude_outside", 0) != 0}};
    if (sizes.empty()) {
        attrs.set("scales", std::move(full_scales));
    }
    return output(c, c.op("resize", {x}, std::move(attrs)));
}

Status handle_depth_space(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    const std::int64_t b = c.int_attr("blocksize", 0);
    if (x->rank() != 4 || b <= 0) {
        return errors::invalid_argument("{} needs a 4-D input and blocksize > 0", c.op_type());
    }
    const Dims& s = x->shape();
    const Dim& channels = s[1];
    if (!channels.is_constant()) {
        return c.unsupported("DepthToSpace / SpaceToDepth with a symbolic channel count");
    }
    const std::int64_t C = channels.value();
    if (c.op_type() == "DepthToSpace") {
        if (C % (b * b) != 0) {
            return errors::invalid_argument("DepthToSpace: channels not divisible by blocksize^2");
        }
        const bool crd = lower(c.string_attr("mode", "DCR")) == "crd";
        Dims split = crd ? Dims{s[0], C / (b * b), b, b, s[2], s[3]} : Dims{s[0], b, b, C / (b * b), s[2], s[3]};
        const std::vector<std::int64_t> perm =
            crd ? std::vector<std::int64_t>{0, 1, 4, 2, 5, 3} : std::vector<std::int64_t>{0, 3, 4, 1, 5, 2};
        II_ASSIGN_OR_RETURN(ir::Value * r, c.reshape(x, std::move(split)));
        II_ASSIGN_OR_RETURN(ir::Value * t, c.op("permute", {r}, {{"perm", perm}}));
        return output(c, c.reshape(t, Dims{s[0], C / (b * b), s[2] * Dim(b), s[3] * Dim(b)}));
    }
    const std::optional<Dim> h = s[2].exact_div(b);
    const std::optional<Dim> w = s[3].exact_div(b);
    if (!h || !w) {
        return c.unsupported("SpaceToDepth of dimensions not divisible by blocksize");
    }
    const bool crd = lower(c.string_attr("mode", "DCR")) == "crd";
    II_ASSIGN_OR_RETURN(ir::Value * r, c.reshape(x, Dims{s[0], C, *h, b, *w, b}));
    II_ASSIGN_OR_RETURN(ir::Value * t, c.op("permute", {r},
                                            {{"perm", crd ? std::vector<std::int64_t>{0, 1, 3, 5, 2, 4}
                                                          : std::vector<std::int64_t>{0, 3, 5, 1, 2, 4}}}));
    return output(c, c.reshape(t, Dims{s[0], C * b * b, *h, *w}));
}

Status handle_trilu(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    if (x->rank() < 2) {
        return errors::invalid_argument("Trilu needs rank >= 2");
    }
    std::int64_t k = 0;
    if (c.has_input(1)) {
        II_ASSIGN_OR_RETURN(const std::vector<std::int64_t> ks, c.ints_input(1));
        if (ks.size() != 1) {
            return errors::invalid_argument("Trilu: k must have one element");
        }
        k = ks[0];
    }
    const bool upper = c.int_attr("upper", 1) != 0;
    const Dims grid = {x->shape()[x->shape().size() - 2], x->shape().back()};
    II_ASSIGN_OR_RETURN(ir::Value * rows, c.op("iota", {}, {{"shape", grid}, {"axis", std::int64_t{0}}}));
    II_ASSIGN_OR_RETURN(ir::Value * cols, c.op("iota", {}, {{"shape", grid}, {"axis", std::int64_t{1}}}));
    II_ASSIGN_OR_RETURN(ir::Value * diff, c.op("sub", {cols, rows}));
    II_ASSIGN_OR_RETURN(ir::Value * kv, c.scalar(static_cast<double>(k), DType::kI64));
    II_ASSIGN_OR_RETURN(ir::Value * mask, c.op(upper ? "ge" : "le", {diff, kv}));
    II_ASSIGN_OR_RETURN(ir::Value * zero, c.scalar(0, x->dtype()));
    return output(c, c.op("where", {mask, x, zero}));
}

Status handle_quantize(NodeContext& c) {
    if (c.int_attr("block_size", 0) != 0) {
        return c.unsupported("blocked QuantizeLinear / DequantizeLinear");
    }
    II_ASSIGN_OR_RETURN(ir::Value * x, c.input(0));
    II_ASSIGN_OR_RETURN(ir::Value * scale, c.input(1));
    II_ASSIGN_OR_RETURN(ir::Value * zero_point, c.optional_input(2));
    const std::int64_t axis = c.int_attr("axis", 1);
    std::vector<ir::Value*> inputs = {x, scale};
    if (zero_point != nullptr) {
        inputs.push_back(zero_point);
    }
    if (c.op_type() == "DequantizeLinear") {
        return output(c, c.op("dequantize", std::move(inputs), {{"axis", axis}}));
    }
    DType dtype = zero_point != nullptr ? zero_point->dtype() : DType::kU8;
    if (const std::int64_t output_dtype = c.int_attr("output_dtype", 0); output_dtype != 0) {
        dtype = dtype_from_onnx(static_cast<std::int32_t>(output_dtype));
    }
    II_ASSIGN_OR_RETURN(inputs[1], c.cast(scale, x->dtype()));
    return output(c, c.op("quantize", std::move(inputs), {{"axis", axis}, {"dtype", dtype}}));
}

Status handle_if(NodeContext& c) {
    if (!c.is_constant_input(0)) {
        return c.unsupported("If with a condition computed at run time");
    }
    II_ASSIGN_OR_RETURN(const std::vector<std::int64_t> cond, c.ints_input(0));
    if (cond.size() != 1) {
        return errors::invalid_argument("If: the condition must have one element");
    }
    const OnnxAttribute* branch = c.attribute(cond[0] != 0 ? "then_branch" : "else_branch");
    if (branch == nullptr || !branch->g) {
        return errors::invalid_argument("If: missing branch");
    }
    II_ASSIGN_OR_RETURN(std::vector<Entry> outputs, c.importer().inline_subgraph(*branch->g));
    if (static_cast<int>(outputs.size()) != c.num_outputs()) {
        return errors::invalid_argument("If: the branch has {} outputs, the node {}", outputs.size(), c.num_outputs());
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (c.has_output(static_cast<int>(i))) {
            II_TRY(c.importer().bind(c.node().outputs[i], outputs[i]));
        }
    }
    return ok_status();
}

Status handle_one_hot(NodeContext& c) {
    II_ASSIGN_OR_RETURN(ir::Value * indices, c.input(0));
    II_ASSIGN_OR_RETURN(const std::vector<std::int64_t> depth_values, c.ints_input(1));
    II_ASSIGN_OR_RETURN(ir::Value * values, c.input(2));
    if (depth_values.size() != 1 || depth_values[0] <= 0) {
        return errors::invalid_argument("OneHot: depth must be one positive value");
    }
    const std::int64_t depth = depth_values[0];
    const int rank = indices->rank() + 1;
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(c.int_attr("axis", -1), rank));
    II_ASSIGN_OR_RETURN(indices, c.cast(indices, DType::kI64));
    // Negative indices count from the back: index + depth * (index < 0).
    II_ASSIGN_OR_RETURN(ir::Value * zero, c.scalar(0, DType::kI64));
    II_ASSIGN_OR_RETURN(ir::Value * d, c.scalar(static_cast<double>(depth), DType::kI64));
    II_ASSIGN_OR_RETURN(ir::Value * negative, c.op("lt", {indices, zero}));
    II_ASSIGN_OR_RETURN(ir::Value * wrapped, c.op("add", {indices, d}));
    II_ASSIGN_OR_RETURN(indices, c.op("where", {negative, wrapped, indices}));
    Dims expanded = indices->shape();
    expanded.insert(expanded.begin() + axis, Dim(1));
    II_ASSIGN_OR_RETURN(ir::Value * column, c.reshape(indices, expanded));
    Dims range_shape(static_cast<std::size_t>(rank), Dim(1));
    range_shape[static_cast<std::size_t>(axis)] = Dim(depth);
    II_ASSIGN_OR_RETURN(ir::Value * range, c.op("iota", {}, {{"shape", range_shape}, {"axis", std::int64_t{axis}}}));
    II_ASSIGN_OR_RETURN(ir::Value * hot, c.op("eq", {column, range}));
    II_ASSIGN_OR_RETURN(ir::Value * off, c.op("slice", {values},
                                              {{"axes", std::vector<std::int64_t>{0}},
                                               {"starts", std::vector<std::int64_t>{0}},
                                               {"ends", std::vector<std::int64_t>{1}}}));
    II_ASSIGN_OR_RETURN(ir::Value * on, c.op("slice", {values},
                                             {{"axes", std::vector<std::int64_t>{0}},
                                              {"starts", std::vector<std::int64_t>{1}},
                                              {"ends", std::vector<std::int64_t>{2}}}));
    II_ASSIGN_OR_RETURN(off, c.reshape(off, Dims{}));
    II_ASSIGN_OR_RETURN(on, c.reshape(on, Dims{}));
    return output(c, c.op("where", {hot, on, off}));
}

// ---------------------------------------------------------------------------------------------------------------

const std::unordered_map<std::string_view, OpHandler>& handlers() {
    static const std::unordered_map<std::string_view, OpHandler> table = [] {
        std::unordered_map<std::string_view, OpHandler> t;
        for (const auto& [name, op] : unary_ops()) {
            t.emplace(name, handle_unary);
        }
        for (const auto& [name, op] : binary_ops()) {
            t.emplace(name, handle_binary);
        }
        for (const std::string_view name :
             {"LeakyRelu", "PRelu", "Elu", "Selu", "Celu", "HardSigmoid", "HardSwish", "Softplus", "Mish", "Softsign",
              "ThresholdedRelu", "Swish", "Gelu", "IsNaN", "IsInf", "Tan", "Sinh", "Cosh"}) {
            t.emplace(name, handle_activation);
        }
        for (const std::string_view name : {"Max", "Min", "Sum", "Mean"}) {
            t.emplace(name, handle_variadic);
        }
        for (const std::string_view name :
             {"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
              "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp"}) {
            t.emplace(name, handle_reduce);
        }
        t.emplace("Pow", handle_pow);
        t.emplace("Mod", handle_mod);
        t.emplace("Where", handle_where);
        t.emplace("Cast", handle_cast);
        t.emplace("CastLike", handle_cast);
        t.emplace("Identity", handle_identity);
        t.emplace("Dropout", handle_identity);
        t.emplace("Clip", handle_clip);
        t.emplace("MatMul", handle_matmul);
        t.emplace("Gemm", handle_gemm);
        t.emplace("Conv", handle_conv);
        t.emplace("ConvTranspose", handle_conv);
        t.emplace("MaxPool", handle_pool);
        t.emplace("AveragePool", handle_pool);
        t.emplace("GlobalAveragePool", handle_global_pool);
        t.emplace("GlobalMaxPool", handle_global_pool);
        t.emplace("BatchNormalization", handle_batch_norm);
        t.emplace("LayerNormalization", handle_layer_norm);
        t.emplace("RMSNormalization", handle_layer_norm);
        t.emplace("InstanceNormalization", handle_instance_norm);
        t.emplace("GroupNormalization", handle_group_norm);
        t.emplace("Softmax", handle_softmax);
        t.emplace("LogSoftmax", handle_softmax);
        t.emplace("LpNormalization", handle_lp_norm);
        t.emplace("ArgMax", handle_arg);
        t.emplace("ArgMin", handle_arg);
        t.emplace("TopK", handle_topk);
        t.emplace("CumSum", handle_cumsum);
        t.emplace("Reshape", handle_reshape);
        t.emplace("Flatten", handle_flatten);
        t.emplace("Transpose", handle_transpose);
        t.emplace("Squeeze", handle_squeeze);
        t.emplace("Unsqueeze", handle_unsqueeze);
        t.emplace("Concat", handle_concat);
        t.emplace("Split", handle_split);
        t.emplace("Slice", handle_slice);
        t.emplace("Gather", handle_gather);
        t.emplace("ScatterElements", handle_scatter);
        t.emplace("Scatter", handle_scatter);
        t.emplace("Expand", handle_expand);
        t.emplace("Tile", handle_tile);
        t.emplace("Pad", handle_pad);
        t.emplace("Constant", handle_constant);
        t.emplace("ConstantOfShape", handle_constant_of_shape);
        t.emplace("Range", handle_range);
        t.emplace("Resize", handle_resize);
        t.emplace("Upsample", handle_resize);
        t.emplace("DepthToSpace", handle_depth_space);
        t.emplace("SpaceToDepth", handle_depth_space);
        t.emplace("Trilu", handle_trilu);
        t.emplace("QuantizeLinear", handle_quantize);
        t.emplace("DequantizeLinear", handle_quantize);
        t.emplace("If", handle_if);
        t.emplace("OneHot", handle_one_hot);
        return t;
    }();
    return table;
}

}  // namespace

OpHandler find_op_handler(std::string_view op_type) noexcept {
    const auto& table = handlers();
    const auto it = table.find(op_type);
    return it != table.end() ? it->second : nullptr;
}

Result<bool> try_known_contents(NodeContext& context) {
    return known_shape_op(context);
}

std::vector<std::string_view> supported_onnx_ops() {
    std::vector<std::string_view> names;
    for (const auto& [name, handler] : handlers()) {
        names.push_back(name);
    }
    for (const std::string_view name : {"Shape", "Size"}) {
        names.push_back(name);
    }
    std::ranges::sort(names);
    return names;
}

}  // namespace ii::formats::onnx
