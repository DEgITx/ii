// SPDX-License-Identifier: MIT
// Composite operations. Each one has a decomposition into simpler operations; backends may implement them natively.
#include "ir/builder.h"
#include "ops/infer_util.h"
#include "ops/opset.h"

#include <cmath>
#include <limits>
#include <numbers>

namespace ii::ops {

namespace {

using ir::AttrSpec;
using ir::Builder;
using ir::Cost;
using ir::CostContext;
using ir::InferContext;
using ir::InputSpec;
using ir::Node;
using ir::OpSchema;
using ir::OutputSpec;
using ir::TensorType;
using ir::Value;

bool is_float_dtype(DType dtype) noexcept {
    return is_float(dtype);
}

bool is_index_dtype(DType dtype) noexcept {
    return dtype == DType::kI32 || dtype == DType::kI64;
}

Status infer_float_unary(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_float_dtype, "a float type"));
    if (context.op.name == "gelu") {
        const std::string_view approximate = context.attrs.get_string("approximate");
        if (approximate != "none" && approximate != "tanh") {
            return errors::invalid_argument("'gelu' approximate must be none or tanh; got '{}'", approximate);
        }
    }
    outputs.push_back(*context.input(0));
    return ok_status();
}

Status infer_softmax(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_float_dtype, "a float type"));
    if (context.input(0)->rank() == 0) {
        return errors::invalid_argument("'{}' requires rank >= 1", context.op.name);
    }
    II_TRY(normalize_axis(context.attrs.get_int("axis"), context.input(0)->rank()));
    outputs.push_back(*context.input(0));
    return ok_status();
}

/// Checks that optional parameter `index` (scale / bias of a normalization) has x's dtype and shape x.shape[axis:].
Status check_norm_param(const InferContext& context, int index, int axis) {
    const TensorType* param = context.input(index);
    if (param == nullptr) {
        return ok_status();
    }
    II_TRY(expect_same_dtype(context, index));
    const TensorType& x = *context.input(0);
    const Dims expected(x.shape.begin() + axis, x.shape.end());
    if (!(param->shape == expected)) {
        return errors::invalid_argument("'{}' parameter {} must have shape {}, got {}", context.op.name, index,
                                        to_string(expected), to_string(param->shape));
    }
    return ok_status();
}

Status infer_norm(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_float_dtype, "a float type"));
    const TensorType& x = *context.input(0);
    if (x.rank() == 0) {
        return errors::invalid_argument("'{}' requires rank >= 1", context.op.name);
    }
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(context.attrs.get_int("axis"), x.rank()));
    if (context.attrs.get_float("epsilon") < 0) {
        return errors::invalid_argument("'{}' epsilon must not be negative", context.op.name);
    }
    for (int i = 1; i < context.num_inputs(); ++i) {
        II_TRY(check_norm_param(context, i, axis));
    }
    outputs.push_back(x);
    return ok_status();
}

Status infer_linear(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_float_dtype, "a float type"));
    const TensorType& x = *context.input(0);
    const TensorType& w = *context.input(1);
    if (x.dtype == DType::kF32 && (w.dtype == DType::kF16 || w.dtype == DType::kBF16)) {
        // 16-bit weights with f32 activations: w is widened, the result is f32.
    } else if (!is_block_quantized(w.dtype)) {
        II_TRY(expect_same_dtype(context, 1));
    } else if (!context.attrs.get_bool("transpose_w")) {
        return errors::invalid_argument("'linear' with a block-quantized weight requires transpose_w = true");
    }
    if (x.rank() == 0 || w.rank() != 2) {
        return errors::invalid_argument("'linear' expects x of rank >= 1 and a 2-D weight, got {} and {}",
                                        x.to_string(), w.to_string());
    }
    const bool transpose = context.attrs.get_bool("transpose_w");
    const Dim& in = transpose ? w.shape[1] : w.shape[0];
    const Dim& out = transpose ? w.shape[0] : w.shape[1];
    if (!(x.shape.back() == in)) {
        return errors::invalid_argument("'linear' input features {} do not match the weight {}",
                                        x.shape.back().to_string(), w.to_string());
    }
    if (const TensorType* bias = context.input(2)) {
        II_TRY(expect_same_dtype(context, 2));
        if (bias->rank() != 1 || !(bias->shape[0] == out)) {
            return errors::invalid_argument("'linear' bias must be [{}], got {}", out.to_string(), bias->to_string());
        }
    }
    Dims shape = x.shape;
    shape.back() = out;
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Cost cost_linear(const CostContext& context) {
    const Shape& x = context.inputs[0];
    const double k = static_cast<double>(x[x.rank() - 1]);
    double bytes = 0;
    for (std::size_t i = 0; i < context.inputs.size(); ++i) {
        if (const Value* input = context.node.input(static_cast<int>(i))) {
            bytes += static_cast<double>(storage_size(input->dtype(), context.inputs[i].numel()));
        }
    }
    bytes += static_cast<double>(storage_size(context.node.output()->dtype(), context.outputs[0].numel()));
    return Cost{2.0 * static_cast<double>(context.outputs[0].numel()) * k, bytes};
}

Status infer_sdpa(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_float_dtype, "a float type"));
    const DType kv = context.input(1)->dtype;
    if (context.input(0)->dtype == DType::kF32 && (kv == DType::kF16 || kv == DType::kBF16)) {
        // A 16-bit KV cache with f32 queries.
        if (context.input(2)->dtype != kv) {
            return errors::invalid_argument("'sdpa' k and v must have one dtype");
        }
    } else {
        II_TRY(expect_same_dtype(context, 1));
        II_TRY(expect_same_dtype(context, 2));
    }
    for (int i = 0; i < 3; ++i) {
        II_TRY(expect_rank(context, i, 4));
    }
    const TensorType& q = *context.input(0);
    const TensorType& k = *context.input(1);
    const TensorType& v = *context.input(2);
    II_TRY(expect_equal_dims(q.shape[0], k.shape[0], "'sdpa' batch of q and k"));
    II_TRY(expect_equal_dims(q.shape[0], v.shape[0], "'sdpa' batch of q and v"));
    II_TRY(expect_equal_dims(k.shape[1], v.shape[1], "'sdpa' heads of k and v"));
    II_TRY(expect_equal_dims(k.shape[2], v.shape[2], "'sdpa' sequence length of k and v"));
    II_TRY(expect_equal_dims(q.shape[3], k.shape[3], "'sdpa' head size of q and k"));
    const Dim& heads_q = q.shape[1];
    const Dim& heads_kv = k.shape[1];
    if (!(heads_q == heads_kv)) {
        if (!heads_q.is_constant() || !heads_kv.is_constant() || heads_kv.value() <= 0 ||
            heads_q.value() % heads_kv.value() != 0) {
            return errors::invalid_argument("'sdpa' query heads {} must be a multiple of key/value heads {}",
                                            heads_q.to_string(), heads_kv.to_string());
        }
    }
    if (context.attrs.get_float("scale") == 0.0 && !q.shape[3].is_constant()) {
        return errors::invalid_argument("'sdpa' needs an explicit scale when the head size is symbolic");
    }
    if (const TensorType* kv_len = context.input(4)) {
        if (!is_index_dtype(kv_len->dtype) || kv_len->rank() != 0) {
            return errors::invalid_argument("'sdpa' kv_len must be an i32 or i64 scalar, got {}", kv_len->to_string());
        }
    }
    const std::int64_t window = context.attrs.get_int("window");
    if (window < 0 || (window > 0 && !context.attrs.get_bool("causal"))) {
        return errors::invalid_argument("'sdpa' window must be >= 0 and requires causal = true, got {}", window);
    }
    if (!(context.attrs.get_float("softcap") >= 0.0)) {
        return errors::invalid_argument("'sdpa' softcap must be >= 0");
    }
    if (const TensorType* mask = context.input(3)) {
        if (mask->dtype != DType::kBool && mask->dtype != q.dtype) {
            return errors::invalid_argument("'sdpa' mask must be bool or {}, got {}", to_string(q.dtype),
                                            to_string(mask->dtype));
        }
        const Dims scores{q.shape[0], q.shape[1], q.shape[2], k.shape[2]};
        II_ASSIGN_OR_RETURN(const Dims broadcast_shape, broadcast(scores, mask->shape));
        if (!(broadcast_shape == scores)) {
            return errors::invalid_argument("'sdpa' mask {} does not broadcast to the scores {}",
                                            to_string(mask->shape), to_string(scores));
        }
    }
    outputs.push_back(TensorType{q.dtype, Dims{q.shape[0], q.shape[1], q.shape[2], v.shape[3]}});
    return ok_status();
}

/// In place: only the update is read and written.
Cost cost_slice_update(const CostContext& context) {
    return Cost{0, 2.0 * static_cast<double>(storage_size(context.node.output()->dtype(), context.inputs[1].numel()))};
}

Cost cost_sdpa(const CostContext& context) {
    const Shape& q = context.inputs[0];
    const Shape& k = context.inputs[1];
    const Shape& v = context.inputs[2];
    const double scores = static_cast<double>(q[0] * q[1] * q[2] * k[2]);
    double bytes = 0;
    for (std::size_t i = 0; i < context.inputs.size(); ++i) {
        if (const Value* input = context.node.input(static_cast<int>(i))) {
            bytes += static_cast<double>(storage_size(input->dtype(), context.inputs[i].numel()));
        }
    }
    bytes += static_cast<double>(storage_size(context.node.output()->dtype(), context.outputs[0].numel()));
    return Cost{2.0 * scores * static_cast<double>(q[3] + v[3]) + 5.0 * scores, bytes};
}

// --- Decompositions ---

std::vector<std::int64_t> trailing_axes(int axis, int rank) {
    std::vector<std::int64_t> axes;
    for (int d = axis; d < rank; ++d) {
        axes.push_back(d);
    }
    return axes;
}

Status decompose_gelu(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* x = node.input(0);
    const DType t = x->dtype();
    Value* inner = nullptr;
    if (node.attrs().get_string("approximate") == "tanh") {
        // tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
        Value* cube = b.op("mul", {b.op("mul", {x, x}), x});
        Value* poly = b.op("add", {x, b.op("mul", {cube, b.scalar(0.044715, t)})});
        inner = b.op("tanh", {b.op("mul", {poly, b.scalar(std::sqrt(2.0 / std::numbers::pi), t)})});
    } else {
        inner = b.op("erf", {b.op("mul", {x, b.scalar(1.0 / std::numbers::sqrt2, t)})});
    }
    Value* half_x = b.op("mul", {x, b.scalar(0.5, t)});
    outputs.push_back(b.op("mul", {half_x, b.op("add", {inner, b.scalar(1.0, t)})}));
    return b.status();
}

Status decompose_silu(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* x = node.input(0);
    outputs.push_back(b.op("mul", {x, b.op("sigmoid", {x})}));
    return b.status();
}

Status decompose_softmax(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* x = node.input(0);
    const std::int64_t axis = node.attrs().get_int("axis");
    const Attributes reduce{{"axes", std::vector<std::int64_t>{axis}}, {"keepdims", true}};
    Value* shifted = b.op("sub", {x, b.op("reduce_max", {x}, reduce)});
    Value* e = b.op("exp", {shifted});
    Value* sum = b.op("reduce_sum", {e}, reduce);
    if (node.op_name() == "log_softmax") {
        outputs.push_back(b.op("sub", {shifted, b.op("log", {sum})}));
    } else {
        outputs.push_back(b.op("div", {e, sum}));
    }
    return b.status();
}

Status decompose_layer_norm(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* x = node.input(0);
    const DType t = x->dtype();
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(node.attrs().get_int("axis"), x->rank()));
    const Attributes reduce{{"axes", trailing_axes(axis, x->rank())}, {"keepdims", true}};
    Value* centered = b.op("sub", {x, b.op("reduce_mean", {x}, reduce)});
    Value* variance = b.op("reduce_mean", {b.op("mul", {centered, centered})}, reduce);
    Value* inv_std = b.op("rsqrt", {b.op("add", {variance, b.scalar(node.attrs().get_float("epsilon"), t)})});
    Value* y = b.op("mul", {centered, inv_std});
    if (Value* scale = node.input(1)) {
        y = b.op("mul", {y, scale});
    }
    if (Value* bias = node.input(2)) {
        y = b.op("add", {y, bias});
    }
    outputs.push_back(y);
    return b.status();
}

Status decompose_rms_norm(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* x = node.input(0);
    const DType t = x->dtype();
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(node.attrs().get_int("axis"), x->rank()));
    const Attributes reduce{{"axes", trailing_axes(axis, x->rank())}, {"keepdims", true}};
    Value* mean_square = b.op("reduce_mean", {b.op("mul", {x, x})}, reduce);
    Value* inv_rms = b.op("rsqrt", {b.op("add", {mean_square, b.scalar(node.attrs().get_float("epsilon"), t)})});
    Value* y = b.op("mul", {x, inv_rms});
    if (Value* scale = node.input(1)) {
        y = b.op("mul", {y, scale});
    }
    outputs.push_back(y);
    return b.status();
}

Status decompose_linear(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* y = b.op("matmul", {node.input(0), node.input(1)}, {{"transpose_b", node.attrs().get_bool("transpose_w")}});
    if (Value* bias = node.input(2)) {
        y = b.op("add", {y, bias});
    }
    outputs.push_back(y);
    return b.status();
}

// --- rope ---

Status infer_rope(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_float_dtype, "a float type"));
    II_TRY(expect_same_dtype(context, 1));
    II_TRY(expect_same_dtype(context, 2));
    const TensorType& x = *context.input(0);
    const TensorType& cos = *context.input(1);
    const TensorType& sin = *context.input(2);
    if (x.rank() == 0 || cos.rank() == 0 || !(cos.shape == sin.shape)) {
        return errors::invalid_argument("'rope' expects x of rank >= 1 and cos, sin of one shape, got {}, {} and {}",
                                        x.to_string(), cos.to_string(), sin.to_string());
    }
    const Dim& size = x.shape.back();
    const Dim& half = cos.shape.back();
    if (!size.is_constant() || !half.is_constant() || half.value() <= 0 || 2 * half.value() > size.value()) {
        return errors::invalid_argument("'rope' rotates 2 * {} of {} features: needs constant sizes with "
                                        "2 * cos.shape[-1] <= x.shape[-1]",
                                        half.to_string(), size.to_string());
    }
    Dims target = x.shape;
    target.back() = half;
    II_ASSIGN_OR_RETURN(const Dims broadcast_shape, broadcast(target, cos.shape));
    if (!(broadcast_shape == target)) {
        return errors::invalid_argument("'rope' cos {} does not broadcast to {}", cos.to_string(), to_string(target));
    }
    outputs.push_back(x);
    return ok_status();
}

Value* slice_last(Builder& b, Value* x, std::int64_t begin, std::int64_t end) {
    return b.op("slice", {x},
                {{"axes", std::vector<std::int64_t>{-1}},
                 {"starts", std::vector<std::int64_t>{begin}},
                 {"ends", std::vector<std::int64_t>{end}}});
}

Status decompose_rope(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* x = node.input(0);
    Value* cos = node.input(1);
    Value* sin = node.input(2);
    const std::int64_t size = x->shape().back().value();
    const std::int64_t half = cos->shape().back().value();
    const std::int64_t axis = x->rank() - 1;
    std::vector<Value*> parts;
    if (node.attrs().get_bool("interleaved")) {
        // Pairs (x[2i], x[2i + 1]) rotate by angle i.
        Dims pairs = x->shape();
        pairs.back() = Dim(half);
        pairs.emplace_back(2);
        Dims column = cos->shape();
        column.emplace_back(1);
        Value* rotated = b.op("reshape", {slice_last(b, x, 0, 2 * half)}, {{"shape", pairs}});
        Value* even = slice_last(b, rotated, 0, 1);
        Value* odd = slice_last(b, rotated, 1, 2);
        Value* c = b.op("reshape", {cos}, {{"shape", column}});
        Value* s = b.op("reshape", {sin}, {{"shape", column}});
        Value* y_even = b.op("sub", {b.op("mul", {even, c}), b.op("mul", {odd, s})});
        Value* y_odd = b.op("add", {b.op("mul", {odd, c}), b.op("mul", {even, s})});
        Dims flat = x->shape();
        flat.back() = Dim(2 * half);
        parts.push_back(b.op("reshape", {b.op("concat", {y_even, y_odd}, {{"axis", axis + 1}})}, {{"shape", flat}}));
    } else {
        // Halves x[:half] and x[half:2 half] rotate (rotate_half).
        Value* x1 = slice_last(b, x, 0, half);
        Value* x2 = slice_last(b, x, half, 2 * half);
        parts.push_back(b.op("sub", {b.op("mul", {x1, cos}), b.op("mul", {x2, sin})}));
        parts.push_back(b.op("add", {b.op("mul", {x2, cos}), b.op("mul", {x1, sin})}));
    }
    if (2 * half < size) {
        parts.push_back(slice_last(b, x, 2 * half, size));
    }
    outputs.push_back(parts.size() == 1 ? parts[0] : b.op("concat", parts, {{"axis", axis}}));
    return b.status();
}

// --- slice_update ---

Status infer_slice_update(const InferContext& context, std::vector<TensorType>& outputs) {
    const TensorType& x = *context.input(0);
    const TensorType& update = *context.input(1);
    II_TRY(expect_input(context, 0, &is_plain, "a non-quantized type"));
    II_TRY(expect_same_dtype(context, 1));
    II_TRY(expect_input(context, 2, &is_index_dtype, "i32 or i64"));
    if (context.input(2)->rank() != 0) {
        return errors::invalid_argument("'slice_update' start must be a scalar");
    }
    if (x.rank() == 0 || x.rank() != update.rank()) {
        return errors::invalid_argument("'slice_update' needs x and update of one rank >= 1, got {} and {}",
                                        x.to_string(), update.to_string());
    }
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(context.attrs.get_int("axis"), x.rank()));
    for (int d = 0; d < x.rank(); ++d) {
        if (d == axis) {
            if (x.shape[d].is_constant() && update.shape[d].is_constant() &&
                update.shape[d].value() > x.shape[d].value()) {
                return errors::invalid_argument("'slice_update' update {} is longer than x {} along axis {}",
                                                update.to_string(), x.to_string(), axis);
            }
            continue;
        }
        II_TRY(expect_equal_dims(x.shape[d], update.shape[d], "'slice_update' dimension of x and update"));
    }
    outputs.push_back(x);
    return ok_status();
}

Status decompose_slice_update(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* update = node.input(1);
    const std::int64_t axis = node.attrs().get_int("axis");
    Value* positions = b.op("iota", {}, {{"shape", update->shape()}, {"axis", axis}, {"dtype", DType::kI64}});
    Value* indices = b.op("add", {positions, b.op("cast", {node.input(2)}, {{"to", DType::kI64}})});
    outputs.push_back(b.op("scatter", {node.input(0), indices, update}, {{"axis", axis}}));
    return b.status();
}

/// Repeats the heads of k or v ([B, Hkv, S, D]) `groups` times: [B, Hkv * groups, S, D].
Value* repeat_heads(Builder& b, Value* x, std::int64_t groups) {
    const Dims& s = x->shape();
    Value* r = b.op("reshape", {x}, {{"shape", Dims{s[0], s[1], 1, s[2], s[3]}}});
    r = b.op("expand", {r}, {{"shape", Dims{s[0], s[1], groups, s[2], s[3]}}});
    return b.op("reshape", {r}, {{"shape", Dims{s[0], s[1] * groups, s[2], s[3]}}});
}

Status decompose_sdpa(Builder& b, const Node& node, std::vector<Value*>& outputs) {
    Value* q = node.input(0);
    Value* k = node.input(1);
    Value* v = node.input(2);
    Value* mask = node.input(3);
    const DType t = q->dtype();
    const Dims& qs = q->shape();
    const Dims& ks = k->shape();
    if (!(qs[1] == ks[1])) {
        const std::int64_t groups = qs[1].value() / ks[1].value();
        k = repeat_heads(b, k, groups);
        v = repeat_heads(b, v, groups);
    }
    double scale = node.attrs().get_float("scale");
    if (scale == 0.0) {
        scale = 1.0 / std::sqrt(static_cast<double>(qs[3].value()));
    }
    Value* scores = b.op("mul", {b.op("matmul", {q, k}, {{"transpose_b", true}}), b.scalar(scale, t)});
    if (const double softcap = node.attrs().get_float("softcap"); softcap > 0.0) {
        scores = b.op("mul", {b.op("tanh", {b.op("mul", {scores, b.scalar(1.0 / softcap, t)})}), b.scalar(softcap, t)});
    }
    const double negative_infinity = -std::numeric_limits<double>::infinity();
    if (mask != nullptr) {
        scores = mask->dtype() == DType::kBool ? b.op("where", {mask, scores, b.scalar(negative_infinity, t)})
                                               : b.op("add", {scores, mask});
    }
    const bool causal = node.attrs().get_bool("causal");
    const std::int64_t window = node.attrs().get_int("window");
    Value* kv_len = node.inputs().size() > 4 ? node.input(4) : nullptr;
    if (causal || kv_len != nullptr) {
        // With L keys in use (kv_len, or Skv), query i is at position p_i = i + L - Sq. Key j is attended iff
        // j < L, and with causal j <= p_i (and j > p_i - window). In terms of reversed indices that need no
        // concrete Sq: p_i - j = (L - 1 - j) - (Sq - 1 - i).
        const Dims grid{qs[2], ks[2]};
        Value* cols = b.op("iota", {}, {{"shape", grid}, {"axis", 1}});
        Value* last_minus_j =
            kv_len != nullptr
                ? b.op("sub",
                       {b.op("sub", {b.op("cast", {kv_len}, {{"to", DType::kI64}}), b.scalar(1, DType::kI64)}), cols})
                : b.op("flip", {cols}, {{"axes", std::vector<std::int64_t>{1}}});
        Value* keep = b.op("ge", {last_minus_j, b.scalar(0, DType::kI64)});
        if (causal) {
            Value* rows = b.op("flip", {b.op("iota", {}, {{"shape", grid}, {"axis", 0}})},
                               {{"axes", std::vector<std::int64_t>{0}}});
            Value* distance = b.op("sub", {last_minus_j, rows});
            keep = b.op("ge", {distance, b.scalar(0, DType::kI64)});
            if (window > 0) {
                keep = b.op("and", {keep, b.op("lt", {distance, b.scalar(static_cast<double>(window), DType::kI64)})});
            }
        }
        scores = b.op("where", {keep, scores, b.scalar(negative_infinity, t)});
    }
    Value* probabilities = b.op("softmax", {scores}, {{"axis", -1}});
    outputs.push_back(b.op("matmul", {probabilities, v}));
    return b.status();
}

}  // namespace

void register_composite_ops(ir::OpRegistry& registry) {
    registry.add(OpSchema{
        .name = "gelu",
        .category = "composite",
        .doc = "Gaussian error linear unit: 0.5 * x * (1 + erf(x / sqrt(2))), or with approximate = tanh "
               "0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"approximate", AttrKind::kString, std::string("none")}},
        .infer = infer_float_unary,
        .decompose = decompose_gelu,
        .flags = ir::kOpElementwise,
    });
    registry.add(OpSchema{
        .name = "silu",
        .category = "composite",
        .doc = "x * sigmoid(x) (also called swish).",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .infer = infer_float_unary,
        .decompose = decompose_silu,
        .flags = ir::kOpElementwise,
    });
    for (const std::string_view name : {std::string_view("softmax"), std::string_view("log_softmax")}) {
        registry.add(OpSchema{
            .name = name,
            .category = "composite",
            .doc = name == "softmax" ? "exp(x - max(x)) / sum(exp(x - max(x))) along `axis`."
                                     : "(x - max(x)) - log(sum(exp(x - max(x)))) along `axis`.",
            .inputs = {InputSpec{"x"}},
            .outputs = {OutputSpec{"y"}},
            .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{-1}}},
            .infer = infer_softmax,
            .decompose = decompose_softmax,
        });
    }
    registry.add(OpSchema{
        .name = "layer_norm",
        .category = "composite",
        .doc = "(x - mean) / sqrt(var + epsilon) * scale + bias, with mean and (biased) variance over the dimensions "
               "from `axis` to the end. scale and bias are optional and shaped like those dimensions.",
        .inputs = {InputSpec{"x"}, InputSpec{"scale", true}, InputSpec{"bias", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{-1}}, AttrSpec{"epsilon", AttrKind::kFloat, 1e-5}},
        .infer = infer_norm,
        .decompose = decompose_layer_norm,
    });
    registry.add(OpSchema{
        .name = "rms_norm",
        .category = "composite",
        .doc = "x / sqrt(mean(x^2) + epsilon) * scale, with the mean over the dimensions from `axis` to the end.",
        .inputs = {InputSpec{"x"}, InputSpec{"scale", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{-1}}, AttrSpec{"epsilon", AttrKind::kFloat, 1e-6}},
        .infer = infer_norm,
        .decompose = decompose_rms_norm,
    });
    registry.add(OpSchema{
        .name = "linear",
        .category = "composite",
        .doc = "x @ w^T + bias for w [out, in] (transpose_w = true, the PyTorch layout) or x @ w + bias for "
               "w [in, out]. w may be f16 / bf16 with an f32 x, or block-quantized (with transpose_w = true); the "
               "result has x's dtype.",
        .inputs = {InputSpec{"x"}, InputSpec{"w"}, InputSpec{"bias", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"transpose_w", AttrKind::kBool, true}},
        .infer = infer_linear,
        .decompose = decompose_linear,
        .cost = cost_linear,
    });
    registry.add(OpSchema{
        .name = "sdpa",
        .category = "composite",
        .doc = "Scaled dot-product attention softmax(q @ k^T * scale + mask) @ v for q [B, Hq, Sq, D], "
               "k [B, Hkv, Skv, D], v [B, Hkv, Skv, Dv]; Hq must be a multiple of Hkv (grouped-query attention). "
               "With an f32 q, k and v may be f16 / bf16 (a 16-bit KV cache); the result has q's dtype. "
               "scale = 0 means 1 / sqrt(D). mask: bool (true = attend) or additive, broadcastable to "
               "[B, Hq, Sq, Skv]. kv_len (i32 or i64 scalar, optional): only the first kv_len keys and values are "
               "used (a preallocated KV cache); L = kv_len, or Skv without it. causal: query i (at position "
               "i + L - Sq) attends key j iff j <= i + L - Sq (keys of earlier steps are at the front, as with a KV "
               "cache); window > 0 (requires causal) also drops keys j <= i + L - Sq - window (sliding-window "
               "attention). softcap > 0 squashes the scaled scores to softcap * tanh(score / softcap) before masking "
               "(gemma2). Fully masked rows produce NaN.",
        .inputs = {InputSpec{"q"}, InputSpec{"k"}, InputSpec{"v"}, InputSpec{"mask", true}, InputSpec{"kv_len", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"causal", AttrKind::kBool, false}, AttrSpec{"scale", AttrKind::kFloat, 0.0},
                  AttrSpec{"window", AttrKind::kInt, std::int64_t{0}}, AttrSpec{"softcap", AttrKind::kFloat, 0.0}},
        .infer = infer_sdpa,
        .decompose = decompose_sdpa,
        .cost = cost_sdpa,
    });
}

void register_llm_ops(ir::OpRegistry& registry) {
    registry.add(OpSchema{
        .name = "rope",
        .category = "composite",
        .doc = "Rotary position embedding: rotates pairs of the first 2 * R features of x (R = cos.shape[-1]) by the "
               "angles whose cosines and sines are given; the remaining features pass through. Pairs are "
               "(x[i], x[i + R]) (rotate_half, GPT-NeoX / Hugging Face) or with interleaved = true "
               "(x[2i], x[2i + 1]) (GPT-J, ggml): y_a = x_a * cos - x_b * sin, y_b = x_b * cos + x_a * sin. cos and "
               "sin broadcast to x.shape[:-1] + [R], e.g. [S, 1, R] for x [B, S, H, D].",
        .inputs = {InputSpec{"x"}, InputSpec{"cos"}, InputSpec{"sin"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"interleaved", AttrKind::kBool, false}},
        .infer = infer_rope,
        .decompose = decompose_rope,
    });
    registry.add(OpSchema{
        .name = "slice_update",
        .category = "composite",
        .doc = "Copy of x with update written at offset `start` (a run-time i32 / i64 scalar) along `axis`: "
               "y[..., start:start + U, ...] = update. Every other dimension of update equals x's. An out-of-range "
               "start is a run-time error. Backends run it in place, so a state updated this way (a KV cache) is "
               "never copied.",
        .inputs = {InputSpec{"x"}, InputSpec{"update"}, InputSpec{"start"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::nullopt}},
        .infer = infer_slice_update,
        .decompose = decompose_slice_update,
        .cost = cost_slice_update,
    });
}

void register_opset(ir::OpRegistry& registry) {
    register_elementwise_ops(registry);
    register_reduction_ops(registry);
    register_linalg_ops(registry);
    register_conv_ops(registry);
    register_image_ops(registry);
    register_movement_ops(registry);
    register_creation_ops(registry);
    register_quantization_ops(registry);
    register_composite_ops(registry);
    register_llm_ops(registry);
    register_internal_ops(registry);
}

}  // namespace ii::ops
