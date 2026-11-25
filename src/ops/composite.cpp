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
    if (!is_block_quantized(w.dtype)) {
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
    II_TRY(expect_same_dtype(context, 1));
    II_TRY(expect_same_dtype(context, 2));
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
    const double negative_infinity = -std::numeric_limits<double>::infinity();
    if (mask != nullptr) {
        scores = mask->dtype() == DType::kBool ? b.op("where", {mask, scores, b.scalar(negative_infinity, t)})
                                               : b.op("add", {scores, mask});
    }
    if (node.attrs().get_bool("causal")) {
        // Query i may attend key j iff j <= i + (Skv - Sq)  <=>  (Skv - 1 - j) >= (Sq - 1 - i).
        const Dims grid{qs[2], ks[2]};
        Value* rows =
            b.op("flip", {b.op("iota", {}, {{"shape", grid}, {"axis", 0}})}, {{"axes", std::vector<std::int64_t>{0}}});
        Value* cols =
            b.op("flip", {b.op("iota", {}, {{"shape", grid}, {"axis", 1}})}, {{"axes", std::vector<std::int64_t>{1}}});
        scores = b.op("where", {b.op("ge", {cols, rows}), scores, b.scalar(negative_infinity, t)});
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
               "w [in, out]. w may be block-quantized (with transpose_w = true).",
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
               "scale = 0 means 1 / sqrt(D). mask: bool (true = attend) or additive, broadcastable to "
               "[B, Hq, Sq, Skv]. causal: query i attends key j iff j <= i + Skv - Sq (keys of earlier steps are "
               "at the front, as with a KV cache). Fully masked rows produce NaN.",
        .inputs = {InputSpec{"q"}, InputSpec{"k"}, InputSpec{"v"}, InputSpec{"mask", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"causal", AttrKind::kBool, false}, AttrSpec{"scale", AttrKind::kFloat, 0.0}},
        .infer = infer_sdpa,
        .decompose = decompose_sdpa,
        .cost = cost_sdpa,
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
    register_internal_ops(registry);
}

}  // namespace ii::ops
