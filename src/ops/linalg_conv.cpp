// SPDX-License-Identifier: MIT
// Linear algebra and sliding-window operations: matmul, conv, conv_transpose, max_pool, avg_pool.
#include "ops/infer_util.h"
#include "ops/opset.h"

namespace ii::ops {

namespace {

using ir::AttrSpec;
using ir::Cost;
using ir::CostContext;
using ir::InferContext;
using ir::InputSpec;
using ir::OpSchema;
using ir::OutputSpec;
using ir::TensorType;

bool is_numeric_dtype(DType dtype) noexcept {
    return is_numeric(dtype);
}
bool is_float_dtype(DType dtype) noexcept {
    return is_float(dtype);
}

double bytes_of(const ir::Value* value, const Shape& shape) {
    return value == nullptr ? 0.0 : static_cast<double>(storage_size(value->dtype(), shape.numel()));
}

double io_bytes(const CostContext& context) {
    double bytes = 0;
    for (std::size_t i = 0; i < context.inputs.size(); ++i) {
        bytes += bytes_of(context.node.input(static_cast<int>(i)), context.inputs[i]);
    }
    for (std::size_t i = 0; i < context.outputs.size(); ++i) {
        bytes += bytes_of(context.node.output(static_cast<int>(i)), context.outputs[i]);
    }
    return bytes;
}

// --- matmul ---

Status infer_matmul(const InferContext& context, std::vector<TensorType>& outputs) {
    const TensorType* a = context.input(0);
    const TensorType* b = context.input(1);
    II_TRY(expect_input(context, 0, &is_numeric_dtype, "a numeric type"));
    const bool transpose_a = context.attrs.get_bool("transpose_a");
    const bool transpose_b = context.attrs.get_bool("transpose_b");
    if (is_block_quantized(b->dtype)) {
        if (!transpose_b || b->rank() != 2 || !is_float(a->dtype)) {
            return errors::invalid_argument(
                "a block-quantized 'matmul' operand b must be [N, K] with transpose_b = true and a float a");
        }
    } else {
        II_TRY(expect_input(context, 1, &is_numeric_dtype, "a numeric type"));
        II_TRY(expect_same_dtype(context, 1));
    }
    if (a->rank() == 0 || b->rank() == 0) {
        return errors::invalid_argument("'matmul' operands must have rank >= 1");
    }
    Dims sa = a->shape;
    Dims sb = b->shape;
    const bool a_vector = sa.size() == 1;
    const bool b_vector = sb.size() == 1;
    if (a_vector) {
        sa.insert(sa.begin(), Dim(1));
    } else if (transpose_a) {
        std::swap(sa[sa.size() - 1], sa[sa.size() - 2]);
    }
    if (b_vector) {
        sb.push_back(Dim(1));
    } else if (transpose_b) {
        std::swap(sb[sb.size() - 1], sb[sb.size() - 2]);
    }
    const Dim& m = sa[sa.size() - 2];
    const Dim& k = sa[sa.size() - 1];
    const Dim& k2 = sb[sb.size() - 2];
    const Dim& n = sb[sb.size() - 1];
    if (!(k == k2)) {
        return errors::invalid_argument("'matmul' inner dimensions differ: {} vs {} (shapes {} and {})", k.to_string(),
                                        k2.to_string(), to_string(a->shape), to_string(b->shape));
    }
    const Dims batch_a(sa.begin(), sa.end() - 2);
    const Dims batch_b(sb.begin(), sb.end() - 2);
    II_ASSIGN_OR_RETURN(Dims shape, broadcast(batch_a, batch_b));
    if (!a_vector) {
        shape.push_back(m);
    }
    if (!b_vector) {
        shape.push_back(n);
    }
    outputs.push_back(TensorType{a->dtype, std::move(shape)});
    return ok_status();
}

Cost cost_matmul(const CostContext& context) {
    const Shape& a = context.inputs[0];
    const bool transpose_a = context.node.attrs().get_bool("transpose_a");
    const std::int64_t k = a.rank() == 1 ? a[0] : (transpose_a ? a[a.rank() - 2] : a[a.rank() - 1]);
    return Cost{2.0 * static_cast<double>(context.outputs[0].numel()) * static_cast<double>(k), io_bytes(context)};
}

// --- conv ---

Status infer_conv(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_float_dtype, "a float type"));
    II_TRY(expect_input(context, 1, &is_float_dtype, "a float type"));
    II_TRY(expect_same_dtype(context, 1));
    II_TRY(expect_same_dtype(context, 2));
    const TensorType& x = *context.input(0);
    const TensorType& w = *context.input(1);
    if (x.rank() < 3 || w.rank() != x.rank()) {
        return errors::invalid_argument("'conv' expects x [N, C, spatial...] and w [M, C/groups, kernel...] of the "
                                        "same rank >= 3, got {} and {}",
                                        x.to_string(), w.to_string());
    }
    const int spatial = x.rank() - 2;
    const std::int64_t groups = context.attrs.get_int("groups");
    if (groups <= 0) {
        return errors::invalid_argument("'conv' groups must be positive");
    }
    II_ASSIGN_OR_RETURN(const WindowAttrs window, parse_window_attrs(context.attrs, spatial));
    const Dim& channels = x.shape[1];
    const Dim& out_channels = w.shape[0];
    if (!(w.shape[1] * groups == channels)) {
        return errors::invalid_argument("'conv' input channels {} != weight channels {} * groups {}",
                                        channels.to_string(), w.shape[1].to_string(), groups);
    }
    if (!out_channels.is_constant() || out_channels.value() % groups != 0) {
        return errors::invalid_argument("'conv' output channels {} must be a constant multiple of groups {}",
                                        out_channels.to_string(), groups);
    }
    if (const TensorType* bias = context.input(2)) {
        if (bias->rank() != 1 || !(bias->shape[0] == out_channels)) {
            return errors::invalid_argument("'conv' bias must be [{}], got {}", out_channels.to_string(),
                                            bias->to_string());
        }
    }
    Dims shape{x.shape[0], out_channels};
    const auto s = static_cast<std::size_t>(spatial);
    for (std::size_t i = 0; i < s; ++i) {
        const Dim& kernel = w.shape[2 + i];
        if (!kernel.is_constant() || kernel.value() <= 0 || kernel.value() > kMaxAttributeSize) {
            return errors::invalid_argument("'conv' kernel sizes must be positive constants, got {}",
                                            to_string(w.shape));
        }
        II_ASSIGN_OR_RETURN(Dim size,
                            window_output_size(x.shape[2 + i], kernel.value(), window.strides[i], window.dilations[i],
                                               window.pads[i], window.pads[s + i], window.auto_pad, false));
        shape.push_back(std::move(size));
    }
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Cost cost_conv(const CostContext& context) {
    // 2 * output elements * (input channels / groups) * kernel volume.
    const Shape& w = context.inputs[1];
    double per_output = 1;
    for (int d = 1; d < w.rank(); ++d) {
        per_output *= static_cast<double>(w[d]);
    }
    return Cost{2.0 * static_cast<double>(context.outputs[0].numel()) * per_output, io_bytes(context)};
}

Status infer_conv_transpose(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_float_dtype, "a float type"));
    II_TRY(expect_input(context, 1, &is_float_dtype, "a float type"));
    II_TRY(expect_same_dtype(context, 1));
    II_TRY(expect_same_dtype(context, 2));
    const TensorType& x = *context.input(0);
    const TensorType& w = *context.input(1);
    if (x.rank() < 3 || w.rank() != x.rank()) {
        return errors::invalid_argument("'conv_transpose' expects x [N, C, spatial...] and w [C, M/groups, kernel...] "
                                        "of the same rank >= 3, got {} and {}",
                                        x.to_string(), w.to_string());
    }
    const int spatial = x.rank() - 2;
    const auto s = static_cast<std::size_t>(spatial);
    const std::int64_t groups = context.attrs.get_int("groups");
    if (groups <= 0) {
        return errors::invalid_argument("'conv_transpose' groups must be positive");
    }
    II_ASSIGN_OR_RETURN(const WindowAttrs window, parse_window_attrs(context.attrs, spatial));
    if (window.auto_pad != "none") {
        return errors::unsupported("'conv_transpose' supports only explicit pads");
    }
    std::vector<std::int64_t> output_padding = context.attrs.get_ints("output_padding");
    if (output_padding.empty()) {
        output_padding.assign(s, 0);
    }
    if (output_padding.size() != s) {
        return errors::invalid_argument("'conv_transpose' output_padding needs {} values", s);
    }
    if (!(x.shape[1] == w.shape[0])) {
        return errors::invalid_argument("'conv_transpose' input channels {} != weight dimension 0 {}",
                                        x.shape[1].to_string(), w.shape[0].to_string());
    }
    if (!w.shape[1].is_constant()) {
        return errors::invalid_argument("'conv_transpose' weight dimension 1 must be constant");
    }
    const Dim out_channels = w.shape[1] * groups;
    if (!x.shape[1].is_constant() || x.shape[1].value() % groups != 0) {
        return errors::invalid_argument("'conv_transpose' input channels must be a constant multiple of groups");
    }
    if (const TensorType* bias = context.input(2)) {
        if (bias->rank() != 1 || !(bias->shape[0] == out_channels)) {
            return errors::invalid_argument("'conv_transpose' bias must be [{}], got {}", out_channels.to_string(),
                                            bias->to_string());
        }
    }
    Dims shape{x.shape[0], out_channels};
    for (std::size_t i = 0; i < s; ++i) {
        const Dim& kernel = w.shape[2 + i];
        if (!kernel.is_constant() || kernel.value() <= 0 || kernel.value() > kMaxAttributeSize) {
            return errors::invalid_argument("'conv_transpose' kernel sizes must be positive constants");
        }
        if (output_padding[i] < 0 || output_padding[i] >= std::max(window.strides[i], window.dilations[i])) {
            return errors::invalid_argument("'conv_transpose' output_padding must be < max(stride, dilation)");
        }
        const Dim size =
            (x.shape[2 + i] - 1) * window.strides[i] + Dim(window.dilations[i] * (kernel.value() - 1) + 1 +
                                                           output_padding[i] - window.pads[i] - window.pads[s + i]);
        if (size.is_constant() && size.value() <= 0) {
            return errors::invalid_argument("'conv_transpose' output size would be {}", size.value());
        }
        shape.push_back(size);
    }
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Cost cost_conv_transpose(const CostContext& context) {
    // 2 * input elements * (output channels / groups) * kernel volume.
    const Shape& w = context.inputs[1];
    double per_input = 1;
    for (int d = 1; d < w.rank(); ++d) {
        per_input *= static_cast<double>(w[d]);
    }
    return Cost{2.0 * static_cast<double>(context.inputs[0].numel()) * per_input, io_bytes(context)};
}

// --- pooling ---

Status infer_pool(const InferContext& context, std::vector<TensorType>& outputs) {
    const bool is_max = context.op.name == "max_pool";
    II_TRY(expect_input(context, 0, is_max ? &is_numeric_dtype : &is_float_dtype,
                        is_max ? "a numeric type" : "a float type"));
    const TensorType& x = *context.input(0);
    if (x.rank() < 3) {
        return errors::invalid_argument("'{}' expects x [N, C, spatial...], got {}", context.op.name, x.to_string());
    }
    const int spatial = x.rank() - 2;
    const auto s = static_cast<std::size_t>(spatial);
    const std::vector<std::int64_t> kernel = context.attrs.get_ints("kernel");
    if (kernel.size() != s) {
        return errors::invalid_argument("'{}' kernel needs {} values", context.op.name, s);
    }
    II_ASSIGN_OR_RETURN(const WindowAttrs window, parse_window_attrs(context.attrs, spatial));
    Dims shape{x.shape[0], x.shape[1]};
    for (std::size_t i = 0; i < s; ++i) {
        II_TRY(expect_in_range(kernel[i], 1, kMaxAttributeSize, "kernel sizes"));
        const bool too_much_padding =
            window.pads[i] >= kernel[i] * window.dilations[i] || window.pads[s + i] >= kernel[i] * window.dilations[i];
        if (too_much_padding) {
            return errors::invalid_argument("'{}' padding must be smaller than the (dilated) kernel", context.op.name);
        }
        II_ASSIGN_OR_RETURN(Dim size,
                            window_output_size(x.shape[2 + i], kernel[i], window.strides[i], window.dilations[i],
                                               window.pads[i], window.pads[s + i], window.auto_pad, window.ceil_mode));
        shape.push_back(std::move(size));
    }
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Cost cost_pool(const CostContext& context) {
    double window = 1;
    for (const std::int64_t k : context.node.attrs().get_ints("kernel")) {
        window *= static_cast<double>(k);
    }
    return Cost{static_cast<double>(context.outputs[0].numel()) * window, io_bytes(context)};
}

std::vector<AttrSpec> window_attrs() {
    return {
        AttrSpec{"strides", AttrKind::kInts, std::vector<std::int64_t>{}},
        AttrSpec{"dilations", AttrKind::kInts, std::vector<std::int64_t>{}},
        AttrSpec{"pads", AttrKind::kInts, std::vector<std::int64_t>{}},
        AttrSpec{"auto_pad", AttrKind::kString, std::string("none")},
    };
}

}  // namespace

void register_linalg_ops(ir::OpRegistry& registry) {
    registry.add(OpSchema{
        .name = "matmul",
        .category = "linalg",
        .doc = "Matrix product with numpy semantics: batch dimensions broadcast, 1-D operands are promoted to "
               "matrices and the promoted dimension is removed. transpose_a / transpose_b swap the last two "
               "dimensions of an operand of rank >= 2. b may be block-quantized ([N, K], transpose_b = true).",
        .inputs = {InputSpec{"a"}, InputSpec{"b"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"transpose_a", AttrKind::kBool, false}, AttrSpec{"transpose_b", AttrKind::kBool, false}},
        .infer = infer_matmul,
        .cost = cost_matmul,
    });
}

void register_conv_ops(ir::OpRegistry& registry) {
    std::vector<AttrSpec> conv_attrs = window_attrs();
    conv_attrs.push_back(AttrSpec{"groups", AttrKind::kInt, std::int64_t{1}});
    registry.add(OpSchema{
        .name = "conv",
        .category = "conv",
        .doc = "N-d convolution (cross-correlation). x [N, C, spatial...], w [M, C/groups, kernel...], optional "
               "bias [M]. pads: all begins, then all ends. auto_pad: none | valid | same_upper | same_lower.",
        .inputs = {InputSpec{"x"}, InputSpec{"w"}, InputSpec{"bias", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = conv_attrs,
        .infer = infer_conv,
        .cost = cost_conv,
    });

    std::vector<AttrSpec> transpose_attrs = window_attrs();
    transpose_attrs.push_back(AttrSpec{"groups", AttrKind::kInt, std::int64_t{1}});
    transpose_attrs.push_back(AttrSpec{"output_padding", AttrKind::kInts, std::vector<std::int64_t>{}});
    registry.add(OpSchema{
        .name = "conv_transpose",
        .category = "conv",
        .doc = "N-d transposed convolution. x [N, C, spatial...], w [C, M/groups, kernel...], optional bias [M]. "
               "Output size: (in - 1) * stride + dilation * (kernel - 1) + 1 + output_padding - pad_begin - pad_end.",
        .inputs = {InputSpec{"x"}, InputSpec{"w"}, InputSpec{"bias", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = transpose_attrs,
        .infer = infer_conv_transpose,
        .cost = cost_conv_transpose,
    });

    std::vector<AttrSpec> pool_attrs = window_attrs();
    pool_attrs.push_back(AttrSpec{"kernel", AttrKind::kInts, std::nullopt});
    pool_attrs.push_back(AttrSpec{"ceil_mode", AttrKind::kBool, false});
    registry.add(OpSchema{
        .name = "max_pool",
        .category = "conv",
        .doc = "N-d max pooling over x [N, C, spatial...]. Padding never contributes values.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = pool_attrs,
        .infer = infer_pool,
        .cost = cost_pool,
    });
    std::vector<AttrSpec> avg_attrs = window_attrs();
    avg_attrs.push_back(AttrSpec{"kernel", AttrKind::kInts, std::nullopt});
    avg_attrs.push_back(AttrSpec{"ceil_mode", AttrKind::kBool, false});
    avg_attrs.push_back(AttrSpec{"count_include_pad", AttrKind::kBool, false});
    registry.add(OpSchema{
        .name = "avg_pool",
        .category = "conv",
        .doc = "N-d average pooling over x [N, C, spatial...]. With count_include_pad the divisor counts explicit "
               "padding (but never the extra positions added by ceil_mode).",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = avg_attrs,
        .infer = infer_pool,
        .cost = cost_pool,
    });
}

}  // namespace ii::ops
