// SPDX-License-Identifier: MIT
// Data movement: reshape, permute, slice, expand, concat, split, pad, gather, scatter, tile, flip, cumsum.
#include "ops/infer_util.h"
#include "ops/opset.h"

namespace ii::ops {

namespace {

using ir::AttrSpec;
using ir::InferContext;
using ir::InputSpec;
using ir::OpSchema;
using ir::OutputSpec;
using ir::TensorType;

/// Selections (slice, gather) read only the elements they produce (plus the indices), not the whole input.
ir::Cost cost_selection(const ir::CostContext& context) {
    const double out = static_cast<double>(storage_size(context.node.output()->dtype(), context.outputs[0].numel()));
    double indices = 0;
    if (const ir::Value* index = context.node.input(1); index != nullptr && context.inputs.size() > 1) {
        indices = static_cast<double>(storage_size(index->dtype(), context.inputs[1].numel()));
    }
    return ir::Cost{0, 2 * out + indices};
}

bool is_plain_dtype(DType dtype) noexcept {
    return is_plain(dtype);
}
bool is_numeric_dtype(DType dtype) noexcept {
    return is_numeric(dtype);
}
bool is_index_dtype(DType dtype) noexcept {
    return dtype == DType::kI32 || dtype == DType::kI64;
}

Status infer_reshape(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const TensorType& x = *context.input(0);
    Dims shape = context.attrs.get_dims("shape");
    int inferred = -1;
    Dim known = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == Dim(-1)) {
            if (inferred >= 0) {
                return errors::invalid_argument("'reshape': more than one dimension is -1");
            }
            inferred = static_cast<int>(i);
            continue;
        }
        if (shape[i].is_constant() && shape[i].value() < 0) {
            return errors::invalid_argument("'reshape': invalid dimension {}", shape[i].value());
        }
        known = known * shape[i];
    }
    const Dim numel = product(x.shape);
    if (inferred >= 0) {
        std::optional<Dim> rest = numel.exact_div(known);
        if (!rest) {
            return errors::invalid_argument("'reshape' cannot infer the -1 dimension of {} from {} elements",
                                            to_string(shape), numel.to_string());
        }
        shape[static_cast<std::size_t>(inferred)] = *rest;
    } else if (!(known == numel)) {
        return errors::invalid_argument("'reshape' cannot reshape {} into {}", to_string(x.shape), to_string(shape));
    }
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Status infer_permute(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const TensorType& x = *context.input(0);
    const std::vector<std::int64_t> perm = context.attrs.get_ints("perm");
    II_TRY(check_permutation(perm, x.rank()));
    Dims shape;
    for (const std::int64_t axis : perm) {
        shape.push_back(x.shape[static_cast<std::size_t>(axis)]);
    }
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Status infer_slice(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const TensorType& x = *context.input(0);
    const std::vector<std::int64_t> axes = context.attrs.get_ints("axes");
    const std::vector<std::int64_t> starts = context.attrs.get_ints("starts");
    const std::vector<std::int64_t> ends = context.attrs.get_ints("ends");
    std::vector<std::int64_t> steps = context.attrs.get_ints("steps");
    if (steps.empty()) {
        steps.assign(axes.size(), 1);
    }
    if (starts.size() != axes.size() || ends.size() != axes.size() || steps.size() != axes.size()) {
        return errors::invalid_argument("'slice' axes, starts, ends and steps must have the same length");
    }
    II_ASSIGN_OR_RETURN(const std::vector<int> normalized, normalize_axes(axes, x.rank()));
    Dims shape = x.shape;
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        const auto axis = static_cast<std::size_t>(normalized[i]);
        II_ASSIGN_OR_RETURN(shape[axis], slice_length(x.shape[axis], starts[i], ends[i], steps[i]));
    }
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Status infer_expand(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    II_ASSIGN_OR_RETURN(Dims shape, broadcast(context.input(0)->shape, context.attrs.get_dims("shape")));
    outputs.push_back(TensorType{context.input(0)->dtype, std::move(shape)});
    return ok_status();
}

Status infer_concat(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const TensorType& first = *context.input(0);
    if (first.rank() == 0) {
        return errors::invalid_argument("'concat' inputs must have rank >= 1");
    }
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(context.attrs.get_int("axis"), first.rank()));
    Dims shape = first.shape;
    for (int i = 1; i < context.num_inputs(); ++i) {
        II_TRY(expect_same_dtype(context, i));
        const TensorType& input = *context.input(i);
        if (input.rank() != first.rank()) {
            return errors::invalid_argument("'concat' inputs must have the same rank, got {} and {}", first.to_string(),
                                            input.to_string());
        }
        for (int d = 0; d < first.rank(); ++d) {
            const auto du = static_cast<std::size_t>(d);
            if (d == axis) {
                shape[du] = shape[du] + input.shape[du];
            } else if (!(input.shape[du] == first.shape[du])) {
                return errors::invalid_argument("'concat' inputs {} and {} differ outside of axis {}",
                                                first.to_string(), input.to_string(), axis);
            }
        }
    }
    outputs.push_back(TensorType{first.dtype, std::move(shape)});
    return ok_status();
}

Status infer_split(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const TensorType& x = *context.input(0);
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(context.attrs.get_int("axis"), x.rank()));
    const std::vector<std::int64_t> sizes = context.attrs.get_ints("sizes");
    if (sizes.empty()) {
        return errors::invalid_argument("'split' requires at least one size");
    }
    std::int64_t total = 0;
    for (const std::int64_t size : sizes) {
        II_TRY(expect_in_range(size, 0, kMaxDimension, "'split' sizes"));
        total += size;
    }
    const Dim& dim = x.shape[static_cast<std::size_t>(axis)];
    if (!(dim == Dim(total))) {
        return errors::invalid_argument("'split' sizes sum to {}, but dimension {} is {}", total, axis,
                                        dim.to_string());
    }
    for (const std::int64_t size : sizes) {
        TensorType type = x;
        type.shape[static_cast<std::size_t>(axis)] = size;
        outputs.push_back(std::move(type));
    }
    return ok_status();
}

Status infer_pad(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const TensorType& x = *context.input(0);
    const std::vector<std::int64_t> pads = context.attrs.get_ints("pads");
    for (const std::int64_t pad : pads) {
        II_TRY(expect_in_range(pad, -kMaxDimension, kMaxDimension, "'pad' pads"));
    }
    const auto rank = static_cast<std::size_t>(x.rank());
    if (pads.size() != 2 * rank) {
        return errors::invalid_argument("'pad' needs {} pads (all begins, then all ends), got {}", 2 * rank,
                                        pads.size());
    }
    const std::string_view mode = context.attrs.get_string("mode");
    if (mode != "constant" && mode != "reflect" && mode != "edge") {
        return errors::invalid_argument("'pad' mode must be constant, reflect or edge; got '{}'", mode);
    }
    Dims shape = x.shape;
    for (std::size_t d = 0; d < rank; ++d) {
        shape[d] = shape[d] + Dim(pads[d] + pads[rank + d]);
        if (shape[d].is_constant() && shape[d].value() < 0) {
            return errors::invalid_argument("'pad' crops dimension {} below zero", d);
        }
        const bool pads_here = pads[d] != 0 || pads[rank + d] != 0;
        if (mode == "reflect" && pads_here && x.shape[d].is_constant() &&
            (pads[d] >= x.shape[d].value() || pads[rank + d] >= x.shape[d].value())) {
            return errors::invalid_argument("'pad' reflect padding must be smaller than the dimension");
        }
        if (mode == "edge" && pads_here && x.shape[d] == Dim(0)) {
            return errors::invalid_argument("'pad' edge padding of an empty dimension");
        }
    }
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Status infer_gather(const InferContext& context, std::vector<TensorType>& outputs) {
    const TensorType& data = *context.input(0);
    if (!is_block_quantized(data.dtype)) {
        II_TRY(expect_input(context, 0, &is_plain_dtype, "a valid type"));
    }
    II_TRY(expect_input(context, 1, &is_index_dtype, "i32 or i64"));
    const TensorType& indices = *context.input(1);
    if (data.rank() == 0) {
        return errors::invalid_argument("'gather' data must have rank >= 1");
    }
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(context.attrs.get_int("axis"), data.rank()));
    if (is_block_quantized(data.dtype) && axis == data.rank() - 1) {
        return errors::invalid_argument("'gather' of block-quantized data must not index the innermost dimension");
    }
    Dims shape(data.shape.begin(), data.shape.begin() + axis);
    shape.insert(shape.end(), indices.shape.begin(), indices.shape.end());
    shape.insert(shape.end(), data.shape.begin() + axis + 1, data.shape.end());
    outputs.push_back(TensorType{data.dtype, std::move(shape)});
    return ok_status();
}

Status infer_scatter(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    II_TRY(expect_input(context, 1, &is_index_dtype, "i32 or i64"));
    II_TRY(expect_same_dtype(context, 2));
    const TensorType& data = *context.input(0);
    const TensorType& indices = *context.input(1);
    const TensorType& updates = *context.input(2);
    if (data.rank() == 0 || indices.rank() != data.rank() || !(indices.shape == updates.shape)) {
        return errors::invalid_argument(
            "'scatter' requires data of rank >= 1, indices of the same rank, and updates shaped like indices");
    }
    II_TRY(normalize_axis(context.attrs.get_int("axis"), data.rank()));
    const std::string_view reduction = context.attrs.get_string("reduction");
    if (reduction != "none" && reduction != "add" && reduction != "mul" && reduction != "max" && reduction != "min") {
        return errors::invalid_argument("'scatter' reduction must be none, add, mul, max or min; got '{}'", reduction);
    }
    if (reduction != "none" && !is_numeric(data.dtype)) {
        return errors::invalid_argument("'scatter' reductions require a numeric type");
    }
    outputs.push_back(data);
    return ok_status();
}

Status infer_tile(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const TensorType& x = *context.input(0);
    const std::vector<std::int64_t> repeats = context.attrs.get_ints("repeats");
    if (repeats.size() != static_cast<std::size_t>(x.rank())) {
        return errors::invalid_argument("'tile' needs one repeat count per dimension");
    }
    Dims shape = x.shape;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        II_TRY(expect_in_range(repeats[d], 0, kMaxDimension, "'tile' repeats"));
        shape[d] = shape[d] * repeats[d];
    }
    outputs.push_back(TensorType{x.dtype, std::move(shape)});
    return ok_status();
}

Status infer_flip(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    II_TRY(normalize_axes(context.attrs.get_ints("axes"), context.input(0)->rank()));
    outputs.push_back(*context.input(0));
    return ok_status();
}

Status infer_cumsum(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_numeric_dtype, "a numeric type"));
    if (context.input(0)->rank() == 0) {
        return errors::invalid_argument("'cumsum' requires rank >= 1");
    }
    II_TRY(normalize_axis(context.attrs.get_int("axis"), context.input(0)->rank()));
    outputs.push_back(*context.input(0));
    return ok_status();
}

}  // namespace

void register_movement_ops(ir::OpRegistry& registry) {
    const std::uint32_t kMove = ir::kOpDataMovement;
    registry.add(OpSchema{
        .name = "reshape",
        .category = "movement",
        .doc = "Same elements in row-major order with a new shape. One dimension may be -1 (inferred).",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"shape", AttrKind::kDims, std::nullopt}},
        .infer = infer_reshape,
        .flags = kMove | ir::kOpView,
    });
    registry.add(OpSchema{
        .name = "permute",
        .category = "movement",
        .doc = "Reorders dimensions: output dimension i is input dimension perm[i].",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"perm", AttrKind::kInts, std::nullopt}},
        .infer = infer_permute,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "slice",
        .category = "movement",
        .doc = "Strided slices along `axes` with ONNX semantics: negative starts/ends count from the back, bounds are "
               "clamped, steps may be negative (but not zero). An end >= 2^31 - 1 means 'until the end'.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axes", AttrKind::kInts, std::nullopt}, AttrSpec{"starts", AttrKind::kInts, std::nullopt},
                  AttrSpec{"ends", AttrKind::kInts, std::nullopt},
                  AttrSpec{"steps", AttrKind::kInts, std::vector<std::int64_t>{}}},
        .infer = infer_slice,
        .cost = cost_selection,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "expand",
        .category = "movement",
        .doc = "Broadcasts x against `shape` (numpy rules in both directions; the result has the broadcast shape).",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"shape", AttrKind::kDims, std::nullopt}},
        .infer = infer_expand,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "concat",
        .category = "movement",
        .doc = "Concatenates inputs along `axis`.",
        .inputs = {InputSpec{"inputs", false, true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::nullopt}},
        .infer = infer_concat,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "split",
        .category = "movement",
        .doc = "Splits x along `axis` into parts of the given `sizes`.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"outputs", true}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{0}}, AttrSpec{"sizes", AttrKind::kInts, std::nullopt}},
        .infer = infer_split,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "pad",
        .category = "movement",
        .doc = "Pads (or crops, with negative pads) every dimension. pads: all begins, then all ends. mode: constant "
               "(fills `value`), reflect (mirror without repeating the edge), edge (repeats the edge).",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"pads", AttrKind::kInts, std::nullopt},
                  AttrSpec{"mode", AttrKind::kString, std::string("constant")},
                  AttrSpec{"value", AttrKind::kFloat, 0.0}},
        .infer = infer_pad,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "gather",
        .category = "movement",
        .doc = "Selects slices of data along `axis` by indices (ONNX Gather). Negative indices count from the back; "
               "out-of-range indices are a runtime error.",
        .inputs = {InputSpec{"data"}, InputSpec{"indices"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{0}}},
        .infer = infer_gather,
        .cost = cost_selection,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "scatter",
        .category = "movement",
        .doc = "Copy of data with elements replaced (or combined by `reduction`) at positions given by indices along "
               "`axis` (ONNX ScatterElements). With reduction = none and duplicate indices the last update wins.",
        .inputs = {InputSpec{"data"}, InputSpec{"indices"}, InputSpec{"updates"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{0}},
                  AttrSpec{"reduction", AttrKind::kString, std::string("none")}},
        .infer = infer_scatter,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "tile",
        .category = "movement",
        .doc = "Repeats x `repeats[d]` times along every dimension d.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"repeats", AttrKind::kInts, std::nullopt}},
        .infer = infer_tile,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "flip",
        .category = "movement",
        .doc = "Reverses the order of elements along `axes`.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axes", AttrKind::kInts, std::nullopt}},
        .infer = infer_flip,
        .flags = kMove,
    });
    registry.add(OpSchema{
        .name = "cumsum",
        .category = "movement",
        .doc = "Cumulative sum along `axis`; `exclusive` excludes the current element, `reverse` sums from the end.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::nullopt}, AttrSpec{"exclusive", AttrKind::kBool, false},
                  AttrSpec{"reverse", AttrKind::kBool, false}},
        .infer = infer_cumsum,
    });
}

}  // namespace ii::ops
