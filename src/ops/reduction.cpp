// SPDX-License-Identifier: MIT
// Reductions: reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod, argmax, argmin, topk.
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

bool is_numeric_dtype(DType dtype) noexcept {
    return is_numeric(dtype);
}
bool is_float_dtype(DType dtype) noexcept {
    return is_float(dtype);
}

/// Output shape of a reduction over `axes` (empty: all axes).
Result<Dims> reduced_shape(const Dims& shape, std::span<const std::int64_t> axes, bool keepdims) {
    const int rank = static_cast<int>(shape.size());
    II_ASSIGN_OR_RETURN(const std::vector<int> normalized, normalize_axes(axes, rank));
    bool reduce[kMaxRank] = {};
    for (const int axis : normalized) {
        reduce[axis] = true;
    }
    const bool all = normalized.empty();
    Dims result;
    for (int d = 0; d < rank; ++d) {
        if (all || reduce[d]) {
            if (keepdims) {
                result.emplace_back(1);
            }
        } else {
            result.push_back(shape[static_cast<std::size_t>(d)]);
        }
    }
    return result;
}

template <bool (*kPredicate)(DType) noexcept>
Status infer_reduce(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, kPredicate, kPredicate == &is_float_dtype ? "a float type" : "a numeric type"));
    const std::vector<std::int64_t> axes = context.attrs.get_ints("axes");
    II_ASSIGN_OR_RETURN(Dims shape, reduced_shape(context.input(0)->shape, axes, context.attrs.get_bool("keepdims")));
    outputs.push_back(TensorType{context.input(0)->dtype, std::move(shape)});
    return ok_status();
}

Status infer_arg_reduce(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_numeric_dtype, "a numeric type"));
    const TensorType& input = *context.input(0);
    if (input.rank() == 0) {
        return errors::invalid_argument("'{}' requires an input of rank >= 1", context.op.name);
    }
    const std::int64_t axis = context.attrs.get_int("axis");
    II_TRY(normalize_axis(axis, input.rank()));
    const std::int64_t axes[] = {axis};
    II_ASSIGN_OR_RETURN(Dims shape, reduced_shape(input.shape, axes, context.attrs.get_bool("keepdims")));
    outputs.push_back(TensorType{DType::kI64, std::move(shape)});
    return ok_status();
}

Status infer_topk(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_numeric_dtype, "a numeric type"));
    const TensorType& input = *context.input(0);
    if (input.rank() == 0) {
        return errors::invalid_argument("'topk' requires an input of rank >= 1");
    }
    II_ASSIGN_OR_RETURN(const int axis, normalize_axis(context.attrs.get_int("axis"), input.rank()));
    const std::int64_t k = context.attrs.get_int("k");
    II_TRY(expect_in_range(k, 0, kMaxDimension, "'topk' k"));
    const Dim& size = input.shape[static_cast<std::size_t>(axis)];
    if (size.is_constant() && k > size.value()) {
        return errors::invalid_argument("'topk': k = {} exceeds the size {} of axis {}", k, size.value(), axis);
    }
    Dims shape = input.shape;
    shape[static_cast<std::size_t>(axis)] = Dim(k);
    outputs.push_back(TensorType{input.dtype, shape});
    outputs.push_back(TensorType{DType::kI64, std::move(shape)});
    return ok_status();
}

}  // namespace

void register_reduction_ops(ir::OpRegistry& registry) {
    struct Def {
        std::string_view name;
        ir::InferFn infer;
        std::string_view doc;
    };
    const Def reductions[] = {
        {"reduce_sum", infer_reduce<is_numeric_dtype>,
         "Sum over `axes` (all axes when empty). An empty reduction is 0."},
        {"reduce_mean", infer_reduce<is_float_dtype>, "Arithmetic mean over `axes`. An empty reduction is NaN."},
        {"reduce_max", infer_reduce<is_numeric_dtype>,
         "Maximum over `axes`; NaN if any element is NaN. An empty reduction is the lowest value of the dtype (-inf "
         "for floats)."},
        {"reduce_min", infer_reduce<is_numeric_dtype>,
         "Minimum over `axes`; NaN if any element is NaN. An empty reduction is the highest value of the dtype (inf "
         "for floats)."},
        {"reduce_prod", infer_reduce<is_numeric_dtype>, "Product over `axes`. An empty reduction is 1."},
    };
    for (const Def& def : reductions) {
        registry.add(OpSchema{
            .name = def.name,
            .category = "reduction",
            .doc = def.doc,
            .inputs = {InputSpec{"x"}},
            .outputs = {OutputSpec{"y"}},
            .attrs = {AttrSpec{"axes", AttrKind::kInts, std::vector<std::int64_t>{}},
                      AttrSpec{"keepdims", AttrKind::kBool, false}},
            .infer = def.infer,
        });
    }
    for (const std::string_view name : {std::string_view("argmax"), std::string_view("argmin")}) {
        registry.add(OpSchema{
            .name = name,
            .category = "reduction",
            .doc = "Index (i64) of the largest / smallest element along `axis`. Ties resolve to the first index, or "
                   "the last with select_last_index. NaN compares greater than every number for argmax and less "
                   "for argmin (the first NaN wins).",
            .inputs = {InputSpec{"x"}},
            .outputs = {OutputSpec{"indices"}},
            .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{0}}, AttrSpec{"keepdims", AttrKind::kBool, false},
                      AttrSpec{"select_last_index", AttrKind::kBool, false}},
            .infer = infer_arg_reduce,
        });
    }
    registry.add(OpSchema{
        .name = "topk",
        .category = "reduction",
        .doc = "The `k` largest (or smallest) elements along `axis` and their indices (i64), in order: descending "
               "for largest, ascending otherwise. Equal values keep the order of their indices; NaN compares greater "
               "than every number.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"values"}, OutputSpec{"indices"}},
        .attrs = {AttrSpec{"k", AttrKind::kInt, std::nullopt}, AttrSpec{"axis", AttrKind::kInt, std::int64_t{-1}},
                  AttrSpec{"largest", AttrKind::kBool, true}},
        .infer = infer_topk,
    });
}

}  // namespace ii::ops
