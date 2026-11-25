// SPDX-License-Identifier: MIT
// Creation (iota, fill, random_uniform, random_normal) and quantization (quantize, dequantize) operations.
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

Status check_created_shape(const Dims& shape) {
    if (shape.size() > static_cast<std::size_t>(kMaxRank)) {
        return errors::invalid_argument("rank {} exceeds the maximum of {}", shape.size(), kMaxRank);
    }
    for (const Dim& dim : shape) {
        if (dim.is_constant() && dim.value() < 0) {
            return errors::invalid_argument("negative dimension in {}", to_string(shape));
        }
    }
    return ok_status();
}

Status infer_iota(const InferContext& context, std::vector<TensorType>& outputs) {
    const Dims shape = context.attrs.get_dims("shape");
    II_TRY(check_created_shape(shape));
    const DType dtype = context.attrs.get_dtype("dtype");
    if (!is_numeric(dtype)) {
        return errors::invalid_argument("'iota' dtype must be numeric, got {}", to_string(dtype));
    }
    if (shape.empty()) {
        return errors::invalid_argument("'iota' requires rank >= 1");
    }
    II_TRY(normalize_axis(context.attrs.get_int("axis"), static_cast<int>(shape.size())));
    outputs.push_back(TensorType{dtype, shape});
    return ok_status();
}

Status infer_fill(const InferContext& context, std::vector<TensorType>& outputs) {
    const Dims shape = context.attrs.get_dims("shape");
    II_TRY(check_created_shape(shape));
    const DType dtype = context.attrs.get_dtype("dtype");
    if (!is_plain(dtype)) {
        return errors::invalid_argument("'fill' dtype must be a non-quantized type, got {}", to_string(dtype));
    }
    outputs.push_back(TensorType{dtype, shape});
    return ok_status();
}

Status infer_random(const InferContext& context, std::vector<TensorType>& outputs) {
    const Dims shape = context.attrs.get_dims("shape");
    II_TRY(check_created_shape(shape));
    const DType dtype = context.attrs.get_dtype("dtype");
    if (!is_float(dtype)) {
        return errors::invalid_argument("'{}' dtype must be a float type, got {}", context.op.name, to_string(dtype));
    }
    outputs.push_back(TensorType{dtype, shape});
    return ok_status();
}

bool is_quantized_int(DType dtype) noexcept {
    return dtype == DType::kI8 || dtype == DType::kU8 || dtype == DType::kI16 || dtype == DType::kU16 ||
           dtype == DType::kI32;
}

/// Checks scale / zero point shapes: rank 0 (per tensor) or [x.shape[axis]] (per axis).
Status check_quant_params(const InferContext& context, const TensorType& x, DType zero_point_dtype) {
    const TensorType* scale = context.input(1);
    const TensorType* zero_point = context.input(2);
    if (scale == nullptr) {
        return errors::invalid_argument("'{}' requires a scale", context.op.name);
    }
    if (!is_float(scale->dtype)) {
        return errors::invalid_argument("'{}' scale must be a float type", context.op.name);
    }
    if (scale->rank() == 1) {
        II_ASSIGN_OR_RETURN(const int axis, normalize_axis(context.attrs.get_int("axis"), x.rank()));
        if (!(scale->shape[0] == x.shape[static_cast<std::size_t>(axis)])) {
            return errors::invalid_argument("'{}' per-axis scale has {} entries for dimension {} of size {}",
                                            context.op.name, scale->shape[0].to_string(), axis,
                                            x.shape[static_cast<std::size_t>(axis)].to_string());
        }
    } else if (scale->rank() != 0) {
        return errors::invalid_argument("'{}' scale must be a scalar or a vector", context.op.name);
    }
    if (zero_point != nullptr) {
        if (zero_point->dtype != zero_point_dtype || !(zero_point->shape == scale->shape)) {
            return errors::invalid_argument("'{}' zero_point must be {} shaped like the scale", context.op.name,
                                            to_string(zero_point_dtype));
        }
    }
    return ok_status();
}

Status infer_quantize(const InferContext& context, std::vector<TensorType>& outputs) {
    const TensorType* x = context.input(0);
    if (!is_float(x->dtype)) {
        return errors::invalid_argument("'quantize' input must be a float type");
    }
    const DType dtype = context.attrs.get_dtype("dtype");
    if (!is_quantized_int(dtype)) {
        return errors::invalid_argument("'quantize' dtype must be i8, u8, i16, u16 or i32; got {}", to_string(dtype));
    }
    II_TRY(check_quant_params(context, *x, dtype));
    if (context.input(1)->dtype != x->dtype) {
        return errors::invalid_argument("'quantize' scale must have the dtype of x");
    }
    outputs.push_back(TensorType{dtype, x->shape});
    return ok_status();
}

Status infer_dequantize(const InferContext& context, std::vector<TensorType>& outputs) {
    const TensorType* x = context.input(0);
    if (is_block_quantized(x->dtype)) {
        if (context.input(1) != nullptr || context.input(2) != nullptr) {
            return errors::invalid_argument("'dequantize' of a block-quantized tensor takes no scale or zero point");
        }
        outputs.push_back(TensorType{DType::kF32, x->shape});
        return ok_status();
    }
    if (!is_quantized_int(x->dtype)) {
        return errors::invalid_argument("'dequantize' input must be i8, u8, i16, u16, i32 or block-quantized");
    }
    II_TRY(check_quant_params(context, *x, x->dtype));
    outputs.push_back(TensorType{context.input(1)->dtype, x->shape});
    return ok_status();
}

}  // namespace

void register_creation_ops(ir::OpRegistry& registry) {
    registry.add(OpSchema{
        .name = "iota",
        .category = "creation",
        .doc = "A tensor whose elements are start + step * index[axis] (converted like `cast`).",
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"shape", AttrKind::kDims, std::nullopt}, AttrSpec{"dtype", AttrKind::kDType, DType::kI64},
                  AttrSpec{"axis", AttrKind::kInt, std::int64_t{0}}, AttrSpec{"start", AttrKind::kFloat, 0.0},
                  AttrSpec{"step", AttrKind::kFloat, 1.0}},
        .infer = infer_iota,
        .flags = ir::kOpNoConstantFold,
    });
    registry.add(OpSchema{
        .name = "fill",
        .category = "creation",
        .doc = "A tensor with every element equal to `value` (converted like `cast`).",
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"shape", AttrKind::kDims, std::nullopt}, AttrSpec{"dtype", AttrKind::kDType, DType::kF32},
                  AttrSpec{"value", AttrKind::kFloat, 0.0}},
        .infer = infer_fill,
        .flags = ir::kOpNoConstantFold,
    });
    registry.add(OpSchema{
        .name = "random_uniform",
        .category = "creation",
        .doc =
            "Pseudo-random values uniform in [low, high). Deterministic for a given seed and shape and identical "
            "on every backend (counter-based Philox4x32-10 with key = seed: element i is word i % 4 of the block with "
            "counter i / 4, mapped to [0, 1) with 24 bits; see src/ops/philox.h).",
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"shape", AttrKind::kDims, std::nullopt}, AttrSpec{"dtype", AttrKind::kDType, DType::kF32},
                  AttrSpec{"low", AttrKind::kFloat, 0.0}, AttrSpec{"high", AttrKind::kFloat, 1.0},
                  AttrSpec{"seed", AttrKind::kInt, std::int64_t{0}}},
        .infer = infer_random,
        .flags = ir::kOpRandom | ir::kOpNoConstantFold,
    });
    registry.add(OpSchema{
        .name = "random_normal",
        .category = "creation",
        .doc = "Pseudo-random normally distributed values (Box-Muller over the same generator as random_uniform).",
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"shape", AttrKind::kDims, std::nullopt}, AttrSpec{"dtype", AttrKind::kDType, DType::kF32},
                  AttrSpec{"mean", AttrKind::kFloat, 0.0}, AttrSpec{"stddev", AttrKind::kFloat, 1.0},
                  AttrSpec{"seed", AttrKind::kInt, std::int64_t{0}}},
        .infer = infer_random,
        .flags = ir::kOpRandom | ir::kOpNoConstantFold,
    });
}

void register_quantization_ops(ir::OpRegistry& registry) {
    registry.add(OpSchema{
        .name = "quantize",
        .category = "quantization",
        .doc = "y = saturate(round_half_even(x / scale) + zero_point), with x / scale computed in single precision "
               "(double "
               "for f64). scale (and zero_point) are scalars or vectors "
               "along `axis`.",
        .inputs = {InputSpec{"x"}, InputSpec{"scale"}, InputSpec{"zero_point", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{1}}, AttrSpec{"dtype", AttrKind::kDType, DType::kI8}},
        .infer = infer_quantize,
    });
    registry.add(OpSchema{
        .name = "dequantize",
        .category = "quantization",
        .doc = "y = (x - zero_point) * scale, in the dtype of scale. A block-quantized x takes no scale and "
               "produces f32.",
        .inputs = {InputSpec{"x"}, InputSpec{"scale", true}, InputSpec{"zero_point", true}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"axis", AttrKind::kInt, std::int64_t{1}}},
        .infer = infer_dequantize,
    });
}

}  // namespace ii::ops
