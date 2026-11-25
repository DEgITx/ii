// SPDX-License-Identifier: MIT
// Elementwise operations: unary, binary (with broadcasting), where, cast, clamp.
#include "ops/infer_util.h"
#include "ops/opset.h"

#include <limits>

namespace ii::ops {

namespace {

using ir::AttrSpec;
using ir::InferContext;
using ir::InputSpec;
using ir::OpSchema;
using ir::OutputSpec;
using ir::TensorType;

bool is_float_dtype(DType dtype) noexcept {
    return is_float(dtype);
}
bool is_signed_numeric(DType dtype) noexcept {
    return is_float(dtype) || is_signed_integer(dtype);
}
bool is_numeric_dtype(DType dtype) noexcept {
    return is_numeric(dtype);
}
bool is_bool_dtype(DType dtype) noexcept {
    return dtype == DType::kBool;
}
bool is_plain_dtype(DType dtype) noexcept {
    return is_plain(dtype);
}

template <bool (*kPredicate)(DType) noexcept>
Status infer_unary(const InferContext& context, std::vector<TensorType>& outputs) {
    constexpr std::string_view kWhat = kPredicate == &is_float_dtype      ? "a float type"
                                       : kPredicate == &is_signed_numeric ? "a float or signed integer type"
                                       : kPredicate == &is_numeric_dtype  ? "a numeric type"
                                       : kPredicate == &is_bool_dtype     ? "bool"
                                                                          : "a supported type";
    II_TRY(expect_input(context, 0, kPredicate, kWhat));
    outputs.push_back(*context.input(0));
    return ok_status();
}

/// Binary op: both inputs have the same dtype satisfying the predicate; the result has `kResult` dtype
/// (kInvalid: the input dtype).
template <bool (*kPredicate)(DType) noexcept, DType kResult>
Status infer_binary(const InferContext& context, std::vector<TensorType>& outputs) {
    constexpr std::string_view kWhat = kPredicate == &is_float_dtype     ? "a float type"
                                       : kPredicate == &is_numeric_dtype ? "a numeric type"
                                       : kPredicate == &is_bool_dtype    ? "bool"
                                                                         : "a non-quantized type";
    II_TRY(expect_input(context, 0, kPredicate, kWhat));
    II_TRY(expect_input(context, 1, kPredicate, kWhat));
    II_TRY(expect_same_dtype(context, 1));
    II_ASSIGN_OR_RETURN(Dims shape, broadcast(context.input(0)->shape, context.input(1)->shape));
    outputs.push_back(TensorType{kResult == DType::kInvalid ? context.input(0)->dtype : kResult, std::move(shape)});
    return ok_status();
}

Status infer_where(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_bool_dtype, "bool"));
    II_TRY(expect_input(context, 1, &is_plain_dtype, "a non-quantized type"));
    II_TRY(expect_input(context, 2, &is_plain_dtype, "a non-quantized type"));
    if (context.input(1)->dtype != context.input(2)->dtype) {
        return errors::invalid_argument("the two value inputs of 'where' must have the same dtype, got {} and {}",
                                        to_string(context.input(1)->dtype), to_string(context.input(2)->dtype));
    }
    II_ASSIGN_OR_RETURN(Dims shape, broadcast(context.input(0)->shape, context.input(1)->shape));
    II_ASSIGN_OR_RETURN(shape, broadcast(shape, context.input(2)->shape));
    outputs.push_back(TensorType{context.input(1)->dtype, std::move(shape)});
    return ok_status();
}

Status infer_cast(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const DType to = context.attrs.get_dtype("to");
    if (!is_plain(to)) {
        return errors::invalid_argument("'cast' target must be a non-quantized type, got {}", to_string(to));
    }
    outputs.push_back(TensorType{to, context.input(0)->shape});
    return ok_status();
}

Status infer_clamp(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_numeric_dtype, "a numeric type"));
    if (context.attrs.get_float("min") > context.attrs.get_float("max")) {
        return errors::invalid_argument("'clamp' requires min <= max");
    }
    outputs.push_back(*context.input(0));
    return ok_status();
}

struct UnaryDef {
    std::string_view name;
    ir::InferFn infer;
    std::string_view doc;
};

struct BinaryDef {
    std::string_view name;
    ir::InferFn infer;
    std::uint32_t flags;
    std::string_view doc;
};

}  // namespace

void register_elementwise_ops(ir::OpRegistry& registry) {
    const UnaryDef unary_ops[] = {
        {"neg", infer_unary<is_signed_numeric>, "-x."},
        {"abs", infer_unary<is_numeric_dtype>, "|x|. For signed integers abs(min) wraps around to min."},
        {"sign", infer_unary<is_numeric_dtype>, "-1, 0 or 1 by the sign of x; NaN stays NaN."},
        {"exp", infer_unary<is_float_dtype>, "e^x."},
        {"log", infer_unary<is_float_dtype>, "Natural logarithm."},
        {"sqrt", infer_unary<is_float_dtype>, "Square root."},
        {"rsqrt", infer_unary<is_float_dtype>, "1 / sqrt(x)."},
        {"recip", infer_unary<is_float_dtype>, "1 / x."},
        {"sin", infer_unary<is_float_dtype>, "Sine."},
        {"cos", infer_unary<is_float_dtype>, "Cosine."},
        {"tanh", infer_unary<is_float_dtype>, "Hyperbolic tangent."},
        {"sigmoid", infer_unary<is_float_dtype>, "1 / (1 + e^-x)."},
        {"erf", infer_unary<is_float_dtype>, "Gauss error function."},
        {"relu", infer_unary<is_signed_numeric>, "max(x, 0)."},
        {"floor", infer_unary<is_float_dtype>, "Round toward negative infinity."},
        {"ceil", infer_unary<is_float_dtype>, "Round toward positive infinity."},
        {"round", infer_unary<is_float_dtype>, "Round to nearest, ties to even."},
        {"not", infer_unary<is_bool_dtype>, "Logical negation."},
    };
    for (const UnaryDef& def : unary_ops) {
        registry.add(OpSchema{
            .name = def.name,
            .category = "unary",
            .doc = def.doc,
            .inputs = {InputSpec{"x"}},
            .outputs = {OutputSpec{"y"}},
            .infer = def.infer,
            .flags = ir::kOpElementwise,
        });
    }

    constexpr DType kSame = DType::kInvalid;
    const BinaryDef binary_ops[] = {
        {"add", infer_binary<is_numeric_dtype, kSame>, ir::kOpCommutative, "a + b (integers wrap around)."},
        {"sub", infer_binary<is_numeric_dtype, kSame>, 0, "a - b (integers wrap around)."},
        {"mul", infer_binary<is_numeric_dtype, kSame>, ir::kOpCommutative, "a * b (integers wrap around)."},
        {"div", infer_binary<is_numeric_dtype, kSame>, 0,
         "a / b. Integer division truncates toward zero; integer division by zero yields 0."},
        {"pow", infer_binary<is_numeric_dtype, kSame>, 0,
         "a ^ b. For integers b must be non-negative (negative exponents yield 0 unless a is 1 or -1)."},
        {"max", infer_binary<is_numeric_dtype, kSame>, ir::kOpCommutative, "Maximum; NaN if either is NaN."},
        {"min", infer_binary<is_numeric_dtype, kSame>, ir::kOpCommutative, "Minimum; NaN if either is NaN."},
        {"mod", infer_binary<is_numeric_dtype, kSame>, 0,
         "Remainder. fmod = false: the result has the sign of b (Python); fmod = true: the sign of a (C fmod). "
         "Integer modulo by zero yields 0."},
        {"eq", infer_binary<is_plain_dtype, DType::kBool>, ir::kOpCommutative, "a == b."},
        {"ne", infer_binary<is_plain_dtype, DType::kBool>, ir::kOpCommutative, "a != b."},
        {"lt", infer_binary<is_numeric_dtype, DType::kBool>, 0, "a < b."},
        {"le", infer_binary<is_numeric_dtype, DType::kBool>, 0, "a <= b."},
        {"gt", infer_binary<is_numeric_dtype, DType::kBool>, 0, "a > b."},
        {"ge", infer_binary<is_numeric_dtype, DType::kBool>, 0, "a >= b."},
        {"and", infer_binary<is_bool_dtype, kSame>, ir::kOpCommutative, "Logical and."},
        {"or", infer_binary<is_bool_dtype, kSame>, ir::kOpCommutative, "Logical or."},
        {"xor", infer_binary<is_bool_dtype, kSame>, ir::kOpCommutative, "Logical exclusive or."},
    };
    for (const BinaryDef& def : binary_ops) {
        OpSchema schema{
            .name = def.name,
            .category = "binary",
            .doc = def.doc,
            .inputs = {InputSpec{"a"}, InputSpec{"b"}},
            .outputs = {OutputSpec{"y"}},
            .infer = def.infer,
            .flags = ir::kOpElementwise | def.flags,
        };
        if (def.name == "mod") {
            schema.attrs = {AttrSpec{"fmod", AttrKind::kBool, false}};
        }
        registry.add(std::move(schema));
    }

    registry.add(OpSchema{
        .name = "where",
        .category = "elementwise",
        .doc = "cond ? a : b, with broadcasting of all three inputs.",
        .inputs = {InputSpec{"cond"}, InputSpec{"a"}, InputSpec{"b"}},
        .outputs = {OutputSpec{"y"}},
        .infer = infer_where,
        .flags = ir::kOpElementwise,
    });
    registry.add(OpSchema{
        .name = "cast",
        .category = "elementwise",
        .doc = "Converts to dtype `to`. Floats to integers: truncation toward zero, saturating, NaN -> 0. "
               "To bool: x != 0. Between floats: round to nearest even.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"to", AttrKind::kDType, std::nullopt}},
        .infer = infer_cast,
        .flags = ir::kOpElementwise,
    });
    registry.add(OpSchema{
        .name = "clamp",
        .category = "elementwise",
        .doc = "min(max(x, min), max). NaN stays NaN.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"min", AttrKind::kFloat, -std::numeric_limits<double>::infinity()},
                  AttrSpec{"max", AttrKind::kFloat, std::numeric_limits<double>::infinity()}},
        .infer = infer_clamp,
        .flags = ir::kOpElementwise,
    });
}

}  // namespace ii::ops
