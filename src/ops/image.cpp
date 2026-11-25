// SPDX-License-Identifier: MIT
// Image operations: resize.
#include "ops/infer_util.h"
#include "ops/opset.h"

#include <cmath>

namespace ii::ops {

namespace {

using ir::AttrSpec;
using ir::InferContext;
using ir::InputSpec;
using ir::OpSchema;
using ir::OutputSpec;
using ir::TensorType;

bool is_plain_dtype(DType dtype) noexcept {
    return is_plain(dtype);
}

bool one_of(std::string_view value, std::initializer_list<std::string_view> allowed) {
    for (const std::string_view a : allowed) {
        if (value == a) {
            return true;
        }
    }
    return false;
}

Status infer_resize(const InferContext& context, std::vector<TensorType>& outputs) {
    II_TRY(expect_input(context, 0, &is_plain_dtype, "a non-quantized type"));
    const TensorType& x = *context.input(0);
    const Dims shape = context.attrs.get_dims("shape");
    if (static_cast<int>(shape.size()) != x.rank()) {
        return errors::invalid_argument("'resize': shape {} must have rank {}", to_string(shape), x.rank());
    }
    for (const Dim& dim : shape) {
        if (dim.is_constant()) {
            II_TRY(expect_in_range(dim.value(), 0, kMaxDimension, "'resize' output dimension"));
        }
    }
    const std::vector<double> scales = context.attrs.get_floats("scales");
    if (!scales.empty() && static_cast<int>(scales.size()) != x.rank()) {
        return errors::invalid_argument("'resize': {} scales for rank {}", scales.size(), x.rank());
    }
    for (const double scale : scales) {
        if (!(scale >= 0) || !std::isfinite(scale)) {
            return errors::invalid_argument("'resize': invalid scale {}", scale);
        }
    }
    const std::string_view mode = context.attrs.get_string("mode");
    if (!one_of(mode, {"nearest", "linear", "cubic"})) {
        return errors::invalid_argument("'resize': unknown mode '{}'", mode);
    }
    if (mode != "nearest" && !is_float(x.dtype)) {
        return errors::invalid_argument("'resize': mode '{}' requires a float type, got {}", mode, to_string(x.dtype));
    }
    const std::string_view coordinates = context.attrs.get_string("coordinate_mode");
    if (!one_of(coordinates,
                {"half_pixel", "half_pixel_symmetric", "pytorch_half_pixel", "align_corners", "asymmetric"})) {
        return errors::invalid_argument("'resize': unknown coordinate_mode '{}'", coordinates);
    }
    const std::string_view nearest = context.attrs.get_string("nearest_mode");
    if (!one_of(nearest, {"round_prefer_floor", "round_prefer_ceil", "floor", "ceil"})) {
        return errors::invalid_argument("'resize': unknown nearest_mode '{}'", nearest);
    }
    outputs.push_back(TensorType{x.dtype, shape});
    return ok_status();
}

}  // namespace

void register_image_ops(ir::OpRegistry& registry) {
    registry.add(OpSchema{
        .name = "resize",
        .category = "image",
        .doc = "Resizes x to `shape` by interpolating along every dimension whose size or scale changes (ONNX Resize). "
               "An output coordinate maps to the input by `coordinate_mode` (half_pixel: (o + 0.5) / scale - 0.5, "
               "half_pixel_symmetric, pytorch_half_pixel, align_corners, asymmetric: o / scale) with scale = "
               "`scales[d]`, or out / in when scales are empty or 0. mode: nearest (rounded by `nearest_mode`), "
               "linear, cubic (`cubic_coeff_a`; `exclude_outside` renormalizes the weights inside the input). "
               "Sample positions are clamped to the input.",
        .inputs = {InputSpec{"x"}},
        .outputs = {OutputSpec{"y"}},
        .attrs = {AttrSpec{"shape", AttrKind::kDims, std::nullopt},
                  AttrSpec{"scales", AttrKind::kFloats, std::vector<double>{}},
                  AttrSpec{"mode", AttrKind::kString, std::string("nearest")},
                  AttrSpec{"coordinate_mode", AttrKind::kString, std::string("half_pixel")},
                  AttrSpec{"nearest_mode", AttrKind::kString, std::string("round_prefer_floor")},
                  AttrSpec{"cubic_coeff_a", AttrKind::kFloat, -0.75},
                  AttrSpec{"exclude_outside", AttrKind::kBool, false}},
        .infer = infer_resize,
    });
}

}  // namespace ii::ops
