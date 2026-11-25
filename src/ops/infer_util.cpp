// SPDX-License-Identifier: MIT
#include "ops/infer_util.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ii::ops {

Result<Dims> broadcast(const Dims& a, const Dims& b) {
    const std::size_t rank = std::max(a.size(), b.size());
    Dims result(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        const Dim da = i < rank - a.size() ? Dim(1) : a[i - (rank - a.size())];
        const Dim db = i < rank - b.size() ? Dim(1) : b[i - (rank - b.size())];
        if (da == db || db == Dim(1)) {
            result[i] = da;
        } else if (da == Dim(1)) {
            result[i] = db;
        } else {
            return errors::invalid_argument("shapes {} and {} are not broadcastable (dimension {} vs {})", to_string(a),
                                            to_string(b), da.to_string(), db.to_string());
        }
    }
    return result;
}

Result<Shape> broadcast(const Shape& a, const Shape& b) {
    const int rank = std::max(a.rank(), b.rank());
    Shape result = Shape::filled(rank, 1);
    for (int i = 0; i < rank; ++i) {
        const std::int64_t da = i < rank - a.rank() ? 1 : a[i - (rank - a.rank())];
        const std::int64_t db = i < rank - b.rank() ? 1 : b[i - (rank - b.rank())];
        if (da == db || db == 1) {
            result[i] = da;
        } else if (da == 1) {
            result[i] = db;
        } else {
            return errors::invalid_argument("shapes {} and {} are not broadcastable", a.to_string(), b.to_string());
        }
    }
    return result;
}

Result<int> normalize_axis(std::int64_t axis, int rank, bool allow_end) {
    const std::int64_t limit = allow_end ? rank + 1 : rank;
    const std::int64_t normalized = axis < 0 ? axis + limit : axis;
    if (normalized < 0 || normalized >= limit) {
        return errors::invalid_argument("axis {} is out of range for rank {}", axis, rank);
    }
    return static_cast<int>(normalized);
}

Result<std::vector<int>> normalize_axes(std::span<const std::int64_t> axes, int rank) {
    std::vector<int> result;
    bool seen[kMaxRank + 1] = {};
    for (const std::int64_t axis : axes) {
        II_ASSIGN_OR_RETURN(const int normalized, normalize_axis(axis, rank));
        if (seen[normalized]) {
            return errors::invalid_argument("axis {} is repeated", axis);
        }
        seen[normalized] = true;
        result.push_back(normalized);
    }
    return result;
}

Status check_permutation(std::span<const std::int64_t> perm, int rank) {
    if (static_cast<int>(perm.size()) != rank) {
        return errors::invalid_argument("permutation has {} entries for rank {}", perm.size(), rank);
    }
    bool seen[kMaxRank] = {};
    for (const std::int64_t axis : perm) {
        if (axis < 0 || axis >= rank || seen[axis]) {
            return errors::invalid_argument("invalid permutation");
        }
        seen[axis] = true;
    }
    return ok_status();
}

bool is_numeric(DType dtype) noexcept {
    return is_float(dtype) || is_integer(dtype);
}

bool is_plain(DType dtype) noexcept {
    return dtype != DType::kInvalid && dtype != DType::kCount && !is_block_quantized(dtype);
}

Status expect_input(const ir::InferContext& context, int index, bool (*predicate)(DType) noexcept,
                    std::string_view what) {
    const ir::TensorType* type = context.input(index);
    if (type == nullptr) {
        return errors::invalid_argument("'{}' requires input {}", context.op.name, index);
    }
    if (!predicate(type->dtype)) {
        return errors::invalid_argument("input {} of '{}' must be {}, got {}", index, context.op.name, what,
                                        to_string(type->dtype));
    }
    return ok_status();
}

Status expect_same_dtype(const ir::InferContext& context, int index) {
    const ir::TensorType* first = context.input(0);
    const ir::TensorType* type = context.input(index);
    if (first != nullptr && type != nullptr && first->dtype != type->dtype) {
        return errors::invalid_argument("inputs of '{}' must have the same dtype, got {} and {}", context.op.name,
                                        to_string(first->dtype), to_string(type->dtype));
    }
    return ok_status();
}

Status expect_rank(const ir::InferContext& context, int index, int rank) {
    const ir::TensorType* type = context.input(index);
    if (type != nullptr && type->rank() != rank) {
        return errors::invalid_argument("input {} of '{}' must have rank {}, got {}", index, context.op.name, rank,
                                        type->to_string());
    }
    return ok_status();
}

Status expect_min_rank(const ir::InferContext& context, int index, int rank) {
    const ir::TensorType* type = context.input(index);
    if (type != nullptr && type->rank() < rank) {
        return errors::invalid_argument("input {} of '{}' must have rank >= {}, got {}", index, context.op.name, rank,
                                        type->to_string());
    }
    return ok_status();
}

bool provably_different(const Dim& a, const Dim& b) noexcept {
    return a.is_constant() && b.is_constant() && a.value() != b.value();
}

Status expect_equal_dims(const Dim& a, const Dim& b, std::string_view what) {
    if (!(a == b)) {
        return errors::invalid_argument("{}: dimensions {} and {} must be equal", what, a.to_string(), b.to_string());
    }
    return ok_status();
}

Status expect_in_range(std::int64_t value, std::int64_t low, std::int64_t high, std::string_view what) {
    if (value < low || value > high) {
        return errors::invalid_argument("{} must be in [{}, {}], got {}", what, low, high, value);
    }
    return ok_status();
}

Dim product(std::span<const Dim> dims) {
    Dim result = 1;
    for (const Dim& dim : dims) {
        result = result * dim;
    }
    return result;
}

SliceRange resolve_slice(std::int64_t size, std::int64_t start, std::int64_t end, std::int64_t step) noexcept {
    SliceRange range;
    range.step = step;
    if (step == 0 || size <= 0) {
        return range;
    }
    // After clamping, start and end lie in [-1, size], so the differences below cannot overflow.
    if (step > 0) {
        start = start < 0 ? std::max<std::int64_t>(start + size, 0) : std::min(start, size);
        end = end < 0 ? std::max<std::int64_t>(end + size, 0) : std::min(end, size);
        range.start = start;
        range.count = end > start ? (end - start - 1) / step + 1 : 0;
    } else {
        const std::int64_t stride =
            step == std::numeric_limits<std::int64_t>::min() ? std::numeric_limits<std::int64_t>::max() : -step;
        start = start < 0 ? std::max<std::int64_t>(start + size, -1) : std::min(start, size - 1);
        end = end < 0 ? std::max<std::int64_t>(end + size, -1) : std::min(end, size - 1);
        range.start = start;
        range.count = start > end ? (start - end - 1) / stride + 1 : 0;
    }
    return range;
}

Result<Dim> slice_length(const Dim& size, std::int64_t start, std::int64_t end, std::int64_t step) {
    if (step == 0) {
        return errors::invalid_argument("slice step must not be zero");
    }
    if (size.is_constant()) {
        return Dim(resolve_slice(size.value(), start, end, step).count);
    }
    if (step < 0) {
        return errors::unsupported("slices with a negative step over the symbolic dimension {}", size.to_string());
    }
    // Symbolic size: assume the bounds are in range. Very large ends mean "until the end".
    constexpr std::int64_t kToEnd = std::numeric_limits<std::int32_t>::max();
    const Dim begin = start < 0 ? size + Dim(start) : Dim(start);
    const Dim stop = end >= kToEnd ? size : (end < 0 ? size + Dim(end) : Dim(end));
    return ceil_div(stop - begin, step);
}

Result<WindowAttrs> parse_window_attrs(const Attributes& attrs, int spatial) {
    WindowAttrs window;
    window.strides = attrs.get_ints("strides");
    window.dilations = attrs.get_ints("dilations");
    window.pads = attrs.get_ints("pads");
    window.auto_pad = std::string(attrs.get_string("auto_pad", "none"));
    window.ceil_mode = attrs.get_bool("ceil_mode", false);
    const auto n = static_cast<std::size_t>(spatial);
    if (window.strides.empty()) {
        window.strides.assign(n, 1);
    }
    if (window.dilations.empty()) {
        window.dilations.assign(n, 1);
    }
    if (window.pads.empty()) {
        window.pads.assign(2 * n, 0);
    }
    if (window.strides.size() != n || window.dilations.size() != n || window.pads.size() != 2 * n) {
        return errors::invalid_argument("strides/dilations need {} values and pads {} values for {} spatial dimensions",
                                        n, 2 * n, n);
    }
    for (std::size_t i = 0; i < n; ++i) {
        II_TRY(expect_in_range(window.strides[i], 1, kMaxAttributeSize, "strides"));
        II_TRY(expect_in_range(window.dilations[i], 1, kMaxAttributeSize, "dilations"));
    }
    for (const std::int64_t pad : window.pads) {
        II_TRY(expect_in_range(pad, 0, kMaxAttributeSize, "pads"));
    }
    if (window.auto_pad != "none" && window.auto_pad != "same_upper" && window.auto_pad != "same_lower" &&
        window.auto_pad != "valid") {
        return errors::invalid_argument("auto_pad must be none, same_upper, same_lower or valid; got '{}'",
                                        window.auto_pad);
    }
    return window;
}

Result<Dim> window_output_size(const Dim& input, std::int64_t kernel, std::int64_t stride, std::int64_t dilation,
                               std::int64_t pad_begin, std::int64_t pad_end, std::string_view auto_pad,
                               bool ceil_mode) {
    if (input.is_constant()) {
        const WindowGeometry g = resolve_window(input.value(), kernel, stride, dilation, pad_begin, pad_end, auto_pad);
        const std::int64_t size = window_output_size(input.value(), g, ceil_mode);
        if (size <= 0) {
            return errors::invalid_argument("the window (kernel {}, dilation {}) does not fit into input size {}",
                                            kernel, dilation, input.value());
        }
        return Dim(size);
    }
    if (auto_pad == "same_upper" || auto_pad == "same_lower") {
        return ceil_div(input, stride);
    }
    if (auto_pad == "valid") {
        pad_begin = 0;
        pad_end = 0;
    }
    if (ceil_mode) {
        return errors::unsupported("ceil_mode with the symbolic input size {}", input.to_string());
    }
    const Dim span = input + Dim(pad_begin + pad_end - dilation * (kernel - 1) - 1);
    return floor_div(span, stride) + 1;
}

WindowGeometry resolve_window(std::int64_t input, std::int64_t kernel, std::int64_t stride, std::int64_t dilation,
                              std::int64_t pad_begin, std::int64_t pad_end, std::string_view auto_pad) {
    WindowGeometry g{kernel, stride, dilation, pad_begin, pad_end};
    if (auto_pad == "valid") {
        g.pad_begin = 0;
        g.pad_end = 0;
    } else if (auto_pad == "same_upper" || auto_pad == "same_lower") {
        const std::int64_t output = (input + stride - 1) / stride;
        const std::int64_t total =
            std::max<std::int64_t>(0, (output - 1) * stride + dilation * (kernel - 1) + 1 - input);
        const std::int64_t small = total / 2;
        g.pad_begin = auto_pad == "same_upper" ? small : total - small;
        g.pad_end = total - g.pad_begin;
    }
    return g;
}

std::int64_t window_output_size(std::int64_t input, const WindowGeometry& g, bool ceil_mode) {
    const std::int64_t span = input + g.pad_begin + g.pad_end - g.dilation * (g.kernel - 1) - 1;
    if (span < 0) {
        return 0;
    }
    if (!ceil_mode) {
        return span / g.stride + 1;
    }
    std::int64_t size = (span + g.stride - 1) / g.stride + 1;
    // The last window must start inside the input or the leading padding (PyTorch / ONNX rule).
    if ((size - 1) * g.stride >= input + g.pad_begin) {
        --size;
    }
    return size;
}

int resize_tap_count(std::string_view mode) noexcept {
    return mode == "nearest" ? 1 : mode == "linear" ? 2 : 4;
}

ResizeTaps resize_taps(std::int64_t input, std::int64_t output, double scale, const ResizeMode& mode) {
    ResizeTaps taps;
    taps.count = resize_tap_count(mode.mode);
    const auto count = static_cast<std::size_t>(taps.count);
    taps.index.assign(static_cast<std::size_t>(output) * count, 0);
    taps.weight.assign(static_cast<std::size_t>(output) * count, 0.0);
    resize_taps_into(input, output, scale, mode, taps.index.data(), taps.weight.data());
    return taps;
}

void resize_taps_into(std::int64_t input, std::int64_t output, double scale, const ResizeMode& mode,
                      std::int64_t* index, double* weight) {
    struct Taps {
        int count;
        std::int64_t* index;
        double* weight;
    } taps{resize_tap_count(mode.mode), index, weight};
    const auto count = static_cast<std::size_t>(taps.count);
    for (std::size_t i = 0; i < static_cast<std::size_t>(output) * count; ++i) {
        taps.index[i] = 0;
        taps.weight[i] = 0;
    }
    if (input <= 0) {
        return;
    }
    if (!(scale > 0)) {
        scale = static_cast<double>(output) / static_cast<double>(input);
    }
    const auto in = static_cast<double>(input);
    const auto out = static_cast<double>(output);
    const auto clamp_index = [&](std::int64_t i) { return std::clamp<std::int64_t>(i, 0, input - 1); };
    for (std::int64_t o = 0; o < output; ++o) {
        const auto x = static_cast<double>(o);
        double source = 0;
        if (mode.coordinate_mode == "half_pixel") {
            source = (x + 0.5) / scale - 0.5;
        } else if (mode.coordinate_mode == "half_pixel_symmetric") {
            const double adjustment = out / (scale * in);
            const double offset = in / 2 * (1 - adjustment);
            source = offset + (x + 0.5) / scale - 0.5;
        } else if (mode.coordinate_mode == "pytorch_half_pixel") {
            source = output > 1 ? (x + 0.5) / scale - 0.5 : 0.0;
        } else if (mode.coordinate_mode == "align_corners") {
            // The resized length is in * scale (not rounded), as in the ONNX reference.
            const double resized = in * scale;
            source = resized == 1 ? 0.0 : x * (in - 1) / (resized - 1);
        } else {  // asymmetric
            source = x / scale;
        }
        const std::size_t base = static_cast<std::size_t>(o) * count;
        const double floor_source = std::floor(source);
        const double t = source - floor_source;
        const auto i0 = static_cast<std::int64_t>(floor_source);
        if (taps.count == 1) {
            std::int64_t i = i0;
            if (t != 0) {
                const std::string_view nearest = mode.nearest_mode;
                const bool up = nearest == "ceil" || (nearest == "round_prefer_floor" && t > 0.5) ||
                                (nearest == "round_prefer_ceil" && t >= 0.5);
                i = up ? i0 + 1 : i0;
            }
            taps.index[base] = clamp_index(i);
            taps.weight[base] = 1.0;
            continue;
        }
        if (taps.count == 2) {
            taps.index[base] = clamp_index(i0);
            taps.index[base + 1] = clamp_index(i0 + 1);
            taps.weight[base] = 1 - t;
            taps.weight[base + 1] = t;
            continue;
        }
        const double a = mode.cubic_coeff_a;
        const auto cubic = [a](double d) {  // Keys kernel, d = |distance|
            if (d <= 1) {
                return ((a + 2) * d - (a + 3)) * d * d + 1;
            }
            if (d < 2) {
                return ((a * d - 5 * a) * d + 8 * a) * d - 4 * a;
            }
            return 0.0;
        };
        double sum = 0;
        for (int k = 0; k < 4; ++k) {
            const std::int64_t i = i0 - 1 + k;
            double w = cubic(std::fabs(t - (k - 1)));
            if (mode.exclude_outside && (i < 0 || i >= input)) {
                w = 0;
            }
            taps.index[base + static_cast<std::size_t>(k)] = clamp_index(i);
            taps.weight[base + static_cast<std::size_t>(k)] = w;
            sum += w;
        }
        if (mode.exclude_outside && sum != 0) {
            for (int k = 0; k < 4; ++k) {
                taps.weight[base + static_cast<std::size_t>(k)] /= sum;
            }
        }
    }
}

}  // namespace ii::ops
