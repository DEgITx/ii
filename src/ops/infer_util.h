// SPDX-License-Identifier: MIT
/// @file
/// Helpers for shape and type inference. Shared by schemas and by kernels (which must interpret attributes the same
/// way the schema did).
#pragma once

#include <ii/dim.h>
#include <ii/dtype.h>
#include <ii/shape.h>
#include <ii/status.h>

#include "ir/graph.h"
#include "ir/op_schema.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ii::ops {

/// numpy broadcasting of two symbolic shapes. Dimensions must be equal or one of them the constant 1.
Result<Dims> broadcast(const Dims& a, const Dims& b);

/// numpy broadcasting of two concrete shapes.
Result<Shape> broadcast(const Shape& a, const Shape& b);

/// Normalizes a possibly negative axis for a tensor of `rank` (`allow_end`: also accept `rank`, for insertion).
Result<int> normalize_axis(std::int64_t axis, int rank, bool allow_end = false);

/// Normalizes a list of axes; rejects duplicates. The order is preserved.
Result<std::vector<int>> normalize_axes(std::span<const std::int64_t> axes, int rank);

/// Validates a permutation of `rank` axes.
Status check_permutation(std::span<const std::int64_t> perm, int rank);

/// Dtype predicates used by schemas.
bool is_numeric(DType dtype) noexcept;  ///< float or integer (not bool, not block-quantized)
bool is_plain(DType dtype) noexcept;    ///< any dtype except invalid and block-quantized

/// Fails unless input `index` exists and satisfies `predicate`.
Status expect_input(const ir::InferContext& context, int index, bool (*predicate)(DType) noexcept,
                    std::string_view what);

/// Fails unless input `index` has the same dtype as input 0.
Status expect_same_dtype(const ir::InferContext& context, int index);

/// Fails unless input `index` has exactly `rank` dimensions.
Status expect_rank(const ir::InferContext& context, int index, int rank);

/// Fails unless the input has at least `rank` dimensions.
Status expect_min_rank(const ir::InferContext& context, int index, int rank);

/// True when two dimensions are provably different (both constant and unequal).
bool provably_different(const Dim& a, const Dim& b) noexcept;

/// Fails if `a` and `b` are provably different, or if they are symbolic and not structurally equal.
Status expect_equal_dims(const Dim& a, const Dim& b, std::string_view what);

/// Product of dimensions.
Dim product(std::span<const Dim> dims);

/// Concrete parameters of a slice along one dimension of size `size`, with ONNX semantics: negative `start`/`end`
/// count from the back, values are clamped, `step` may be negative (but not zero).
struct SliceRange {
    std::int64_t start = 0;  ///< first index
    std::int64_t count = 0;  ///< number of elements
    std::int64_t step = 1;
};
SliceRange resolve_slice(std::int64_t size, std::int64_t start, std::int64_t end, std::int64_t step) noexcept;

/// Symbolic slice length: exact for constant sizes; for symbolic sizes assumes in-range positive-step bounds.
Result<Dim> slice_length(const Dim& size, std::int64_t start, std::int64_t end, std::int64_t step);

/// Upper bound for sizes in attributes that take part in shape arithmetic (kernel sizes, strides, pads, ...).
/// Keeps every intermediate value of shape computations far from int64 overflow.
inline constexpr std::int64_t kMaxAttributeSize = std::int64_t{1} << 31;

/// Upper bound for a constant tensor dimension (2^48 elements is far beyond any real tensor).
inline constexpr std::int64_t kMaxDimension = std::int64_t{1} << 48;

/// Fails unless `value` is in [low, high].
Status expect_in_range(std::int64_t value, std::int64_t low, std::int64_t high, std::string_view what);

/// Sentinel used for "to the end" in slice `ends` (any value >= this means "until the end").
inline constexpr std::int64_t kSliceEnd = std::int64_t{1} << 62;

/// Spatial convolution / pooling geometry for one spatial dimension.
struct WindowGeometry {
    std::int64_t kernel = 1;
    std::int64_t stride = 1;
    std::int64_t dilation = 1;
    std::int64_t pad_begin = 0;
    std::int64_t pad_end = 0;
};

/// Parsed convolution / pooling attributes for `spatial` spatial dimensions (strides, dilations, pads, auto_pad).
struct WindowAttrs {
    std::vector<std::int64_t> strides;
    std::vector<std::int64_t> dilations;
    std::vector<std::int64_t> pads;  ///< 2 * spatial: all begins, then all ends
    std::string auto_pad;            ///< "none", "same_upper", "same_lower", "valid"
    bool ceil_mode = false;
};
Result<WindowAttrs> parse_window_attrs(const Attributes& attrs, int spatial);

/// Output size of a sliding window over `input` (symbolic).
Result<Dim> window_output_size(const Dim& input, std::int64_t kernel, std::int64_t stride, std::int64_t dilation,
                               std::int64_t pad_begin, std::int64_t pad_end, std::string_view auto_pad, bool ceil_mode);

/// Concrete geometry (with auto_pad resolved) for an input of size `input`.
WindowGeometry resolve_window(std::int64_t input, std::int64_t kernel, std::int64_t stride, std::int64_t dilation,
                              std::int64_t pad_begin, std::int64_t pad_end, std::string_view auto_pad);

/// Concrete output size of a sliding window (same rules as `window_output_size`).
std::int64_t window_output_size(std::int64_t input, const WindowGeometry& geometry, bool ceil_mode);

/// Interpolation taps of `resize` along one dimension: output position `o` reads input positions
/// `index[o * count + t]` with weights `weight[o * count + t]` for t < count (1 tap for nearest, 2 for linear, 4 for
/// cubic). Positions are clamped to the input (edge replication).
struct ResizeTaps {
    int count = 1;
    std::vector<std::int64_t> index;
    std::vector<double> weight;
};

/// Parsed `resize` attributes for one dimension.
struct ResizeMode {
    std::string_view mode;             ///< nearest, linear, cubic
    std::string_view coordinate_mode;  ///< half_pixel, ...
    std::string_view nearest_mode;     ///< round_prefer_floor, ...
    double cubic_coeff_a = -0.75;
    bool exclude_outside = false;
};

/// The taps of one dimension resized from `input` to `output` positions; `scale` <= 0 means output / input.
ResizeTaps resize_taps(std::int64_t input, std::int64_t output, double scale, const ResizeMode& mode);

/// Number of taps per output position of a resize mode (1, 2 or 4).
int resize_tap_count(std::string_view mode) noexcept;

/// Like `resize_taps`, writing `output * resize_tap_count(mode)` indices and weights into caller memory (no
/// allocation).
void resize_taps_into(std::int64_t input, std::int64_t output, double scale, const ResizeMode& mode,
                      std::int64_t* index, double* weight);

}  // namespace ii::ops
