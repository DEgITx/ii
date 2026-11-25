// SPDX-License-Identifier: MIT
/// @file
/// Building models in code.
///
/// @code
/// ii::GraphBuilder b("mlp");
/// ii::Var x  = b.input("x", ii::DType::kF32, {"N", 784});
/// ii::Var w1 = b.constant(w1_tensor);                     // [256, 784]
/// ii::Var h  = ii::relu(ii::linear(x, w1, b.constant(b1_tensor)));
/// b.output("logits", ii::linear(h, b.constant(w2_tensor)));
/// auto model = ii::Model::from_builder(context, std::move(b)).value();
/// @endcode
///
/// Shapes and types are inferred and checked as the graph is built. Errors are "sticky": the first error is
/// recorded in the builder, operations after it return invalid `Var`s, and `Model::from_builder` reports the error —
/// so model-building code does not need to check every call.
#pragma once

#include <ii/attribute.h>
#include <ii/dim.h>
#include <ii/dtype.h>
#include <ii/export.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

II_BEGIN_DLL_EXPORTS

namespace ii {

class GraphBuilder;

namespace detail {
struct GraphBuilderImpl;
}  // namespace detail

/// A tensor value in a graph under construction. Default-constructed: invalid (also used for absent optional
/// operands).
class II_API Var {
public:
    Var() noexcept = default;

    bool valid() const noexcept { return builder_ != nullptr && id_ >= 0; }
    explicit operator bool() const noexcept { return valid(); }

    GraphBuilder* builder() const noexcept { return builder_; }
    DType dtype() const;
    Dims shape() const;
    int rank() const;

    /// Internal: index of the value in its builder.
    int id() const noexcept { return id_; }

private:
    friend class GraphBuilder;
    Var(GraphBuilder* builder, int id) noexcept : builder_(builder), id_(id) {}
    GraphBuilder* builder_ = nullptr;
    int id_ = -1;
};

/// Builds a model graph.
class II_API GraphBuilder {
public:
    explicit GraphBuilder(std::string name = {});
    ~GraphBuilder();
    GraphBuilder(GraphBuilder&&) noexcept;
    GraphBuilder& operator=(GraphBuilder&&) noexcept;
    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;

    /// Declares an input. Dimensions may be symbolic: `{"N", 3, 224, 224}`.
    Var input(std::string_view name, DType dtype, Dims shape);

    /// A constant (weights). The tensor's memory is shared, not copied.
    Var constant(const Tensor& tensor);

    /// A rank-0 constant of `dtype` holding `value`.
    Var scalar(double value, DType dtype);

    /// Declares a state: a value kept between runs (see `update_state`). `initial` (optional) is its value before
    /// the first run; zeros by default.
    Var state(std::string_view name, DType dtype, Dims shape, const Tensor& initial = {});

    /// The value written to `state` at the end of each run.
    void update_state(const Var& state, const Var& value);

    /// Declares a named output.
    void output(std::string_view name, const Var& value);

    /// Adds an operation of the OpSet and returns its outputs. Absent optional inputs are invalid `Var`s.
    std::vector<Var> op(std::string_view name, std::span<const Var> inputs, const Attributes& attrs = {});
    std::vector<Var> op(std::string_view name, std::initializer_list<Var> inputs, const Attributes& attrs = {});

    /// Like `op`, returning the first (usually only) output.
    Var op1(std::string_view name, std::initializer_list<Var> inputs, const Attributes& attrs = {});

    /// The first error, or OK.
    const Status& status() const noexcept;

    /// Records an error (if none is recorded yet).
    void fail(Status status);

    /// The graph built so far in the IR text format.
    std::string to_text() const;

    /// Internal: the implementation object.
    detail::GraphBuilderImpl& impl() const noexcept { return *impl_; }

private:
    std::unique_ptr<detail::GraphBuilderImpl> impl_;
};

// ---------------------------------------------------------------------------------------------------------------
// Operations. Each function adds one OpSet operation to the builder of its operands; see the OpSet reference
// (`docs/opset.md`) for exact semantics.

// Unary
II_API Var neg(const Var& x);
II_API Var abs(const Var& x);
II_API Var sign(const Var& x);
II_API Var exp(const Var& x);
II_API Var log(const Var& x);
II_API Var sqrt(const Var& x);
II_API Var rsqrt(const Var& x);
II_API Var recip(const Var& x);
II_API Var sin(const Var& x);
II_API Var cos(const Var& x);
II_API Var tanh(const Var& x);
II_API Var sigmoid(const Var& x);
II_API Var erf(const Var& x);
II_API Var relu(const Var& x);
II_API Var floor(const Var& x);
II_API Var ceil(const Var& x);
II_API Var round(const Var& x);
II_API Var logical_not(const Var& x);

// Binary (numpy broadcasting; both operands must have the same dtype)
II_API Var add(const Var& a, const Var& b);
II_API Var sub(const Var& a, const Var& b);
II_API Var mul(const Var& a, const Var& b);
II_API Var div(const Var& a, const Var& b);
II_API Var pow(const Var& a, const Var& b);
II_API Var maximum(const Var& a, const Var& b);
II_API Var minimum(const Var& a, const Var& b);
II_API Var mod(const Var& a, const Var& b, bool fmod = false);
II_API Var equal(const Var& a, const Var& b);
II_API Var not_equal(const Var& a, const Var& b);
II_API Var less(const Var& a, const Var& b);
II_API Var less_equal(const Var& a, const Var& b);
II_API Var greater(const Var& a, const Var& b);
II_API Var greater_equal(const Var& a, const Var& b);
II_API Var logical_and(const Var& a, const Var& b);
II_API Var logical_or(const Var& a, const Var& b);
II_API Var logical_xor(const Var& a, const Var& b);

II_API Var operator+(const Var& a, const Var& b);
II_API Var operator-(const Var& a, const Var& b);
II_API Var operator*(const Var& a, const Var& b);
II_API Var operator/(const Var& a, const Var& b);
II_API Var operator-(const Var& a);
/// Arithmetic with a scalar converted to the dtype of the tensor operand.
II_API Var operator+(const Var& a, double b);
II_API Var operator-(const Var& a, double b);
II_API Var operator*(const Var& a, double b);
II_API Var operator/(const Var& a, double b);
II_API Var operator+(double a, const Var& b);
II_API Var operator-(double a, const Var& b);
II_API Var operator*(double a, const Var& b);
II_API Var operator/(double a, const Var& b);

// Other elementwise
II_API Var where(const Var& cond, const Var& a, const Var& b);
II_API Var cast(const Var& x, DType to);
II_API Var clamp(const Var& x, double min, double max);

// Reductions (empty axes: all axes)
II_API Var reduce_sum(const Var& x, std::vector<std::int64_t> axes = {}, bool keepdims = false);
II_API Var reduce_mean(const Var& x, std::vector<std::int64_t> axes = {}, bool keepdims = false);
II_API Var reduce_max(const Var& x, std::vector<std::int64_t> axes = {}, bool keepdims = false);
II_API Var reduce_min(const Var& x, std::vector<std::int64_t> axes = {}, bool keepdims = false);
II_API Var reduce_prod(const Var& x, std::vector<std::int64_t> axes = {}, bool keepdims = false);
II_API Var argmax(const Var& x, std::int64_t axis, bool keepdims = false);
II_API Var argmin(const Var& x, std::int64_t axis, bool keepdims = false);

// Linear algebra and convolution
II_API Var matmul(const Var& a, const Var& b, bool transpose_a = false, bool transpose_b = false);

/// Sliding-window parameters; empty lists mean the defaults (strides and dilations 1, no padding).
struct WindowOptions {
    std::vector<std::int64_t> strides;
    std::vector<std::int64_t> dilations;
    std::vector<std::int64_t> pads;  ///< all begins, then all ends
    std::string auto_pad = "none";   ///< none | valid | same_upper | same_lower
};
II_API Var conv(const Var& x, const Var& w, const Var& bias = {}, const WindowOptions& window = {},
                std::int64_t groups = 1);
II_API Var conv_transpose(const Var& x, const Var& w, const Var& bias = {}, const WindowOptions& window = {},
                          std::int64_t groups = 1, std::vector<std::int64_t> output_padding = {});
II_API Var max_pool(const Var& x, std::vector<std::int64_t> kernel, const WindowOptions& window = {},
                    bool ceil_mode = false);
II_API Var avg_pool(const Var& x, std::vector<std::int64_t> kernel, const WindowOptions& window = {},
                    bool ceil_mode = false, bool count_include_pad = false);

// Data movement
II_API Var reshape(const Var& x, Dims shape);
II_API Var permute(const Var& x, std::vector<std::int64_t> perm);
II_API Var slice(const Var& x, std::vector<std::int64_t> axes, std::vector<std::int64_t> starts,
                 std::vector<std::int64_t> ends, std::vector<std::int64_t> steps = {});
II_API Var expand(const Var& x, Dims shape);
II_API Var concat(std::span<const Var> inputs, std::int64_t axis);
II_API Var concat(std::initializer_list<Var> inputs, std::int64_t axis);
II_API std::vector<Var> split(const Var& x, std::vector<std::int64_t> sizes, std::int64_t axis = 0);
II_API Var pad(const Var& x, std::vector<std::int64_t> pads, std::string_view mode = "constant", double value = 0);
II_API Var gather(const Var& data, const Var& indices, std::int64_t axis = 0);
II_API Var scatter(const Var& data, const Var& indices, const Var& updates, std::int64_t axis = 0,
                   std::string_view reduction = "none");
II_API Var tile(const Var& x, std::vector<std::int64_t> repeats);
II_API Var flip(const Var& x, std::vector<std::int64_t> axes);
II_API Var cumsum(const Var& x, std::int64_t axis, bool exclusive = false, bool reverse = false);

// Creation
II_API Var iota(GraphBuilder& builder, Dims shape, DType dtype = DType::kI64, std::int64_t axis = 0, double start = 0,
                double step = 1);
II_API Var fill(GraphBuilder& builder, Dims shape, DType dtype, double value);
II_API Var random_uniform(GraphBuilder& builder, Dims shape, DType dtype = DType::kF32, double low = 0, double high = 1,
                          std::int64_t seed = 0);
II_API Var random_normal(GraphBuilder& builder, Dims shape, DType dtype = DType::kF32, double mean = 0,
                         double stddev = 1, std::int64_t seed = 0);

// Quantization
II_API Var quantize(const Var& x, const Var& scale, const Var& zero_point = {}, DType dtype = DType::kI8,
                    std::int64_t axis = 1);
II_API Var dequantize(const Var& x, const Var& scale = {}, const Var& zero_point = {}, std::int64_t axis = 1);

// Composites
II_API Var gelu(const Var& x, bool tanh_approximation = false);
II_API Var silu(const Var& x);
II_API Var softmax(const Var& x, std::int64_t axis = -1);
II_API Var log_softmax(const Var& x, std::int64_t axis = -1);
II_API Var layer_norm(const Var& x, const Var& scale = {}, const Var& bias = {}, std::int64_t axis = -1,
                      double epsilon = 1e-5);
II_API Var rms_norm(const Var& x, const Var& scale = {}, std::int64_t axis = -1, double epsilon = 1e-6);
/// x @ w^T + bias for w [out, in] (transpose_w, the PyTorch layout) or x @ w + bias for w [in, out].
II_API Var linear(const Var& x, const Var& w, const Var& bias = {}, bool transpose_w = true);
/// Scaled dot-product attention; scale 0 means 1 / sqrt(head size).
II_API Var sdpa(const Var& q, const Var& k, const Var& v, const Var& mask = {}, bool causal = false, double scale = 0);
/// Attention over a preallocated KV cache: only the first `kv_len` (an i32 / i64 scalar) keys and values are used;
/// `window` > 0 limits causal attention to the last `window` positions; `softcap` > 0 squashes the scores.
II_API Var sdpa(const Var& q, const Var& k, const Var& v, const Var& mask, bool causal, double scale, const Var& kv_len,
                std::int64_t window = 0, double softcap = 0);
/// Rotary position embedding of the first 2 * cos.shape[-1] features of x (see `rope` in docs/opset.md).
II_API Var rope(const Var& x, const Var& cos, const Var& sin, bool interleaved = false);
/// x with `update` written at offset `start` (an i32 / i64 scalar) along `axis`.
II_API Var slice_update(const Var& x, const Var& update, const Var& start, std::int64_t axis);

}  // namespace ii

II_END_DLL_EXPORTS
