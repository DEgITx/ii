// SPDX-License-Identifier: MIT
// Reference kernels for composite operations, implemented directly (not via their decompositions) so that the
// decompositions themselves can be checked against them.
#include "backends/ref/common.h"

#include <numbers>
#include <string>

namespace ii::ref {

namespace {

// ------------------------------------------------------------------------------------------------------------------
// gelu, silu

Result<std::unique_ptr<Kernel>> gelu_kernel(const ir::Node& node) {
    const bool tanh_approximation = node.attrs().get_string("approximate") == "tanh";
    return make_kernel([tanh_approximation](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S* in = x.as<S>();
            S* out = context.output(0).as<S>();
            for (std::int64_t i = 0; i < x.numel(); ++i) {
                const auto v = static_cast<double>(load(in[i]));
                const double inner = tanh_approximation
                                         ? std::tanh(std::sqrt(2.0 / std::numbers::pi) * (v + 0.044715 * v * v * v))
                                         : std::erf(v / std::numbers::sqrt2);
                out[i] = store<S>(0.5 * v * (1.0 + inner));
            }
            return ok_status();
        });
    });
}

Result<std::unique_ptr<Kernel>> silu_kernel(const ir::Node&) {
    return make_kernel([](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S* in = x.as<S>();
            S* out = context.output(0).as<S>();
            for (std::int64_t i = 0; i < x.numel(); ++i) {
                const auto v = static_cast<double>(load(in[i]));
                out[i] = store<S>(v / (1.0 + std::exp(-v)));
            }
            return ok_status();
        });
    });
}

// ------------------------------------------------------------------------------------------------------------------
// softmax, log_softmax

template <bool kLog>
Result<std::unique_ptr<Kernel>> softmax_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel([axis_attr](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const int axis = axis_of(axis_attr, x.rank());
        const std::int64_t outer = product(x.shape, 0, axis);
        const std::int64_t n = x.shape[axis];
        const std::int64_t inner = product(x.shape, axis + 1, x.rank());
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S* in = x.as<S>();
            S* out = context.output(0).as<S>();
            for (std::int64_t o = 0; o < outer; ++o) {
                for (std::int64_t i = 0; i < inner; ++i) {
                    auto element = [&](std::int64_t k) { return (o * n + k) * inner + i; };
                    double max = -std::numeric_limits<double>::infinity();
                    bool nan = false;
                    for (std::int64_t k = 0; k < n; ++k) {
                        const auto v = static_cast<double>(load(in[element(k)]));
                        nan = nan || std::isnan(v);
                        max = std::max(max, v);
                    }
                    if (nan) {
                        max = std::numeric_limits<double>::quiet_NaN();
                    }
                    double sum = 0.0;
                    for (std::int64_t k = 0; k < n; ++k) {
                        sum += std::exp(static_cast<double>(load(in[element(k)])) - max);
                    }
                    for (std::int64_t k = 0; k < n; ++k) {
                        const double shifted = static_cast<double>(load(in[element(k)])) - max;
                        out[element(k)] = store<S>(kLog ? shifted - std::log(sum) : std::exp(shifted) / sum);
                    }
                }
            }
            return ok_status();
        });
    });
}

// ------------------------------------------------------------------------------------------------------------------
// layer_norm, rms_norm

template <bool kRms>
Result<std::unique_ptr<Kernel>> norm_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const double epsilon = node.attrs().get_float("epsilon");
    return make_kernel([axis_attr, epsilon](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& scale = context.input(1);
        const TensorView no_bias;
        const TensorView& bias = context.inputs.size() > 2 ? context.input(2) : no_bias;
        const int axis = axis_of(axis_attr, x.rank());
        const std::int64_t rows = product(x.shape, 0, axis);
        const std::int64_t n = product(x.shape, axis, x.rank());
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S* in = x.as<S>();
            S* out = context.output(0).as<S>();
            for (std::int64_t r = 0; r < rows; ++r) {
                const S* row = in + r * n;
                double mean = 0.0;
                if constexpr (!kRms) {
                    for (std::int64_t j = 0; j < n; ++j) {
                        mean += static_cast<double>(load(row[j]));
                    }
                    mean /= static_cast<double>(n);
                }
                double variance = 0.0;
                for (std::int64_t j = 0; j < n; ++j) {
                    const double d = static_cast<double>(load(row[j])) - mean;
                    variance += d * d;
                }
                variance /= static_cast<double>(n);
                const double inv = 1.0 / std::sqrt(variance + epsilon);
                for (std::int64_t j = 0; j < n; ++j) {
                    double y = (static_cast<double>(load(row[j])) - mean) * inv;
                    if (scale.present()) {
                        y *= static_cast<double>(load(at<S>(scale, j)));
                    }
                    if (bias.present()) {
                        y += static_cast<double>(load(at<S>(bias, j)));
                    }
                    out[r * n + j] = store<S>(y);
                }
            }
            return ok_status();
        });
    });
}

// ------------------------------------------------------------------------------------------------------------------
// linear

Status check_linear(const ir::Node& node) {
    if (is_block_quantized(node.input(1)->dtype()) && node.input(0)->dtype() == DType::kF64) {
        return errors::unsupported("linear of f64 with block-quantized weights");
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> linear_kernel(const ir::Node& node) {
    if (is_block_quantized(node.input(1)->dtype())) {
        return make_kernel(
            [](const KernelContext& context) -> Status {
                const TensorView& x = context.input(0);
                const TensorView& w = context.input(1);
                const std::int64_t k = x.shape[x.rank() - 1];
                const std::int64_t rows = k == 0 ? 0 : x.numel() / k;
                return block_quantized_product(x.dtype, x.data, rows, k, w.data, w.dtype, w.shape[0],
                                               context.input(2).present() ? context.input(2).data : nullptr,
                                               context.output(0).data,
                                               reinterpret_cast<float*>(context.workspace.data()));
            },
            [](std::span<const Shape> inputs, std::span<const Shape>) { return block_quantized_workspace(inputs[1]); });
    }
    const bool transpose_w = node.attrs().get_bool("transpose_w");
    return make_kernel([transpose_w](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& w = context.input(1);
        const TensorView& bias = context.input(2);
        const std::int64_t in_features = x.shape[x.rank() - 1];
        const std::int64_t out_features = transpose_w ? w.shape[0] : w.shape[1];
        const std::int64_t rows = out_features == 0 ? 0 : context.output(0).numel() / out_features;
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            // W: the dtype of w (S, or f16 / bf16 weights with an f32 x).
            auto multiply = [&](auto weight_tag) -> Status {
                using W = typename decltype(weight_tag)::type;
                const S* px = x.as<S>();
                const W* pw = w.as<W>();
                S* py = context.output(0).as<S>();
                for (std::int64_t r = 0; r < rows; ++r) {
                    for (std::int64_t o = 0; o < out_features; ++o) {
                        double acc = bias.present() ? static_cast<double>(load(at<S>(bias, o))) : 0.0;
                        for (std::int64_t k = 0; k < in_features; ++k) {
                            const W& weight = transpose_w ? pw[o * in_features + k] : pw[k * out_features + o];
                            acc +=
                                static_cast<double>(load(px[r * in_features + k])) * static_cast<double>(load(weight));
                        }
                        py[r * out_features + o] = store<S>(acc);
                    }
                }
                return ok_status();
            };
            if (w.dtype == x.dtype) {
                return multiply(Tag<S>{});
            }
            if constexpr (std::is_same_v<S, float>) {
                if (w.dtype == DType::kF16) {
                    return multiply(Tag<f16>{});
                }
                if (w.dtype == DType::kBF16) {
                    return multiply(Tag<bf16>{});
                }
            }
            return errors::unsupported("linear of {} and {}", to_string(x.dtype), to_string(w.dtype));
        });
    });
}

// ------------------------------------------------------------------------------------------------------------------
// sdpa

class SdpaKernel final : public Kernel {
public:
    SdpaKernel(bool causal, double scale, std::int64_t window, double softcap)
        : causal_(causal), scale_(scale), window_(window), softcap_(softcap) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape>) const override {
        return static_cast<std::size_t>(inputs[1][2]) * sizeof(double);  // one row of scores
    }

    Status run(const KernelContext& context) const override {
        const TensorView& q = context.input(0);
        const TensorView& k = context.input(1);
        const TensorView& v = context.input(2);
        const TensorView no_mask;
        const TensorView& mask = context.inputs.size() > 3 ? context.input(3) : no_mask;
        const TensorView& kv_len_view = context.inputs.size() > 4 ? context.input(4) : no_mask;
        const std::int64_t batch = q.shape[0];
        const std::int64_t heads_q = q.shape[1];
        const std::int64_t seq_q = q.shape[2];
        const std::int64_t head_size = q.shape[3];
        const std::int64_t heads_kv = k.shape[1];
        const std::int64_t seq_kv = k.shape[2];
        const std::int64_t value_size = v.shape[3];
        const std::int64_t group = heads_q / heads_kv;
        std::int64_t used = seq_kv;  // L: keys in use
        if (kv_len_view.present()) {
            used = kv_len_view.dtype == DType::kI32 ? *kv_len_view.as<const std::int32_t>()
                                                    : *kv_len_view.as<const std::int64_t>();
            if (used < 0 || used > seq_kv) {
                return errors::out_of_range("'sdpa' kv_len {} is outside [0, {}]", used, seq_kv);
            }
        }
        const double scale = scale_ != 0.0 ? scale_ : 1.0 / std::sqrt(static_cast<double>(head_size));
        const Shape scores_shape{batch, heads_q, seq_q, seq_kv};
        const Strides mask_strides = mask.present() ? broadcast_strides(mask.shape, scores_shape) : Strides();
        auto* scores = reinterpret_cast<double*>(context.workspace.data());
        return dispatch(q.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            // K: the dtype of keys and values (S, or 16-bit with an f32 q).
            auto attend = [&](auto kv_tag) -> Status {
                using K = typename decltype(kv_tag)::type;
                const S* pq = q.as<S>();
                const K* pk = k.as<K>();
                const K* pv = v.as<K>();
                S* py = context.output(0).as<S>();
                for (std::int64_t b = 0; b < batch; ++b) {
                    for (std::int64_t h = 0; h < heads_q; ++h) {
                        const std::int64_t hk = h / group;
                        const K* keys = pk + (b * heads_kv + hk) * seq_kv * head_size;
                        const K* values = pv + (b * heads_kv + hk) * seq_kv * value_size;
                        for (std::int64_t i = 0; i < seq_q; ++i) {
                            const S* query = pq + ((b * heads_q + h) * seq_q + i) * head_size;
                            double max = -std::numeric_limits<double>::infinity();
                            for (std::int64_t j = 0; j < used; ++j) {
                                double s = 0.0;
                                for (std::int64_t d = 0; d < head_size; ++d) {
                                    s += static_cast<double>(load(query[d])) *
                                         static_cast<double>(load(keys[j * head_size + d]));
                                }
                                s *= scale;
                                if (softcap_ > 0.0) {
                                    s = softcap_ * std::tanh(s / softcap_);
                                }
                                if (mask.present()) {
                                    const std::int64_t index[4] = {b, h, i, j};
                                    const std::int64_t offset = offset_of(index, mask_strides);
                                    if (mask.dtype == DType::kBool) {
                                        if (mask.as<const std::uint8_t>()[offset] == 0) {
                                            s = -std::numeric_limits<double>::infinity();
                                        }
                                    } else {
                                        s += static_cast<double>(load(mask.as<const S>()[offset]));
                                    }
                                }
                                const std::int64_t position = i + used - seq_q;
                                if (j >= used || (causal_ && j > position) ||
                                    (window_ > 0 && j <= position - window_)) {
                                    s = -std::numeric_limits<double>::infinity();
                                }
                                scores[j] = s;
                                max = std::max(max, s);
                            }
                            double sum = 0.0;
                            for (std::int64_t j = 0; j < used; ++j) {
                                scores[j] = std::exp(scores[j] - max);
                                sum += scores[j];
                            }
                            S* out = py + ((b * heads_q + h) * seq_q + i) * value_size;
                            for (std::int64_t d = 0; d < value_size; ++d) {
                                double acc = 0.0;
                                for (std::int64_t j = 0; j < used; ++j) {
                                    acc += scores[j] * static_cast<double>(load(values[j * value_size + d]));
                                }
                                out[d] = store<S>(acc / sum);
                            }
                        }
                    }
                }
                return ok_status();
            };
            if (k.dtype == q.dtype) {
                return attend(Tag<S>{});
            }
            if constexpr (std::is_same_v<S, float>) {
                if (k.dtype == DType::kF16) {
                    return attend(Tag<f16>{});
                }
                if (k.dtype == DType::kBF16) {
                    return attend(Tag<bf16>{});
                }
            }
            return errors::unsupported("sdpa of {} and {}", to_string(q.dtype), to_string(k.dtype));
        });
    }

private:
    bool causal_;
    double scale_;
    std::int64_t window_;
    double softcap_;
};

Result<std::unique_ptr<Kernel>> sdpa_kernel(const ir::Node& node) {
    return std::unique_ptr<Kernel>(
        std::make_unique<SdpaKernel>(node.attrs().get_bool("causal"), node.attrs().get_float("scale"),
                                     node.attrs().get_int("window"), node.attrs().get_float("softcap")));
}

// ------------------------------------------------------------------------------------------------------------------
// rope

Result<std::unique_ptr<Kernel>> rope_kernel(const ir::Node& node) {
    const bool interleaved = node.attrs().get_bool("interleaved");
    return make_kernel([interleaved](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& cos = context.input(1);
        const TensorView& sin = context.input(2);
        const std::int64_t size = x.shape[x.rank() - 1];
        const std::int64_t half = cos.shape[cos.rank() - 1];
        const std::int64_t rows = size == 0 ? 0 : x.numel() / size;
        // cos / sin are read through broadcast strides over x.shape[:-1] + [half].
        Shape table = x.shape;
        table[table.rank() - 1] = half;
        const Strides strides = broadcast_strides(cos.shape, table);
        const Shape rows_shape = [&] {
            Shape s = x.shape;
            s[s.rank() - 1] = 1;
            return s;
        }();
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            if constexpr (!kIsFloatStorage<S>) {
                return errors::unsupported("'rope' of a non-float type");
            } else {
                const S* px = x.as<S>();
                S* py = context.output(0).as<S>();
                std::int64_t index[kMaxRank] = {};
                for (std::int64_t r = 0; r < rows; ++r) {
                    unravel_index(r, rows_shape, index);
                    const S* in = px + r * size;
                    S* out = py + r * size;
                    for (std::int64_t i = 0; i < half; ++i) {
                        index[table.rank() - 1] = i;
                        const std::int64_t t = offset_of(index, strides);
                        const double c = static_cast<double>(load(cos.as<const S>()[t]));
                        const double s = static_cast<double>(load(sin.as<const S>()[t]));
                        const std::int64_t a = interleaved ? 2 * i : i;
                        const std::int64_t b = interleaved ? 2 * i + 1 : i + half;
                        const double xa = static_cast<double>(load(in[a]));
                        const double xb = static_cast<double>(load(in[b]));
                        out[a] = store<S>(xa * c - xb * s);
                        out[b] = store<S>(xb * c + xa * s);
                    }
                    for (std::int64_t i = 2 * half; i < size; ++i) {
                        out[i] = in[i];
                    }
                }
                return ok_status();
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------------------------
// slice_update

/// Writes `update` into a copy of x (or into x itself when the runtime runs the step in place).
class SliceUpdateKernel final : public Kernel {
public:
    explicit SliceUpdateKernel(std::int64_t axis) : axis_(axis) {}

    bool can_run_in_place(int input) const override { return input == 0; }

    Status run(const KernelContext& context) const override {
        const TensorView& x = context.input(0);
        const TensorView& update = context.input(1);
        const TensorView& start_view = context.input(2);
        const TensorView& y = context.output(0);
        const int axis = axis_of(axis_, x.rank());
        const std::int64_t start = start_view.dtype == DType::kI32 ? *start_view.as<const std::int32_t>()
                                                                   : *start_view.as<const std::int64_t>();
        const std::int64_t length = update.shape[axis];
        if (start < 0 || start > x.shape[axis] - length) {
            return errors::out_of_range("'slice_update' writes [{}, {}) of an axis of size {}", start, start + length,
                                        x.shape[axis]);
        }
        const std::size_t element = element_size(x.dtype);
        if (y.data != x.data) {
            copy_elements(y.data, x.data, x.numel(), element);
        }
        const std::int64_t outer = product(x.shape, 0, axis);
        const std::int64_t inner = product(x.shape, axis + 1, x.rank());
        const auto row = static_cast<std::int64_t>(element) * inner;
        for (std::int64_t o = 0; o < outer; ++o) {
            copy_elements(static_cast<std::byte*>(y.data) + (o * x.shape[axis] + start) * row,
                          static_cast<const std::byte*>(update.data) + o * length * row, length * row, 1);
        }
        return ok_status();
    }

private:
    std::int64_t axis_;
};

Result<std::unique_ptr<Kernel>> slice_update_kernel(const ir::Node& node) {
    return std::unique_ptr<Kernel>(std::make_unique<SliceUpdateKernel>(node.attrs().get_int("axis")));
}

}  // namespace

void register_composite_kernels(backend::KernelBackend& backend) {
    backend.add_kernel("gelu", gelu_kernel);
    backend.add_kernel("silu", silu_kernel);
    backend.add_kernel("softmax", softmax_kernel<false>);
    backend.add_kernel("log_softmax", softmax_kernel<true>);
    backend.add_kernel("layer_norm", norm_kernel<false>);
    backend.add_kernel("rms_norm", norm_kernel<true>);
    backend.add_kernel("linear", linear_kernel, check_linear);
    backend.add_kernel("sdpa", sdpa_kernel);
    backend.add_kernel("rope", rope_kernel);
    backend.add_kernel("slice_update", slice_update_kernel);
}

}  // namespace ii::ref
