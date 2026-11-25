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
    if (is_block_quantized(node.input(1)->dtype())) {
        return errors::unsupported("linear with block-quantized weights is not implemented yet");
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> linear_kernel(const ir::Node& node) {
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
            const S* px = x.as<S>();
            const S* pw = w.as<S>();
            S* py = context.output(0).as<S>();
            for (std::int64_t r = 0; r < rows; ++r) {
                for (std::int64_t o = 0; o < out_features; ++o) {
                    double acc = bias.present() ? static_cast<double>(load(at<S>(bias, o))) : 0.0;
                    for (std::int64_t k = 0; k < in_features; ++k) {
                        const S& weight = transpose_w ? pw[o * in_features + k] : pw[k * out_features + o];
                        acc += static_cast<double>(load(px[r * in_features + k])) * static_cast<double>(load(weight));
                    }
                    py[r * out_features + o] = store<S>(acc);
                }
            }
            return ok_status();
        });
    });
}

// ------------------------------------------------------------------------------------------------------------------
// sdpa

class SdpaKernel final : public Kernel {
public:
    SdpaKernel(bool causal, double scale) : causal_(causal), scale_(scale) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape>) const override {
        return static_cast<std::size_t>(inputs[1][2]) * sizeof(double);  // one row of scores
    }

    Status run(const KernelContext& context) const override {
        const TensorView& q = context.input(0);
        const TensorView& k = context.input(1);
        const TensorView& v = context.input(2);
        const TensorView no_mask;
        const TensorView& mask = context.inputs.size() > 3 ? context.input(3) : no_mask;
        const std::int64_t batch = q.shape[0];
        const std::int64_t heads_q = q.shape[1];
        const std::int64_t seq_q = q.shape[2];
        const std::int64_t head_size = q.shape[3];
        const std::int64_t heads_kv = k.shape[1];
        const std::int64_t seq_kv = k.shape[2];
        const std::int64_t value_size = v.shape[3];
        const std::int64_t group = heads_q / heads_kv;
        const double scale = scale_ != 0.0 ? scale_ : 1.0 / std::sqrt(static_cast<double>(head_size));
        const Shape scores_shape{batch, heads_q, seq_q, seq_kv};
        const Strides mask_strides = mask.present() ? broadcast_strides(mask.shape, scores_shape) : Strides();
        auto* scores = reinterpret_cast<double*>(context.workspace.data());
        return dispatch(q.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S* pq = q.as<S>();
            const S* pk = k.as<S>();
            const S* pv = v.as<S>();
            S* py = context.output(0).as<S>();
            for (std::int64_t b = 0; b < batch; ++b) {
                for (std::int64_t h = 0; h < heads_q; ++h) {
                    const std::int64_t hk = h / group;
                    const S* keys = pk + (b * heads_kv + hk) * seq_kv * head_size;
                    const S* values = pv + (b * heads_kv + hk) * seq_kv * value_size;
                    for (std::int64_t i = 0; i < seq_q; ++i) {
                        const S* query = pq + ((b * heads_q + h) * seq_q + i) * head_size;
                        double max = -std::numeric_limits<double>::infinity();
                        for (std::int64_t j = 0; j < seq_kv; ++j) {
                            double s = 0.0;
                            for (std::int64_t d = 0; d < head_size; ++d) {
                                s += static_cast<double>(load(query[d])) *
                                     static_cast<double>(load(keys[j * head_size + d]));
                            }
                            s *= scale;
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
                            if (causal_ && j > i + seq_kv - seq_q) {
                                s = -std::numeric_limits<double>::infinity();
                            }
                            scores[j] = s;
                            max = std::max(max, s);
                        }
                        double sum = 0.0;
                        for (std::int64_t j = 0; j < seq_kv; ++j) {
                            scores[j] = std::exp(scores[j] - max);
                            sum += scores[j];
                        }
                        S* out = py + ((b * heads_q + h) * seq_q + i) * value_size;
                        for (std::int64_t d = 0; d < value_size; ++d) {
                            double acc = 0.0;
                            for (std::int64_t j = 0; j < seq_kv; ++j) {
                                acc += scores[j] * static_cast<double>(load(values[j * value_size + d]));
                            }
                            out[d] = store<S>(acc / sum);
                        }
                    }
                }
            }
            return ok_status();
        });
    }

private:
    bool causal_;
    double scale_;
};

Result<std::unique_ptr<Kernel>> sdpa_kernel(const ir::Node& node) {
    return std::unique_ptr<Kernel>(
        std::make_unique<SdpaKernel>(node.attrs().get_bool("causal"), node.attrs().get_float("scale")));
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
}

}  // namespace ii::ref
