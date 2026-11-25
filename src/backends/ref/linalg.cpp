// SPDX-License-Identifier: MIT
// Reference kernels: matmul, conv, conv_transpose, max_pool, avg_pool.
#include "backends/ref/common.h"
#include "ops/infer_util.h"
#include "tensor/quant.h"

#include <array>
#include <vector>

namespace ii::ref {

namespace {

// ------------------------------------------------------------------------------------------------------------------
// matmul

Status check_matmul(const ir::Node& node) {
    if (is_block_quantized(node.input(1)->dtype()) &&
        (node.input(0)->dtype() == DType::kF64 || node.attrs().get_bool("transpose_a"))) {
        return errors::unsupported("matmul with block-quantized weights needs a non-transposed f32, f16 or bf16 a");
    }
    return ok_status();
}

/// a [..., M, K] times a block-quantized b [N, K] (transpose_b).
std::unique_ptr<Kernel> block_matmul_kernel() {
    return make_kernel(
        [](const KernelContext& context) -> Status {
            const TensorView& a = context.input(0);
            const TensorView& b = context.input(1);
            const std::int64_t k = a.shape[a.rank() - 1];
            const std::int64_t rows = k == 0 ? 0 : a.numel() / k;
            return block_quantized_product(a.dtype, a.data, rows, k, b.data, b.dtype, b.shape[0], nullptr,
                                           context.output(0).data, reinterpret_cast<float*>(context.workspace.data()));
        },
        [](std::span<const Shape> inputs, std::span<const Shape>) { return block_quantized_workspace(inputs[1]); });
}

Result<std::unique_ptr<Kernel>> matmul_kernel(const ir::Node& node) {
    if (is_block_quantized(node.input(1)->dtype())) {
        return block_matmul_kernel();
    }
    const bool transpose_a = node.attrs().get_bool("transpose_a");
    const bool transpose_b = node.attrs().get_bool("transpose_b");
    return make_kernel([transpose_a, transpose_b](const KernelContext& context) -> Status {
        const TensorView& a = context.input(0);
        const TensorView& b = context.input(1);
        const TensorView& y = context.output(0);
        const bool a_vector = a.rank() == 1;
        const bool b_vector = b.rank() == 1;
        const bool ta = transpose_a && !a_vector;
        const bool tb = transpose_b && !b_vector;
        // Logical matrix sizes.
        const std::int64_t m = a_vector ? 1 : (ta ? a.shape[a.rank() - 1] : a.shape[a.rank() - 2]);
        const std::int64_t k = a_vector ? a.shape[0] : (ta ? a.shape[a.rank() - 2] : a.shape[a.rank() - 1]);
        const std::int64_t n = b_vector ? 1 : (tb ? b.shape[b.rank() - 2] : b.shape[b.rank() - 1]);
        // Batch dimensions and broadcasting (in units of whole matrices).
        Shape a_batch;
        Shape b_batch;
        for (int d = 0; d + 2 < a.rank(); ++d) {
            a_batch.push_back(a.shape[d]);
        }
        for (int d = 0; d + 2 < b.rank(); ++d) {
            b_batch.push_back(b.shape[d]);
        }
        Shape batch;
        const int batch_rank = std::max(a_batch.rank(), b_batch.rank());
        for (int d = 0; d < batch_rank; ++d) {
            const int da = d - (batch_rank - a_batch.rank());
            const int db = d - (batch_rank - b_batch.rank());
            const std::int64_t sa = da < 0 ? 1 : a_batch[da];
            const std::int64_t sb = db < 0 ? 1 : b_batch[db];
            batch.push_back(std::max(sa, sb) == 1 ? 1 : (sa == 1 ? sb : sa));
        }
        const Strides a_strides = broadcast_strides(a_batch, batch);
        const Strides b_strides = broadcast_strides(b_batch, batch);
        const std::int64_t batches = batch.numel();
        return dispatch(a.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            using C = ComputeOf<S>;
            // W: the dtype of b (S, or f16 / bf16 weights with an f32 a).
            auto multiply = [&](auto weight_tag) -> Status {
                using W = typename decltype(weight_tag)::type;
                const S* pa = a.as<S>();
                const W* pb = b.as<W>();
                S* py = y.as<S>();
                std::int64_t index[kMaxRank] = {};
                for (std::int64_t bi = 0; bi < batches; ++bi) {
                    unravel_index(bi, batch, index);
                    const S* ma = pa + offset_of(index, a_strides) * m * k;
                    const W* mb = pb + offset_of(index, b_strides) * k * n;
                    S* my = py + bi * m * n;
                    for (std::int64_t i = 0; i < m; ++i) {
                        for (std::int64_t j = 0; j < n; ++j) {
                            C acc{};
                            for (std::int64_t p = 0; p < k; ++p) {
                                const C x = load(ma[ta ? p * m + i : i * k + p]);
                                const C w = static_cast<C>(load(mb[tb ? j * k + p : p * n + j]));
                                if constexpr (std::is_floating_point_v<C>) {
                                    acc += x * w;
                                } else if constexpr (std::is_same_v<C, bool>) {
                                    acc = acc || (x && w);
                                } else {
                                    acc = wrap_add(acc, wrap_mul(x, w));
                                }
                            }
                            my[i * n + j] = store<S>(acc);
                        }
                    }
                }
                return ok_status();
            };
            if (b.dtype == a.dtype) {
                return multiply(Tag<S>{});
            }
            if constexpr (std::is_same_v<S, float>) {
                if (b.dtype == DType::kF16) {
                    return multiply(Tag<f16>{});
                }
                if (b.dtype == DType::kBF16) {
                    return multiply(Tag<bf16>{});
                }
            }
            return errors::unsupported("matmul of {} and {}", to_string(a.dtype), to_string(b.dtype));
        });
    });
}

}  // namespace

Status block_quantized_product(DType dtype, const void* x, std::int64_t rows, std::int64_t k, const void* w,
                               DType w_dtype, std::int64_t n, const void* bias, void* y, float* scratch) {
    II_TRY(quant::check_row(w_dtype, k));
    const std::size_t row_bytes = storage_size(w_dtype, k);
    return dispatch(dtype, [&](auto tag) -> Status {
        using S = typename decltype(tag)::type;
        if constexpr (!kIsFloatStorage<S>) {
            return errors::unsupported("block-quantized products need a float input");
        } else {
            const auto* px = static_cast<const S*>(x);
            const auto* pbias = static_cast<const S*>(bias);
            auto* py = static_cast<S*>(y);
            for (std::int64_t j = 0; j < n; ++j) {
                quant::dequantize(w_dtype, static_cast<const std::byte*>(w) + static_cast<std::size_t>(j) * row_bytes,
                                  scratch, k);
                for (std::int64_t r = 0; r < rows; ++r) {
                    double acc = pbias != nullptr ? static_cast<double>(load(pbias[j])) : 0.0;
                    const S* row = px + r * k;
                    for (std::int64_t p = 0; p < k; ++p) {
                        acc += static_cast<double>(load(row[p])) * static_cast<double>(scratch[p]);
                    }
                    py[r * n + j] = store<S>(acc);
                }
            }
            return ok_status();
        }
    });
}

namespace {

// ------------------------------------------------------------------------------------------------------------------
// Sliding windows

struct Window {
    int spatial = 0;
    std::vector<std::int64_t> kernel;
    std::vector<std::int64_t> strides;
    std::vector<std::int64_t> dilations;
    std::vector<std::int64_t> pads;  // begins, then ends
    std::string auto_pad;
    bool ceil_mode = false;

    /// Concrete per-dimension geometry for `input` spatial sizes (no allocation: kernels call it in `run`).
    std::array<ops::WindowGeometry, kMaxRank> resolve(const Shape& input) const {
        std::array<ops::WindowGeometry, kMaxRank> result{};
        const auto s = static_cast<std::size_t>(spatial);
        for (std::size_t d = 0; d < s; ++d) {
            result[d] = ops::resolve_window(input[static_cast<int>(2 + d)], kernel[d], strides[d], dilations[d],
                                            pads[d], pads[s + d], auto_pad);
        }
        return result;
    }
};

Result<Window> parse_window(const ir::Node& node, int spatial, std::vector<std::int64_t> kernel) {
    II_ASSIGN_OR_RETURN(const ops::WindowAttrs attrs, ops::parse_window_attrs(node.attrs(), spatial));
    Window window;
    window.spatial = spatial;
    window.kernel = std::move(kernel);
    window.strides = attrs.strides;
    window.dilations = attrs.dilations;
    window.pads = attrs.pads;
    window.auto_pad = attrs.auto_pad;
    window.ceil_mode = attrs.ceil_mode;
    return window;
}

Shape spatial_shape(const Shape& shape) {
    Shape result;
    for (int d = 2; d < shape.rank(); ++d) {
        result.push_back(shape[d]);
    }
    return result;
}

Result<std::unique_ptr<Kernel>> conv_kernel(const ir::Node& node) {
    const int spatial = node.input(0)->rank() - 2;
    std::vector<std::int64_t> kernel;
    for (int d = 0; d < spatial; ++d) {
        kernel.push_back(node.input(1)->shape()[static_cast<std::size_t>(2 + d)].value());
    }
    II_ASSIGN_OR_RETURN(Window window, parse_window(node, spatial, std::move(kernel)));
    const std::int64_t groups = node.attrs().get_int("groups");
    return make_kernel([window, groups](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& w = context.input(1);
        const TensorView& bias = context.input(2);
        const TensorView& y = context.output(0);
        const auto geometry = window.resolve(x.shape);
        const Shape in_spatial = spatial_shape(x.shape);
        const Shape out_spatial = spatial_shape(y.shape);
        const Shape k_spatial = spatial_shape(w.shape);
        const std::int64_t batch = x.shape[0];
        const std::int64_t channels = x.shape[1];
        const std::int64_t out_channels = y.shape[1];
        const std::int64_t group_in = channels / groups;
        const std::int64_t group_out = out_channels / groups;
        const std::int64_t in_size = in_spatial.numel();
        const std::int64_t out_size = out_spatial.numel();
        const std::int64_t k_size = k_spatial.numel();
        const int s = window.spatial;
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S* px = x.as<S>();
            const S* pw = w.as<S>();
            S* py = y.as<S>();
            std::int64_t out_index[kMaxRank] = {};
            std::int64_t k_index[kMaxRank] = {};
            for (std::int64_t nb = 0; nb < batch; ++nb) {
                for (std::int64_t oc = 0; oc < out_channels; ++oc) {
                    const std::int64_t g = oc / group_out;
                    for (std::int64_t o = 0; o < out_size; ++o) {
                        unravel_index(o, out_spatial, out_index);
                        double acc = bias.present() ? static_cast<double>(load(at<S>(bias, oc))) : 0.0;
                        for (std::int64_t c = 0; c < group_in; ++c) {
                            const std::int64_t ic = g * group_in + c;
                            const S* x_plane = px + (nb * channels + ic) * in_size;
                            const S* w_kernel = pw + (oc * group_in + c) * k_size;
                            for (std::int64_t kp = 0; kp < k_size; ++kp) {
                                unravel_index(kp, k_spatial, k_index);
                                std::int64_t offset = 0;
                                bool inside = true;
                                for (int d = 0; d < s; ++d) {
                                    const auto& gd = geometry[static_cast<std::size_t>(d)];
                                    const std::int64_t pos =
                                        out_index[d] * gd.stride - gd.pad_begin + k_index[d] * gd.dilation;
                                    if (pos < 0 || pos >= in_spatial[d]) {
                                        inside = false;
                                        break;
                                    }
                                    offset = offset * in_spatial[d] + pos;
                                }
                                if (inside) {
                                    acc += static_cast<double>(load(x_plane[offset])) *
                                           static_cast<double>(load(w_kernel[kp]));
                                }
                            }
                        }
                        py[(nb * out_channels + oc) * out_size + o] = store<S>(acc);
                    }
                }
            }
            return ok_status();
        });
    });
}

Result<std::unique_ptr<Kernel>> conv_transpose_kernel(const ir::Node& node) {
    const int spatial = node.input(0)->rank() - 2;
    std::vector<std::int64_t> kernel;
    for (int d = 0; d < spatial; ++d) {
        kernel.push_back(node.input(1)->shape()[static_cast<std::size_t>(2 + d)].value());
    }
    II_ASSIGN_OR_RETURN(Window window, parse_window(node, spatial, std::move(kernel)));
    const std::int64_t groups = node.attrs().get_int("groups");
    return make_kernel([window, groups](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& w = context.input(1);
        const TensorView& bias = context.input(2);
        const TensorView& y = context.output(0);
        const Shape in_spatial = spatial_shape(x.shape);
        const Shape out_spatial = spatial_shape(y.shape);
        const Shape k_spatial = spatial_shape(w.shape);
        const std::int64_t batch = x.shape[0];
        const std::int64_t channels = x.shape[1];
        const std::int64_t out_channels = y.shape[1];
        const std::int64_t group_in = channels / groups;
        const std::int64_t group_out = out_channels / groups;
        const std::int64_t in_size = in_spatial.numel();
        const std::int64_t out_size = out_spatial.numel();
        const std::int64_t k_size = k_spatial.numel();
        const int s = window.spatial;
        const auto sp = static_cast<std::size_t>(s);
        // Accumulate in double in the workspace, then round once.
        auto* acc = reinterpret_cast<double*>(context.workspace.data());
        for (std::int64_t nb = 0; nb < batch; ++nb) {
            for (std::int64_t oc = 0; oc < out_channels; ++oc) {
                for (std::int64_t o = 0; o < out_size; ++o) {
                    acc[(nb * out_channels + oc) * out_size + o] = 0.0;
                }
            }
        }
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S* px = x.as<S>();
            const S* pw = w.as<S>();
            S* py = y.as<S>();
            std::int64_t in_index[kMaxRank] = {};
            std::int64_t k_index[kMaxRank] = {};
            for (std::int64_t nb = 0; nb < batch; ++nb) {
                for (std::int64_t ic = 0; ic < channels; ++ic) {
                    const std::int64_t g = ic / group_in;
                    for (std::int64_t i = 0; i < in_size; ++i) {
                        unravel_index(i, in_spatial, in_index);
                        const double xv = static_cast<double>(load(px[(nb * channels + ic) * in_size + i]));
                        for (std::int64_t m = 0; m < group_out; ++m) {
                            const std::int64_t oc = g * group_out + m;
                            const S* w_kernel = pw + (ic * group_out + m) * k_size;
                            for (std::int64_t kp = 0; kp < k_size; ++kp) {
                                unravel_index(kp, k_spatial, k_index);
                                std::int64_t offset = 0;
                                bool inside = true;
                                for (std::size_t d = 0; d < sp; ++d) {
                                    const auto di = static_cast<int>(d);
                                    const std::int64_t pos = in_index[di] * window.strides[d] - window.pads[d] +
                                                             k_index[di] * window.dilations[d];
                                    if (pos < 0 || pos >= out_spatial[di]) {
                                        inside = false;
                                        break;
                                    }
                                    offset = offset * out_spatial[di] + pos;
                                }
                                if (inside) {
                                    acc[(nb * out_channels + oc) * out_size + offset] +=
                                        xv * static_cast<double>(load(w_kernel[kp]));
                                }
                            }
                        }
                    }
                }
            }
            for (std::int64_t nb = 0; nb < batch; ++nb) {
                for (std::int64_t oc = 0; oc < out_channels; ++oc) {
                    const double b = bias.present() ? static_cast<double>(load(at<S>(bias, oc))) : 0.0;
                    for (std::int64_t o = 0; o < out_size; ++o) {
                        const std::int64_t index = (nb * out_channels + oc) * out_size + o;
                        py[index] = store<S>(acc[index] + b);
                    }
                }
            }
            return ok_status();
        });
    });
}

/// conv_transpose accumulates in a double workspace.
class ConvTransposeKernel final : public Kernel {
public:
    explicit ConvTransposeKernel(std::unique_ptr<Kernel> inner) : inner_(std::move(inner)) {}
    std::size_t workspace_size(std::span<const Shape>, std::span<const Shape> outputs) const override {
        return static_cast<std::size_t>(outputs[0].numel()) * sizeof(double);
    }
    Status run(const KernelContext& context) const override { return inner_->run(context); }

private:
    std::unique_ptr<Kernel> inner_;
};

Result<std::unique_ptr<Kernel>> conv_transpose_factory(const ir::Node& node) {
    II_ASSIGN_OR_RETURN(std::unique_ptr<Kernel> inner, conv_transpose_kernel(node));
    return std::unique_ptr<Kernel>(std::make_unique<ConvTransposeKernel>(std::move(inner)));
}

template <bool kMax>
Result<std::unique_ptr<Kernel>> pool_kernel(const ir::Node& node) {
    const int spatial = node.input(0)->rank() - 2;
    II_ASSIGN_OR_RETURN(Window window, parse_window(node, spatial, node.attrs().get_ints("kernel")));
    const bool count_include_pad = node.attrs().get_bool("count_include_pad");
    return make_kernel([window, count_include_pad](const KernelContext& context) -> Status {
        (void)count_include_pad;  // used by avg_pool only
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        const auto geometry = window.resolve(x.shape);
        const Shape in_spatial = spatial_shape(x.shape);
        const Shape out_spatial = spatial_shape(y.shape);
        Shape k_spatial;
        for (const std::int64_t k : window.kernel) {
            k_spatial.push_back(k);
        }
        const std::int64_t planes = x.shape[0] * x.shape[1];
        const std::int64_t in_size = in_spatial.numel();
        const std::int64_t out_size = out_spatial.numel();
        const std::int64_t k_size = k_spatial.numel();
        const int s = window.spatial;
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            using C = ComputeOf<S>;
            const S* px = x.as<S>();
            S* py = y.as<S>();
            std::int64_t out_index[kMaxRank] = {};
            std::int64_t k_index[kMaxRank] = {};
            for (std::int64_t plane = 0; plane < planes; ++plane) {
                const S* x_plane = px + plane * in_size;
                for (std::int64_t o = 0; o < out_size; ++o) {
                    unravel_index(o, out_spatial, out_index);
                    C best{};
                    bool have = false;
                    bool nan = false;
                    double sum = 0.0;
                    std::int64_t valid = 0;
                    std::int64_t padded = 0;
                    for (std::int64_t kp = 0; kp < k_size; ++kp) {
                        unravel_index(kp, k_spatial, k_index);
                        std::int64_t offset = 0;
                        bool inside = true;
                        bool in_padding_area = true;
                        for (int d = 0; d < s; ++d) {
                            const auto& gd = geometry[static_cast<std::size_t>(d)];
                            const std::int64_t pos = out_index[d] * gd.stride - gd.pad_begin + k_index[d] * gd.dilation;
                            if (pos < -gd.pad_begin || pos >= in_spatial[d] + gd.pad_end) {
                                in_padding_area = false;
                            }
                            if (pos < 0 || pos >= in_spatial[d]) {
                                inside = false;
                            } else {
                                offset = offset * in_spatial[d] + pos;
                            }
                        }
                        if (in_padding_area) {
                            ++padded;
                        }
                        if (!inside) {
                            continue;
                        }
                        const C v = load(x_plane[offset]);
                        ++valid;
                        if constexpr (kMax) {
                            if constexpr (std::is_floating_point_v<C>) {
                                nan = nan || std::isnan(v);
                            }
                            if (!have || v > best) {
                                best = v;
                                have = true;
                            }
                        } else {
                            sum += static_cast<double>(v);
                        }
                    }
                    if constexpr (kMax) {
                        if (nan) {
                            py[plane * out_size + o] = store<S>(std::numeric_limits<double>::quiet_NaN());
                        } else if (!have) {
                            py[plane * out_size + o] = store<S>(-std::numeric_limits<double>::infinity());
                        } else {
                            py[plane * out_size + o] = store<S>(best);
                        }
                    } else {
                        const std::int64_t divisor = count_include_pad ? padded : valid;
                        py[plane * out_size + o] = store<S>(divisor == 0 ? std::numeric_limits<double>::quiet_NaN()
                                                                         : sum / static_cast<double>(divisor));
                    }
                }
            }
            return ok_status();
        });
    });
}

}  // namespace

void register_linalg_kernels(backend::KernelBackend& backend) {
    backend.add_kernel("matmul", matmul_kernel, check_matmul);
    backend.add_kernel("conv", conv_kernel);
    backend.add_kernel("conv_transpose", conv_transpose_factory);
    backend.add_kernel("max_pool", pool_kernel<true>);
    backend.add_kernel("avg_pool", pool_kernel<false>);
}

}  // namespace ii::ref
