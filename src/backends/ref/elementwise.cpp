// SPDX-License-Identifier: MIT
// Reference kernels: unary, binary, where, cast, clamp, quantize, dequantize.
#include "backends/ref/common.h"
#include "tensor/quant.h"

namespace ii::ref {

namespace {

double round_half_even(double value) noexcept {
    const double rounded = std::round(value);  // ties away from zero
    if (std::fabs(value - std::trunc(value)) == 0.5) {
        return 2.0 * std::round(value / 2.0);
    }
    return rounded;
}

// ------------------------------------------------------------------------------------------------------------------
// Unary

template <class Op>
Status run_unary(const KernelContext& context) {
    const TensorView& x = context.input(0);
    const TensorView& y = context.output(0);
    return dispatch(x.dtype, [&](auto tag) -> Status {
        using S = typename decltype(tag)::type;
        const S* in = x.as<S>();
        S* out = y.as<S>();
        const std::int64_t n = x.numel();
        for (std::int64_t i = 0; i < n; ++i) {
            out[i] = store<S>(Op::apply(load(in[i])));
        }
        return ok_status();
    });
}

template <class C>
constexpr bool kFloatCompute = std::is_floating_point_v<C>;

struct Neg {
    template <class C>
    static C apply(C a) {
        if constexpr (kFloatCompute<C>) {
            return -a;
        } else if constexpr (std::is_same_v<C, bool>) {
            return a;
        } else {
            return wrap_sub(C{0}, a);
        }
    }
};
struct Abs {
    template <class C>
    static C apply(C a) {
        if constexpr (kFloatCompute<C>) {
            return std::fabs(a);
        } else if constexpr (std::is_same_v<C, std::int64_t>) {
            return a < 0 ? wrap_sub(std::int64_t{0}, a) : a;
        } else {
            return a;
        }
    }
};
struct Sign {
    template <class C>
    static C apply(C a) {
        if constexpr (kFloatCompute<C>) {
            if (std::isnan(a)) {
                return a;
            }
            return a > 0 ? 1.0 : (a < 0 ? -1.0 : 0.0);
        } else if constexpr (std::is_same_v<C, std::int64_t>) {
            return a > 0 ? 1 : (a < 0 ? -1 : 0);
        } else if constexpr (std::is_same_v<C, bool>) {
            return a;
        } else {
            return a > 0 ? C{1} : C{0};
        }
    }
};
struct Relu {
    template <class C>
    static C apply(C a) {
        if constexpr (kFloatCompute<C>) {
            return std::isnan(a) ? a : (a > 0 ? a : 0.0);
        } else if constexpr (std::is_same_v<C, bool>) {
            return a;
        } else {
            return a > 0 ? a : C{0};
        }
    }
};
struct Not {
    template <class C>
    static C apply(C a) {
        return static_cast<C>(!a);
    }
};

/// Float-only unary functions; integer instantiations exist only to satisfy the dispatcher (never executed:
/// shape inference rejects non-float inputs).
#define II_REF_FLOAT_UNARY(Name, expression)                                                                           \
    struct Name {                                                                                                      \
        template <class C>                                                                                             \
        static C apply(C value) {                                                                                      \
            if constexpr (kFloatCompute<C>) {                                                                          \
                const double a = value;                                                                                \
                return (expression);                                                                                   \
            } else {                                                                                                   \
                return value;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    };

II_REF_FLOAT_UNARY(Exp, std::exp(a))
II_REF_FLOAT_UNARY(Log, std::log(a))
II_REF_FLOAT_UNARY(Sqrt, std::sqrt(a))
II_REF_FLOAT_UNARY(Rsqrt, 1.0 / std::sqrt(a))
II_REF_FLOAT_UNARY(Recip, 1.0 / a)
II_REF_FLOAT_UNARY(Sin, std::sin(a))
II_REF_FLOAT_UNARY(Cos, std::cos(a))
II_REF_FLOAT_UNARY(Tanh, std::tanh(a))
II_REF_FLOAT_UNARY(Sigmoid, 1.0 / (1.0 + std::exp(-a)))
II_REF_FLOAT_UNARY(Erf, std::erf(a))
II_REF_FLOAT_UNARY(Floor, std::floor(a))
II_REF_FLOAT_UNARY(Ceil, std::ceil(a))
II_REF_FLOAT_UNARY(Round, round_half_even(a))
#undef II_REF_FLOAT_UNARY

template <class Op>
Result<std::unique_ptr<Kernel>> unary_kernel(const ir::Node&) {
    return make_kernel([](const KernelContext& context) { return run_unary<Op>(context); });
}

// ------------------------------------------------------------------------------------------------------------------
// Binary (broadcasting)

template <class Op>
Status run_binary(const KernelContext& context, const Op& op) {
    const TensorView& a = context.input(0);
    const TensorView& b = context.input(1);
    const TensorView& y = context.output(0);
    const Shape& shape = y.shape;
    const Strides out_strides = contiguous_strides(shape);
    const Strides a_strides = broadcast_strides(a.shape, shape);
    const Strides b_strides = broadcast_strides(b.shape, shape);
    const int rank = shape.rank();
    const std::int64_t inner = rank == 0 ? 1 : shape[rank - 1];
    const std::int64_t a_inner = rank == 0 ? 0 : a_strides[rank - 1];
    const std::int64_t b_inner = rank == 0 ? 0 : b_strides[rank - 1];
    const Strides* const strides[3] = {&out_strides, &a_strides, &b_strides};
    return dispatch(a.dtype, [&](auto tag) -> Status {
        using S = typename decltype(tag)::type;
        using C = ComputeOf<S>;
        using R = decltype(op(C{}, C{}));
        using Out = std::conditional_t<std::is_same_v<R, bool>, Bool8, S>;
        const S* pa = a.as<S>();
        const S* pb = b.as<S>();
        Out* py = y.as<Out>();
        for_each_row<3>(shape, strides, [&](const std::int64_t* offsets) {
            for (std::int64_t j = 0; j < inner; ++j) {
                py[offsets[0] + j] =
                    store<Out>(op(load(pa[offsets[1] + j * a_inner]), load(pb[offsets[2] + j * b_inner])));
            }
        });
        return ok_status();
    });
}

std::int64_t int_pow(std::int64_t base, std::int64_t exponent) {
    if (exponent < 0) {
        if (base == 1) {
            return 1;
        }
        if (base == -1) {
            return (exponent % 2 == 0) ? 1 : -1;
        }
        return 0;
    }
    std::int64_t result = 1;
    while (exponent > 0) {
        if ((exponent & 1) != 0) {
            result = wrap_mul(result, base);
        }
        base = wrap_mul(base, base);
        exponent >>= 1;
    }
    return result;
}

std::uint64_t uint_pow(std::uint64_t base, std::uint64_t exponent) {
    std::uint64_t result = 1;
    while (exponent > 0) {
        if ((exponent & 1) != 0) {
            result *= base;
        }
        base *= base;
        exponent >>= 1;
    }
    return result;
}

struct AddOp {
    template <class C>
    C operator()(C a, C b) const {
        if constexpr (kFloatCompute<C>) {
            return a + b;
        } else if constexpr (std::is_same_v<C, bool>) {
            return a || b;
        } else {
            return wrap_add(a, b);
        }
    }
};
struct SubOp {
    template <class C>
    C operator()(C a, C b) const {
        if constexpr (kFloatCompute<C>) {
            return a - b;
        } else if constexpr (std::is_same_v<C, bool>) {
            return a != b;
        } else {
            return wrap_sub(a, b);
        }
    }
};
struct MulOp {
    template <class C>
    C operator()(C a, C b) const {
        if constexpr (kFloatCompute<C>) {
            return a * b;
        } else if constexpr (std::is_same_v<C, bool>) {
            return a && b;
        } else {
            return wrap_mul(a, b);
        }
    }
};
struct DivOp {
    template <class C>
    C operator()(C a, C b) const {
        if constexpr (kFloatCompute<C>) {
            return a / b;
        } else if constexpr (std::is_same_v<C, bool>) {
            return a;
        } else {
            if (b == 0) {
                return 0;
            }
            if constexpr (std::is_same_v<C, std::int64_t>) {
                if (a == std::numeric_limits<std::int64_t>::min() && b == -1) {
                    return a;  // wraps around
                }
            }
            return a / b;
        }
    }
};
struct PowOp {
    template <class C>
    C operator()(C a, C b) const {
        if constexpr (kFloatCompute<C>) {
            return std::pow(a, b);
        } else if constexpr (std::is_same_v<C, std::int64_t>) {
            return int_pow(a, b);
        } else if constexpr (std::is_same_v<C, std::uint64_t>) {
            return uint_pow(a, b);
        } else {
            return a || !b;
        }
    }
};
struct MaxOp {
    template <class C>
    C operator()(C a, C b) const {
        if constexpr (kFloatCompute<C>) {
            if (std::isnan(a) || std::isnan(b)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        return a < b ? b : a;
    }
};
struct MinOp {
    template <class C>
    C operator()(C a, C b) const {
        if constexpr (kFloatCompute<C>) {
            if (std::isnan(a) || std::isnan(b)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        return b < a ? b : a;
    }
};
struct ModOp {
    bool fmod = false;
    template <class C>
    C operator()(C a, C b) const {
        if constexpr (kFloatCompute<C>) {
            double r = std::fmod(a, b);
            if (!fmod && r != 0 && ((r < 0) != (b < 0))) {
                r += b;
            }
            return r;
        } else if constexpr (std::is_same_v<C, std::int64_t>) {
            if (b == 0 || b == -1) {
                return 0;  // x % -1 == 0; avoids the INT64_MIN % -1 trap
            }
            std::int64_t r = a % b;
            if (!fmod && r != 0 && ((r < 0) != (b < 0))) {
                r += b;
            }
            return r;
        } else if constexpr (std::is_same_v<C, std::uint64_t>) {
            return b == 0 ? 0 : a % b;
        } else {
            return false;
        }
    }
};

#define II_REF_COMPARE(Name, expression)                                                                               \
    struct Name {                                                                                                      \
        template <class C>                                                                                             \
        bool operator()(C a, C b) const {                                                                              \
            return (expression);                                                                                       \
        }                                                                                                              \
    };
II_REF_COMPARE(EqOp, a == b)
II_REF_COMPARE(NeOp, a != b)
II_REF_COMPARE(LtOp, a < b)
II_REF_COMPARE(LeOp, a <= b)
II_REF_COMPARE(GtOp, a > b)
II_REF_COMPARE(GeOp, a >= b)
II_REF_COMPARE(AndOp, static_cast<bool>(a) && static_cast<bool>(b))
II_REF_COMPARE(OrOp, static_cast<bool>(a) || static_cast<bool>(b))
II_REF_COMPARE(XorOp, static_cast<bool>(a) != static_cast<bool>(b))
#undef II_REF_COMPARE

template <class Op>
Result<std::unique_ptr<Kernel>> binary_kernel(const ir::Node&) {
    return make_kernel([](const KernelContext& context) { return run_binary(context, Op{}); });
}

Result<std::unique_ptr<Kernel>> mod_kernel(const ir::Node& node) {
    const ModOp op{node.attrs().get_bool("fmod")};
    return make_kernel([op](const KernelContext& context) { return run_binary(context, op); });
}

// ------------------------------------------------------------------------------------------------------------------
// where, cast, clamp

Result<std::unique_ptr<Kernel>> where_kernel(const ir::Node&) {
    return make_kernel([](const KernelContext& context) -> Status {
        const TensorView& cond = context.input(0);
        const TensorView& a = context.input(1);
        const TensorView& b = context.input(2);
        const TensorView& y = context.output(0);
        const Shape& shape = y.shape;
        const Strides out_strides = contiguous_strides(shape);
        const Strides c_strides = broadcast_strides(cond.shape, shape);
        const Strides a_strides = broadcast_strides(a.shape, shape);
        const Strides b_strides = broadcast_strides(b.shape, shape);
        const int rank = shape.rank();
        const std::int64_t inner = rank == 0 ? 1 : shape[rank - 1];
        const std::int64_t ci = rank == 0 ? 0 : c_strides[rank - 1];
        const std::int64_t ai = rank == 0 ? 0 : a_strides[rank - 1];
        const std::int64_t bi = rank == 0 ? 0 : b_strides[rank - 1];
        const Strides* const strides[4] = {&out_strides, &c_strides, &a_strides, &b_strides};
        const std::size_t size = element_size(y.dtype);
        const auto* pc = cond.as<const std::uint8_t>();
        const auto* pa = a.as<const std::byte>();
        const auto* pb = b.as<const std::byte>();
        auto* py = y.as<std::byte>();
        const auto s = static_cast<std::int64_t>(size);
        for_each_row<4>(shape, strides, [&](const std::int64_t* o) {
            for (std::int64_t j = 0; j < inner; ++j) {
                const std::byte* source = pc[o[1] + j * ci] != 0 ? pa + (o[2] + j * ai) * s : pb + (o[3] + j * bi) * s;
                std::memcpy(py + (o[0] + j) * s, source, size);
            }
        });
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> cast_kernel(const ir::Node&) {
    return make_kernel([](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        return dispatch(x.dtype, [&](auto in_tag) -> Status {
            using In = typename decltype(in_tag)::type;
            return dispatch(y.dtype, [&](auto out_tag) -> Status {
                using Out = typename decltype(out_tag)::type;
                const In* in = x.as<In>();
                Out* out = y.as<Out>();
                for (std::int64_t i = 0; i < x.numel(); ++i) {
                    out[i] = store<Out>(load(in[i]));
                }
                return ok_status();
            });
        });
    });
}

Result<std::unique_ptr<Kernel>> clamp_kernel(const ir::Node& node) {
    const double low = node.attrs().get_float("min");
    const double high = node.attrs().get_float("max");
    return make_kernel([low, high](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S* in = x.as<S>();
            S* out = y.as<S>();
            for (std::int64_t i = 0; i < x.numel(); ++i) {
                const auto value = load(in[i]);
                if constexpr (std::is_floating_point_v<decltype(value)>) {
                    out[i] = std::isnan(value) ? in[i] : store<S>(std::min(std::max(value, low), high));
                } else {
                    const auto as_double = static_cast<double>(value);
                    if (as_double < low) {
                        out[i] = store<S>(std::ceil(low));
                    } else if (as_double > high) {
                        out[i] = store<S>(std::floor(high));
                    } else {
                        out[i] = in[i];
                    }
                }
            }
            return ok_status();
        });
    });
}

// ------------------------------------------------------------------------------------------------------------------
// Quantization (per-tensor or per-axis integer formats)

struct QuantGeometry {
    std::int64_t channels = 1;  // 1 for per-tensor
    std::int64_t inner = 1;     // elements per channel step
};

QuantGeometry quant_geometry(const TensorView& x, const TensorView& scale, std::int64_t axis) {
    QuantGeometry g;
    if (scale.rank() == 1) {
        const int a = axis_of(axis, x.rank());
        g.channels = x.shape[a];
        g.inner = product(x.shape, a + 1, x.rank());
    }
    return g;
}

Result<std::unique_ptr<Kernel>> quantize_kernel(const ir::Node& node) {
    const std::int64_t axis = node.attrs().get_int("axis");
    return make_kernel([axis](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& scale = context.input(1);
        const TensorView& zero_point = context.input(2);
        const TensorView& y = context.output(0);
        const QuantGeometry g = quant_geometry(x, scale, axis);
        return dispatch(x.dtype, [&](auto x_tag) -> Status {
            using X = typename decltype(x_tag)::type;
            return dispatch(y.dtype, [&](auto y_tag) -> Status {
                using Y = typename decltype(y_tag)::type;
                for (std::int64_t i = 0; i < x.numel(); ++i) {
                    const std::int64_t channel = g.channels == 1 ? 0 : (i / g.inner) % g.channels;
                    const double s = static_cast<double>(load(at<X>(scale, channel)));
                    const double zp =
                        zero_point.present() ? static_cast<double>(load(at<Y>(zero_point, channel))) : 0.0;
                    // x / scale in single precision (double for f64 inputs), like ONNX Runtime.
                    double ratio = static_cast<double>(load(at<X>(x, i))) / s;
                    if constexpr (!std::is_same_v<X, double>) {
                        ratio = static_cast<double>(static_cast<float>(load(at<X>(x, i))) / static_cast<float>(s));
                    }
                    const double q = round_half_even(ratio) + zp;
                    y.as<Y>()[i] = store<Y>(q);
                }
                return ok_status();
            });
        });
    });
}

Status check_dequantize(const ir::Node&) {
    return ok_status();
}

Result<std::unique_ptr<Kernel>> dequantize_kernel(const ir::Node& node) {
    if (is_block_quantized(node.input(0)->dtype())) {
        return make_kernel([](const KernelContext& context) -> Status {
            const TensorView& x = context.input(0);
            const std::int64_t row = x.rank() == 0 ? 1 : x.shape[x.rank() - 1];
            II_TRY(quant::check_row(x.dtype, row));
            quant::dequantize(x.dtype, x.data, context.output(0).as<float>(), x.numel());
            return ok_status();
        });
    }
    const std::int64_t axis = node.attrs().get_int("axis");
    return make_kernel([axis](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& scale = context.input(1);
        const TensorView& zero_point = context.input(2);
        const TensorView& y = context.output(0);
        const QuantGeometry g = quant_geometry(x, scale, axis);
        return dispatch(x.dtype, [&](auto x_tag) -> Status {
            using X = typename decltype(x_tag)::type;
            return dispatch(y.dtype, [&](auto y_tag) -> Status {
                using Y = typename decltype(y_tag)::type;
                for (std::int64_t i = 0; i < x.numel(); ++i) {
                    const std::int64_t channel = g.channels == 1 ? 0 : (i / g.inner) % g.channels;
                    const double s = static_cast<double>(load(at<Y>(scale, channel)));
                    const double zp =
                        zero_point.present() ? static_cast<double>(load(at<X>(zero_point, channel))) : 0.0;
                    y.as<Y>()[i] = store<Y>((static_cast<double>(load(at<X>(x, i))) - zp) * s);
                }
                return ok_status();
            });
        });
    });
}

}  // namespace

void register_elementwise_kernels(backend::KernelBackend& backend) {
    backend.add_kernel("neg", unary_kernel<Neg>);
    backend.add_kernel("abs", unary_kernel<Abs>);
    backend.add_kernel("sign", unary_kernel<Sign>);
    backend.add_kernel("exp", unary_kernel<Exp>);
    backend.add_kernel("log", unary_kernel<Log>);
    backend.add_kernel("sqrt", unary_kernel<Sqrt>);
    backend.add_kernel("rsqrt", unary_kernel<Rsqrt>);
    backend.add_kernel("recip", unary_kernel<Recip>);
    backend.add_kernel("sin", unary_kernel<Sin>);
    backend.add_kernel("cos", unary_kernel<Cos>);
    backend.add_kernel("tanh", unary_kernel<Tanh>);
    backend.add_kernel("sigmoid", unary_kernel<Sigmoid>);
    backend.add_kernel("erf", unary_kernel<Erf>);
    backend.add_kernel("relu", unary_kernel<Relu>);
    backend.add_kernel("floor", unary_kernel<Floor>);
    backend.add_kernel("ceil", unary_kernel<Ceil>);
    backend.add_kernel("round", unary_kernel<Round>);
    backend.add_kernel("not", unary_kernel<Not>);

    backend.add_kernel("add", binary_kernel<AddOp>);
    backend.add_kernel("sub", binary_kernel<SubOp>);
    backend.add_kernel("mul", binary_kernel<MulOp>);
    backend.add_kernel("div", binary_kernel<DivOp>);
    backend.add_kernel("pow", binary_kernel<PowOp>);
    backend.add_kernel("max", binary_kernel<MaxOp>);
    backend.add_kernel("min", binary_kernel<MinOp>);
    backend.add_kernel("mod", mod_kernel);
    backend.add_kernel("eq", binary_kernel<EqOp>);
    backend.add_kernel("ne", binary_kernel<NeOp>);
    backend.add_kernel("lt", binary_kernel<LtOp>);
    backend.add_kernel("le", binary_kernel<LeOp>);
    backend.add_kernel("gt", binary_kernel<GtOp>);
    backend.add_kernel("ge", binary_kernel<GeOp>);
    backend.add_kernel("and", binary_kernel<AndOp>);
    backend.add_kernel("or", binary_kernel<OrOp>);
    backend.add_kernel("xor", binary_kernel<XorOp>);

    backend.add_kernel("where", where_kernel);
    backend.add_kernel("cast", cast_kernel);
    backend.add_kernel("clamp", clamp_kernel);
    backend.add_kernel("quantize", quantize_kernel);
    backend.add_kernel("dequantize", dequantize_kernel, check_dequantize);
}

}  // namespace ii::ref
