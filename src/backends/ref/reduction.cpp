// SPDX-License-Identifier: MIT
// Reference kernels: reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod, argmax, argmin, topk.
#include "backends/ref/common.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace ii::ref {

namespace {

enum class ReduceKind { kSum, kMean, kMax, kMin, kProd };

/// Splits the dimensions of `shape` into kept and reduced ones.
struct ReducePlan {
    Shape kept_shape;
    Strides kept_strides;
    Shape reduced_shape;
    Strides reduced_strides;
};

ReducePlan plan_reduction(const Shape& shape, const std::vector<std::int64_t>& axes) {
    bool reduce[kMaxRank] = {};
    for (const std::int64_t axis : axes) {
        reduce[axis_of(axis, shape.rank())] = true;
    }
    const bool all = axes.empty();
    const Strides strides = contiguous_strides(shape);
    ReducePlan plan;
    for (int d = 0; d < shape.rank(); ++d) {
        if (all || reduce[d]) {
            plan.reduced_shape.push_back(shape[d]);
            plan.reduced_strides.push_back(strides[d]);
        } else {
            plan.kept_shape.push_back(shape[d]);
            plan.kept_strides.push_back(strides[d]);
        }
    }
    return plan;
}

template <ReduceKind kKind>
Result<std::unique_ptr<Kernel>> reduce_kernel(const ir::Node& node) {
    const std::vector<std::int64_t> axes = node.attrs().get_ints("axes");
    return make_kernel([axes](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        const ReducePlan plan = plan_reduction(x.shape, axes);
        const std::int64_t outputs = plan.kept_shape.numel();
        const std::int64_t count = plan.reduced_shape.numel();
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            using C = ComputeOf<S>;
            const S* in = x.as<S>();
            S* out = y.as<S>();
            std::int64_t kept_index[kMaxRank] = {};
            std::int64_t reduced_index[kMaxRank] = {};
            for (std::int64_t o = 0; o < outputs; ++o) {
                unravel_index(o, plan.kept_shape, kept_index);
                const std::int64_t base = offset_of(kept_index, plan.kept_strides);
                C acc{};
                if constexpr (kKind == ReduceKind::kProd) {
                    acc = C{1};
                }
                if constexpr ((kKind == ReduceKind::kMax || kKind == ReduceKind::kMin) && !kIsBoolStorage<S>) {
                    // The value of an empty reduction: the identity of max / min.
                    if constexpr (std::is_floating_point_v<C>) {
                        acc = kKind == ReduceKind::kMax ? -std::numeric_limits<double>::infinity()
                                                        : std::numeric_limits<double>::infinity();
                    } else {
                        acc = kKind == ReduceKind::kMax ? static_cast<C>(std::numeric_limits<S>::lowest())
                                                        : static_cast<C>(std::numeric_limits<S>::max());
                    }
                }
                bool nan = false;
                for (std::int64_t r = 0; r < count; ++r) {
                    unravel_index(r, plan.reduced_shape, reduced_index);
                    const C v = load(in[base + offset_of(reduced_index, plan.reduced_strides)]);
                    if constexpr (kKind == ReduceKind::kSum || kKind == ReduceKind::kMean) {
                        if constexpr (std::is_floating_point_v<C>) {
                            acc += v;
                        } else if constexpr (!std::is_same_v<C, bool>) {
                            acc = wrap_add(acc, v);
                        }
                    } else if constexpr (kKind == ReduceKind::kProd) {
                        if constexpr (std::is_floating_point_v<C>) {
                            acc *= v;
                        } else if constexpr (!std::is_same_v<C, bool>) {
                            acc = wrap_mul(acc, v);
                        }
                    } else {
                        if constexpr (std::is_floating_point_v<C>) {
                            nan = nan || std::isnan(v);
                        }
                        const bool better = kKind == ReduceKind::kMax ? v > acc : v < acc;
                        if (better) {
                            acc = v;
                        }
                    }
                }
                if constexpr (kKind == ReduceKind::kMean) {
                    out[o] = store<S>(count == 0 ? std::numeric_limits<double>::quiet_NaN()
                                                 : static_cast<double>(acc) / static_cast<double>(count));
                } else if constexpr (std::is_floating_point_v<C>) {
                    out[o] = store<S>(nan ? std::numeric_limits<double>::quiet_NaN() : acc);
                } else {
                    out[o] = store<S>(acc);
                }
            }
            return ok_status();
        });
    });
}

template <bool kMax>
Result<std::unique_ptr<Kernel>> arg_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const bool select_last = node.attrs().get_bool("select_last_index");
    return make_kernel([axis_attr, select_last](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        auto* out = context.output(0).as<std::int64_t>();
        const int axis = axis_of(axis_attr, x.rank());
        const std::int64_t outer = product(x.shape, 0, axis);
        const std::int64_t size = x.shape[axis];
        const std::int64_t inner = product(x.shape, axis + 1, x.rank());
        if (size == 0 && outer * inner > 0) {
            return errors::invalid_argument("{} over an empty axis", kMax ? "argmax" : "argmin");
        }
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            using C = ComputeOf<S>;
            const S* in = x.as<S>();
            for (std::int64_t o = 0; o < outer; ++o) {
                for (std::int64_t i = 0; i < inner; ++i) {
                    std::int64_t best_index = 0;
                    C best = load(in[o * size * inner + i]);
                    bool best_nan = false;
                    if constexpr (std::is_floating_point_v<C>) {
                        best_nan = std::isnan(best);
                    }
                    for (std::int64_t k = 1; k < size && !best_nan; ++k) {
                        const C v = load(in[(o * size + k) * inner + i]);
                        if constexpr (std::is_floating_point_v<C>) {
                            if (std::isnan(v)) {  // the first NaN wins
                                best_index = k;
                                best_nan = true;
                                break;
                            }
                        }
                        const bool better = kMax ? v > best : v < best;
                        if (better || (select_last && v == best)) {
                            best = v;
                            best_index = k;
                        }
                    }
                    out[o * inner + i] = best_index;
                }
            }
            return ok_status();
        });
    });
}

Result<std::unique_ptr<Kernel>> topk_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const bool largest = node.attrs().get_bool("largest", true);
    return make_kernel([axis_attr, largest](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& values = context.output(0);
        auto* indices = context.output(1).as<std::int64_t>();
        const int axis = axis_of(axis_attr, x.rank());
        const std::int64_t outer = product(x.shape, 0, axis);
        const std::int64_t size = x.shape[axis];
        const std::int64_t inner = product(x.shape, axis + 1, x.rank());
        const std::int64_t k = values.shape[axis];
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            using C = ComputeOf<S>;
            const S* in = x.as<S>();
            S* out = values.as<S>();
            std::vector<std::int64_t> order(static_cast<std::size_t>(size));
            for (std::int64_t o = 0; o < outer; ++o) {
                for (std::int64_t i = 0; i < inner; ++i) {
                    const auto value = [&](std::int64_t j) { return load(in[(o * size + j) * inner + i]); };
                    // "a before b": NaN is greater than every number; equal values keep index order.
                    const auto greater = [&](C a, C b) {
                        if constexpr (std::is_floating_point_v<C>) {
                            if (std::isnan(a) || std::isnan(b)) {
                                return std::isnan(a) && !std::isnan(b);
                            }
                        }
                        return a > b;
                    };
                    std::iota(order.begin(), order.end(), std::int64_t{0});
                    std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
                        return largest ? greater(value(a), value(b)) : greater(value(b), value(a));
                    });
                    for (std::int64_t j = 0; j < k; ++j) {
                        const std::int64_t source = order[static_cast<std::size_t>(j)];
                        out[(o * k + j) * inner + i] = in[(o * size + source) * inner + i];
                        indices[(o * k + j) * inner + i] = source;
                    }
                }
            }
            return ok_status();
        });
    });
}

}  // namespace

void register_reduction_kernels(backend::KernelBackend& backend) {
    backend.add_kernel("reduce_sum", reduce_kernel<ReduceKind::kSum>);
    backend.add_kernel("reduce_mean", reduce_kernel<ReduceKind::kMean>);
    backend.add_kernel("reduce_max", reduce_kernel<ReduceKind::kMax>);
    backend.add_kernel("reduce_min", reduce_kernel<ReduceKind::kMin>);
    backend.add_kernel("reduce_prod", reduce_kernel<ReduceKind::kProd>);
    backend.add_kernel("argmax", arg_kernel<true>);
    backend.add_kernel("argmin", arg_kernel<false>);
    backend.add_kernel("topk", topk_kernel);
}

}  // namespace ii::ref
