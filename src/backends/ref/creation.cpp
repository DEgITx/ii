// SPDX-License-Identifier: MIT
// Reference kernels: iota, fill, random_uniform, random_normal.
#include "backends/ref/common.h"
#include "ops/philox.h"

namespace ii::ref {

namespace {

Result<std::unique_ptr<Kernel>> iota_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const double start = node.attrs().get_float("start");
    const double step = node.attrs().get_float("step");
    return make_kernel([axis_attr, start, step](const KernelContext& context) -> Status {
        const TensorView& y = context.output(0);
        const int axis = axis_of(axis_attr, y.rank());
        const std::int64_t size = y.shape[axis];
        const std::int64_t inner = product(y.shape, axis + 1, y.rank());
        return dispatch(y.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            S* out = y.as<S>();
            for (std::int64_t i = 0; i < y.numel(); ++i) {
                const std::int64_t position = size == 0 ? 0 : (i / inner) % size;
                out[i] = store<S>(start + step * static_cast<double>(position));
            }
            return ok_status();
        });
    });
}

Result<std::unique_ptr<Kernel>> fill_kernel(const ir::Node& node) {
    const double value = node.attrs().get_float("value");
    return make_kernel([value](const KernelContext& context) -> Status {
        const TensorView& y = context.output(0);
        return dispatch(y.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S converted = store<S>(value);
            S* out = y.as<S>();
            for (std::int64_t i = 0; i < y.numel(); ++i) {
                out[i] = converted;
            }
            return ok_status();
        });
    });
}

template <bool kNormal>
Result<std::unique_ptr<Kernel>> random_kernel(const ir::Node& node) {
    const auto seed = static_cast<std::uint64_t>(node.attrs().get_int("seed"));
    const double a = node.attrs().get_float(kNormal ? "mean" : "low");
    const double b = node.attrs().get_float(kNormal ? "stddev" : "high");
    return make_kernel([seed, a, b](const KernelContext& context) -> Status {
        const TensorView& y = context.output(0);
        return dispatch(y.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            S* out = y.as<S>();
            for (std::int64_t i = 0; i < y.numel(); ++i) {
                const auto index = static_cast<std::uint64_t>(i);
                const double value = kNormal ? a + b * ops::normal_element(index, seed)
                                             : a + (b - a) * ops::uniform_element(index, seed);
                out[i] = store<S>(value);
            }
            return ok_status();
        });
    });
}

}  // namespace

void register_creation_kernels(backend::KernelBackend& backend) {
    backend.add_kernel("iota", iota_kernel);
    backend.add_kernel("fill", fill_kernel);
    backend.add_kernel("random_uniform", random_kernel<false>);
    backend.add_kernel("random_normal", random_kernel<true>);
}

}  // namespace ii::ref
