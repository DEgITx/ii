// SPDX-License-Identifier: MIT
// Reference kernels: resize.
#include "backends/ref/common.h"
#include "ops/infer_util.h"

#include <vector>

namespace ii::ref {

namespace {

Result<std::unique_ptr<Kernel>> resize_kernel(const ir::Node& node) {
    const std::vector<double> scales = node.attrs().get_floats("scales");
    const std::string mode(node.attrs().get_string("mode"));
    const std::string coordinate_mode(node.attrs().get_string("coordinate_mode"));
    const std::string nearest_mode(node.attrs().get_string("nearest_mode"));
    const double cubic_coeff_a = node.attrs().get_float("cubic_coeff_a");
    const bool exclude_outside = node.attrs().get_bool("exclude_outside");
    return make_kernel([=](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        const int rank = x.rank();
        const ops::ResizeMode resize_mode{mode, coordinate_mode, nearest_mode, cubic_coeff_a, exclude_outside};
        // Taps per dimension; dimensions with identity mapping use a single tap.
        std::vector<ops::ResizeTaps> taps(static_cast<std::size_t>(rank));
        for (int d = 0; d < rank; ++d) {
            const double scale = scales.empty() ? 0.0 : scales[static_cast<std::size_t>(d)];
            const bool identity = x.shape[d] == y.shape[d] && (scale == 0 || scale == 1);
            if (identity) {
                ops::ResizeTaps& t = taps[static_cast<std::size_t>(d)];
                t.count = 1;
                for (std::int64_t i = 0; i < y.shape[d]; ++i) {
                    t.index.push_back(i);
                    t.weight.push_back(1.0);
                }
            } else {
                taps[static_cast<std::size_t>(d)] = ops::resize_taps(x.shape[d], y.shape[d], scale, resize_mode);
            }
        }
        if (y.numel() == 0) {
            return ok_status();
        }
        if (x.numel() == 0) {
            return errors::invalid_argument("'resize' of an empty tensor to a non-empty one");
        }
        const Strides in_strides = contiguous_strides(x.shape);
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            using C = ComputeOf<S>;
            const S* px = x.as<S>();
            S* py = y.as<S>();
            std::int64_t index[kMaxRank] = {};
            int tap[kMaxRank] = {};
            for (std::int64_t i = 0; i < y.numel(); ++i) {
                unravel_index(i, y.shape, index);
                if constexpr (std::is_floating_point_v<C>) {
                    double sum = 0;
                    // Iterate over the cartesian product of the taps of every dimension.
                    std::fill(tap, tap + rank, 0);
                    while (true) {
                        double weight = 1;
                        std::int64_t offset = 0;
                        for (int d = 0; d < rank; ++d) {
                            const ops::ResizeTaps& t = taps[static_cast<std::size_t>(d)];
                            const std::size_t k = static_cast<std::size_t>(index[d] * t.count + tap[d]);
                            weight *= t.weight[k];
                            offset += t.index[k] * in_strides[d];
                        }
                        if (weight != 0) {
                            sum += weight * load(px[offset]);
                        }
                        int d = rank - 1;
                        while (d >= 0 && ++tap[d] == taps[static_cast<std::size_t>(d)].count) {
                            tap[d] = 0;
                            --d;
                        }
                        if (d < 0) {
                            break;
                        }
                    }
                    py[i] = store<S>(sum);
                } else {
                    // Non-float types are only resized with nearest (one tap per dimension).
                    std::int64_t offset = 0;
                    for (int d = 0; d < rank; ++d) {
                        const ops::ResizeTaps& t = taps[static_cast<std::size_t>(d)];
                        offset += t.index[static_cast<std::size_t>(index[d] * t.count)] * in_strides[d];
                    }
                    py[i] = px[offset];
                }
            }
            return ok_status();
        });
    });
}

}  // namespace

void register_image_kernels(backend::KernelBackend& backend) {
    backend.add_kernel("resize", resize_kernel);
}

}  // namespace ii::ref
