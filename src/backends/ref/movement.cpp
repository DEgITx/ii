// SPDX-License-Identifier: MIT
// Reference kernels: reshape, permute, slice, expand, concat, split, pad, gather, scatter, tile, flip, cumsum.
#include "backends/ref/common.h"
#include "ops/infer_util.h"

#include <string>
#include <vector>

namespace ii::ref {

namespace {

/// Copies element `src_index` of `src` to element `dst_index` of `dst` (dtype-agnostic).
inline void copy_element(const TensorView& dst, std::int64_t dst_index, const TensorView& src, std::int64_t src_index,
                         std::size_t size) noexcept {
    std::memcpy(static_cast<std::byte*>(dst.data) + dst_index * static_cast<std::int64_t>(size),
                static_cast<const std::byte*>(src.data) + src_index * static_cast<std::int64_t>(size), size);
}

/// Generic gather-style copy: every output element reads input element `map(output index tuple)`.
template <class Map>
void gather_copy(const TensorView& x, const TensorView& y, Map&& map) {
    const std::size_t size = element_size(y.dtype);
    std::int64_t index[kMaxRank] = {};
    for (std::int64_t i = 0; i < y.numel(); ++i) {
        unravel_index(i, y.shape, index);
        copy_element(y, i, x, map(index), size);
    }
}

Result<std::unique_ptr<Kernel>> reshape_kernel(const ir::Node&) {
    return make_kernel([](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        if (x.data != y.data) {
            copy_elements(y.data, x.data, x.numel(), element_size(x.dtype));
        }
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> permute_kernel(const ir::Node& node) {
    const std::vector<std::int64_t> perm = node.attrs().get_ints("perm");
    return make_kernel([perm](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const Strides in_strides = contiguous_strides(x.shape);
        gather_copy(x, context.output(0), [&](const std::int64_t* index) {
            std::int64_t offset = 0;
            for (std::size_t d = 0; d < perm.size(); ++d) {
                offset += index[d] * in_strides[static_cast<int>(perm[d])];
            }
            return offset;
        });
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> slice_kernel(const ir::Node& node) {
    const int rank = node.input(0)->rank();
    std::vector<std::int64_t> axes = node.attrs().get_ints("axes");
    const std::vector<std::int64_t> starts = node.attrs().get_ints("starts");
    const std::vector<std::int64_t> ends = node.attrs().get_ints("ends");
    std::vector<std::int64_t> steps = node.attrs().get_ints("steps");
    if (steps.empty()) {
        steps.assign(axes.size(), 1);
    }
    for (std::int64_t& axis : axes) {
        axis = axis_of(axis, rank);
    }
    return make_kernel([axes, starts, ends, steps](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        Shape start = Shape::filled(x.rank(), 0);
        Shape step = Shape::filled(x.rank(), 1);
        for (std::size_t i = 0; i < axes.size(); ++i) {
            const int axis = static_cast<int>(axes[i]);
            const ops::SliceRange range = ops::resolve_slice(x.shape[axis], starts[i], ends[i], steps[i]);
            start[axis] = range.start;
            step[axis] = range.step;
        }
        const Strides in_strides = contiguous_strides(x.shape);
        gather_copy(x, context.output(0), [&](const std::int64_t* index) {
            std::int64_t offset = 0;
            for (int d = 0; d < x.rank(); ++d) {
                offset += (start[d] + index[d] * step[d]) * in_strides[d];
            }
            return offset;
        });
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> expand_kernel(const ir::Node&) {
    return make_kernel([](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        const Strides strides = broadcast_strides(x.shape, y.shape);
        gather_copy(x, y, [&](const std::int64_t* index) { return offset_of(index, strides); });
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> concat_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel([axis_attr](const KernelContext& context) -> Status {
        const TensorView& y = context.output(0);
        const int axis = axis_of(axis_attr, y.rank());
        const std::int64_t outer = product(y.shape, 0, axis);
        const std::int64_t inner = product(y.shape, axis + 1, y.rank());
        const std::size_t size = element_size(y.dtype);
        std::int64_t offset = 0;  // along the axis
        for (const TensorView& x : context.inputs) {
            const std::int64_t chunk = x.shape[axis] * inner;
            for (std::int64_t o = 0; o < outer; ++o) {
                copy_elements(static_cast<std::byte*>(y.data) +
                                  (o * y.shape[axis] * inner + offset * inner) * static_cast<std::int64_t>(size),
                              static_cast<const std::byte*>(x.data) + o * chunk * static_cast<std::int64_t>(size),
                              chunk, size);
            }
            offset += x.shape[axis];
        }
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> split_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel([axis_attr](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const int axis = axis_of(axis_attr, x.rank());
        const std::int64_t outer = product(x.shape, 0, axis);
        const std::int64_t inner = product(x.shape, axis + 1, x.rank());
        const std::size_t size = element_size(x.dtype);
        std::int64_t offset = 0;
        for (const TensorView& y : context.outputs) {
            const std::int64_t chunk = y.shape[axis] * inner;
            for (std::int64_t o = 0; o < outer; ++o) {
                copy_elements(static_cast<std::byte*>(y.data) + o * chunk * static_cast<std::int64_t>(size),
                              static_cast<const std::byte*>(x.data) +
                                  (o * x.shape[axis] * inner + offset * inner) * static_cast<std::int64_t>(size),
                              chunk, size);
            }
            offset += y.shape[axis];
        }
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> pad_kernel(const ir::Node& node) {
    const std::vector<std::int64_t> pads = node.attrs().get_ints("pads");
    const std::string mode(node.attrs().get_string("mode"));
    const double value = node.attrs().get_float("value");
    return make_kernel([pads, mode, value](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        const int rank = x.rank();
        const Strides in_strides = contiguous_strides(x.shape);
        const std::size_t size = element_size(x.dtype);
        // The fill value, converted once.
        std::byte fill[8] = {};
        II_TRY(dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            const S converted = store<S>(value);
            std::memcpy(fill, &converted, sizeof(S));
            return ok_status();
        }));
        std::int64_t index[kMaxRank] = {};
        for (std::int64_t i = 0; i < y.numel(); ++i) {
            unravel_index(i, y.shape, index);
            std::int64_t offset = 0;
            bool inside = true;
            for (int d = 0; d < rank; ++d) {
                std::int64_t pos = index[d] - pads[static_cast<std::size_t>(d)];
                const std::int64_t n = x.shape[d];
                if (pos < 0 || pos >= n) {
                    if (mode == "constant") {
                        inside = false;
                        break;
                    }
                    if (mode == "edge") {
                        pos = pos < 0 ? 0 : n - 1;
                    } else {  // reflect: mirror without repeating the edge
                        const std::int64_t period = 2 * (n - 1);
                        pos = period == 0 ? 0 : ((pos % period) + period) % period;
                        if (pos >= n) {
                            pos = period - pos;
                        }
                    }
                }
                offset += pos * in_strides[d];
            }
            auto* dst = static_cast<std::byte*>(y.data) + i * static_cast<std::int64_t>(size);
            if (inside) {
                std::memcpy(dst, static_cast<const std::byte*>(x.data) + offset * static_cast<std::int64_t>(size),
                            size);
            } else {
                std::memcpy(dst, fill, size);
            }
        }
        return ok_status();
    });
}

/// Reads index `i` of an i32 / i64 indices tensor.
inline std::int64_t read_index(const TensorView& indices, std::int64_t i) noexcept {
    return indices.dtype == DType::kI32 ? indices.as<const std::int32_t>()[i] : indices.as<const std::int64_t>()[i];
}

Result<std::unique_ptr<Kernel>> gather_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel([axis_attr](const KernelContext& context) -> Status {
        const TensorView& data = context.input(0);
        const TensorView& indices = context.input(1);
        const TensorView& y = context.output(0);
        const int axis = axis_of(axis_attr, data.rank());
        const std::int64_t outer = product(data.shape, 0, axis);
        const std::int64_t size_axis = data.shape[axis];
        const std::int64_t inner = product(data.shape, axis + 1, data.rank());
        const std::int64_t count = indices.numel();
        // Whole slices of `inner` elements are copied, so block-quantized rows work too.
        const auto slice_bytes = static_cast<std::int64_t>(storage_size(data.dtype, inner));
        for (std::int64_t j = 0; j < count; ++j) {
            std::int64_t index = read_index(indices, j);
            if (index < 0) {
                index += size_axis;
            }
            if (index < 0 || index >= size_axis) {
                return errors::out_of_range("gather index {} is out of range for axis size {}", read_index(indices, j),
                                            size_axis);
            }
            for (std::int64_t o = 0; o < outer; ++o) {
                copy_elements(static_cast<std::byte*>(y.data) + (o * count + j) * slice_bytes,
                              static_cast<const std::byte*>(data.data) + (o * size_axis + index) * slice_bytes,
                              slice_bytes, 1);
            }
        }
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> scatter_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const std::string reduction(node.attrs().get_string("reduction"));
    return make_kernel([axis_attr, reduction](const KernelContext& context) -> Status {
        const TensorView& data = context.input(0);
        const TensorView& indices = context.input(1);
        const TensorView& updates = context.input(2);
        const TensorView& y = context.output(0);
        const int axis = axis_of(axis_attr, data.rank());
        copy_elements(y.data, data.data, data.numel(), element_size(data.dtype));
        const Strides out_strides = contiguous_strides(y.shape);
        return dispatch(data.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            using C = ComputeOf<S>;
            S* py = y.as<S>();
            const S* pu = updates.as<S>();
            std::int64_t index[kMaxRank] = {};
            for (std::int64_t i = 0; i < indices.numel(); ++i) {
                unravel_index(i, indices.shape, index);
                std::int64_t target = read_index(indices, i);
                if (target < 0) {
                    target += data.shape[axis];
                }
                if (target < 0 || target >= data.shape[axis]) {
                    return errors::out_of_range("scatter index {} is out of range for axis size {}",
                                                read_index(indices, i), data.shape[axis]);
                }
                index[axis] = target;
                // Indices may be smaller than data along non-axis dimensions; they address data directly.
                S& out = py[offset_of(index, out_strides)];
                const C u = load(pu[i]);
                const C current = load(out);
                if (reduction == "none") {
                    out = pu[i];
                } else if constexpr (!std::is_same_v<C, bool>) {
                    C combined = current;
                    if (reduction == "add") {
                        if constexpr (std::is_floating_point_v<C>) {
                            combined = current + u;
                        } else {
                            combined = wrap_add(current, u);
                        }
                    } else if (reduction == "mul") {
                        if constexpr (std::is_floating_point_v<C>) {
                            combined = current * u;
                        } else {
                            combined = wrap_mul(current, u);
                        }
                    } else if (reduction == "max") {
                        combined = u > current ? u : current;
                    } else {
                        combined = u < current ? u : current;
                    }
                    out = store<S>(combined);
                }
            }
            return ok_status();
        });
    });
}

Result<std::unique_ptr<Kernel>> tile_kernel(const ir::Node&) {
    return make_kernel([](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const Strides in_strides = contiguous_strides(x.shape);
        gather_copy(x, context.output(0), [&](const std::int64_t* index) {
            std::int64_t offset = 0;
            for (int d = 0; d < x.rank(); ++d) {
                offset += (index[d] % x.shape[d]) * in_strides[d];
            }
            return offset;
        });
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> flip_kernel(const ir::Node& node) {
    const int rank = node.input(0)->rank();
    bool flip[kMaxRank] = {};
    for (const std::int64_t axis : node.attrs().get_ints("axes")) {
        flip[axis_of(axis, rank)] = true;
    }
    std::vector<bool> flags(flip, flip + rank);
    return make_kernel([flags](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const Strides in_strides = contiguous_strides(x.shape);
        gather_copy(x, context.output(0), [&](const std::int64_t* index) {
            std::int64_t offset = 0;
            for (int d = 0; d < x.rank(); ++d) {
                const std::int64_t pos = flags[static_cast<std::size_t>(d)] ? x.shape[d] - 1 - index[d] : index[d];
                offset += pos * in_strides[d];
            }
            return offset;
        });
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> cumsum_kernel(const ir::Node& node) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const bool exclusive = node.attrs().get_bool("exclusive");
    const bool reverse = node.attrs().get_bool("reverse");
    return make_kernel([axis_attr, exclusive, reverse](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        const int axis = axis_of(axis_attr, x.rank());
        const std::int64_t outer = product(x.shape, 0, axis);
        const std::int64_t n = x.shape[axis];
        const std::int64_t inner = product(x.shape, axis + 1, x.rank());
        return dispatch(x.dtype, [&](auto tag) -> Status {
            using S = typename decltype(tag)::type;
            using C = ComputeOf<S>;
            const S* px = x.as<S>();
            S* py = y.as<S>();
            for (std::int64_t o = 0; o < outer; ++o) {
                for (std::int64_t i = 0; i < inner; ++i) {
                    C acc{};
                    for (std::int64_t step = 0; step < n; ++step) {
                        const std::int64_t k = reverse ? n - 1 - step : step;
                        const std::int64_t index = (o * n + k) * inner + i;
                        const C v = load(px[index]);
                        if (exclusive) {
                            py[index] = store<S>(acc);
                        }
                        if constexpr (std::is_floating_point_v<C>) {
                            acc += v;
                        } else if constexpr (!std::is_same_v<C, bool>) {
                            acc = wrap_add(acc, v);
                        }
                        if (!exclusive) {
                            py[index] = store<S>(acc);
                        }
                    }
                }
            }
            return ok_status();
        });
    });
}

}  // namespace

void register_movement_kernels(backend::KernelBackend& backend) {
    backend.add_kernel("reshape", reshape_kernel);
    backend.add_kernel("permute", permute_kernel);
    backend.add_kernel("slice", slice_kernel);
    backend.add_kernel("expand", expand_kernel);
    backend.add_kernel("concat", concat_kernel);
    backend.add_kernel("split", split_kernel);
    backend.add_kernel("pad", pad_kernel);
    backend.add_kernel("gather", gather_kernel);
    backend.add_kernel("scatter", scatter_kernel);
    backend.add_kernel("tile", tile_kernel);
    backend.add_kernel("flip", flip_kernel);
    backend.add_kernel("cumsum", cumsum_kernel);
}

}  // namespace ii::ref
