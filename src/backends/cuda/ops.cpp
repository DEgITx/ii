// SPDX-License-Identifier: MIT
// ii-cuda operations backed by the nvcc-compiled kernels: reductions, softmax and normalizations, data movement,
// pooling and resize.
#include "backends/cuda/cuda_backend.h"

#include "ops/infer_util.h"

#include <ii/half.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace ii::cuda {

namespace {

int axis_of(std::int64_t axis, int rank) {
    return static_cast<int>(axis < 0 ? axis + rank : axis);
}

Status check_float(const ir::Node& node) {
    return check_dtypes(node, {DType::kF32, DType::kF16, DType::kBF16});
}

Status check_numeric(const ir::Node& node) {
    return check_dtypes(node, {DType::kF32, DType::kF16, DType::kBF16, DType::kI32, DType::kI64});
}

Status check_rank(const ir::Node& node) {
    for (const ir::Value* value : node.inputs()) {
        if (value != nullptr && value->rank() > kMaxRank) {
            return errors::unsupported("rank above {}", kMaxRank);
        }
    }
    for (const ir::Value* value : node.outputs()) {
        if (value->rank() > kMaxRank) {
            return errors::unsupported("rank above {}", kMaxRank);
        }
    }
    return ok_status();
}

/// Plain dtypes of 1, 2, 4 or 8 bytes (data movement).
Status check_plain(const ir::Node& node) {
    II_TRY(check_rank(node));
    for (const ir::Value* value : node.outputs()) {
        if (!ops::is_plain(value->dtype())) {
            return errors::unsupported("'{}' of {}", node.op_name(), to_string(value->dtype()));
        }
    }
    return ok_status();
}

// --- reductions -------------------------------------------------------------------------------------------------

template <ReduceOp kOp>
Result<std::unique_ptr<Kernel>> create_reduce(const ir::Node& node, const CudaDevice& device) {
    const std::vector<std::int64_t> axes = node.attrs().get_ints("axes");
    return make_kernel(device, [axes](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        bool reduce[kMaxRank] = {};
        for (const std::int64_t axis : axes) {
            reduce[axis_of(axis, x.rank())] = true;
        }
        std::int64_t strides[kMaxRank] = {};
        contiguous_strides(x.shape, strides);
        ReduceParams p;
        p.outputs = y.numel();
        p.count = 1;
        for (int d = 0; d < x.rank(); ++d) {
            if (axes.empty() || reduce[d]) {
                p.reduced_dims[p.reduced_rank] = x.shape[d];
                p.reduced_strides[p.reduced_rank++] = strides[d];
                p.count *= x.shape[d];
            } else {
                p.kept_dims[p.kept_rank] = x.shape[d];
                p.kept_strides[p.kept_rank++] = strides[d];
            }
        }
        return launched(cuda::reduce(y.data, x.data, *scalar_of(x.dtype), kOp, p, stream), "reduce");
    });
}

Status check_reduce(const ir::Node& node) {
    II_TRY(check_rank(node));
    return check_numeric(node);
}

template <bool kMax>
Result<std::unique_ptr<Kernel>> create_arg(const ir::Node& node, const CudaDevice& device) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const bool select_last = node.attrs().get_bool("select_last_index");
    return make_kernel(device, [axis_attr, select_last](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const int axis = axis_of(axis_attr, x.rank());
        ArgParams p;
        p.outer = 1;
        p.inner = 1;
        for (int d = 0; d < axis; ++d) {
            p.outer *= x.shape[d];
        }
        p.size = x.shape[axis];
        for (int d = axis + 1; d < x.rank(); ++d) {
            p.inner *= x.shape[d];
        }
        p.max = kMax;
        p.select_last = select_last;
        if (p.size == 0 && p.outer * p.inner > 0) {
            return errors::invalid_argument("{} over an empty axis", kMax ? "argmax" : "argmin");
        }
        return launched(arg_reduce(context.output(0).as<std::int64_t>(), x.data, *scalar_of(x.dtype), p, stream),
                        "arg_reduce");
    });
}

Status check_arg(const ir::Node& node) {
    II_TRY(check_rank(node));
    return check_numeric(node);
}

// --- softmax, normalization -------------------------------------------------------------------------------------

template <bool kLog>
Result<std::unique_ptr<Kernel>> create_softmax(const ir::Node& node, const CudaDevice& device) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel(
        device,
        [axis_attr](const KernelContext& context, cudaStream_t stream) -> Status {
            const TensorView& x = context.input(0);
            const int axis = axis_of(axis_attr, x.rank());
            std::int64_t outer = 1;
            std::int64_t inner = 1;
            for (int d = 0; d < axis; ++d) {
                outer *= x.shape[d];
            }
            for (int d = axis + 1; d < x.rank(); ++d) {
                inner *= x.shape[d];
            }
            return launched(
                softmax(context.output(0).data, x.data, *scalar_of(x.dtype), outer, x.shape[axis], inner, kLog, stream),
                "softmax");
        },
        nullptr, 0);
}

template <bool kRms>
Result<std::unique_ptr<Kernel>> create_norm(const ir::Node& node, const CudaDevice& device) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const auto epsilon = static_cast<float>(node.attrs().get_float("epsilon"));
    return make_kernel(
        device,
        [axis_attr, epsilon](const KernelContext& context, cudaStream_t stream) -> Status {
            const TensorView& x = context.input(0);
            const TensorView& scale = context.input(1);
            const TensorView& bias = context.input(2);
            const int axis = axis_of(axis_attr, x.rank());
            std::int64_t rows = 1;
            std::int64_t size = 1;
            for (int d = 0; d < axis; ++d) {
                rows *= x.shape[d];
            }
            for (int d = axis; d < x.rank(); ++d) {
                size *= x.shape[d];
            }
            return launched(normalize(context.output(0).data, x.data, scale.present() ? scale.data : nullptr,
                                      bias.present() ? bias.data : nullptr, *scalar_of(x.dtype), rows, size, epsilon,
                                      kRms, stream),
                            "normalize");
        },
        nullptr, 0);
}

Status check_norm(const ir::Node& node) {
    II_TRY(check_float(node));
    for (const ir::Value* input : node.inputs()) {
        if (input != nullptr && input->dtype() != node.input(0)->dtype()) {
            return errors::unsupported("'{}' of mixed dtypes", node.op_name());
        }
    }
    return ok_status();
}

// --- data movement ----------------------------------------------------------------------------------------------

Status copy(const CopyParams& p, void* dst, const void* src, DType dtype, cudaStream_t stream) {
    return launched(copy_strided(dst, src, static_cast<int>(element_size(dtype)), p, stream), "copy_strided");
}

const std::byte* offset(const void* data, std::int64_t elements, DType dtype) {
    return static_cast<const std::byte*>(data) + elements * static_cast<std::int64_t>(element_size(dtype));
}

Result<std::unique_ptr<Kernel>> create_reshape(const ir::Node&, const CudaDevice& device) {
    return make_kernel(
        device,
        [](const KernelContext& context, cudaStream_t stream) -> Status {
            const TensorView& x = context.input(0);
            const TensorView& y = context.output(0);
            if (x.data != y.data && y.nbytes() > 0) {
                II_CUDA_TRY(cudaMemcpyAsync(y.data, x.data, y.nbytes(), cudaMemcpyDeviceToDevice, stream));
            }
            return ok_status();
        },
        nullptr, 0);
}

Status check_reshape(const ir::Node& node) {
    if (is_block_quantized(node.input(0)->dtype())) {
        return ok_status();  // a copy of bytes
    }
    return check_plain(node);
}

Result<std::unique_ptr<Kernel>> create_permute(const ir::Node& node, const CudaDevice& device) {
    const std::vector<std::int64_t> perm = node.attrs().get_ints("perm");
    return make_kernel(device, [perm](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        std::int64_t in_strides[kMaxRank] = {};
        contiguous_strides(x.shape, in_strides);
        CopyParams p;
        p.rank = y.rank();
        contiguous_strides(y.shape, p.dst_strides);
        for (int d = 0; d < p.rank; ++d) {
            p.dims[d] = y.shape[d];
            p.src_strides[d] = in_strides[perm[static_cast<std::size_t>(d)]];
        }
        return copy(p, y.data, x.data, x.dtype, stream);
    });
}

Result<std::unique_ptr<Kernel>> create_slice(const ir::Node& node, const CudaDevice& device) {
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
    return make_kernel(
        device, [axes, starts, ends, steps](const KernelContext& context, cudaStream_t stream) -> Status {
            const TensorView& x = context.input(0);
            const TensorView& y = context.output(0);
            std::int64_t in_strides[kMaxRank] = {};
            contiguous_strides(x.shape, in_strides);
            CopyParams p;
            p.rank = y.rank();
            contiguous_strides(y.shape, p.dst_strides);
            std::int64_t base = 0;
            for (int d = 0; d < p.rank; ++d) {
                p.dims[d] = y.shape[d];
                p.src_strides[d] = in_strides[d];
            }
            for (std::size_t i = 0; i < axes.size(); ++i) {
                const int axis = static_cast<int>(axes[i]);
                const ops::SliceRange range = ops::resolve_slice(x.shape[axis], starts[i], ends[i], steps[i]);
                base += range.start * in_strides[axis];
                p.src_strides[axis] = in_strides[axis] * range.step;
            }
            if (y.numel() == 0) {
                return ok_status();
            }
            return copy(p, y.data, offset(x.data, base, x.dtype), x.dtype, stream);
        });
}

Result<std::unique_ptr<Kernel>> create_expand(const ir::Node&, const CudaDevice& device) {
    return make_kernel(device, [](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        CopyParams p;
        p.rank = y.rank();
        contiguous_strides(y.shape, p.dst_strides);
        std::int64_t stride = 1;
        for (int d = y.rank() - 1; d >= 0; --d) {
            p.dims[d] = y.shape[d];
            const int xd = d - (y.rank() - x.rank());
            if (xd >= 0) {
                p.src_strides[d] = x.shape[xd] == 1 ? 0 : stride;
                stride *= x.shape[xd];
            }
        }
        return copy(p, y.data, x.data, x.dtype, stream);
    });
}

Result<std::unique_ptr<Kernel>> create_concat(const ir::Node& node, const CudaDevice& device) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel(device, [axis_attr](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& y = context.output(0);
        const int axis = axis_of(axis_attr, y.rank());
        std::int64_t y_strides[kMaxRank] = {};
        contiguous_strides(y.shape, y_strides);
        std::int64_t position = 0;
        for (const TensorView& x : context.inputs) {
            CopyParams p;
            p.rank = x.rank();
            contiguous_strides(x.shape, p.src_strides);
            for (int d = 0; d < p.rank; ++d) {
                p.dims[d] = x.shape[d];
                p.dst_strides[d] = y_strides[d];
            }
            if (x.numel() > 0) {
                II_TRY(copy(p, const_cast<std::byte*>(offset(y.data, position * y_strides[axis], y.dtype)), x.data,
                            x.dtype, stream));
            }
            position += x.shape[axis];
        }
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> create_split(const ir::Node& node, const CudaDevice& device) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel(device, [axis_attr](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const int axis = axis_of(axis_attr, x.rank());
        std::int64_t x_strides[kMaxRank] = {};
        contiguous_strides(x.shape, x_strides);
        std::int64_t position = 0;
        for (const TensorView& y : context.outputs) {
            CopyParams p;
            p.rank = y.rank();
            contiguous_strides(y.shape, p.dst_strides);
            for (int d = 0; d < p.rank; ++d) {
                p.dims[d] = y.shape[d];
                p.src_strides[d] = x_strides[d];
            }
            if (y.numel() > 0) {
                II_TRY(copy(p, y.data, offset(x.data, position * x_strides[axis], x.dtype), x.dtype, stream));
            }
            position += y.shape[axis];
        }
        return ok_status();
    });
}

Result<std::unique_ptr<Kernel>> create_flip(const ir::Node& node, const CudaDevice& device) {
    const int rank = node.input(0)->rank();
    std::vector<bool> flip(static_cast<std::size_t>(rank), false);
    for (const std::int64_t axis : node.attrs().get_ints("axes")) {
        flip[static_cast<std::size_t>(axis_of(axis, rank))] = true;
    }
    return make_kernel(device, [flip](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        CopyParams p;
        p.rank = x.rank();
        contiguous_strides(x.shape, p.src_strides);
        contiguous_strides(y.shape, p.dst_strides);
        std::int64_t base = 0;
        for (int d = 0; d < p.rank; ++d) {
            p.dims[d] = x.shape[d];
            if (flip[static_cast<std::size_t>(d)] && x.shape[d] > 0) {
                base += (x.shape[d] - 1) * p.src_strides[d];
                p.src_strides[d] = -p.src_strides[d];
            }
        }
        if (y.numel() == 0) {
            return ok_status();
        }
        return copy(p, y.data, offset(x.data, base, x.dtype), x.dtype, stream);
    });
}

/// The bits of `value` converted to `dtype` (the `cast` rules).
std::uint64_t fill_bits(DType dtype, double value) {
    std::uint64_t bits = 0;
    auto put = [&](auto converted) { std::memcpy(&bits, &converted, sizeof(converted)); };
    auto saturate = [&](auto type_tag) {
        using T = decltype(type_tag);
        if (std::isnan(value)) {
            return T{0};
        }
        const double t = std::trunc(value);
        if (t <= static_cast<double>(std::numeric_limits<T>::lowest())) {
            return std::numeric_limits<T>::lowest();
        }
        if (t >= static_cast<double>(std::numeric_limits<T>::max())) {
            return std::numeric_limits<T>::max();
        }
        return static_cast<T>(t);
    };
    switch (dtype) {
    case DType::kF32:
        put(static_cast<float>(value));
        break;
    case DType::kF16:
        put(f16(static_cast<float>(value)).bits);
        break;
    case DType::kBF16:
        put(bf16(static_cast<float>(value)).bits);
        break;
    case DType::kF64:
        put(value);
        break;
    case DType::kBool:
        put(static_cast<std::uint8_t>(value != 0.0 ? 1 : 0));
        break;
    case DType::kI8:
        put(saturate(std::int8_t{}));
        break;
    case DType::kU8:
        put(saturate(std::uint8_t{}));
        break;
    case DType::kI16:
        put(saturate(std::int16_t{}));
        break;
    case DType::kU16:
        put(saturate(std::uint16_t{}));
        break;
    case DType::kI32:
        put(saturate(std::int32_t{}));
        break;
    case DType::kU32:
        put(saturate(std::uint32_t{}));
        break;
    case DType::kI64:
        put(saturate(std::int64_t{}));
        break;
    case DType::kU64:
        put(saturate(std::uint64_t{}));
        break;
    default:
        break;
    }
    return bits;
}

Result<std::unique_ptr<Kernel>> create_pad(const ir::Node& node, const CudaDevice& device) {
    const std::vector<std::int64_t> pads = node.attrs().get_ints("pads");
    const std::string_view mode_name = node.attrs().get_string("mode");
    const PadMode mode =
        mode_name == "edge" ? PadMode::kEdge : (mode_name == "reflect" ? PadMode::kReflect : PadMode::kConstant);
    const std::uint64_t fill = fill_bits(node.input(0)->dtype(), node.attrs().get_float("value"));
    return make_kernel(device, [pads, mode, fill](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        PadParams p;
        p.rank = x.rank();
        p.mode = mode;
        p.fill = fill;
        for (int d = 0; d < p.rank; ++d) {
            p.out_dims[d] = y.shape[d];
            p.in_dims[d] = x.shape[d];
            p.pad_begin[d] = pads[static_cast<std::size_t>(d)];
        }
        if (mode != PadMode::kConstant && x.numel() == 0 && y.numel() > 0) {
            return errors::invalid_argument("'pad' of an empty tensor in mode edge or reflect");
        }
        return launched(cuda::pad(y.data, x.data, static_cast<int>(element_size(x.dtype)), p, stream), "pad");
    });
}

Status check_pad(const ir::Node& node) {
    II_TRY(check_plain(node));
    const std::string_view mode = node.attrs().get_string("mode");
    if (mode != "constant" && mode != "edge" && mode != "reflect") {
        return errors::unsupported("'pad' mode {}", mode);
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_tile(const ir::Node&, const CudaDevice& device) {
    return make_kernel(device, [](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        if (y.numel() == 0) {
            return ok_status();
        }
        PadParams p;
        p.rank = x.rank();
        p.mode = PadMode::kWrap;
        for (int d = 0; d < p.rank; ++d) {
            p.out_dims[d] = y.shape[d];
            p.in_dims[d] = x.shape[d];
        }
        return launched(cuda::pad(y.data, x.data, static_cast<int>(element_size(x.dtype)), p, stream), "tile");
    });
}

Result<std::unique_ptr<Kernel>> create_gather(const ir::Node& node, const CudaDevice& device) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel(device, [axis_attr](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& data = context.input(0);
        const TensorView& indices = context.input(1);
        const int axis = axis_of(axis_attr, data.rank());
        GatherParams p;
        p.outer = 1;
        std::int64_t inner = 1;
        for (int d = 0; d < axis; ++d) {
            p.outer *= data.shape[d];
        }
        for (int d = axis + 1; d < data.rank(); ++d) {
            inner *= data.shape[d];
        }
        p.axis_size = data.shape[axis];
        p.count = indices.numel();
        p.slice_bytes = static_cast<std::int64_t>(storage_size(data.dtype, inner));
        p.index64 = indices.dtype == DType::kI64;
        p.error = error_of(context);
        return launched(cuda::gather(context.output(0).data, data.data, indices.data, p, stream), "gather");
    });
}

Status check_gather(const ir::Node& node) {
    II_TRY(check_rank(node));
    const DType index = node.input(1)->dtype();
    if (index != DType::kI32 && index != DType::kI64) {
        return errors::unsupported("'gather' with {} indices", to_string(index));
    }
    return ok_status();
}

// --- creation ---------------------------------------------------------------------------------------------------

Result<std::unique_ptr<Kernel>> create_fill(const ir::Node& node, const CudaDevice& device) {
    const std::uint64_t bits = fill_bits(node.output()->dtype(), node.attrs().get_float("value"));
    return make_kernel(device, [bits](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& y = context.output(0);
        return launched(cuda::fill(y.data, static_cast<int>(element_size(y.dtype)), bits, y.numel(), stream), "fill");
    });
}

Result<std::unique_ptr<Kernel>> create_iota(const ir::Node& node, const CudaDevice& device) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    const double start = node.attrs().get_float("start");
    const double step = node.attrs().get_float("step");
    return make_kernel(device, [axis_attr, start, step](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& y = context.output(0);
        const int axis = axis_of(axis_attr, y.rank());
        std::int64_t inner = 1;
        for (int d = axis + 1; d < y.rank(); ++d) {
            inner *= y.shape[d];
        }
        return launched(cuda::iota(y.data, *scalar_of(y.dtype), start, step, y.numel(),
                                   y.rank() == 0 ? 1 : y.shape[axis], inner, stream),
                        "iota");
    });
}

// --- pooling, resize --------------------------------------------------------------------------------------------

template <bool kMax>
Result<std::unique_ptr<Kernel>> create_pool(const ir::Node& node, const CudaDevice& device) {
    const int spatial = node.input(0)->rank() - 2;
    II_ASSIGN_OR_RETURN(ops::WindowAttrs window, ops::parse_window_attrs(node.attrs(), spatial));
    const std::vector<std::int64_t> kernel = node.attrs().get_ints("kernel");
    const bool count_include_pad = node.attrs().get_bool("count_include_pad");
    return make_kernel(device, [=](const KernelContext& context, cudaStream_t stream) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        PoolParams p;
        p.planes = x.shape[0] * x.shape[1];
        p.max = kMax;
        p.count_include_pad = count_include_pad;
        const auto s = static_cast<std::size_t>(spatial);
        for (int d = 0; d < spatial; ++d) {
            const auto i = static_cast<std::size_t>(d);
            const ops::WindowGeometry g =
                ops::resolve_window(x.shape[2 + d], kernel[i], window.strides[i], window.dilations[i], window.pads[i],
                                    window.pads[s + i], window.auto_pad);
            const bool width = d == spatial - 1;
            (width ? p.in_w : p.in_h) = x.shape[2 + d];
            (width ? p.out_w : p.out_h) = y.shape[2 + d];
            (width ? p.kernel_w : p.kernel_h) = g.kernel;
            (width ? p.stride_w : p.stride_h) = g.stride;
            (width ? p.pad_w : p.pad_h) = g.pad_begin;
            (width ? p.pad_w_end : p.pad_h_end) = g.pad_end;
            (width ? p.dilation_w : p.dilation_h) = g.dilation;
        }
        return launched(pool(y.data, x.data, *scalar_of(x.dtype), p, stream), "pool");
    });
}

Status check_pool(const ir::Node& node) {
    II_TRY(check_float(node));
    const int rank = node.input(0)->rank();
    if (rank != 3 && rank != 4) {
        return errors::unsupported("'{}' with {} spatial dimensions", node.op_name(), rank - 2);
    }
    return ok_status();
}

/// Resize with taps precomputed for the static shapes of the node, kept in device memory.
class ResizeKernel final : public Kernel {
public:
    static Result<std::unique_ptr<Kernel>> create(const ir::Node& node, const CudaDevice& device) {
        const Shape in = *node.input(0)->type().static_shape();
        const Shape out = *node.output()->type().static_shape();
        const std::vector<double> scales = node.attrs().get_floats("scales");
        const std::string_view mode = node.attrs().get_string("mode");
        const ops::ResizeMode resize_mode{
            mode, node.attrs().get_string("coordinate_mode"), node.attrs().get_string("nearest_mode"),
            node.attrs().get_float("cubic_coeff_a"), node.attrs().get_bool("exclude_outside")};
        std::vector<std::int64_t> index;
        std::vector<float> weight;
        auto kernel = std::unique_ptr<ResizeKernel>(new ResizeKernel(device, in, out));
        std::vector<std::size_t> first(static_cast<std::size_t>(in.rank()));
        for (int d = 0; d < in.rank(); ++d) {
            const double scale = scales.empty() ? 0.0 : scales[static_cast<std::size_t>(d)];
            first[static_cast<std::size_t>(d)] = index.size();
            if (in[d] == out[d] && (scale == 0 || scale == 1)) {
                kernel->params_.taps[d] = 1;
                for (std::int64_t i = 0; i < out[d]; ++i) {
                    index.push_back(i);
                    weight.push_back(1.0f);
                }
                continue;
            }
            const ops::ResizeTaps taps = ops::resize_taps(in[d], out[d], scale, resize_mode);
            kernel->params_.taps[d] = taps.count;
            index.insert(index.end(), taps.index.begin(), taps.index.end());
            for (const double w : taps.weight) {
                weight.push_back(static_cast<float>(w));
            }
        }
        II_ASSIGN_OR_RETURN(kernel->indices_, device.allocate_owned(index.size() * sizeof(std::int64_t)));
        II_ASSIGN_OR_RETURN(kernel->weights_, device.allocate_owned(weight.size() * sizeof(float)));
        II_CUDA_TRY(cudaMemcpy(kernel->indices_.get(), index.data(), index.size() * sizeof(std::int64_t),
                               cudaMemcpyHostToDevice));
        II_CUDA_TRY(
            cudaMemcpy(kernel->weights_.get(), weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));
        kernel->params_.rank = in.rank();
        std::int64_t in_strides[kMaxRank] = {};
        contiguous_strides(in, in_strides);
        for (int d = 0; d < in.rank(); ++d) {
            kernel->params_.out_dims[d] = out[d];
            kernel->params_.in_strides[d] = in_strides[d];
            kernel->params_.index[d] =
                static_cast<const std::int64_t*>(kernel->indices_.get()) + first[static_cast<std::size_t>(d)];
            kernel->params_.weight[d] =
                static_cast<const float*>(kernel->weights_.get()) + first[static_cast<std::size_t>(d)];
        }
        return std::unique_ptr<Kernel>(std::move(kernel));
    }

    Status run(const KernelContext& context) const override {
        II_TRY(device_.activate());
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        if (x.shape != in_ || y.shape != out_) {
            return errors::internal("resize compiled for {} -> {} ran with {} -> {}", in_.to_string(), out_.to_string(),
                                    x.shape.to_string(), y.shape.to_string());
        }
        return launched(resize(y.data, x.data, *scalar_of(x.dtype), params_, stream_of(context)), "resize");
    }

private:
    ResizeKernel(const CudaDevice& device, Shape in, Shape out) : device_(device), in_(in), out_(out) {}

    const CudaDevice& device_;
    Shape in_;
    Shape out_;
    ResizeParams params_;
    std::shared_ptr<void> indices_;
    std::shared_ptr<void> weights_;
};

Status check_resize(const ir::Node& node) {
    II_TRY(check_float(node));
    II_TRY(check_rank(node));
    if (!node.input(0)->type().is_static() || !node.output()->type().is_static()) {
        return errors::unsupported("'resize' with dynamic shapes");
    }
    if (node.input(0)->type().static_shape()->numel() == 0) {
        return errors::unsupported("'resize' of an empty tensor");
    }
    return ok_status();
}

}  // namespace

void register_ops_kernels(CudaBackend& backend) {
    backend.add_kernel("reduce_sum", create_reduce<ReduceOp::kSum>, check_reduce);
    backend.add_kernel("reduce_mean", create_reduce<ReduceOp::kMean>, check_reduce);
    backend.add_kernel("reduce_max", create_reduce<ReduceOp::kMax>, check_reduce);
    backend.add_kernel("reduce_min", create_reduce<ReduceOp::kMin>, check_reduce);
    backend.add_kernel("reduce_prod", create_reduce<ReduceOp::kProd>, check_reduce);
    backend.add_kernel("argmax", create_arg<true>, check_arg);
    backend.add_kernel("argmin", create_arg<false>, check_arg);
    backend.add_kernel("softmax", create_softmax<false>, check_float);
    backend.add_kernel("log_softmax", create_softmax<true>, check_float);
    backend.add_kernel("layer_norm", create_norm<false>, check_norm);
    backend.add_kernel("rms_norm", create_norm<true>, check_norm);
    backend.add_kernel("reshape", create_reshape, check_reshape);
    backend.add_kernel("permute", create_permute, check_plain);
    backend.add_kernel("slice", create_slice, check_plain);
    backend.add_kernel("expand", create_expand, check_plain);
    backend.add_kernel("concat", create_concat, check_plain);
    backend.add_kernel("split", create_split, check_plain);
    backend.add_kernel("flip", create_flip, check_plain);
    backend.add_kernel("pad", create_pad, check_pad);
    backend.add_kernel("tile", create_tile, check_plain);
    backend.add_kernel("gather", create_gather, check_gather);
    backend.add_kernel("fill", create_fill, check_plain);
    backend.add_kernel("iota", create_iota, check_numeric);
    backend.add_kernel("max_pool", create_pool<true>, check_pool);
    backend.add_kernel("avg_pool", create_pool<false>, check_pool);
    backend.add_kernel("resize", ResizeKernel::create, check_resize);
}

}  // namespace ii::cuda
