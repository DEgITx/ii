// SPDX-License-Identifier: MIT
// ii-cpu data movement (any dtype except block-quantized): reshape, permute, concat. Copies are dtype-agnostic
// (by element size) and parallel over rows.
#include "backends/cpu/cpu_backend.h"
#include "ir/op_schema.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace ii::cpu {

namespace {

constexpr int kMaxRank = 8;

Status check_plain(const ir::Node& node) {
    const DType dtype = node.output()->dtype();
    if (is_block_quantized(dtype) || node.output()->rank() > kMaxRank) {
        return errors::unsupported("'{}' of {} is not accelerated", node.op_name(), to_string(dtype));
    }
    return ok_status();
}

// --- reshape ---

class ReshapeKernel final : public Kernel {
public:
    bool can_run_in_place(int input) const override { return input == 0; }

    Status run(const KernelContext& context) const override {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        if (x.data != y.data && y.nbytes() > 0) {
            const auto bytes = static_cast<std::int64_t>(y.nbytes());
            parallel_range(context, (bytes + 65535) / 65536, 65536.0, [&](std::int64_t begin, std::int64_t end) {
                const std::int64_t from = begin * 65536;
                const std::int64_t to = std::min(bytes, end * 65536);
                std::memcpy(static_cast<std::byte*>(y.data) + from, static_cast<const std::byte*>(x.data) + from,
                            static_cast<std::size_t>(to - from));
            });
        }
        return ok_status();
    }
};

Result<std::unique_ptr<Kernel>> create_reshape(const ir::Node&, const CpuConfig&) {
    return std::unique_ptr<Kernel>(std::make_unique<ReshapeKernel>());
}

// --- permute ---

template <class T>
void permute_rows(const T* in, T* out, const std::int64_t* sizes, const std::int64_t* in_strides, int rank,
                  std::int64_t begin, std::int64_t end) {
    // Output is walked row by row (last output dimension contiguous); the input is gathered with in_strides.
    const std::int64_t inner = sizes[rank - 1];
    const std::int64_t inner_stride = in_strides[rank - 1];
    std::int64_t index[kMaxRank] = {};
    std::int64_t rest = begin;
    std::int64_t offset = 0;
    for (int d = rank - 2; d >= 0; --d) {
        index[d] = rest % sizes[d];
        rest /= sizes[d];
        offset += index[d] * in_strides[d];
    }
    for (std::int64_t row = begin; row < end; ++row) {
        T* dst = out + row * inner;
        const T* src = in + offset;
        if (inner_stride == 1) {
            std::memcpy(dst, src, static_cast<std::size_t>(inner) * sizeof(T));
        } else {
            for (std::int64_t i = 0; i < inner; ++i) {
                dst[i] = src[i * inner_stride];
            }
        }
        for (int d = rank - 2; d >= 0; --d) {
            offset += in_strides[d];
            if (++index[d] < sizes[d]) {
                break;
            }
            offset -= index[d] * in_strides[d];
            index[d] = 0;
        }
    }
}

/// Transposes blocks of the two innermost output dimensions when the input is contiguous along the second-to-last
/// output dimension (the common [.., a, b] -> [.., b, a] case): tiles keep both sides in cache.
template <class T>
void transpose_tiles(const T* in, T* out, std::int64_t rows, std::int64_t cols, std::int64_t in_row_stride,
                     std::int64_t in_col_stride) {
    constexpr std::int64_t kTile = 32;
    for (std::int64_t r0 = 0; r0 < rows; r0 += kTile) {
        for (std::int64_t c0 = 0; c0 < cols; c0 += kTile) {
            const std::int64_t r1 = std::min(rows, r0 + kTile);
            const std::int64_t c1 = std::min(cols, c0 + kTile);
            for (std::int64_t r = r0; r < r1; ++r) {
                for (std::int64_t c = c0; c < c1; ++c) {
                    out[r * cols + c] = in[r * in_row_stride + c * in_col_stride];
                }
            }
        }
    }
}

template <class T>
void permute_typed(const KernelContext& context, const std::int64_t* sizes, const std::int64_t* in_strides, int rank) {
    const T* in = context.input(0).as<T>();
    T* out = context.output(0).as<T>();
    std::int64_t rows = 1;
    for (int d = 0; d + 1 < rank; ++d) {
        rows *= sizes[d];
    }
    const std::int64_t inner = sizes[rank - 1];
    if (rank >= 2 && in_strides[rank - 1] != 1 && in_strides[rank - 2] == 1) {
        // Tiled transpose of the last two dimensions, parallel over the remaining ones.
        const std::int64_t matrices = rows / sizes[rank - 2];
        const std::int64_t matrix = sizes[rank - 2] * inner;
        parallel_range(context, matrices, static_cast<double>(matrix * sizeof(T)),
                       [&](std::int64_t begin, std::int64_t end) {
                           for (std::int64_t m = begin; m < end; ++m) {
                               std::int64_t rest = m;
                               std::int64_t offset = 0;
                               for (int d = rank - 3; d >= 0; --d) {
                                   offset += (rest % sizes[d]) * in_strides[d];
                                   rest /= sizes[d];
                               }
                               transpose_tiles(in + offset, out + m * matrix, sizes[rank - 2], inner,
                                               in_strides[rank - 2], in_strides[rank - 1]);
                           }
                       });
        return;
    }
    parallel_range(
        context, rows, static_cast<double>(inner * static_cast<std::int64_t>(sizeof(T))),
        [&](std::int64_t begin, std::int64_t end) { permute_rows(in, out, sizes, in_strides, rank, begin, end); });
}

Result<std::unique_ptr<Kernel>> create_permute(const ir::Node& node, const CpuConfig&) {
    const std::vector<std::int64_t> perm = node.attrs().get_ints("perm");
    return make_kernel([perm](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& y = context.output(0);
        const int rank = y.rank();
        if (y.numel() == 0) {
            return ok_status();
        }
        if (rank == 0) {
            std::memcpy(y.data, x.data, y.nbytes());
            return ok_status();
        }
        std::int64_t x_strides[kMaxRank];
        std::int64_t stride = 1;
        for (int d = rank - 1; d >= 0; --d) {
            x_strides[d] = stride;
            stride *= x.shape[d];
        }
        // Collapse output dimensions that are also adjacent (in order) in the input; drop size-1 dimensions.
        std::int64_t sizes[kMaxRank];
        std::int64_t strides[kMaxRank];
        int crank = 0;
        for (int d = 0; d < rank; ++d) {
            const std::int64_t size = y.shape[d];
            const std::int64_t in_stride = x_strides[perm[static_cast<std::size_t>(d)]];
            if (size == 1) {
                continue;
            }
            if (crank > 0 && strides[crank - 1] == in_stride * size) {
                sizes[crank - 1] *= size;
                strides[crank - 1] = in_stride;
            } else {
                sizes[crank] = size;
                strides[crank] = in_stride;
                ++crank;
            }
        }
        if (crank == 0) {
            sizes[0] = 1;
            strides[0] = 1;
            crank = 1;
        }
        switch (element_size(y.dtype)) {
        case 1:
            permute_typed<std::uint8_t>(context, sizes, strides, crank);
            break;
        case 2:
            permute_typed<std::uint16_t>(context, sizes, strides, crank);
            break;
        case 4:
            permute_typed<std::uint32_t>(context, sizes, strides, crank);
            break;
        case 8:
            permute_typed<std::uint64_t>(context, sizes, strides, crank);
            break;
        default:
            return errors::unsupported("permute of {}-byte elements", element_size(y.dtype));
        }
        return ok_status();
    });
}

Status check_permute(const ir::Node& node) {
    II_TRY(check_plain(node));
    const std::size_t size = element_size(node.output()->dtype());
    if (size != 1 && size != 2 && size != 4 && size != 8) {
        return errors::unsupported("permute of {}-byte elements", size);
    }
    return ok_status();
}

// --- concat ---

Result<std::unique_ptr<Kernel>> create_concat(const ir::Node& node, const CpuConfig&) {
    const std::int64_t axis_attr = node.attrs().get_int("axis");
    return make_kernel([axis_attr](const KernelContext& context) -> Status {
        const TensorView& y = context.output(0);
        const int axis = static_cast<int>(axis_attr < 0 ? axis_attr + y.rank() : axis_attr);
        std::int64_t outer = 1;
        std::int64_t inner = 1;
        for (int d = 0; d < y.rank(); ++d) {
            if (d < axis) {
                outer *= y.shape[d];
            } else if (d > axis) {
                inner *= y.shape[d];
            }
        }
        const auto size = static_cast<std::int64_t>(element_size(y.dtype));
        const std::int64_t row = y.shape[axis] * inner * size;  // bytes per output row
        parallel_range(context, outer, static_cast<double>(row), [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t o = begin; o < end; ++o) {
                std::int64_t offset = 0;
                for (const TensorView& x : context.inputs) {
                    const std::int64_t chunk = x.shape[axis] * inner * size;
                    if (chunk > 0) {
                        std::memcpy(static_cast<std::byte*>(y.data) + o * row + offset,
                                    static_cast<const std::byte*>(x.data) + o * chunk, static_cast<std::size_t>(chunk));
                    }
                    offset += chunk;
                }
            }
        });
        return ok_status();
    });
}

}  // namespace

void register_movement_kernels(CpuBackend& backend) {
    backend.add_kernel("reshape", create_reshape, check_plain);
    backend.add_kernel("permute", create_permute, check_permute);
    backend.add_kernel("concat", create_concat, check_plain);
}

}  // namespace ii::cpu
