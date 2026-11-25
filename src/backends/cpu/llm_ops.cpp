// SPDX-License-Identifier: MIT
// ii-cpu kernels of the LLM operations: rope (f32) and slice_update (any non-quantized dtype, in place).
#include "backends/cpu/cpu_backend.h"
#include "ir/graph.h"
#include "ir/op_schema.h"

#include <cstring>

namespace ii::cpu {

namespace {

Status check_rope(const ir::Node& node) {
    for (int i = 0; i < 3; ++i) {
        if (node.input(i)->dtype() != DType::kF32) {
            return errors::unsupported("'rope' is accelerated for f32 only");
        }
    }
    if (node.input(0)->rank() > kMaxRank || node.input(1)->rank() > kMaxRank) {
        return errors::unsupported("rank above {}", kMaxRank);
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_rope(const ir::Node& node, const CpuConfig&) {
    const bool interleaved = node.attrs().get_bool("interleaved");
    return make_kernel([interleaved](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const TensorView& cos = context.input(1);
        const TensorView& sin = context.input(2);
        const TensorView& y = context.output(0);
        const int rank = x.rank();
        const std::int64_t size = x.shape[rank - 1];
        const std::int64_t half = cos.shape[cos.rank() - 1];
        const std::int64_t rows = size == 0 ? 0 : x.numel() / size;
        // Element strides of the cos / sin table over the dimensions of x (0 along broadcast dimensions).
        std::int64_t strides[kMaxRank] = {};
        {
            const int offset = rank - cos.rank();
            std::int64_t stride = 1;
            for (int d = cos.rank() - 1; d >= 0; --d) {
                strides[d + offset] = cos.shape[d] == 1 ? 0 : stride;
                stride *= cos.shape[d];
            }
        }
        const float* px = x.as<float>();
        const float* pc = cos.as<float>();
        const float* ps = sin.as<float>();
        float* py = y.as<float>();
        parallel_range(context, rows, static_cast<double>(size) * 8.0, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t r = begin; r < end; ++r) {
                // Offset of row r in the table: unravel r over x's dimensions except the last.
                std::int64_t rest = r;
                std::int64_t table = 0;
                for (int d = rank - 2; d >= 0; --d) {
                    table += (rest % x.shape[d]) * strides[d];
                    rest /= x.shape[d];
                }
                const float* c = pc + table;
                const float* s = ps + table;
                const float* in = px + r * size;
                float* out = py + r * size;
                if (interleaved) {
                    for (std::int64_t i = 0; i < half; ++i) {
                        const float a = in[2 * i];
                        const float b = in[2 * i + 1];
                        out[2 * i] = a * c[i] - b * s[i];
                        out[2 * i + 1] = b * c[i] + a * s[i];
                    }
                } else {
                    for (std::int64_t i = 0; i < half; ++i) {
                        const float a = in[i];
                        const float b = in[i + half];
                        out[i] = a * c[i] - b * s[i];
                        out[i + half] = b * c[i] + a * s[i];
                    }
                }
                if (out != in) {
                    std::memcpy(out + 2 * half, in + 2 * half, static_cast<std::size_t>(size - 2 * half) * 4);
                }
            }
        });
        return ok_status();
    });
}

Status check_slice_update(const ir::Node& node) {
    if (is_block_quantized(node.input(0)->dtype()) || node.input(0)->rank() > kMaxRank) {
        return errors::unsupported("'slice_update' of block-quantized data or rank above {}", kMaxRank);
    }
    return ok_status();
}

/// Writes `update` into x's memory (when the runtime runs the step in place, e.g. into a KV cache state) or into a
/// copy of x.
class SliceUpdateKernel final : public Kernel {
public:
    explicit SliceUpdateKernel(std::int64_t axis) : axis_(axis) {}

    bool can_run_in_place(int input) const override { return input == 0; }

    Status run(const KernelContext& context) const override {
        const TensorView& x = context.input(0);
        const TensorView& update = context.input(1);
        const TensorView& start_view = context.input(2);
        const TensorView& y = context.output(0);
        const int axis = static_cast<int>(axis_ < 0 ? axis_ + x.rank() : axis_);
        const std::int64_t start = start_view.dtype == DType::kI32 ? *start_view.as<const std::int32_t>()
                                                                   : *start_view.as<const std::int64_t>();
        const std::int64_t length = update.shape[axis];
        if (start < 0 || start > x.shape[axis] - length) {
            return errors::out_of_range("'slice_update' writes [{}, {}) of an axis of size {}", start, start + length,
                                        x.shape[axis]);
        }
        const std::size_t element = element_size(x.dtype);
        if (y.data != x.data) {
            std::memcpy(y.data, x.data, static_cast<std::size_t>(x.numel()) * element);
        }
        std::int64_t outer = 1;
        for (int d = 0; d < axis; ++d) {
            outer *= x.shape[d];
        }
        std::int64_t inner = 1;
        for (int d = axis + 1; d < x.rank(); ++d) {
            inner *= x.shape[d];
        }
        const auto row = static_cast<std::size_t>(inner) * element;
        const auto chunk = static_cast<std::size_t>(length) * row;
        for (std::int64_t o = 0; o < outer; ++o) {
            std::memcpy(static_cast<std::byte*>(y.data) + (static_cast<std::size_t>(o * x.shape[axis] + start)) * row,
                        static_cast<const std::byte*>(update.data) + static_cast<std::size_t>(o) * chunk, chunk);
        }
        return ok_status();
    }

private:
    std::int64_t axis_;
};

Result<std::unique_ptr<Kernel>> create_slice_update(const ir::Node& node, const CpuConfig&) {
    return std::unique_ptr<Kernel>(std::make_unique<SliceUpdateKernel>(node.attrs().get_int("axis")));
}

}  // namespace

void register_llm_kernels(CpuBackend& backend) {
    backend.add_kernel("rope", create_rope, check_rope);
    backend.add_kernel("slice_update", create_slice_update, check_slice_update);
}

}  // namespace ii::cpu
