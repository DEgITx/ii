// SPDX-License-Identifier: MIT
// ii-cpu reductions, softmax and normalizations (f32). Each operation is viewed as rows of contiguous memory:
// [outer, reduced, inner]; the ISA kernels process ranges of the outer dimension in parallel.
#include "backends/cpu/cpu_backend.h"
#include "ir/op_schema.h"

#include <algorithm>
#include <vector>

namespace ii::cpu {

namespace {

Status expect_f32(const ir::Node& node) {
    for (const ir::Value* value : node.inputs()) {
        if (value != nullptr && value->dtype() != DType::kF32) {
            return errors::unsupported("'{}' is accelerated for f32 only", node.op_name());
        }
    }
    return ok_status();
}

int normalized_axis(std::int64_t axis, int rank) {
    return static_cast<int>(axis < 0 ? axis + rank : axis);
}

/// The reduced axes as a contiguous range [first, last), or an error when they are not contiguous.
Result<std::pair<int, int>> axis_range(const ir::Node& node) {
    const int rank = node.input(0)->rank();
    std::vector<std::int64_t> axes = node.attrs().get_ints("axes");
    if (axes.empty()) {
        return std::make_pair(0, rank);
    }
    std::vector<int> normalized;
    for (const std::int64_t axis : axes) {
        normalized.push_back(normalized_axis(axis, rank));
    }
    std::sort(normalized.begin(), normalized.end());
    for (std::size_t i = 1; i < normalized.size(); ++i) {
        if (normalized[i] != normalized[i - 1] + 1) {
            return errors::unsupported("reductions over non-adjacent axes are not accelerated");
        }
    }
    return std::make_pair(normalized.front(), normalized.back() + 1);
}

/// Splits `shape` into outer x middle x inner around the dimension range [first, last).
struct Split3 {
    std::int64_t outer = 1;
    std::int64_t middle = 1;
    std::int64_t inner = 1;
};

Split3 split(const Shape& shape, int first, int last) {
    Split3 s;
    for (int d = 0; d < shape.rank(); ++d) {
        (d < first ? s.outer : d < last ? s.middle : s.inner) *= shape[d];
    }
    return s;
}

// --- reduce_* ---

Status check_reduce(const ir::Node& node) {
    II_TRY(expect_f32(node));
    II_ASSIGN_OR_RETURN(const auto range, axis_range(node));
    (void)range;
    return ok_status();
}

Result<std::unique_ptr<Kernel>> create_reduce(const ir::Node& node, const CpuConfig& config) {
    II_ASSIGN_OR_RETURN(const auto range, axis_range(node));
    const std::string_view op = node.op_name();
    const ReduceKind kind = op == "reduce_sum"    ? ReduceKind::kSum
                            : op == "reduce_mean" ? ReduceKind::kMean
                            : op == "reduce_max"  ? ReduceKind::kMax
                            : op == "reduce_min"  ? ReduceKind::kMin
                                                  : ReduceKind::kProd;
    const KernelTable* table = config.table;
    return make_kernel([table, kind, range, op](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const Split3 s = split(x.shape, range.first, range.second);
        if (s.middle == 0 && (kind == ReduceKind::kMax || kind == ReduceKind::kMin) && s.outer * s.inner > 0) {
            return errors::invalid_argument("'{}' of an empty range", op);
        }
        const float* in = x.as<float>();
        float* out = context.output(0).as<float>();
        parallel_range(context, s.outer, static_cast<double>(s.middle * s.inner),
                       [&](std::int64_t begin, std::int64_t end) {
                           table->reduce(kind, in + begin * s.middle * s.inner, end - begin, s.middle, s.inner,
                                         out + begin * s.inner);
                       });
        return ok_status();
    });
}

// --- softmax / log_softmax ---

Result<std::unique_ptr<Kernel>> create_softmax(const ir::Node& node, const CpuConfig& config) {
    const int axis = normalized_axis(node.attrs().get_int("axis"), node.input(0)->rank());
    const bool log = node.op_name() == "log_softmax";
    const KernelTable* table = config.table;
    return make_kernel([table, axis, log](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const Split3 s = split(x.shape, axis, axis + 1);
        const float* in = x.as<float>();
        float* out = context.output(0).as<float>();
        const std::int64_t row = s.middle * s.inner;
        parallel_range(context, s.outer, 4.0 * static_cast<double>(row), [&](std::int64_t begin, std::int64_t end) {
            table->softmax(in + begin * row, out + begin * row, end - begin, s.middle, s.inner, log);
        });
        return ok_status();
    });
}

// --- layer_norm / rms_norm ---

Result<std::unique_ptr<Kernel>> create_norm(const ir::Node& node, const CpuConfig& config) {
    const int axis = normalized_axis(node.attrs().get_int("axis"), node.input(0)->rank());
    const auto epsilon = static_cast<float>(node.attrs().get_float("epsilon"));
    const bool rms = node.op_name() == "rms_norm";
    const KernelTable* table = config.table;
    return make_kernel([table, axis, epsilon, rms](const KernelContext& context) -> Status {
        const TensorView& x = context.input(0);
        const Split3 s = split(x.shape, axis, x.rank());
        const std::int64_t n = s.middle;
        const float* in = x.as<float>();
        const float* scale = context.input(1).present() ? context.input(1).as<float>() : nullptr;
        const float* bias = context.input(2).present() ? context.input(2).as<float>() : nullptr;
        float* out = context.output(0).as<float>();
        parallel_range(context, s.outer, 3.0 * static_cast<double>(n), [&](std::int64_t begin, std::int64_t end) {
            if (rms) {
                table->rms_norm(in + begin * n, scale, out + begin * n, end - begin, n, epsilon);
            } else {
                table->layer_norm(in + begin * n, scale, bias, out + begin * n, end - begin, n, epsilon);
            }
        });
        return ok_status();
    });
}

}  // namespace

void register_reduction_kernels(CpuBackend& backend) {
    for (const std::string_view op : {"reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod"}) {
        backend.add_kernel(op, create_reduce, check_reduce);
    }
    backend.add_kernel("softmax", create_softmax, expect_f32);
    backend.add_kernel("log_softmax", create_softmax, expect_f32);
    backend.add_kernel("layer_norm", create_norm, expect_f32);
    backend.add_kernel("rms_norm", create_norm, expect_f32);
}

}  // namespace ii::cpu
