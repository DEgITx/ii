// SPDX-License-Identifier: MIT
#include "backends/cuda/cuda_backend.h"

#include "core/assert.h"
#include "ir/op_schema.h"

#include <charconv>
#include <format>

namespace ii::cuda {

CudaBackend::CudaBackend(std::string name, std::unique_ptr<CudaDevice> device)
    : name_(std::move(name)), device_(std::move(device)) {
    register_elementwise_kernels(*this);
    register_gemm_kernels(*this);
    register_ops_kernels(*this);
}

Status CudaBackend::supports(const ir::Node& node) const {
    const auto it = kernels_.find(node.op_name());
    if (it == kernels_.end()) {
        return errors::unsupported("{}: no kernel for '{}'", name_, node.op_name());
    }
    if (it->second.check != nullptr) {
        if (Status status = it->second.check(node); !status.ok()) {
            return errors::unsupported("{}: {}", name_, status.message());
        }
    }
    return ok_status();
}

Result<std::unique_ptr<Kernel>> CudaBackend::create_kernel(const ir::Node& node) const {
    II_TRY(supports(node));
    II_TRY(device_->activate());
    return kernels_.at(node.op_name()).factory(node, *device_);
}

void CudaBackend::add_kernel(std::string_view op, Factory factory, SupportCheck check) {
    const ir::OpSchema& schema = ir::builtin_op(op);
    II_CHECK(!kernels_.contains(schema.name), "kernel for '{}' registered twice", op);
    kernels_.emplace(schema.name, Entry{factory, check});
}

Status FunctionKernel::run(const KernelContext& context) const {
    if (context.queue == nullptr) {
        return errors::internal("ii-cuda kernel run without a device queue");
    }
    II_TRY(device_.activate());
    return run_(context, stream_of(context));
}

std::optional<Scalar> scalar_of(DType dtype) noexcept {
    switch (dtype) {
    case DType::kF32:
        return Scalar::kF32;
    case DType::kF16:
        return Scalar::kF16;
    case DType::kBF16:
        return Scalar::kBF16;
    case DType::kI32:
        return Scalar::kI32;
    case DType::kI64:
        return Scalar::kI64;
    default:
        return std::nullopt;
    }
}

Status check_dtypes(const ir::Node& node, std::initializer_list<DType> allowed) {
    auto check = [&](const ir::Value* value) -> Status {
        if (value == nullptr) {
            return ok_status();
        }
        for (const DType dtype : allowed) {
            if (value->dtype() == dtype) {
                return ok_status();
            }
        }
        return errors::unsupported("'{}' of {}", node.op_name(), to_string(value->dtype()));
    };
    for (const ir::Value* input : node.inputs()) {
        II_TRY(check(input));
    }
    for (const ir::Value* output : node.outputs()) {
        II_TRY(check(output));
    }
    return ok_status();
}

void contiguous_strides(const Shape& shape, std::int64_t* strides) noexcept {
    std::int64_t stride = 1;
    for (int d = shape.rank() - 1; d >= 0; --d) {
        strides[d] = stride;
        stride *= shape[d];
    }
}

Status launched(cudaError_t error, std::string_view kernel) {
    if (error == cudaSuccess) {
        return ok_status();
    }
    return check(error, kernel);
}

Result<std::vector<std::unique_ptr<backend::Backend>>>
create_cuda_backends(const Options& options, std::string_view library_path, std::string_view devices, Logger logger) {
    if (devices == "none") {
        return std::vector<std::unique_ptr<backend::Backend>>();
    }
    II_ASSIGN_OR_RETURN(const int count, device_count());
    II_ASSIGN_OR_RETURN(const Libraries* libraries, load_libraries(library_path));

    std::vector<int> indices;
    if (devices.empty() || devices == "all") {
        for (int i = 0; i < count; ++i) {
            indices.push_back(i);
        }
    } else {
        while (!devices.empty()) {
            const std::size_t comma = devices.find(',');
            const std::string_view item = devices.substr(0, comma);
            int index = -1;
            const auto [end, error] = std::from_chars(item.data(), item.data() + item.size(), index);
            if (error != std::errc() || end != item.data() + item.size() || index < 0 || index >= count) {
                return errors::invalid_argument("cuda.devices: '{}' is not a device index (0..{})", item, count - 1);
            }
            indices.push_back(index);
            devices = comma == std::string_view::npos ? std::string_view() : devices.substr(comma + 1);
        }
    }

    std::vector<std::unique_ptr<backend::Backend>> backends;
    for (const int index : indices) {
        Result<std::unique_ptr<CudaDevice>> device = CudaDevice::create(index, *libraries, options);
        if (!device.ok()) {
            logger.warning("ii-cuda: device {} skipped: {}", index, device.status().message());
            continue;
        }
        logger.debug("ii-cuda: device {}: {}, {} MB usable, {:.0f} GFLOP/s, {:.0f} GB/s; {}", index,
                     (*device)->info().name, (*device)->info().memory_bytes >> 20, (*device)->info().gflops,
                     (*device)->info().bandwidth_gbs, libraries->description);
        const std::string name =
            backends.empty() ? std::string(kBackendName) : std::format("{}.{}", kBackendName, index);
        backends.push_back(std::make_unique<CudaBackend>(name, std::move(*device)));
    }
    if (backends.empty()) {
        return errors::unavailable("no usable CUDA device");
    }
    return backends;
}

}  // namespace ii::cuda
