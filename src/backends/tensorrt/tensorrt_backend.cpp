// SPDX-License-Identifier: MIT
#include "backends/tensorrt/tensorrt_backend.h"

#include "backends/tensorrt/engine.h"
#include "backends/tensorrt/library.h"
#include "backends/tensorrt/translate.h"
#include "ir/op_schema.h"
#include "ops/fused.h"

#include <algorithm>
#include <charconv>
#include <format>

namespace ii::tensorrt {

namespace {

/// Host-side cost of enqueueing an engine (setting shapes and addresses, enqueueV3), microseconds.
constexpr double kEnqueueUs = 5;
/// TensorRT's kernels (implicit-GEMM convolutions, fused attention and normalization) are assumed this much closer to
/// the roofline than a sequence of separate kernels.
constexpr double kEfficiency = 0.7;

/// Operations TensorRT runs as kernels of their own; elementwise operations and moves fuse into them.
bool is_kernel(const ir::Node& node) {
    return !node.op().has_flag(ir::kOpElementwise) && !node.op().has_flag(ir::kOpDataMovement) &&
           !node.op().has_flag(ir::kOpView);
}

class TensorRTBackend final : public backend::Backend {
public:
    TensorRTBackend(std::unique_ptr<TrtDevice> device, std::unique_ptr<Compiler> compiler)
        : device_(std::move(device)),
          compiler_(std::move(compiler)),
          name_(device_->index() == 0 ? std::string("ii-tensorrt") : std::format("ii-tensorrt.{}", device_->index())) {}

    std::string_view name() const noexcept override { return name_; }
    const backend::DeviceInfo& device_info() const noexcept override { return device_->info(); }
    backend::DeviceApi* device_api() const noexcept override { return device_.get(); }
    bool compiles_subgraphs() const override { return true; }

    Status supports(const ir::Node& node) const override {
        if (node.op_name() == ops::kTransfer) {
            return errors::unsupported("{}: no kernel for 'transfer'", name_);
        }
        if (Status status = supports_node(node); !status.ok()) {
            return errors::unsupported("{}: {}", name_, status.message());
        }
        return ok_status();
    }

    /// A part costs its roofline (times kEfficiency), a launch per operation that is a kernel of its own
    /// (elementwise operations and moves fuse into them) and the host's work of enqueueing the engine; a single node
    /// like any kernel.
    double estimate_us(const ir::CostContext& context) const override {
        if (compiler_->options().prefer) {
            return 0;
        }
        if (context.node.op_name() != ops::kSubgraph) {
            return Backend::estimate_us(context);
        }
        const backend::DeviceInfo& info = device_info();
        const ir::Cost cost = ir::estimate_cost(context);
        const double compute = cost.flops / (std::max(info.gflops, 1e-3) * 1e3);
        const double memory = cost.bytes / (std::max(info.bandwidth_gbs, 1e-3) * 1e3);
        int kernels = 1;
        if (Result<std::shared_ptr<const ir::Graph>> program = ops::parse_subgraph_program(context.node);
            program.ok()) {
            kernels = std::max<int>(
                1, static_cast<int>(std::count_if((*program)->nodes().begin(), (*program)->nodes().end(),
                                                  [](const std::unique_ptr<ir::Node>& n) { return is_kernel(*n); })));
        }
        return kEfficiency * std::max(compute, memory) + info.launch_us * kernels + kEnqueueUs;
    }

    Result<std::unique_ptr<backend::Kernel>> create_kernel(const ir::Node& node) const override {
        II_TRY(supports(node));
        return create_engine_kernel(*compiler_, node);
    }

private:
    std::unique_ptr<TrtDevice> device_;
    std::unique_ptr<Compiler> compiler_;
    std::string name_;
};

Result<std::vector<int>> parse_devices(std::string_view spec, int count) {
    std::vector<int> devices;
    if (spec.empty() || spec == "all") {
        for (int i = 0; i < count; ++i) {
            devices.push_back(i);
        }
        return devices;
    }
    if (spec == "none") {
        return devices;
    }
    while (!spec.empty()) {
        const std::size_t comma = spec.find(',');
        const std::string_view item = spec.substr(0, comma);
        int index = -1;
        const auto [end, error] = std::from_chars(item.data(), item.data() + item.size(), index);
        if (error != std::errc() || end != item.data() + item.size() || index < 0) {
            return errors::invalid_argument("tensorrt.devices: expected 'all', 'none' or indices, got '{}'", item);
        }
        if (index < count) {
            devices.push_back(index);
        }
        spec = comma == std::string_view::npos ? std::string_view() : spec.substr(comma + 1);
    }
    return devices;
}

}  // namespace

Result<std::vector<std::unique_ptr<backend::Backend>>> create_tensorrt_backends(const Options& options,
                                                                                std::string_view library_path,
                                                                                std::string_view devices,
                                                                                Logger logger) {
    std::vector<std::unique_ptr<backend::Backend>> backends;
    if (devices == "none") {
        return backends;
    }
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess || count == 0) {
        cudaGetLastError();
        return errors::unavailable("no CUDA device{}", error != cudaSuccess && error != cudaErrorNoDevice
                                                           ? std::format(" ({})", cudaGetErrorString(error))
                                                           : "");
    }
    II_ASSIGN_OR_RETURN(const std::vector<int> indices, parse_devices(devices, count));
    II_ASSIGN_OR_RETURN(const Library* library, load_library(library_path));
    std::string skipped;
    for (const int index : indices) {
        Result<std::unique_ptr<TrtDevice>> device = TrtDevice::create(index, options);
        if (!device.ok()) {
            skipped += std::format("{}device {}: {}", skipped.empty() ? "" : "; ", index, device.status().message());
            continue;
        }
        II_ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler, Compiler::create(*library, **device, options, logger));
        backends.push_back(std::make_unique<TensorRTBackend>(std::move(*device), std::move(compiler)));
    }
    if (backends.empty() && !indices.empty()) {
        return errors::unavailable("no usable GPU: {}", skipped);
    }
    return backends;
}

}  // namespace ii::tensorrt
