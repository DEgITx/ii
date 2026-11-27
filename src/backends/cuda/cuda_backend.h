// SPDX-License-Identifier: MIT
/// @file
/// ii-cuda: the backend of one NVIDIA GPU (built as the plugin ii-backend-cuda).
///
/// Kernels enqueue their work on the CUDA stream of the session's queue. Matrix products and convolutions use
/// cuBLASLt (gemm.cpp); elementwise operations and fused elementwise chains are generated as CUDA C++ and compiled
/// with NVRTC when the kernel is created (elementwise.cpp); reductions, normalizations, pooling and data movement
/// are kernels compiled by nvcc (kernels_*.cu, launched from ops.cpp). Operations, dtypes or attributes it does not
/// support are left to the CPU by the planner.
#pragma once

#include <ii/log.h>

#include "backend/backend.h"
#include "backends/cuda/cuda_device.h"
#include "backends/cuda/kernels.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ii::cuda {

using backend::Kernel;
using backend::KernelContext;
using backend::TensorView;

/// Name of the backend of device 0; device i > 0 is "ii-cuda.i".
inline constexpr std::string_view kBackendName = "ii-cuda";

class CudaBackend final : public backend::Backend {
public:
    using Factory = Result<std::unique_ptr<Kernel>> (*)(const ir::Node& node, const CudaDevice& device);
    using SupportCheck = Status (*)(const ir::Node& node);

    CudaBackend(std::string name, std::unique_ptr<CudaDevice> device);

    std::string_view name() const noexcept override { return name_; }
    const backend::DeviceInfo& device_info() const noexcept override { return device_->info(); }
    backend::DeviceApi* device_api() const noexcept override { return device_.get(); }
    Status supports(const ir::Node& node) const override;
    Result<std::unique_ptr<Kernel>> create_kernel(const ir::Node& node) const override;

    void add_kernel(std::string_view op, Factory factory, SupportCheck check = nullptr);

private:
    struct Entry {
        Factory factory = nullptr;
        SupportCheck check = nullptr;
    };
    std::string name_;
    std::unique_ptr<CudaDevice> device_;
    std::unordered_map<std::string_view, Entry> kernels_;
};

/// A kernel defined by a function (most kernels of ii-cuda).
class FunctionKernel final : public Kernel {
public:
    using Run = std::function<Status(const KernelContext& context, cudaStream_t stream)>;
    using Workspace = std::function<std::size_t(std::span<const Shape> inputs, std::span<const Shape> outputs)>;

    FunctionKernel(const CudaDevice& device, Run run, Workspace workspace = nullptr, int in_place_input = -1)
        : device_(device), run_(std::move(run)), workspace_(std::move(workspace)), in_place_input_(in_place_input) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        return workspace_ ? workspace_(inputs, outputs) : 0;
    }
    bool can_run_in_place(int input) const override { return input == in_place_input_; }
    Status run(const KernelContext& context) const override;

private:
    const CudaDevice& device_;
    Run run_;
    Workspace workspace_;
    int in_place_input_;
};

/// Shorthand for creating a FunctionKernel.
inline Result<std::unique_ptr<Kernel>> make_kernel(const CudaDevice& device, FunctionKernel::Run run,
                                                   FunctionKernel::Workspace workspace = nullptr,
                                                   int in_place_input = -1) {
    return std::unique_ptr<Kernel>(
        std::make_unique<FunctionKernel>(device, std::move(run), std::move(workspace), in_place_input));
}

/// The kernel element type of a float or i32 / i64 dtype.
std::optional<Scalar> scalar_of(DType dtype) noexcept;

/// OK if every present input and output of `node` has a dtype in `allowed`.
Status check_dtypes(const ir::Node& node, std::initializer_list<DType> allowed);

/// Element strides of a contiguous shape.
void contiguous_strides(const Shape& shape, std::int64_t* strides) noexcept;

/// Converts a launch status.
Status launched(cudaError_t error, std::string_view kernel);

/// A compiled elementwise program (elementwise.cpp): one kernel per indexing variant.
struct ElementwiseModule {
    cudaKernel_t flat32 = nullptr;
    cudaKernel_t nd32 = nullptr;
    cudaKernel_t flat64 = nullptr;
    cudaKernel_t nd64 = nullptr;
};

/// Runs an elementwise program on `inputs` (broadcast to the output's shape) into `output`. The output may be
/// input 0 when both have the same shape and element size.
Status launch_elementwise(const ElementwiseModule& module, std::span<const TensorView> inputs, const TensorView& output,
                          cudaStream_t stream);

/// The epilogue program of a fused_conv node; with `device` nullptr only checks that it can be generated (and
/// returns nullptr).
Result<const ElementwiseModule*> compile_epilogue(const ir::Node& node, const CudaDevice* device);

void register_elementwise_kernels(CudaBackend& backend);
void register_gemm_kernels(CudaBackend& backend);
void register_ops_kernels(CudaBackend& backend);

/// Creates the backends of the CUDA devices for a plugin instance.
Result<std::vector<std::unique_ptr<backend::Backend>>>
create_cuda_backends(const Options& options, std::string_view library_path, std::string_view devices, Logger logger);

}  // namespace ii::cuda
