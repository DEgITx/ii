// SPDX-License-Identifier: MIT
/// @file
/// A CUDA device of ii-cuda: memory (cudaMalloc), queues (CUDA streams), the cuBLASLt handle and the figures for the
/// planner.
#pragma once

#include "backend/backend.h"
#include "backends/cuda/autotune.h"
#include "backends/cuda/kernels.h"
#include "backends/cuda/libraries.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <string_view>

namespace ii::cuda {

/// OK, or an error naming `what` (kOutOfMemory for failed allocations).
Status check(cudaError_t error, std::string_view what);

/// Returns the error status of a CUDA runtime call.
#define II_CUDA_TRY(expr) II_TRY(::ii::cuda::check((expr), #expr))

/// Plugin options ("cuda.*" keys of ContextOptions::plugin_options or II_PLUGIN_OPTIONS).
struct Options {
    /// "cuda.prefer=1": report the device as very fast, so the planner puts every operation it supports on it
    /// (tests; like a test device with `launch=0,...`).
    bool prefer = false;
    /// "cuda.tf32=1": f32 matrix products and convolutions may use TF32 tensor cores (faster, ~1e-3 relative error).
    bool tf32 = false;
    /// "cuda.graphs=0": no CUDA graphs (the runtime then launches every kernel of every run).
    bool graphs = true;
    /// "cuda.autotune=0": no measurements (cuBLASLt's first algorithm, the default warps per weight row, the
    /// device's nominal figures for the planner) and no cache file.
    bool autotune = true;
    /// "cuda.cache": the autotuning cache file; empty: in ii's cache directory; "none": kept in memory only.
    std::string cache;
};

class CudaDevice final : public backend::DeviceApi {
public:
    static Result<std::unique_ptr<CudaDevice>> create(int index, const Libraries& libraries, const Options& options);
    ~CudaDevice() override;
    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(const CudaDevice&) = delete;

    const backend::DeviceInfo& info() const noexcept override { return info_; }
    Result<std::shared_ptr<Buffer>> allocate(std::size_t size) override;
    Result<std::unique_ptr<backend::Queue>> create_queue() override;
    Result<std::shared_ptr<Buffer>> allocate_host(std::size_t size) override;

    /// Makes the device current on the calling thread.
    Status activate() const;

    int index() const noexcept { return index_; }
    int sm_count() const noexcept { return sm_count_; }
    /// e.g. 120 for compute capability 12.0.
    int compute_capability() const noexcept { return compute_capability_; }
    const Libraries& libraries() const noexcept { return libraries_; }
    cublasLtHandle_t blas() const noexcept { return blas_; }
    const Options& options() const noexcept { return options_; }

    /// Device memory owned by a kernel (e.g. precomputed tables), freed with it.
    Result<std::shared_ptr<void>> allocate_owned(std::size_t size) const;

    /// Measured choices kept between runs; nullptr without autotuning.
    TuningCache* tuning() const noexcept { return tuning_.get(); }

private:
    CudaDevice(int index, const Libraries& libraries, const Options& options)
        : index_(index), libraries_(libraries), options_(options) {}

    int index_;
    const Libraries& libraries_;
    Options options_;
    backend::DeviceInfo info_;
    int sm_count_ = 1;
    int compute_capability_ = 0;
    cublasLtHandle_t blas_ = nullptr;
    std::unique_ptr<TuningCache> tuning_;
};

/// A CUDA stream.
class CudaQueue final : public backend::Queue {
public:
    static Result<std::unique_ptr<CudaQueue>> create(const CudaDevice& device);
    ~CudaQueue() override;
    CudaQueue(const CudaQueue&) = delete;
    CudaQueue& operator=(const CudaQueue&) = delete;

    Status copy_to_device(void* dst, const void* src, std::size_t size) override;
    Status copy_to_host(void* dst, const void* src, std::size_t size) override;
    Status copy(void* dst, const void* src, std::size_t size) override;
    Status fill_zero(void* dst, std::size_t size) override;
    Status synchronize() override;
    /// CUDA graphs: stream capture, then cudaGraphLaunch.
    Status begin_recording() override;
    Result<std::unique_ptr<backend::Recording>> end_recording() override;
    Status replay(const backend::Recording& recording) override;

    cudaStream_t stream() const noexcept { return stream_; }
    const CudaDevice& device() const noexcept { return device_; }
    void* native_handle() const noexcept override { return stream_; }

    /// Where kernels on this queue report errors (device address of host memory).
    DeviceError* error() const noexcept { return error_device_; }

private:
    CudaQueue(const CudaDevice& device, cudaStream_t stream, DeviceError* host, DeviceError* device_address)
        : device_(device), stream_(stream), error_host_(host), error_device_(device_address) {}

    const CudaDevice& device_;
    cudaStream_t stream_;
    DeviceError* error_host_;
    DeviceError* error_device_;
};

/// The stream a kernel of ii-cuda runs on.
inline cudaStream_t stream_of(const backend::KernelContext& context) noexcept {
    return static_cast<const CudaQueue*>(context.queue)->stream();
}

/// Where a kernel of ii-cuda reports errors.
inline DeviceError* error_of(const backend::KernelContext& context) noexcept {
    return static_cast<const CudaQueue*>(context.queue)->error();
}

/// The number of CUDA devices; an kUnavailable error without a driver or a device.
Result<int> device_count();

}  // namespace ii::cuda
