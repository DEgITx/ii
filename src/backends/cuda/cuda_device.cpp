// SPDX-License-Identifier: MIT
#include "backends/cuda/cuda_device.h"

#include <algorithm>
#include <cstring>
#include <format>

namespace ii::cuda {

Status check(cudaError_t error, std::string_view what) {
    if (error == cudaSuccess) {
        return ok_status();
    }
    cudaGetLastError();  // clears a non-sticky error so that later calls are not reported as failed
    if (error == cudaErrorMemoryAllocation) {
        return errors::out_of_memory("{}: {}", what, cudaGetErrorString(error));
    }
    return errors::internal("{}: {} ({})", what, cudaGetErrorString(error), cudaGetErrorName(error));
}

Result<int> device_count() {
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || (error == cudaSuccess && count == 0)) {
        cudaGetLastError();
        return errors::unavailable("no CUDA device");
    }
    if (error != cudaSuccess) {
        cudaGetLastError();
        return errors::unavailable("no usable NVIDIA driver: {}", cudaGetErrorString(error));
    }
    return count;
}

namespace {

/// FP32 lanes per streaming multiprocessor.
int fp32_lanes(int major, int minor) {
    if (major == 8 && minor == 0) {
        return 64;  // A100
    }
    if (major >= 8) {
        return 128;
    }
    return 64;
}

int attribute(cudaDeviceAttr attribute, int device) {
    int value = 0;
    if (cudaDeviceGetAttribute(&value, attribute, device) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return value;
}

}  // namespace

Result<std::unique_ptr<CudaDevice>> CudaDevice::create(int index, const Libraries& libraries, const Options& options) {
    std::unique_ptr<CudaDevice> device(new CudaDevice(index, libraries, options));
    II_TRY(device->activate());
    cudaDeviceProp properties{};
    II_CUDA_TRY(cudaGetDeviceProperties(&properties, index));
    device->sm_count_ = properties.multiProcessorCount;
    device->compute_capability_ = properties.major * 10 + properties.minor;
    if (properties.major < 7 || (properties.major == 7 && properties.minor < 5)) {
        return errors::unavailable("{} (compute capability {}.{}) is older than the minimum 7.5", properties.name,
                                   properties.major, properties.minor);
    }

    std::size_t free = 0;
    std::size_t total = 0;
    II_CUDA_TRY(cudaMemGetInfo(&free, &total));
    const std::size_t reserve = std::size_t{512} << 20;  // CUDA context, cuBLASLt, other applications

    backend::DeviceInfo& info = device->info_;
    info.device = Device{DeviceKind::kCuda, index};
    info.name = properties.name;
    info.memory_bytes = free > reserve ? free - reserve : free / 2;
    const double clock_ghz = attribute(cudaDevAttrClockRate, index) * 1e-6;
    const double memory_clock_ghz = attribute(cudaDevAttrMemoryClockRate, index) * 1e-6;
    info.gflops =
        std::max(1.0, properties.multiProcessorCount * fp32_lanes(properties.major, properties.minor) * 2 * clock_ghz);
    info.bandwidth_gbs = std::max(1.0, 2 * memory_clock_ghz * properties.memoryBusWidth / 8);
    info.launch_us = 4;
    info.link_gbs = 20;
    info.link_latency_us = 10;
    if (options.prefer) {
        info.gflops = 1e9;
        info.bandwidth_gbs = 1e9;
        info.launch_us = 0;
        info.link_gbs = 1e6;
        info.link_latency_us = 0;
    }

    const cublasStatus_t status = libraries.lt_create(&device->blas_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return errors::unavailable("cublasLtCreate: {}", libraries.lt_status_string(status));
    }
    return device;
}

CudaDevice::~CudaDevice() {
    if (blas_ != nullptr) {
        libraries_.lt_destroy(blas_);
    }
}

Status CudaDevice::activate() const {
    int current = -1;
    if (cudaGetDevice(&current) == cudaSuccess && current == index_) {
        return ok_status();
    }
    II_CUDA_TRY(cudaSetDevice(index_));
    return ok_status();
}

Result<std::shared_ptr<Buffer>> CudaDevice::allocate(std::size_t size) {
    II_ASSIGN_OR_RETURN(std::shared_ptr<void> memory, allocate_owned(size));
    void* data = memory.get();
    return Buffer::wrap(data, size, std::move(memory), info_.device);
}

Result<std::shared_ptr<void>> CudaDevice::allocate_owned(std::size_t size) const {
    II_TRY(activate());
    void* data = nullptr;
    if (const cudaError_t error = cudaMalloc(&data, std::max<std::size_t>(size, 1)); error != cudaSuccess) {
        cudaGetLastError();
        return errors::out_of_memory("{}: allocating {} bytes: {}", info_.name, size, cudaGetErrorString(error));
    }
    const int index = index_;
    return std::shared_ptr<void>(data, [index](void* p) {
        int current = -1;
        if (cudaGetDevice(&current) == cudaSuccess && current != index) {
            cudaSetDevice(index);
        }
        cudaFree(p);
    });
}

Result<std::unique_ptr<backend::Queue>> CudaDevice::create_queue() {
    II_ASSIGN_OR_RETURN(std::unique_ptr<CudaQueue> queue, CudaQueue::create(*this));
    return std::unique_ptr<backend::Queue>(std::move(queue));
}

// ---------------------------------------------------------------------------------------------------------------------

Result<std::unique_ptr<CudaQueue>> CudaQueue::create(const CudaDevice& device) {
    II_TRY(device.activate());
    void* host = nullptr;
    II_CUDA_TRY(cudaHostAlloc(&host, sizeof(DeviceError), cudaHostAllocMapped));
    std::memset(host, 0, sizeof(DeviceError));
    void* device_address = nullptr;
    cudaStream_t stream = nullptr;
    if (const Status status = check(cudaHostGetDevicePointer(&device_address, host, 0), "cudaHostGetDevicePointer");
        !status.ok()) {
        cudaFreeHost(host);
        return status;
    }
    if (const Status status = check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        !status.ok()) {
        cudaFreeHost(host);
        return status;
    }
    return std::unique_ptr<CudaQueue>(
        new CudaQueue(device, stream, static_cast<DeviceError*>(host), static_cast<DeviceError*>(device_address)));
}

CudaQueue::~CudaQueue() {
    if (device_.activate().ok()) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        cudaFreeHost(error_host_);
    }
}

Status CudaQueue::copy_to_device(void* dst, const void* src, std::size_t size) {
    if (size == 0) {
        return ok_status();
    }
    // From pageable memory the copy goes through a staging buffer before the call returns, so `src` may be reused.
    II_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream_));
    return ok_status();
}

Status CudaQueue::copy_to_host(void* dst, const void* src, std::size_t size) {
    if (size == 0) {
        return ok_status();
    }
    II_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream_));
    return ok_status();
}

Status CudaQueue::copy(void* dst, const void* src, std::size_t size) {
    if (size == 0) {
        return ok_status();
    }
    II_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream_));
    return ok_status();
}

Status CudaQueue::fill_zero(void* dst, std::size_t size) {
    if (size == 0) {
        return ok_status();
    }
    II_CUDA_TRY(cudaMemsetAsync(dst, 0, size, stream_));
    return ok_status();
}

Status CudaQueue::synchronize() {
    II_CUDA_TRY(cudaStreamSynchronize(stream_));
    volatile DeviceError* error = error_host_;
    const auto code = static_cast<DeviceErrorCode>(error->code);
    if (code == DeviceErrorCode::kNone) {
        return ok_status();
    }
    const long long value = error->value;
    const long long limit = error->limit;
    error->code = 0;
    switch (code) {
    case DeviceErrorCode::kGatherIndex:
        return errors::out_of_range("gather index {} is out of range for axis size {}", value, limit);
    default:
        return errors::internal("device error {}", static_cast<long long>(code));
    }
}

}  // namespace ii::cuda
