// SPDX-License-Identifier: MIT
// Memory and queues of a GPU for ii-tensorrt when it runs without ii-cuda.
#include "backends/tensorrt/tensorrt_backend.h"

#include <algorithm>
#include <format>

namespace ii::tensorrt {

Status check(cudaError_t error, std::string_view what) {
    if (error == cudaSuccess) {
        return ok_status();
    }
    cudaGetLastError();  // clears a non-sticky error
    if (error == cudaErrorMemoryAllocation) {
        return errors::out_of_memory("{}: {}", what, cudaGetErrorString(error));
    }
    return errors::internal("{}: {} ({})", what, cudaGetErrorString(error), cudaGetErrorName(error));
}

namespace {

#define II_TRT_CUDA_TRY(call) II_TRY(::ii::tensorrt::check((call), #call))

int attribute(cudaDeviceAttr attribute, int device) {
    int value = 0;
    if (cudaDeviceGetAttribute(&value, attribute, device) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return value;
}

/// A CUDA stream of the device.
class TrtQueue final : public backend::Queue {
public:
    TrtQueue(const TrtDevice& device, cudaStream_t stream) noexcept : device_(device), stream_(stream) {}
    ~TrtQueue() override {
        if (device_.activate().ok()) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }
    TrtQueue(const TrtQueue&) = delete;
    TrtQueue& operator=(const TrtQueue&) = delete;

    Status copy_to_device(void* dst, const void* src, std::size_t size) override {
        // From pageable memory the copy is staged before the call returns: `src` may be reused.
        return size == 0 ? ok_status()
                         : check(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream_), "copy");
    }
    Status copy_to_host(void* dst, const void* src, std::size_t size) override {
        return size == 0 ? ok_status()
                         : check(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream_), "copy");
    }
    Status copy(void* dst, const void* src, std::size_t size) override {
        return size == 0 ? ok_status()
                         : check(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream_), "copy");
    }
    Status fill_zero(void* dst, std::size_t size) override {
        return size == 0 ? ok_status() : check(cudaMemsetAsync(dst, 0, size, stream_), "cudaMemsetAsync");
    }
    Status synchronize() override { return check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }
    void* native_handle() const noexcept override { return stream_; }

private:
    const TrtDevice& device_;
    cudaStream_t stream_;
};

}  // namespace

Result<std::unique_ptr<TrtDevice>> TrtDevice::create(int index, const Options& options) {
    std::unique_ptr<TrtDevice> device(new TrtDevice());
    device->index_ = index;
    II_TRY(device->activate());
    cudaDeviceProp properties{};
    II_TRT_CUDA_TRY(cudaGetDeviceProperties(&properties, index));
    device->compute_capability_ = properties.major * 10 + properties.minor;
    if (device->compute_capability_ < 75) {
        return errors::unavailable("{} (compute capability {}.{}) is older than TensorRT 10's minimum 7.5",
                                   properties.name, properties.major, properties.minor);
    }
    for (const char byte : properties.uuid.bytes) {
        device->uuid_ += std::format("{:02x}", static_cast<unsigned char>(byte));
    }
    std::size_t free = 0;
    std::size_t total = 0;
    II_TRT_CUDA_TRY(cudaMemGetInfo(&free, &total));
    const std::size_t reserve = std::size_t{512} << 20;
    backend::DeviceInfo& info = device->info_;
    info.device = Device{DeviceKind::kCuda, index};
    info.name = properties.name;
    info.memory_bytes = free > reserve ? free - reserve : free / 2;
    const double clock_ghz = attribute(cudaDevAttrClockRate, index) * 1e-6;
    const double memory_clock_ghz = attribute(cudaDevAttrMemoryClockRate, index) * 1e-6;
    const int lanes = properties.major >= 8 && !(properties.major == 8 && properties.minor == 0) ? 128 : 64;
    info.gflops = std::max(1.0, properties.multiProcessorCount * lanes * 2 * clock_ghz);
    info.bandwidth_gbs = std::max(1.0, 2 * memory_clock_ghz * properties.memoryBusWidth / 8);
    info.launch_us = 2;  // a kernel in a recorded run (CUDA graph), as for ii-cuda
    info.link_gbs = 20;
    info.link_latency_us = 10;
    if (options.prefer) {
        info.gflops = 1e9;
        info.bandwidth_gbs = 1e9;
        info.launch_us = 0;
        info.link_gbs = 1e6;
        info.link_latency_us = 0;
    }
    return device;
}

Status TrtDevice::activate() const {
    return check(cudaSetDevice(index_), "cudaSetDevice");
}

Result<std::shared_ptr<Buffer>> TrtDevice::allocate(std::size_t size) {
    II_TRY(activate());
    void* data = nullptr;
    II_TRT_CUDA_TRY(cudaMalloc(&data, std::max<std::size_t>(size, 1)));
    const TrtDevice* self = this;
    std::shared_ptr<void> owner(data, [self](void* p) {
        if (self->activate().ok()) {
            cudaFree(p);
        }
    });
    return Buffer::wrap(data, size, std::move(owner), info_.device);
}

Result<std::unique_ptr<backend::Queue>> TrtDevice::create_queue() {
    II_TRY(activate());
    cudaStream_t stream = nullptr;
    II_TRT_CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return std::unique_ptr<backend::Queue>(std::make_unique<TrtQueue>(*this, stream));
}

}  // namespace ii::tensorrt
