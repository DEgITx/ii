// SPDX-License-Identifier: MIT
/// @file
/// Devices and memory domains.
///
/// Every backend executes on one device. Backends on the same device share its memory domain: buffers move between
/// them without copies (ii-ref and ii-cpu share the host; ii-cuda and TensorRT share a GPU). Values that cross
/// domains are copied by `transfer` steps the runtime inserts into the plan.
///
/// A device other than the host is driven through a `DeviceApi` (memory) and per-session `Queue`s (ordered,
/// possibly asynchronous work: a CUDA stream). Kernels of a device backend enqueue their work on the queue they get
/// in `KernelContext::queue`; the runtime synchronizes before the host reads results.
#pragma once

#include <ii/status.h>
#include <ii/tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ii::backend {

/// Capacity and speed of a device: the inputs of the planner's cost model. Estimates are fine; they only need to
/// rank devices and backends sensibly.
struct DeviceInfo {
    Device device;
    std::string name;                ///< e.g. "Intel(R) Core(TM) Ultra 7 265KF", "NVIDIA GeForce RTX 5080"
    std::uint64_t memory_bytes = 0;  ///< memory ii may use on the device; 0: unlimited (host memory)
    double gflops = 1;               ///< peak arithmetic throughput, GFLOP/s
    double bandwidth_gbs = 1;        ///< memory bandwidth, GB/s
    double launch_us = 0;            ///< fixed cost of executing one kernel, microseconds
    double link_gbs = 0;             ///< host <-> device copies, GB/s (unused for the host)
    double link_latency_us = 0;      ///< fixed cost of one host <-> device copy
};

/// Ordered work on a device for one session (a CUDA stream). Not thread-safe; a session uses it from one thread.
///
/// Operations execute in the order they are enqueued, possibly after the call returns. Pointers are device
/// addresses unless stated otherwise.
class Queue {
public:
    virtual ~Queue() = default;

    /// Copies `size` bytes from host memory `src` to `dst`, after the work enqueued before. `src` may be reused as
    /// soon as the call returns.
    virtual Status copy_to_device(void* dst, const void* src, std::size_t size) = 0;

    /// Copies `size` bytes from `src` to host memory `dst`, after the work enqueued before. `dst` holds the data
    /// once `synchronize()` returns.
    virtual Status copy_to_host(void* dst, const void* src, std::size_t size) = 0;

    /// Copies `size` bytes within the device (the ranges do not overlap).
    virtual Status copy(void* dst, const void* src, std::size_t size) = 0;

    /// Sets `size` bytes at `dst` to zero.
    virtual Status fill_zero(void* dst, std::size_t size) = 0;

    /// Waits until all enqueued work has finished; returns the first error of that work.
    virtual Status synchronize() = 0;

protected:
    Queue() = default;
    Queue(const Queue&) = default;
    Queue& operator=(const Queue&) = default;
};

/// Memory and queues of a device other than the host. Thread-safe.
class DeviceApi {
public:
    virtual ~DeviceApi() = default;

    virtual const DeviceInfo& info() const noexcept = 0;

    /// Allocates `size` bytes of device memory (at least 256-byte aligned, contents undefined). The memory is freed
    /// when the last reference to the buffer goes away; the device API must outlive it.
    virtual Result<std::shared_ptr<Buffer>> allocate(std::size_t size) = 0;

    /// Creates a queue (one per session).
    virtual Result<std::unique_ptr<Queue>> create_queue() = 0;

protected:
    DeviceApi() = default;
    DeviceApi(const DeviceApi&) = default;
    DeviceApi& operator=(const DeviceApi&) = default;
};

/// Rough performance figures of this machine's CPU for `threads` threads with `flops_per_cycle` single-precision
/// operations per cycle and core (8 for SSE2, 16 for NEON, 32 for AVX2 with FMA, 64 for AVX-512).
DeviceInfo host_device_info(int threads, double flops_per_cycle);

}  // namespace ii::backend
