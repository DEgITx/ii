// SPDX-License-Identifier: MIT
/// @file
/// ii-tensorrt: NVIDIA TensorRT as a graph backend (plugin ii-backend-tensorrt).
///
/// The backend compiles parts of a model into TensorRT engines. After planning, the compiler hands it the parts of
/// the graph on its GPU that it can translate (`subgraph` nodes, runtime/partition.h) when its estimate beats the
/// planned kernels; with ii-cuda loaded both run on the same GPU, in the same memory and on the same CUDA stream (the
/// queue's native handle), so a plan mixes TensorRT engines and ii-cuda kernels without copies.
///
/// - Translation (translate.h): ii operations to TensorRT layers in a strongly typed network, so that TensorRT keeps
///   the model's types (f32 stays f32; TF32 only with `tensorrt.tf32`). Composite operations without a layer of their
///   own are decomposed by ii first.
/// - Dynamic shapes: symbolic dimensions of a part are dynamic dimensions of the engine; an engine has one
///   optimization profile per range of the symbols' values (by default 1 .. twice the value seen first, the opt point
///   at that value), and a value outside every profile builds another engine.
/// - Engine cache: serialized engines in `<cache>/tensorrt/` keyed by the part (program and weights), the profile,
///   the options, the TensorRT version and the GPU; TensorRT's timing cache next to them shortens new builds.
#pragma once

#include <ii/log.h>
#include <ii/status.h>

#include "backend/backend.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ii::tensorrt {

/// Plugin options ("tensorrt.*" keys of ContextOptions::plugin_options or II_PLUGIN_OPTIONS).
struct Options {
    bool tf32 = false;    ///< f32 products may use TF32 tensor cores
    bool prefer = false;  ///< the planner gives the GPU everything and TensorRT every part it can compile (tests)
    std::string cache;    ///< engine cache directory; "none": none; empty: `<ii cache>/tensorrt`
    int level = 3;        ///< TensorRT builder optimization level (0-5)
    std::string profile;  ///< symbol ranges of the optimization profiles, e.g. "N=1:8:64,S=1:128:512" (min:opt:max)
};

Status check(cudaError_t error, std::string_view what);

/// Memory and queues of one GPU when ii-cuda does not provide them (TensorRT loaded alone).
class TrtDevice final : public backend::DeviceApi {
public:
    static Result<std::unique_ptr<TrtDevice>> create(int index, const Options& options);

    const backend::DeviceInfo& info() const noexcept override { return info_; }
    Result<std::shared_ptr<Buffer>> allocate(std::size_t size) override;
    Result<std::unique_ptr<backend::Queue>> create_queue() override;
    Result<std::shared_ptr<Buffer>> allocate_host(std::size_t size) override;

    int index() const noexcept { return index_; }
    int compute_capability() const noexcept { return compute_capability_; }
    const std::string& uuid() const noexcept { return uuid_; }
    /// Makes the device current on the calling thread.
    Status activate() const;

private:
    TrtDevice() = default;

    int index_ = 0;
    int compute_capability_ = 0;
    std::string uuid_;
    backend::DeviceInfo info_;
};

/// Creates one backend per usable GPU (`devices`: "all", indices such as "0,2", or "none").
Result<std::vector<std::unique_ptr<backend::Backend>>> create_tensorrt_backends(const Options& options,
                                                                                std::string_view library_path,
                                                                                std::string_view devices,
                                                                                Logger logger);

}  // namespace ii::tensorrt
