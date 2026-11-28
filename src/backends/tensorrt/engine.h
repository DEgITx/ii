// SPDX-License-Identifier: MIT
/// @file
/// TensorRT engines of ii-tensorrt: building (with the engine and timing caches) and the kernel that runs them.
#pragma once

#include <ii/log.h>
#include <ii/status.h>

#include "backend/backend.h"
#include "backends/tensorrt/library.h"
#include "backends/tensorrt/tensorrt_backend.h"

#include <NvInfer.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ii::tensorrt {

/// A range of values of a symbol that an engine accepts, and the value it is optimized for.
struct SymbolRange {
    std::int64_t min = 1;
    std::int64_t opt = 1;
    std::int64_t max = 1;
};

/// Forwards TensorRT's messages to the ii logger (errors at debug level: a failed build is reported as a status)
/// and keeps the last error of the calling thread.
class TrtLogger final : public nvinfer1::ILogger {
public:
    explicit TrtLogger(Logger logger) : logger_(std::move(logger)) {}
    void log(Severity severity, const char* message) noexcept override;
    /// The last error TensorRT reported on this thread ("" if none), cleared by reading.
    static std::string take_error();

private:
    Logger logger_;
};

/// Builds engines for one GPU: the TensorRT builder and runtime, the engine cache and the timing cache.
class Compiler {
public:
    static Result<std::unique_ptr<Compiler>> create(const Library& library, const TrtDevice& device,
                                                    const Options& options, Logger logger);
    ~Compiler();
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    /// An engine for `program` (inputs "x<i>", outputs "y<i>") whose dynamic dimensions take the ranges given per
    /// symbol, from the engine cache or newly built. `key` identifies the program and its weights.
    Result<std::shared_ptr<nvinfer1::ICudaEngine>>
    engine(const ir::Graph& program, const std::map<std::string, SymbolRange>& ranges, const std::string& key);

    const TrtDevice& device() const noexcept { return device_; }
    const Options& options() const noexcept { return options_; }
    /// The profile ranges configured for a symbol (`tensorrt.profile`), if any.
    std::optional<SymbolRange> configured_range(const std::string& symbol) const;

private:
    Compiler(const Library& library, const TrtDevice& device, const Options& options, Logger logger);
    void save_timing_cache();

    const Library& library_;
    const TrtDevice& device_;
    Options options_;
    TrtLogger logger_;
    std::map<std::string, SymbolRange> configured_;
    std::optional<std::filesystem::path> cache_dir_;  // none: no engine cache
    std::mutex mutex_;                                // one build at a time
    nvinfer1::IBuilder* builder_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ITimingCache* timing_ = nullptr;  // owned by a builder config while a build runs
    std::string identity_;                      // TensorRT version, GPU and options: part of every key
};

/// Creates the kernel of a `subgraph` node (or of a single node, as a one-node program).
Result<std::unique_ptr<backend::Kernel>> create_engine_kernel(Compiler& compiler, const ir::Node& node);

}  // namespace ii::tensorrt
