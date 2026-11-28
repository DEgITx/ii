// SPDX-License-Identifier: MIT
/// @file
/// The TensorRT library (nvinfer), loaded when the plugin starts.
///
/// It is not linked: a machine with an NVIDIA driver but without TensorRT gets a plugin that reports itself
/// unavailable (with the places searched) instead of a plugin that fails to load. TensorRT's C++ objects are reached
/// through the two factory functions it exports; everything else is virtual calls.
#pragma once

#include <ii/status.h>

#include <NvInfer.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace ii::tensorrt {

struct Library {
    void* (*create_builder)(void* logger, std::int32_t version) = nullptr;  // createInferBuilder_INTERNAL
    void* (*create_runtime)(void* logger, std::int32_t version) = nullptr;  // createInferRuntime_INTERNAL
    std::int32_t version = 0;                                               // getInferLibVersion()
    std::string path;                                                       // the loaded file

    nvinfer1::IBuilder* builder(nvinfer1::ILogger& logger) const {
        return static_cast<nvinfer1::IBuilder*>(create_builder(&logger, NV_TENSORRT_VERSION));
    }
    nvinfer1::IRuntime* runtime(nvinfer1::ILogger& logger) const {
        return static_cast<nvinfer1::IRuntime*>(create_runtime(&logger, NV_TENSORRT_VERSION));
    }
    /// "10.16.1".
    std::string version_string() const;
};

/// Loads nvinfer (once per process) from `directory` if given, else next to the plugin, TENSORRT_ROOT / its bin
/// directory, the TensorRT the plugin was built with, and the system search path. An kUnavailable error says where
/// it looked.
Result<const Library*> load_library(std::string_view directory);

}  // namespace ii::tensorrt
