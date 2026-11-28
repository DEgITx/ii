// SPDX-License-Identifier: MIT
// ii-backend-tensorrt: the ii-tensorrt backends (one per NVIDIA GPU) as a backend plugin.
//
// Options (ContextOptions::plugin_options or II_PLUGIN_OPTIONS):
//   tensorrt.devices       "all" (default), device indices, e.g. "0,2", or "none"
//   tensorrt.library_path  a directory with nvinfer (searched before TENSORRT_ROOT and the system path)
//   tensorrt.tf32          "1": f32 products may use TF32 tensor cores
//   tensorrt.prefer        "1": the planner gives the GPU everything and TensorRT every part it can compile (tests)
//   tensorrt.cache         the engine cache directory ("none": none; default: tensorrt in ii's cache directory)
//   tensorrt.level         the builder optimization level, 0-5 (default 3)
//   tensorrt.profile       ranges of symbolic dimensions for the engines, e.g. "N=1:8:64,S=1:128:512" (min:opt:max)
#include <ii/version.h>

#include "backends/tensorrt/tensorrt_backend.h"
#include "plugin/sdk.h"

#include <charconv>

namespace {

ii::Result<ii::plugin::BackendList> create_backends(const ii::plugin::PluginHost& host) {
    ii::tensorrt::Options options;
    options.prefer = host.option("tensorrt.prefer") == "1";
    options.tf32 = host.option("tensorrt.tf32") == "1";
    options.cache = host.option("tensorrt.cache");
    options.profile = host.option("tensorrt.profile");
    if (const std::string level = host.option("tensorrt.level"); !level.empty()) {
        int value = -1;
        const auto [end, error] = std::from_chars(level.data(), level.data() + level.size(), value);
        if (error != std::errc() || end != level.data() + level.size() || value < 0 || value > 5) {
            return ii::errors::invalid_argument("tensorrt.level: expected 0-5, got '{}'", level);
        }
        options.level = value;
    }
    return ii::tensorrt::create_tensorrt_backends(options, host.option("tensorrt.library_path"),
                                                  host.option("tensorrt.devices"), host.logger());
}

constexpr ii::plugin::PluginDefinition kPlugin{"ii-backend-tensorrt", II_VERSION_STRING, &create_backends, true};

}  // namespace

II_DEFINE_BACKEND_PLUGIN(kPlugin)
