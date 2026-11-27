// SPDX-License-Identifier: MIT
// ii-backend-cuda: the ii-cuda backends (one per NVIDIA GPU) as a backend plugin.
//
// Options (ContextOptions::plugin_options or II_PLUGIN_OPTIONS):
//   cuda.devices       "all" (default), device indices, e.g. "0,2", or "none"
//   cuda.library_path  a directory with cuBLASLt and NVRTC (searched before CUDA_PATH and the system path)
//   cuda.tf32          "1": f32 matrix products may use TF32 tensor cores
//   cuda.prefer        "1": the planner puts everything the GPU supports on it (tests)
//   cuda.graphs        "0": no CUDA graphs (repeated runs of kernels are otherwise recorded once and replayed)
//   cuda.autotune      "0": no measurements of algorithms and device figures (see autotune.h)
//   cuda.cache         the autotuning cache file ("none": none; default: cuda-<device uuid>.json in ii's cache
//   directory)
#include <ii/version.h>

#include "backends/cuda/cuda_backend.h"
#include "plugin/sdk.h"

namespace {

ii::Result<ii::plugin::BackendList> create_backends(const ii::plugin::PluginHost& host) {
    ii::cuda::Options options;
    options.prefer = host.option("cuda.prefer") == "1";
    options.tf32 = host.option("cuda.tf32") == "1";
    options.graphs = host.option("cuda.graphs") != "0";
    options.autotune = host.option("cuda.autotune") != "0";
    options.cache = host.option("cuda.cache");
    return ii::cuda::create_cuda_backends(options, host.option("cuda.library_path"), host.option("cuda.devices"),
                                          host.logger());
}

constexpr ii::plugin::PluginDefinition kPlugin{"ii-backend-cuda", II_VERSION_STRING, &create_backends};

}  // namespace

II_DEFINE_BACKEND_PLUGIN(kPlugin)
