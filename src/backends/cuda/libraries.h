// SPDX-License-Identifier: MIT
/// @file
/// cuBLASLt and NVRTC, loaded when the plugin starts.
///
/// They are not linked: a machine with an NVIDIA driver but without these libraries then gets a plugin that reports
/// itself unavailable (with the places searched) instead of a plugin that fails to load. The CUDA runtime is linked
/// statically and talks to the driver directly.
#pragma once

#include <ii/status.h>

#include <cublasLt.h>
#include <nvrtc.h>

#include <string>
#include <string_view>

namespace ii::cuda {

struct Libraries {
    // cuBLASLt
    decltype(&cublasLtCreate) lt_create = nullptr;
    decltype(&cublasLtDestroy) lt_destroy = nullptr;
    decltype(&cublasLtGetVersion) lt_version = nullptr;
    decltype(&cublasLtGetStatusString) lt_status_string = nullptr;
    decltype(&cublasLtMatmulDescCreate) matmul_desc_create = nullptr;
    decltype(&cublasLtMatmulDescDestroy) matmul_desc_destroy = nullptr;
    decltype(&cublasLtMatmulDescSetAttribute) matmul_desc_set = nullptr;
    decltype(&cublasLtMatrixLayoutCreate) layout_create = nullptr;
    decltype(&cublasLtMatrixLayoutDestroy) layout_destroy = nullptr;
    decltype(&cublasLtMatrixLayoutSetAttribute) layout_set = nullptr;
    decltype(&cublasLtMatmulPreferenceCreate) preference_create = nullptr;
    decltype(&cublasLtMatmulPreferenceDestroy) preference_destroy = nullptr;
    decltype(&cublasLtMatmulPreferenceSetAttribute) preference_set = nullptr;
    decltype(&cublasLtMatmulAlgoGetHeuristic) algo_heuristic = nullptr;
    decltype(&cublasLtMatmul) matmul = nullptr;
    /// Descriptors in caller-provided (stack) memory: no allocation when a kernel runs.
    decltype(&cublasLtMatmulDescInit_internal) matmul_desc_init = nullptr;
    decltype(&cublasLtMatrixLayoutInit_internal) layout_init = nullptr;

    // NVRTC
    decltype(&nvrtcVersion) nvrtc_version = nullptr;
    decltype(&nvrtcGetErrorString) nvrtc_error_string = nullptr;
    decltype(&nvrtcCreateProgram) nvrtc_create = nullptr;
    decltype(&nvrtcDestroyProgram) nvrtc_destroy = nullptr;
    decltype(&nvrtcCompileProgram) nvrtc_compile = nullptr;
    decltype(&nvrtcGetProgramLogSize) nvrtc_log_size = nullptr;
    decltype(&nvrtcGetProgramLog) nvrtc_log = nullptr;
    decltype(&nvrtcGetCUBINSize) nvrtc_cubin_size = nullptr;
    decltype(&nvrtcGetCUBIN) nvrtc_cubin = nullptr;
    decltype(&nvrtcGetSupportedArchs) nvrtc_supported_archs = nullptr;
    decltype(&nvrtcGetNumSupportedArchs) nvrtc_num_supported_archs = nullptr;

    std::string description;  ///< versions and where the libraries were found
};

/// Loads the libraries (once per process; later calls return the same result). `directory`: searched first (the
/// plugin option "cuda.library_path"); then the plugin's directory, CUDA_PATH, the toolkit ii was built with and
/// the system's library path. An `kUnavailable` error lists the places searched.
Result<const Libraries*> load_libraries(std::string_view directory);

}  // namespace ii::cuda
