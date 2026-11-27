# SPDX-License-Identifier: MIT

# ii-backend-cuda: the NVIDIA GPU backend, a plugin (ii/plugin.h) built where the CUDA Toolkit is found. nvcc needs
# MSVC as the host compiler on Windows, so the plugin is built by the MSVC presets; ii built with any compiler loads
# it. cuBLASLt and NVRTC are loaded when the plugin starts (not linked), so a machine without them reports the
# plugin as unavailable instead of failing to load it.
option(II_BUILD_CUDA "Build the ii-backend-cuda plugin when the CUDA Toolkit is found" ON)

if(NOT II_BUILD_CUDA)
    return()
endif()
if(ANDROID OR APPLE OR II_SANITIZE OR II_BUILD_FUZZERS OR (WIN32 AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "ii: CUDA compiler not found; the ii-backend-cuda plugin is not built")
    return()
endif()

# Real code for the GPUs most likely in use plus PTX that newer GPUs compile at load time. Override with
# -DCMAKE_CUDA_ARCHITECTURES=native (this machine's GPU only; fastest build).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75-virtual;86-real;89-real;120-real")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "ii: building ii-backend-cuda with CUDA ${CUDAToolkit_VERSION} for ${CMAKE_CUDA_ARCHITECTURES}")

# Kernels compiled by nvcc. Kept apart from the plugin's C++ sources, which get the project's warning flags.
add_library(ii_cuda_kernels STATIC
    kernels_conv.cu
    kernels_movement.cu
    kernels_norm.cu
    kernels_reduce.cu)
target_include_directories(ii_cuda_kernels PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_features(ii_cuda_kernels PRIVATE cuda_std_17)
target_compile_options(ii_cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=$<IF:$<CXX_COMPILER_ID:MSVC>,/W3,-Wall>>)
set_target_properties(ii_cuda_kernels PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    # The plugin uses the DLL runtime (/MD), like the rest of the project.
    set_target_properties(ii_cuda_kernels PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

ii_add_backend_plugin(ii-backend-cuda SOURCES
    cuda_backend.cpp
    cuda_device.cpp
    elementwise.cpp
    gemm.cpp
    libraries.cpp
    ops.cpp
    plugin.cpp)
target_link_libraries(ii-backend-cuda PRIVATE ii_cuda_kernels CUDA::cudart_static)
target_include_directories(ii-backend-cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(ii-backend-cuda PRIVATE
    II_CUDA_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
    II_CUDA_TOOLKIT_BIN_DIR="${CUDAToolkit_BIN_DIR}"
    II_CUDA_TOOLKIT_LIBRARY_DIR="${CUDAToolkit_LIBRARY_DIR}")
set_property(GLOBAL PROPERTY II_CUDA_PLUGIN_BUILT TRUE)
