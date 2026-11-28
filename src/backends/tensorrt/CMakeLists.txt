# SPDX-License-Identifier: MIT

# ii-backend-tensorrt: NVIDIA TensorRT as a graph backend, a plugin (ii/plugin.h) built where the CUDA Toolkit and the
# TensorRT headers are found. The plugin compiles parts of a model into TensorRT engines; nvinfer is loaded when the
# plugin starts (not linked), so a machine without TensorRT reports the plugin as unavailable instead of failing to
# load it.
option(II_BUILD_TENSORRT "Build the ii-backend-tensorrt plugin when CUDA and TensorRT are found" ON)
set(II_TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT installation (include/NvInfer.h, bin or lib)")

if(NOT II_BUILD_TENSORRT)
    return()
endif()
if(ANDROID OR APPLE OR II_SANITIZE OR II_BUILD_FUZZERS OR (WIN32 AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    return()
endif()
find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
    return()
endif()

set(_tensorrt_hints "${II_TENSORRT_ROOT}")
if(WIN32)
    list(APPEND _tensorrt_hints "C:/TensorRT")
else()
    list(APPEND _tensorrt_hints "/usr" "/usr/local/tensorrt" "/opt/tensorrt")
endif()
find_path(II_TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${_tensorrt_hints} PATH_SUFFIXES include include/x86_64-linux-gnu)
if(NOT II_TENSORRT_INCLUDE_DIR)
    message(STATUS "ii: TensorRT headers not found (set II_TENSORRT_ROOT); the ii-backend-tensorrt plugin is not built")
    return()
endif()
file(STRINGS "${II_TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_major REGEX "^#define TRT_MAJOR_[A-Z]+ [0-9]+")
string(REGEX MATCH "[0-9]+$" II_TENSORRT_MAJOR "${_trt_major}")
if(NOT II_TENSORRT_MAJOR OR II_TENSORRT_MAJOR LESS 10)
    message(STATUS "ii: TensorRT ${II_TENSORRT_MAJOR} is older than 10; the ii-backend-tensorrt plugin is not built")
    return()
endif()
get_filename_component(_tensorrt_root "${II_TENSORRT_INCLUDE_DIR}" DIRECTORY)
message(STATUS "ii: building ii-backend-tensorrt with TensorRT ${II_TENSORRT_MAJOR} (${_tensorrt_root})")

ii_add_backend_plugin(ii-backend-tensorrt SOURCES
    device.cpp
    engine.cpp
    library.cpp
    plugin.cpp
    tensorrt_backend.cpp
    translate.cpp)
target_link_libraries(ii-backend-tensorrt PRIVATE CUDA::cudart_static)
target_include_directories(ii-backend-tensorrt SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS} "${II_TENSORRT_INCLUDE_DIR}")
target_compile_definitions(ii-backend-tensorrt PRIVATE
    II_TENSORRT_MAJOR=${II_TENSORRT_MAJOR}
    II_TENSORRT_DEFAULT_DIR="${_tensorrt_root}")
