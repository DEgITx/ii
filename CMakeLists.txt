# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)

project(ii
    VERSION 0.1.0
    DESCRIPTION "Efficient embeddable C++ library for neural network inference and training"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------------------
option(II_BUILD_SHARED "Build ii as a shared library" OFF)
option(II_BUILD_CLI "Build the ii command-line tool" ${PROJECT_IS_TOP_LEVEL})
option(II_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(II_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(II_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(II_INSTALL "Generate install rules" ${PROJECT_IS_TOP_LEVEL})
option(II_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(II_BUILD_FUZZERS "Build libFuzzer targets (Clang); otherwise fuzz corpora are replayed as tests" OFF)
option(II_USE_SYSTEM_DEPS "Use find_package() for test/benchmark dependencies instead of downloading them" OFF)
set(II_SANITIZE "" CACHE STRING "Sanitizers to enable (GCC/Clang): e.g. 'address;undefined' or 'thread'")
set(II_CPU_ISAS "auto" CACHE STRING "CPU ISA variants to compile kernels for ('auto' = all for the target architecture)")

# ---------------------------------------------------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(IiTargetArch)
include(IiCompilerOptions)
include(IiIsa)
include(IiComponents)
include(IiVersion)

find_package(Threads REQUIRED)

# Android: ctest runs executables on a device or emulator through adb (scripts/android_run.py).
if(ANDROID AND PROJECT_IS_TOP_LEVEL AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        set(CMAKE_CROSSCOMPILING_EMULATOR "${Python3_EXECUTABLE};${PROJECT_SOURCE_DIR}/scripts/android_run.py")
    endif()
endif()

if(II_BUILD_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "ii: II_BUILD_FUZZERS requires Clang (libFuzzer)")
    endif()
    # Coverage instrumentation for all code, so that fuzzing explores the library, not just the fuzz targets.
    add_compile_options(-fsanitize=fuzzer-no-link)
endif()

# ---------------------------------------------------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------------------------------------------------
if(II_BUILD_TESTS)
    enable_testing()  # before any directory that registers tests (tests, CLI smoke tests, examples)
endif()

add_subdirectory(src)

if(II_BUILD_CLI)
    add_subdirectory(tools/ii)
endif()

if(II_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(II_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(II_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(II_INSTALL)
    include(IiInstall)
endif()

message(STATUS "ii ${PROJECT_VERSION}: arch=${II_TARGET_ARCH}, ISA variants=[${II_ISA_VARIANTS}], "
               "shared=${II_BUILD_SHARED}, tests=${II_BUILD_TESTS}, benchmarks=${II_BUILD_BENCHMARKS}")
