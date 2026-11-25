# SPDX-License-Identifier: MIT
#
# Third-party dependencies of tests and benchmarks. The ii library itself has no third-party dependencies.
# Archives are pinned by version and SHA-256. Set II_USE_SYSTEM_DEPS=ON to use installed packages instead.

include(FetchContent)

function(ii_require_googletest)
    if(TARGET GTest::gtest_main)
        return()
    endif()
    if(II_USE_SYSTEM_DEPS)
        find_package(GTest REQUIRED)
        return()
    endif()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
        URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        SYSTEM
        EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(googletest)
endfunction()

function(ii_require_benchmark)
    if(TARGET benchmark::benchmark)
        return()
    endif()
    if(II_USE_SYSTEM_DEPS)
        find_package(benchmark REQUIRED)
        return()
    endif()
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(benchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.10.0.tar.gz
        URL_HASH SHA256=acc5c7c0fbed855c4aaa2d7c3c36fb82b0c15c28f4e151e3f679c2fc0aa404af
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        SYSTEM
        EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(benchmark)
endfunction()
