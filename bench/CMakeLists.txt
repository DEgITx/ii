# SPDX-License-Identifier: MIT

include(IiDependencies)
ii_require_benchmark()

add_executable(ii_bench_micro
    micro/bench_cpu.cpp
    micro/bench_half.cpp
    micro/bench_quant.cpp
    micro/bench_runtime.cpp
    micro/bench_thread_pool.cpp)

target_link_libraries(ii_bench_micro PRIVATE ii_internal ii_build_options benchmark::benchmark_main)
target_include_directories(ii_bench_micro PRIVATE "${PROJECT_SOURCE_DIR}/src")
