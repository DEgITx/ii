# SPDX-License-Identifier: MIT

include(IiDependencies)
ii_require_benchmark()

add_executable(ii_bench_micro
    micro/bench_half.cpp
    micro/bench_thread_pool.cpp)

target_link_libraries(ii_bench_micro PRIVATE ii_internal ii_build_options benchmark::benchmark_main)
