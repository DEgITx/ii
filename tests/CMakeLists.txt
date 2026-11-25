# SPDX-License-Identifier: MIT

include(IiDependencies)
include(GoogleTest)
ii_require_googletest()

# A tiny shared library loaded by the DynLib tests.
add_library(ii_test_dynlib SHARED support/test_dynlib.cpp)
target_link_libraries(ii_test_dynlib PRIVATE ii_build_options)

add_executable(ii_tests
    unit/core/allocator_test.cpp
    unit/core/function_ref_test.cpp
    unit/core/half_test.cpp
    unit/core/isa_test.cpp
    unit/core/log_test.cpp
    unit/core/roofline_test.cpp
    unit/core/small_vector_test.cpp
    unit/core/status_test.cpp
    unit/core/system_info_test.cpp
    unit/core/thread_pool_test.cpp
    unit/platform/dynlib_test.cpp
    unit/platform/mapped_file_test.cpp
    unit/platform/platform_test.cpp)

target_link_libraries(ii_tests PRIVATE ii_internal ii_build_options GTest::gtest_main GTest::gmock)
target_compile_definitions(ii_tests PRIVATE "II_TEST_DYNLIB_PATH=\"$<TARGET_FILE:ii_test_dynlib>\"")
add_dependencies(ii_tests ii_test_dynlib)

# PRE_TEST discovery lists tests when ctest runs (not at build time), which keeps cross-compiled builds working.
gtest_discover_tests(ii_tests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 60
    PROPERTIES TIMEOUT 300)
