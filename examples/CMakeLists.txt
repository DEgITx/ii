# SPDX-License-Identifier: MIT
#
# Examples. Built as part of the ii build (II_BUILD_EXAMPLES), or standalone against an installed ii:
#
#   cmake -S examples -B build-examples -DCMAKE_PREFIX_PATH=<ii install prefix>
#   cmake --build build-examples
cmake_minimum_required(VERSION 3.25)
project(ii_examples LANGUAGES C CXX)

if(NOT TARGET ii::ii)
    find_package(ii REQUIRED)
endif()

add_executable(ii_example_mlp mlp/main.cpp)
target_link_libraries(ii_example_mlp PRIVATE ii::ii)
target_compile_features(ii_example_mlp PRIVATE cxx_std_20)

add_executable(ii_example_c_api c_api/main.c)
target_link_libraries(ii_example_c_api PRIVATE ii::ii)
set_target_properties(ii_example_c_api PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    # ii is a C++ library: a static build needs the C++ runtime at link time.
    LINKER_LANGUAGE CXX)

if(TARGET ii_build_options)
    target_link_libraries(ii_example_mlp PRIVATE ii_build_options)
endif()

# The examples double as smoke tests.
if(BUILD_TESTING OR II_BUILD_TESTS)
    add_test(NAME example_mlp COMMAND ii_example_mlp)
    add_test(NAME example_c_api COMMAND ii_example_c_api)
    if(WIN32)
        # Find ii.dll of a shared build next to the test executables.
        set_tests_properties(example_mlp example_c_api PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:ii::ii>")
    endif()
endif()
