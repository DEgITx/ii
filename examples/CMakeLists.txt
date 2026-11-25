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

add_executable(ii_example_llm llm/main.cpp)
target_link_libraries(ii_example_llm PRIVATE ii::ii)
target_compile_features(ii_example_llm PRIVATE cxx_std_20)

add_executable(ii_example_c_api c_api/main.c)
target_link_libraries(ii_example_c_api PRIVATE ii::ii)
set_target_properties(ii_example_c_api PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    # ii is a C++ library: a static build needs the C++ runtime at link time.
    LINKER_LANGUAGE CXX)

# Object detection with YOLOv8 (images are decoded with stb_image, fetched at configure time).
if(COMMAND ii_require_stb)
    ii_require_stb()
else()
    include(FetchContent)
    FetchContent_Declare(stb
        URL https://github.com/nothings/stb/archive/aeb69073d7570a481245a665dc17726d68827062.tar.gz
        URL_HASH SHA256=a3bf405a0625a153ba9d51511ec0ef82e1ce0995f273e8a85858e956154c83a2
        DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(stb)
    add_library(ii_stb INTERFACE)
    target_include_directories(ii_stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
endif()
add_executable(ii_example_yolo yolo/main.cpp)
target_link_libraries(ii_example_yolo PRIVATE ii::ii ii_stb)
target_compile_features(ii_example_yolo PRIVATE cxx_std_20)

if(TARGET ii_build_options)
    target_link_libraries(ii_example_mlp PRIVATE ii_build_options)
    target_link_libraries(ii_example_llm PRIVATE ii_build_options)
    target_link_libraries(ii_example_yolo PRIVATE ii_build_options)
endif()

# The examples double as smoke tests.
if(BUILD_TESTING OR II_BUILD_TESTS)
    add_test(NAME example_mlp COMMAND ii_example_mlp)
    add_test(NAME example_c_api COMMAND ii_example_c_api)
    # A tiny random model of the test data (part of the ii repository, not of a standalone examples build).
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/data/llm/qwen3.gguf")
        add_test(NAME example_llm COMMAND ii_example_llm "${CMAKE_CURRENT_SOURCE_DIR}/../tests/data/llm/qwen3.gguf" "Hi")
        set_tests_properties(example_llm PROPERTIES PASS_REGULAR_EXPRESSION "generated")
    endif()
    # The YOLO example needs a model and an image (not in the repository): set II_EXAMPLE_YOLO_MODEL and
    # II_EXAMPLE_YOLO_IMAGE to run it as a test (the model tests in CI do).
    if(II_EXAMPLE_YOLO_MODEL AND II_EXAMPLE_YOLO_IMAGE)
        add_test(NAME example_yolo COMMAND ii_example_yolo "${II_EXAMPLE_YOLO_MODEL}" "${II_EXAMPLE_YOLO_IMAGE}")
        set_tests_properties(example_yolo PROPERTIES PASS_REGULAR_EXPRESSION "person")
    endif()
    if(WIN32)
        # Find ii.dll of a shared build next to the test executables.
        set_tests_properties(example_mlp example_c_api PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:ii::ii>")
        if(TEST example_llm)
            set_tests_properties(example_llm PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:ii::ii>")
        endif()
    endif()
endif()
