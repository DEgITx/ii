# SPDX-License-Identifier: MIT
#
# Fuzz targets. With II_BUILD_FUZZERS (Clang only) they are libFuzzer executables and ctest runs each for 60 seconds
# on its corpus ("fuzz smoke"); otherwise they are built with a replay driver and ctest runs the corpus once, so every
# build checks the corpus as a regression suite.

function(ii_add_fuzzer name corpus)
    set(corpus_dir "${CMAKE_CURRENT_SOURCE_DIR}/corpus/${corpus}")
    if(II_BUILD_FUZZERS)
        add_executable(${name} ${name}.cpp)
        target_compile_options(${name} PRIVATE -fsanitize=fuzzer)
        target_link_options(${name} PRIVATE -fsanitize=fuzzer)
        # libFuzzer adds new inputs to the first corpus directory: keep them out of the source tree.
        set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/${corpus}-work")
        file(MAKE_DIRECTORY "${work_dir}")
        add_test(NAME ${name} COMMAND ${name} -max_total_time=60 -timeout=10 -rss_limit_mb=2048 "${work_dir}"
                                      "${corpus_dir}")
    else()
        add_executable(${name} ${name}.cpp replay_main.cpp)
        add_test(NAME ${name}_replay COMMAND ${name} "${corpus_dir}")
    endif()
    target_link_libraries(${name} PRIVATE ii_internal ii_build_options)
endfunction()

ii_add_fuzzer(fuzz_ir_text ir_text)
ii_add_fuzzer(fuzz_json json)
