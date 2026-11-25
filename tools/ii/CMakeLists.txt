# SPDX-License-Identifier: MIT

add_executable(ii_cli
    main.cpp
    args.cpp
    console.cpp
    command_info.cpp
    command_inspect.cpp
    command_llm.cpp
    command_ops.cpp
    command_run.cpp
    image.cpp)

include(IiDependencies)
ii_require_stb()
target_link_libraries(ii_cli PRIVATE ii::ii ii_build_options ii_stb)
if(WIN32)
    target_link_libraries(ii_cli PRIVATE shell32)
endif()

set_target_properties(ii_cli PROPERTIES
    OUTPUT_NAME ii
    PDB_NAME ii_cli)

# An installed shared-library build finds libii next to the executable's prefix.
if(APPLE)
    set_target_properties(ii_cli PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_target_properties(ii_cli PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

# Smoke tests of the commands (registered only when tests are enabled; the top-level project calls enable_testing()).
if(II_BUILD_TESTS)
    set(_ii_example_model "${PROJECT_SOURCE_DIR}/examples/models/attention.ii.txt")
    add_test(NAME cli_info COMMAND ii_cli info)
    add_test(NAME cli_inspect COMMAND ii_cli inspect "${_ii_example_model}" --graph)
    add_test(NAME cli_run COMMAND ii_cli run "${_ii_example_model}" --dims B=2,S=16 --repeat 2 --profile trace.json)
    add_test(NAME cli_run_generic COMMAND ii_cli run "${_ii_example_model}" --shapes x=1x7x8 --isa generic --no-fuse)
    set_tests_properties(cli_run PROPERTIES PASS_REGULAR_EXPRESSION "trace written")
    unset(_ii_example_model)

    # Language model commands on the tiny test models.
    set(_ii_llm "${PROJECT_SOURCE_DIR}/tests/data/llm")
    add_test(NAME cli_generate COMMAND ii_cli generate "${_ii_llm}/qwen3.gguf" -p "Hello world" -n 8 --temp 0)
    add_test(NAME cli_generate_chat COMMAND ii_cli generate "${_ii_llm}/gemma2" --chat -p "Hi" -n 4 --seed 1
                                            --kv-type f16)
    add_test(NAME cli_tokenize COMMAND ii_cli tokenize "${_ii_llm}/llama.gguf" -p "Hello world" --special)
    add_test(NAME cli_bench_llm COMMAND ii_cli bench "${_ii_llm}/llama.gguf" -p 16 -n 4 -r 1)
    add_test(NAME cli_perplexity COMMAND ii_cli perplexity "${_ii_llm}/phi3.gguf" -f "${_ii_llm}/chat_templates.json"
                                         --ctx 32 --chunks 2)
    add_test(NAME cli_inspect_gguf COMMAND ii_cli inspect "${_ii_llm}/qwen2.gguf" --graph)
    set_tests_properties(cli_generate PROPERTIES PASS_REGULAR_EXPRESSION "generated [0-9]+ tokens")
    if(NOT CMAKE_CROSSCOMPILING)  # writes a file on the host
        add_test(NAME cli_quantize COMMAND ii_cli quantize "${_ii_llm}/llama.gguf"
                                           "${CMAKE_CURRENT_BINARY_DIR}/llama-q4_k_m.gguf" --type q4_k_m)
        add_test(NAME cli_quantized_generate COMMAND ii_cli generate "${CMAKE_CURRENT_BINARY_DIR}/llama-q4_k_m.gguf"
                                                     -p "Hello" -n 4 --temp 0)
        set_tests_properties(cli_quantized_generate PROPERTIES DEPENDS cli_quantize
                                                               PASS_REGULAR_EXPRESSION "generated [0-9]+ tokens")
    endif()
    set_tests_properties(cli_tokenize PROPERTIES PASS_REGULAR_EXPRESSION "tokens")
    set_tests_properties(cli_bench_llm PROPERTIES PASS_REGULAR_EXPRESSION "tg4")
    set_tests_properties(cli_perplexity PROPERTIES PASS_REGULAR_EXPRESSION "Final estimate: PPL")
    set_tests_properties(cli_inspect_gguf PROPERTIES PASS_REGULAR_EXPRESSION "qwen2.block_count")
    unset(_ii_llm)
endif()
