# SPDX-License-Identifier: MIT

add_executable(ii_cli
    main.cpp
    args.cpp
    console.cpp
    command_info.cpp
    command_inspect.cpp
    command_ops.cpp
    command_run.cpp)

target_link_libraries(ii_cli PRIVATE ii::ii ii_build_options)
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
endif()
