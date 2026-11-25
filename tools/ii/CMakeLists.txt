# SPDX-License-Identifier: MIT

add_executable(ii_cli
    main.cpp
    args.cpp
    console.cpp
    command_info.cpp
    command_inspect.cpp
    command_ops.cpp)

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
