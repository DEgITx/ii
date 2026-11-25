# SPDX-License-Identifier: MIT

add_executable(ii_cli
    main.cpp
    args.cpp
    console.cpp
    command_info.cpp)

target_link_libraries(ii_cli PRIVATE ii::ii ii_build_options)
if(WIN32)
    target_link_libraries(ii_cli PRIVATE shell32)
endif()

set_target_properties(ii_cli PROPERTIES
    OUTPUT_NAME ii
    PDB_NAME ii_cli)
