# SPDX-License-Identifier: MIT

# Loading backend plugins (part of the library).
ii_add_component(plugin SOURCES
    host.cpp)

# The plugin SDK (sdk.h) for plugins built with the ii sources. A plugin links it and so carries its own copy of the
# parts of ii it uses; only the C interface of ii/plugin.h crosses to the ii that loads it. ii_internal is created by
# ii_finalize_library() later; CMake resolves the name when it generates the build.
add_library(ii_plugin_sdk STATIC sdk.cpp)
target_link_libraries(ii_plugin_sdk PUBLIC ii_internal PRIVATE ii_build_options)
set_target_properties(ii_plugin_sdk PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# ii_add_backend_plugin(<target> SOURCES <src>...)
#   Builds a backend plugin (a MODULE library) with the SDK into <build>/plugins (plus the configuration directory of
#   multi-configuration generators), away from the executables: plugins next to the ii library are loaded
#   automatically, and the tests choose when to load them.
function(ii_add_backend_plugin target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES")
    add_library(${target} MODULE ${ARG_SOURCES})
    target_link_libraries(${target} PRIVATE ii_plugin_sdk ii_build_options)
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/plugins"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/plugins")
endfunction()
