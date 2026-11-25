# SPDX-License-Identifier: MIT
#
# Library components.
#
# Each architectural layer (core, ir, runtime, backends, ...) is an OBJECT library registered as a component. Its
# objects are compiled once and linked into two products:
#
#   ii            the public library (static or shared, per II_BUILD_SHARED); exports only II_API symbols
#   ii_internal   a static library with the same objects, used by tests and benchmarks to reach internal code
#
#   ii_add_component(<name> SOURCES <src>...)
#       Adds OBJECT library ii_<name>.
#   ii_add_isa_component(<name> ISA <isa> SOURCES <src>...)
#       Adds OBJECT library ii_<name>_<isa> compiled with the ISA's flags and II_ISA_NS=<isa>.
#       Does nothing when <isa> is not in II_ISA_VARIANTS.
#   ii_add_link_libraries(<lib>...)
#       System libraries required by components (linked into ii and ii_internal).
#   ii_finalize_library()
#       Creates ii and ii_internal from all registered components. Call once, after all components.

function(_ii_configure_component target)
    target_link_libraries(${target} PRIVATE ii_build_options Threads::Threads)
    target_include_directories(${target}
        PUBLIC
            "${PROJECT_SOURCE_DIR}/include"
            "${PROJECT_BINARY_DIR}/include"
        PRIVATE
            "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(${target} PRIVATE II_BUILDING_LIBRARY ${II_ISA_DEFINITIONS})
    if(NOT II_BUILD_SHARED)
        target_compile_definitions(${target} PRIVATE II_STATIC_DEFINE)
    endif()
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    set_property(GLOBAL APPEND PROPERTY II_COMPONENTS ${target})
endfunction()

function(ii_add_component name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES")
    add_library(ii_${name} OBJECT ${ARG_SOURCES})
    _ii_configure_component(ii_${name})
endfunction()

function(ii_add_isa_component name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "ISA" "SOURCES")
    if(NOT ARG_ISA IN_LIST II_ISA_VARIANTS)
        return()
    endif()
    set(target ii_${name}_${ARG_ISA})
    add_library(${target} OBJECT ${ARG_SOURCES})
    _ii_configure_component(${target})
    ii_isa_compile_options(${ARG_ISA} _flags)
    target_compile_options(${target} PRIVATE ${_flags})
    target_compile_definitions(${target} PRIVATE II_ISA_NS=${ARG_ISA})
endfunction()

function(ii_add_link_libraries)
    set_property(GLOBAL APPEND PROPERTY II_LINK_LIBRARIES ${ARGN})
endfunction()

function(ii_finalize_library)
    get_property(components GLOBAL PROPERTY II_COMPONENTS)
    get_property(link_libraries GLOBAL PROPERTY II_LINK_LIBRARIES)

    set(objects "")
    foreach(component IN LISTS components)
        list(APPEND objects $<TARGET_OBJECTS:${component}>)
    endforeach()

    # --- Public library ---
    if(II_BUILD_SHARED)
        add_library(ii SHARED ${objects})
    else()
        add_library(ii STATIC ${objects})
        target_compile_definitions(ii PUBLIC II_STATIC_DEFINE)
    endif()
    add_library(ii::ii ALIAS ii)
    target_include_directories(ii PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_compile_features(ii PUBLIC cxx_std_20)
    target_link_libraries(ii PRIVATE Threads::Threads ${link_libraries})
    set_target_properties(ii PROPERTIES
        OUTPUT_NAME ii
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        # The CLI executable is also called "ii"; keep their PDB files apart.
        PDB_NAME ii_library)

    # --- Internal static library for tests and benchmarks ---
    add_library(ii_internal STATIC ${objects})
    target_include_directories(ii_internal PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_BINARY_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src")
    target_compile_definitions(ii_internal PUBLIC II_STATIC_DEFINE ${II_ISA_DEFINITIONS})
    target_compile_features(ii_internal PUBLIC cxx_std_20)
    target_link_libraries(ii_internal PUBLIC Threads::Threads ${link_libraries})
endfunction()
