# SPDX-License-Identifier: MIT
#
# Compiler options shared by all ii targets.
#
#   ii_build_options   INTERFACE target with warnings and platform definitions. Every target built by this project
#                      links it PRIVATE. It is never exported.
#
# Sanitizers are applied globally (including third-party test dependencies), because mixing instrumented and
# non-instrumented code produces false positives.

add_library(ii_build_options INTERFACE)

set(_ii_msvc_frontend FALSE)
if(MSVC)
    set(_ii_msvc_frontend TRUE)  # MSVC or clang-cl
endif()

if(_ii_msvc_frontend)
    target_compile_options(ii_build_options INTERFACE
        /W4
        /utf-8
        /wd4324  # structure was padded due to alignment specifier (intentional for cache-line alignment)
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(ii_build_options INTERFACE /permissive- /Zc:__cplusplus /Zc:preprocessor)
    endif()
    if(II_WARNINGS_AS_ERRORS)
        target_compile_options(ii_build_options INTERFACE /WX)
    endif()
else()
    target_compile_options(ii_build_options INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wimplicit-fallthrough
        -Wformat=2
    )
    if(II_WARNINGS_AS_ERRORS)
        target_compile_options(ii_build_options INTERFACE -Werror)
    endif()
endif()

if(WIN32)
    target_compile_definitions(ii_build_options INTERFACE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE)
endif()

# --- Sanitizers ------------------------------------------------------------------------------------------------------
if(II_SANITIZE)
    if(_ii_msvc_frontend)
        if(II_SANITIZE STREQUAL "address")
            add_compile_options(/fsanitize=address)
        else()
            message(FATAL_ERROR "ii: only II_SANITIZE=address is supported with MSVC")
        endif()
    else()
        list(JOIN II_SANITIZE "," _ii_sanitizers)
        add_compile_options(-fsanitize=${_ii_sanitizers} -fno-omit-frame-pointer -fno-sanitize-recover=all)
        add_link_options(-fsanitize=${_ii_sanitizers})
        unset(_ii_sanitizers)
    endif()
endif()

unset(_ii_msvc_frontend)
