# SPDX-License-Identifier: MIT
#
# Determines the target CPU architecture as II_TARGET_ARCH: x86_64, arm64 or generic.
#
# Kernels are compiled for one architecture per build. Multi-architecture builds (e.g. macOS universal binaries) are
# produced by building each architecture separately and merging the results (lipo), because ISA-specific kernels need
# per-architecture compiler flags.

if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _ii_osx_arch_count)
    if(_ii_osx_arch_count GREATER 1)
        message(FATAL_ERROR
            "ii: CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}' lists several architectures. "
            "Build each architecture separately and merge them with lipo.")
    endif()
    set(_ii_arch "${CMAKE_OSX_ARCHITECTURES}")
elseif(MSVC AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID)
    # With MSVC CMAKE_SYSTEM_PROCESSOR describes the host, not the target.
    set(_ii_arch "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
else()
    set(_ii_arch "${CMAKE_SYSTEM_PROCESSOR}")
endif()

string(TOLOWER "${_ii_arch}" _ii_arch)

if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND _ii_arch MATCHES "^(x86_64|amd64|x64)$")
    set(II_TARGET_ARCH "x86_64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8 AND _ii_arch MATCHES "^(aarch64|arm64|arm64e|armv8.*|armv9.*)$")
    set(II_TARGET_ARCH "arm64")
else()
    set(II_TARGET_ARCH "generic")
endif()

unset(_ii_arch)
unset(_ii_osx_arch_count)
