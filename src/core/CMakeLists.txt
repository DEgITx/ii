# SPDX-License-Identifier: MIT

set(core_sources
    allocator.cpp
    build_info.cpp
    compress.cpp
    fatal.cpp
    isa.cpp
    json.cpp
    log.cpp
    status.cpp
    system_info.cpp
    thread_pool.cpp
    perf/roofline.cpp
)

# --- Platform layer: the only place with OS-specific code ---
if(WIN32)
    list(APPEND core_sources
        platform/windows/win32_dynlib.cpp
        platform/windows/win32_mapped_file.cpp
        platform/windows/win32_platform.cpp
        platform/windows/win32_topology.cpp)
    ii_add_link_libraries(advapi32)
elseif(CMAKE_SYSTEM_NAME MATCHES "^(Linux|Android|Darwin)$")
    list(APPEND core_sources
        platform/posix/posix_dynlib.cpp
        platform/posix/posix_mapped_file.cpp
        platform/posix/posix_platform.cpp)
    if(APPLE)
        list(APPEND core_sources platform/apple/apple_system.cpp)
    else()
        list(APPEND core_sources platform/linux/linux_system.cpp)
    endif()
    ii_add_link_libraries(${CMAKE_DL_LIBS})
else()
    message(FATAL_ERROR "ii: unsupported operating system '${CMAKE_SYSTEM_NAME}'")
endif()

if(II_TARGET_ARCH STREQUAL "x86_64")
    list(APPEND core_sources platform/cpu_features_x86.cpp)
elseif(II_TARGET_ARCH STREQUAL "arm64")
    list(APPEND core_sources platform/cpu_features_arm.cpp)
else()
    list(APPEND core_sources platform/cpu_features_generic.cpp)
endif()

ii_add_component(core SOURCES ${core_sources})

# Build metadata for build_info().
set_property(SOURCE build_info.cpp TARGET_DIRECTORY ii_core APPEND PROPERTY COMPILE_DEFINITIONS
    "II_GIT_HASH=\"${II_GIT_HASH}\""
    "II_BUILD_CONFIG=\"$<CONFIG>\""
    "II_TARGET_ARCH_NAME=\"${II_TARGET_ARCH}\""
    "II_BUILD_SHARED_LIBRARY=$<BOOL:${II_BUILD_SHARED}>")

# --- ISA-specific roofline kernels ---
ii_add_isa_component(core_perf ISA generic SOURCES perf/fma_kernel_generic.cpp)
ii_add_isa_component(core_perf ISA avx2 SOURCES perf/fma_kernel_avx2.cpp)
ii_add_isa_component(core_perf ISA avx512 SOURCES perf/fma_kernel_avx512.cpp)
ii_add_isa_component(core_perf ISA neon SOURCES perf/fma_kernel_neon.cpp)
