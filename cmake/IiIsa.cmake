# SPDX-License-Identifier: MIT
#
# Multi-ISA kernel compilation.
#
# Kernels that use instruction-set extensions are compiled several times with different compiler flags, and the best
# variant supported by the running CPU is selected at runtime. This keeps one binary portable to every CPU of the
# target architecture.
#
#   II_ISA_VARIANTS        list of ISA variants compiled in this build (always starts with "generic")
#   II_ISA_DEFINITIONS     compile definitions II_HAVE_ISA_<NAME>=1 for every compiled variant (applied to all
#                          library components so dispatchers know which variants exist)
#   ii_isa_compile_options(<isa> <out-var>)
#
# Rules for ISA-specific translation units (see docs/architecture.md, "ODR trap"):
#   * all code lives in namespace `ii::...::II_ISA_NS` (II_ISA_NS is defined to the variant name);
#   * they include only headers without inline functions or templates that could be instantiated with ISA flags
#     (intrinsics headers, <cstddef>, <cstdint>, and kernel declaration headers).

set(_ii_isas_x86_64 generic avx2 avx_vnni avx512)
set(_ii_isas_arm64 generic neon neon_dotprod)
set(_ii_isas_generic generic)

set(_ii_available_isas ${_ii_isas_${II_TARGET_ARCH}})

if(II_CPU_ISAS STREQUAL "auto")
    set(II_ISA_VARIANTS ${_ii_available_isas})
else()
    set(II_ISA_VARIANTS ${II_CPU_ISAS})
    foreach(_isa IN LISTS II_ISA_VARIANTS)
        if(NOT _isa IN_LIST _ii_available_isas)
            message(FATAL_ERROR "ii: ISA '${_isa}' is not available for ${II_TARGET_ARCH} "
                                "(available: ${_ii_available_isas})")
        endif()
    endforeach()
    if(NOT "generic" IN_LIST II_ISA_VARIANTS)
        list(PREPEND II_ISA_VARIANTS generic)
    endif()
endif()

set(II_ISA_DEFINITIONS "")
foreach(_isa IN LISTS II_ISA_VARIANTS)
    string(TOUPPER "${_isa}" _isa_upper)
    list(APPEND II_ISA_DEFINITIONS "II_HAVE_ISA_${_isa_upper}=1")
endforeach()

function(ii_isa_compile_options isa out_var)
    set(_flags "")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(isa STREQUAL "avx2" OR isa STREQUAL "avx_vnni")
            set(_flags /arch:AVX2)
        elseif(isa STREQUAL "avx512")
            set(_flags /arch:AVX512)
        endif()
    else()
        # GCC, Clang and clang-cl accept GCC-style -m flags.
        if(isa STREQUAL "avx2")
            set(_flags -mavx2 -mfma -mf16c)
        elseif(isa STREQUAL "avx_vnni")
            set(_flags -mavx2 -mfma -mf16c -mavxvnni)
        elseif(isa STREQUAL "avx512")
            set(_flags -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx2 -mfma -mf16c)
        elseif(isa STREQUAL "neon_dotprod")
            set(_flags -march=armv8.2-a+dotprod)
        endif()
        # MinGW GCC does not realign the stack for 32/64-byte vector spills (GCC bug 54412) and may emit aligned
        # moves to misaligned stack slots. Ask the assembler to turn them into unaligned moves.
        if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND isa MATCHES "^(avx2|avx_vnni|avx512)$")
            list(APPEND _flags -Wa,-muse-unaligned-vector-move)
        endif()
    endif()
    set(${out_var} ${_flags} PARENT_SCOPE)
endfunction()

unset(_ii_isas_x86_64)
unset(_ii_isas_arm64)
unset(_ii_isas_generic)
unset(_ii_available_isas)
