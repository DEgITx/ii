// SPDX-License-Identifier: MIT
#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>
#elif (defined(_M_ARM64) || defined(_M_ARM)) && defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace ii {

/// Hints the CPU that the calling thread is spin-waiting (reduces power use and pipeline flushes).
inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif (defined(_M_ARM64) || defined(_M_ARM)) && defined(_MSC_VER) && !defined(__clang__)
    __yield();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#endif
}

}  // namespace ii
