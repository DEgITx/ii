// SPDX-License-Identifier: MIT
/// @file
/// Symbol visibility macros.
///
/// `II_API` marks declarations that are part of the binary interface of the ii library. Everything else is hidden
/// when ii is built as a shared library. Consumers of a static build must define `II_STATIC_DEFINE` (the CMake target
/// `ii::ii` and the pkg-config file do this automatically).
#pragma once

#if defined(II_STATIC_DEFINE)
#define II_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#if defined(II_BUILDING_LIBRARY)
#define II_API __declspec(dllexport)
#else
#define II_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define II_API __attribute__((visibility("default")))
#else
#define II_API
#endif

/// Exported C++ classes hold standard-library members (`std::shared_ptr`, `std::vector`, ...). MSVC warns (C4251)
/// that those members have no DLL interface; that is harmless because the C++ API requires the application and ii to
/// use the same compiler and standard library (the C ABI in `ii/c_api.h` is the boundary between toolchains).
/// Public headers wrap their declarations in these macros so the warning reaches neither ii nor its users.
#if defined(_MSC_VER)
#define II_BEGIN_DLL_EXPORTS __pragma(warning(push)) __pragma(warning(disable : 4251))
#define II_END_DLL_EXPORTS __pragma(warning(pop))
#else
#define II_BEGIN_DLL_EXPORTS
#define II_END_DLL_EXPORTS
#endif
