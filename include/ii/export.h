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
