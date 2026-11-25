// SPDX-License-Identifier: MIT
// A minimal shared library used by the DynLib tests.

#if defined(_WIN32)
#define II_TEST_EXPORT __declspec(dllexport)
#else
#define II_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" II_TEST_EXPORT int ii_test_answer() {
    return 42;
}
