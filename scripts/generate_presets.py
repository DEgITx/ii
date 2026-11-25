#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generates CMakePresets.json.

Every toolchain gets one configure preset (Ninja Multi-Config) and, per build configuration, a build preset, a test
preset and a workflow preset with the same name, e.g.:

    cmake --workflow --preset windows-msvc-release      # configure + build + test
    cmake --preset windows-msvc && cmake --build --preset windows-msvc-debug

CI runs exactly these workflow presets, so every CI job can be reproduced locally with one command.

Usage: python scripts/generate_presets.py   (rewrites CMakePresets.json in the repository root)
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

WINDOWS = {"type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows"}
LINUX = {"type": "equals", "lhs": "${hostSystemName}", "rhs": "Linux"}
DARWIN = {"type": "equals", "lhs": "${hostSystemName}", "rhs": "Darwin"}

# name -> (description, condition, cache variables, configurations, test environment)
TOOLCHAINS = {
    "windows-msvc": (
        "Windows, MSVC (run from a Visual Studio developer prompt)",
        WINDOWS,
        {"CMAKE_C_COMPILER": "cl", "CMAKE_CXX_COMPILER": "cl"},
        ["debug", "release"],
        {},
    ),
    "windows-msvc-shared": (
        "Windows, MSVC, ii as a DLL",
        WINDOWS,
        {"CMAKE_C_COMPILER": "cl", "CMAKE_CXX_COMPILER": "cl", "II_BUILD_SHARED": "ON"},
        ["release"],
        {},
    ),
    "windows-msvc-asan": (
        "Windows, MSVC with AddressSanitizer",
        WINDOWS,
        {"CMAKE_C_COMPILER": "cl", "CMAKE_CXX_COMPILER": "cl", "II_SANITIZE": "address"},
        ["relwithdebinfo"],
        # Newer Windows builds (e.g. Insider 28000) contain function prologues the MSVC ASan runtime cannot patch;
        # continuing only skips interception of those few functions and still reports them in the output.
        {"ASAN_WIN_CONTINUE_ON_INTERCEPTION_FAILURE": "1"},
    ),
    "windows-clang-cl": (
        "Windows, clang-cl (run from a Visual Studio developer prompt)",
        WINDOWS,
        {"CMAKE_C_COMPILER": "clang-cl", "CMAKE_CXX_COMPILER": "clang-cl"},
        ["debug", "release"],
        {},
    ),
    "windows-mingw-gcc": (
        "Windows, MinGW-w64 GCC",
        WINDOWS,
        {"CMAKE_C_COMPILER": "gcc", "CMAKE_CXX_COMPILER": "g++"},
        ["debug", "release"],
        {},
    ),
    "windows-mingw-clang": (
        "Windows, MinGW-w64 Clang",
        WINDOWS,
        {"CMAKE_C_COMPILER": "clang", "CMAKE_CXX_COMPILER": "clang++"},
        ["debug", "release"],
        {},
    ),
    "linux-gcc": (
        "Linux, GCC",
        LINUX,
        {"CMAKE_C_COMPILER": "gcc", "CMAKE_CXX_COMPILER": "g++"},
        ["debug", "release"],
        {},
    ),
    "linux-gcc-shared": (
        "Linux, GCC, ii as a shared library",
        LINUX,
        {"CMAKE_C_COMPILER": "gcc", "CMAKE_CXX_COMPILER": "g++", "II_BUILD_SHARED": "ON"},
        ["release"],
        {},
    ),
    "linux-clang": (
        "Linux, Clang",
        LINUX,
        {"CMAKE_C_COMPILER": "clang", "CMAKE_CXX_COMPILER": "clang++"},
        ["debug", "release"],
        {},
    ),
    "linux-clang-asan": (
        "Linux, Clang with AddressSanitizer + UndefinedBehaviorSanitizer",
        LINUX,
        {"CMAKE_C_COMPILER": "clang", "CMAKE_CXX_COMPILER": "clang++", "II_SANITIZE": "address;undefined"},
        ["relwithdebinfo"],
        {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1:strict_string_checks=1",
         "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"},
    ),
    "linux-clang-fuzz": (
        "Linux, Clang with libFuzzer + AddressSanitizer + UndefinedBehaviorSanitizer (fuzz smoke tests)",
        LINUX,
        {"CMAKE_C_COMPILER": "clang", "CMAKE_CXX_COMPILER": "clang++", "II_SANITIZE": "address;undefined",
         "II_BUILD_FUZZERS": "ON"},
        ["relwithdebinfo"],
        {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1",
         "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"},
    ),
    "linux-clang-tsan": (
        "Linux, Clang with ThreadSanitizer",
        LINUX,
        {"CMAKE_C_COMPILER": "clang", "CMAKE_CXX_COMPILER": "clang++", "II_SANITIZE": "thread"},
        ["relwithdebinfo"],
        {"TSAN_OPTIONS": "halt_on_error=1:second_deadlock_stack=1"},
    ),
    "macos-clang": (
        "macOS, Apple Clang",
        DARWIN,
        {"CMAKE_C_COMPILER": "clang", "CMAKE_CXX_COMPILER": "clang++", "CMAKE_OSX_DEPLOYMENT_TARGET": "13.3"},
        ["debug", "release"],
        {},
    ),
}

CONFIGS = {"debug": "Debug", "release": "Release", "relwithdebinfo": "RelWithDebInfo"}


def main() -> None:
    configure = [
        {
            "name": "base",
            "hidden": True,
            "generator": "Ninja Multi-Config",
            "binaryDir": "${sourceDir}/build/${presetName}",
            "installDir": "${sourceDir}/install/${presetName}",
            "cacheVariables": {
                "CMAKE_CONFIGURATION_TYPES": "Debug;Release;RelWithDebInfo",
                "CMAKE_DEFAULT_BUILD_TYPE": "Debug",
                "II_BUILD_TESTS": "ON",
                "II_BUILD_BENCHMARKS": "ON",
                "II_WARNINGS_AS_ERRORS": "ON",
            },
        }
    ]
    build, test, workflow = [], [], []
    for name, (description, condition, cache, configs, test_env) in TOOLCHAINS.items():
        configure.append({
            "name": name,
            "displayName": description,
            "inherits": "base",
            "condition": condition,
            "cacheVariables": cache,
        })
        for config in configs:
            preset = f"{name}-{config}"
            build.append({"name": preset, "configurePreset": name, "configuration": CONFIGS[config]})
            test_preset = {
                "name": preset,
                "configurePreset": name,
                "configuration": CONFIGS[config],
                "output": {"outputOnFailure": True},
                "execution": {"noTestsAction": "error"},
            }
            if test_env:
                test_preset["environment"] = test_env
            test.append(test_preset)
            workflow.append({
                "name": preset,
                "steps": [
                    {"type": "configure", "name": name},
                    {"type": "build", "name": preset},
                    {"type": "test", "name": preset},
                ],
            })

    presets = {
        "version": 6,
        "cmakeMinimumRequired": {"major": 3, "minor": 25, "patch": 0},
        "configurePresets": configure,
        "buildPresets": build,
        "testPresets": test,
        "workflowPresets": workflow,
    }
    (ROOT / "CMakePresets.json").write_text(json.dumps(presets, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
