// SPDX-License-Identifier: MIT
#include <ii/version.h>

#define II_STRINGIFY_IMPL(x) #x
#define II_STRINGIFY(x) II_STRINGIFY_IMPL(x)

#if defined(__clang__)
#define II_COMPILER_NAME "Clang " __clang_version__
#elif defined(_MSC_VER)
#define II_COMPILER_NAME "MSVC " II_STRINGIFY(_MSC_FULL_VER)
#elif defined(__GNUC__)
#define II_COMPILER_NAME "GCC " __VERSION__
#else
#define II_COMPILER_NAME "unknown compiler"
#endif

namespace ii {

const BuildInfo& build_info() noexcept {
    static const BuildInfo info{
        .version = II_VERSION_STRING,
        .git_hash = II_GIT_HASH,
        .compiler = II_COMPILER_NAME,
        .config = II_BUILD_CONFIG,
        .target_arch = II_TARGET_ARCH_NAME,
        .shared = II_BUILD_SHARED_LIBRARY != 0,
    };
    return info;
}

}  // namespace ii
