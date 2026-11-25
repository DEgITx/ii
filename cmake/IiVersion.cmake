# SPDX-License-Identifier: MIT
#
# Generates include/ii/version.h in the build tree and determines the git revision.

set(II_GIT_HASH "unknown")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE _ii_git_hash
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _ii_git_result)
    if(_ii_git_result EQUAL 0 AND _ii_git_hash)
        set(II_GIT_HASH "${_ii_git_hash}")
    endif()
    unset(_ii_git_hash)
    unset(_ii_git_result)
endif()

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/version.h.in"
    "${PROJECT_BINARY_DIR}/include/ii/version.h"
    @ONLY)
