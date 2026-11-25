# SPDX-License-Identifier: MIT
#
# Install rules and CMake package export: find_package(ii) provides the imported target ii::ii.

include(CMakePackageConfigHelpers)

set(II_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/ii")

install(TARGETS ii
    EXPORT iiTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/ii" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${PROJECT_BINARY_DIR}/include/ii/version.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ii")

if(TARGET ii_cli)
    install(TARGETS ii_cli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

install(EXPORT iiTargets
    NAMESPACE ii::
    DESTINATION "${II_INSTALL_CMAKEDIR}")

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/iiConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/iiConfig.cmake"
    INSTALL_DESTINATION "${II_INSTALL_CMAKEDIR}")

# Pre-1.0 releases may break compatibility between minor versions.
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/iiConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

install(FILES
    "${PROJECT_BINARY_DIR}/iiConfig.cmake"
    "${PROJECT_BINARY_DIR}/iiConfigVersion.cmake"
    DESTINATION "${II_INSTALL_CMAKEDIR}")

# pkg-config
if(II_BUILD_SHARED)
    set(II_PC_CFLAGS "")
else()
    set(II_PC_CFLAGS "-DII_STATIC_DEFINE")
endif()
configure_file("${PROJECT_SOURCE_DIR}/cmake/ii.pc.in" "${PROJECT_BINARY_DIR}/ii.pc" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/ii.pc" DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/ii")
