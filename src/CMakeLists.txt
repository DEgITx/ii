# Сборка C++ раннера TFLite + внешний NPU-делегат делегата + Wayland/EGL вывода.
#
# Пример (на устройстве/в SDK BSP, где TFLite уже установлен):
#     cmake -S . -B build -DTFLITE_ROOT=/usr
#     cmake --build build -j
#
# Если TFLite собран отдельно — укажите пути явно:
#     cmake -S . -B build \
#         -DTFLITE_INCLUDE_DIR=/opt/tflite/include \
#         -DTFLITE_LIB=/opt/tflite/lib/libtensorflowlite.so
#
# Без графического вывода (только инференс, без зависимости от Wayland):
#     cmake -S . -B build -DUSE_DISPLAY=OFF

cmake_minimum_required(VERSION 3.16)
project(ii CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(USE_DISPLAY "Сборка с Wayland/EGL/GLES2 backend’ом для отображения" ON)

# ---- TFLite ---------------------------------------------------------------
set(TFLITE_ROOT "" CACHE PATH "Корень TFLite (содержит include/ и lib/)")

find_path(TFLITE_INCLUDE_DIR
    NAMES tensorflow/lite/interpreter.h
    HINTS ${TFLITE_ROOT}/include /usr/include /usr/local/include)

find_library(TFLITE_LIB
    NAMES tensorflowlite tensorflow-lite
    HINTS ${TFLITE_ROOT}/lib /usr/lib /usr/local/lib
          /usr/lib/aarch64-linux-gnu)

if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIB)
    message(FATAL_ERROR
        "Не найден TFLite. Передайте -DTFLITE_ROOT=/path или "
        "-DTFLITE_INCLUDE_DIR=... -DTFLITE_LIB=...")
endif()

message(STATUS "TFLite include: ${TFLITE_INCLUDE_DIR}")
message(STATUS "TFLite lib:     ${TFLITE_LIB}")

# ---- stb (одноголовники для image/resize, без внешних зависимостей) -------
include(FetchContent)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(stb)

# ---- Бинарь ---------------------------------------------------------------
add_executable(ii ii.cpp)
target_include_directories(ii PRIVATE
    ${TFLITE_INCLUDE_DIR}
    ${stb_SOURCE_DIR})
target_link_libraries(ii PRIVATE ${TFLITE_LIB} dl pthread)
target_compile_options(ii PRIVATE -Wall -Wextra -O3)

# ---- Backend для отображения ---------------------------------------------
if(USE_DISPLAY)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(WAYLAND REQUIRED IMPORTED_TARGET
        wayland-client wayland-egl egl glesv2)

    # Генерация кода xdg-shell через wayland-scanner.
    find_program(WAYLAND_SCANNER NAMES wayland-scanner REQUIRED)
    pkg_get_variable(WAYLAND_PROTOCOLS_DIR wayland-protocols pkgdatadir)
    if(NOT WAYLAND_PROTOCOLS_DIR)
        message(FATAL_ERROR
            "Не нашли wayland-protocols. Установите libwayland-dev и "
            "wayland-protocols, либо соберите без дисплея: -DUSE_DISPLAY=OFF")
    endif()

    set(XDG_SHELL_XML
        "${WAYLAND_PROTOCOLS_DIR}/stable/xdg-shell/xdg-shell.xml")
    set(XDG_SHELL_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-client-protocol.h")
    set(XDG_SHELL_SOURCE
        "${CMAKE_CURRENT_BINARY_DIR}/xdg-shell-protocol.c")

    add_custom_command(
        OUTPUT ${XDG_SHELL_HEADER}
        COMMAND ${WAYLAND_SCANNER} client-header
                ${XDG_SHELL_XML} ${XDG_SHELL_HEADER}
        DEPENDS ${XDG_SHELL_XML}
        VERBATIM)
    add_custom_command(
        OUTPUT ${XDG_SHELL_SOURCE}
        COMMAND ${WAYLAND_SCANNER} private-code
                ${XDG_SHELL_XML} ${XDG_SHELL_SOURCE}
        DEPENDS ${XDG_SHELL_XML}
        VERBATIM)

    target_sources(ii PRIVATE
        display_wayland.cpp
        ${XDG_SHELL_HEADER}
        ${XDG_SHELL_SOURCE})
    target_include_directories(ii PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(ii PRIVATE PkgConfig::WAYLAND)
else()
    target_sources(ii PRIVATE display_stub.cpp)
endif()
