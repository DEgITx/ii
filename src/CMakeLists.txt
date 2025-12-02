# Сборка C++ раннера TFLite + (опционального) внешний NPU-делегат делегата + Wayland/EGL
# вывода / V4L2 камеры.
#
# Поддерживаемые платформы:
#   * Linux / BSP — полный набор (Wayland/EGL/GLES2, V4L2);
#   * Linux dev-host без видеокарты — собирайте с -DUSE_DISPLAY=OFF;
#   * Windows / macOS dev-host — Wayland/V4L2 не собираются, делегата нет,
#     раннер работает в CPU-режиме (или с альтернативным бэкендом, когда
#     он появится — TensorRT / DirectML).
#
# Пример (Linux, на устройстве/в SDK BSP):
#     cmake -S . -B build -DTFLITE_ROOT=/usr
#     cmake --build build -j
#
# Если TFLite собран отдельно — укажите пути явно:
#     cmake -S . -B build \
#         -DTFLITE_INCLUDE_DIR=/opt/tflite/include \
#         -DTFLITE_LIB=/opt/tflite/lib/libtensorflowlite.so
#
# Без графического вывода / без камеры:
#     cmake -S . -B build -DUSE_DISPLAY=OFF -DUSE_CAMERA=OFF
#
# Windows (через vcpkg, MSVC):
#     cmake -S . -B build -DTFLITE_ROOT=C:/path/to/tflite-prebuilt
#     cmake --build build --config Release -j
#   (USE_DISPLAY / USE_CAMERA по умолчанию OFF — Wayland/V4L2 на Windows нет.)

cmake_minimum_required(VERSION 3.16)
project(ii CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Опции по умолчанию ---------------------------------------------------
# Display (Wayland/EGL/GLES2) и Camera (V4L2) — Linux-only зависимости.
# На остальных платформах форсим OFF, чтобы ii собрался со стабами и
# make_display()/make_camera() корректно вернули nullptr.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    option(USE_DISPLAY "Сборка с Wayland/EGL/GLES2 backend'ом для отображения" ON)
    option(USE_CAMERA  "Сборка с V4L2 backend'ом для захвата с камеры"          ON)
else()
    option(USE_DISPLAY "Сборка с Wayland/EGL/GLES2 backend'ом для отображения" OFF)
    option(USE_CAMERA  "Сборка с V4L2 backend'ом для захвата с камеры"          OFF)
endif()

# Бэкенды инференса. TFLite пока единственный реализованный, поэтому
# по-умолчанию ON. TensorRT/DirectML — placeholder под будущие реализации
# (см. inference.cpp:make_engine). Когда они появятся, эти опции включат
# линковку соответствующих *.cpp + библиотек.
option(USE_TFLITE    "Сборка с бэкендом TensorFlow Lite"                       ON)
option(USE_TENSORRT  "Сборка с бэкендом NVIDIA TensorRT (placeholder)"        OFF)
option(USE_DIRECTML  "Сборка с бэкендом DirectML/Windows ML (placeholder)"    OFF)

# ---- Бинарь (без бэкендов; они подключаются ниже опционально) ------------
add_executable(ii
    ii.cpp
    inference.cpp
    yolo.cpp)

# ---- Платформо-зависимые флаги компилятора и линкер-флаги ----------------
if(MSVC)
    # /W4 = -Wall -Wextra эквивалент; /utf-8 нужен из-за русских комментариев
    # и литералов в исходниках. /O2 — релизная оптимизация по-умолчанию,
    # CMake её и так подставит в Release; добавляем явно для других конфигов.
    # NOMINMAX отключает конфликтующие макросы из <windows.h> (min/max),
    # а _CRT_SECURE_NO_WARNINGS — глушит шум по std::fopen/strncpy.
    target_compile_options(ii PRIVATE /W4 /utf-8 /O2)
    target_compile_definitions(ii PRIVATE
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(ii PRIVATE -Wall -Wextra -O3)
endif()

# dl/pthread — POSIX; на Windows их нет (а TFLite сам линкуется со своими
# системными зависимостями через .lib импорт-библиотеку).
if(UNIX)
    target_link_libraries(ii PRIVATE dl pthread)
endif()

# ---- stb (одноголовники для image/resize, без внешних зависимостей) -------
include(FetchContent)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(stb)
target_include_directories(ii PRIVATE ${stb_SOURCE_DIR})

# ---- Бэкенд: TFLite -------------------------------------------------------
if(USE_TFLITE)
    set(TFLITE_ROOT "" CACHE PATH "Корень TFLite (содержит include/ и lib/)")

    find_path(TFLITE_INCLUDE_DIR
        NAMES tensorflow/lite/interpreter.h
        HINTS ${TFLITE_ROOT}/include
              /usr/include /usr/local/include)

    # find_library уже знает про платформенные расширения:
    #   * .so / .a  на Linux;
    #   * .dylib    на macOS;
    #   * .lib      на Windows (импорт-либ для .dll).
    find_library(TFLITE_LIB
        NAMES tensorflowlite tensorflow-lite tensorflowlite_c
        HINTS ${TFLITE_ROOT}/lib
              ${TFLITE_ROOT}/bin
              /usr/lib /usr/local/lib
              /usr/lib/aarch64-linux-gnu)

    if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIB)
        message(FATAL_ERROR
            "Не найден TFLite. Передайте -DTFLITE_ROOT=/path или "
            "-DTFLITE_INCLUDE_DIR=... -DTFLITE_LIB=... (или соберите без "
            "TFLite: -DUSE_TFLITE=OFF, когда будут другие бэкенды).")
    endif()

    message(STATUS "TFLite include: ${TFLITE_INCLUDE_DIR}")
    message(STATUS "TFLite lib:     ${TFLITE_LIB}")

    target_sources(ii PRIVATE inference_tflite.cpp)
    target_include_directories(ii PRIVATE ${TFLITE_INCLUDE_DIR})
    target_link_libraries(ii PRIVATE ${TFLITE_LIB})
    target_compile_definitions(ii PRIVATE INF_HAS_TFLITE=1)
endif()

# ---- Бэкенд: TensorRT (заготовка) ----------------------------------------
# Когда появится inference_tensorrt.cpp, здесь будет find_package(CUDA) +
# подключение nvinfer/nvonnxparser и target_sources(ii PRIVATE inference_tensorrt.cpp).
if(USE_TENSORRT)
    message(FATAL_ERROR
        "USE_TENSORRT=ON, но бэкенд TensorRT пока не реализован. "
        "Добавьте inference_tensorrt.cpp перед включением опции.")
endif()

# ---- Бэкенд: DirectML (заготовка) ----------------------------------------
if(USE_DIRECTML)
    message(FATAL_ERROR
        "USE_DIRECTML=ON, но бэкенд DirectML пока не реализован. "
        "Добавьте inference_directml.cpp перед включением опции.")
endif()

# ---- Backend для камеры (V4L2) -------------------------------------------
if(USE_CAMERA)
    target_sources(ii PRIVATE camera_v4l2.cpp)
    target_compile_definitions(ii PRIVATE NPU_HAS_CAMERA=1)
else()
    target_sources(ii PRIVATE camera_stub.cpp)
endif()

# ---- Backend для отображения (Wayland/EGL/GLES2) -------------------------
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

# ---- Финальный summary ---------------------------------------------------
message(STATUS "ii build summary:")
message(STATUS "  platform : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  TFLite   : ${USE_TFLITE}")
message(STATUS "  TensorRT : ${USE_TENSORRT}")
message(STATUS "  DirectML : ${USE_DIRECTML}")
message(STATUS "  Display  : ${USE_DISPLAY}")
message(STATUS "  Camera   : ${USE_CAMERA}")
