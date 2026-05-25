# Сборка C++ раннера: бэкенды инференса (TFLite / TensorRT / DirectML) +
# Wayland/EGL вывод + V4L2 камера.
#
# Поддерживаемые платформы:
#   * Linux (в т.ч. встраиваемый / SBC) — полный набор (Wayland/EGL/GLES2,
#     V4L2);
#   * Linux dev-host без видеокарты — собирайте с -DUSE_DISPLAY=OFF;
#   * Windows / macOS dev-host — Wayland/V4L2 не собираются, делегата нет,
#     раннер работает в CPU-режиме (или с альтернативным бэкендом —
#     TensorRT / DirectML).
#
# Доступные бэкенды инференса (включаются опциями CMake):
#   * USE_TFLITE   (ON  по умолчанию) — TensorFlow Lite + опциональный
#                                       внешний делегат (NPU/GPU);
#   * USE_TENSORRT (OFF по умолчанию) — NVIDIA TensorRT 10+, GPU;
#                                       требует CUDA Runtime и nvinfer
#                                       (опционально nvonnxparser для
#                                       сборки .engine из .onnx);
#   * USE_DIRECTML (OFF по умолчанию) — ONNX Runtime + DirectML EP,
#                                       NPU/GPU/iGPU под Windows
#                                       (на не-Windows собирается только
#                                       CPU EP — полезно для регрессий).
#
# Пример (Linux):
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

# Бэкенды инференса. TFLite по-умолчанию ON (CPU + опциональный внешний
# делегат). TensorRT и DirectML — опциональны: их сборка требует внешних
# SDK (CUDA+TensorRT и ONNX Runtime соответственно), поэтому по-умолчанию
# OFF; включаются опциями ниже + указанием путей через TRT_ROOT / ORT_ROOT
# (см. блоки ниже).
option(USE_TFLITE    "Сборка с бэкендом TensorFlow Lite"                       ON)
option(USE_TENSORRT  "Сборка с бэкендом NVIDIA TensorRT (требует CUDA+nvinfer)" OFF)
option(USE_DIRECTML  "Сборка с бэкендом DirectML/Windows ML (требует ORT)"      OFF)

# Опциональный модуль внешнего NPU-делегата TFLite (см. delegate_neutron.cpp).
# Подставляет платформенный дефолтный путь к .so-плагину делегата на Linux.
# По умолчанию OFF — ядро раннера не зависит ни от какого конкретного
# ускорителя; путь к делегату всегда можно задать через --delegate.
option(USE_NEUTRON_DELEGATE
    "Подключить опциональный модуль внешнего NPU-делегата (Linux)"            OFF)

# ---- Бинарь (без бэкендов; они подключаются ниже опционально) ------------
add_executable(ii
    ii.cpp
    cli.cpp
    preprocess.cpp
    tensor_utils.cpp
    yolo_render.cpp
    inference.cpp
    delegate.cpp
    image_proc.cpp
    tile.cpp
    parallel.cpp
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

# std::thread / std::async (модуль parallel.cpp — пул потоков). MSVC тащит
# поддержку потоков сам; glibc уже покрыт pthread выше, но Threads::Threads
# даёт правильный флаг (-pthread) и для MinGW, и для прочих тулчейнов.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ii PRIVATE Threads::Threads)

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

# ---- Опциональный модуль внешнего NPU-делегата ---------------------------
# Платформенно-специфичная мелочь: дефолтный путь к .so-плагину делегата.
# Отделён от ядра, чтобы раннер не был привязан ни к одному ускорителю.
if(USE_NEUTRON_DELEGATE)
    target_sources(ii PRIVATE delegate_neutron.cpp)
    target_compile_definitions(ii PRIVATE INF_HAS_DELEGATE_MODULE=1)
    message(STATUS "Delegate module: внешний NPU-делегат (ON)")
endif()

# ---- Бэкенд: TensorRT ----------------------------------------------------
# Требует TensorRT 10+ (enqueueV3 / setTensorAddress) и CUDA Runtime.
# Пути к TRT задаются через -DTRT_ROOT=/path или явно
# -DTRT_INCLUDE_DIR=... -DTRT_NVINFER_LIB=... ; nvonnxparser ищется
# опционально (без него бэкенд работает, но только с .engine/.plan).
if(USE_TENSORRT)
    # CUDA Runtime: используем современный модуль CUDAToolkit (CMake 3.17+),
    # он подбирает CUDA::cudart как импортированную цель без ручного
    # прописывания путей. На старых CMake — find_package(CUDA REQUIRED).
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
        find_package(CUDA REQUIRED)
        set(_INF_CUDART_LIB ${CUDA_CUDART_LIBRARY})
        set(_INF_CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
    else()
        # Используем импортированный таргет ниже.
    endif()

    set(TRT_ROOT "" CACHE PATH "Корень TensorRT (содержит include/ и lib/)")

    find_path(TRT_INCLUDE_DIR
        NAMES NvInfer.h
        HINTS ${TRT_ROOT}/include
              /usr/include /usr/local/include
              /usr/include/x86_64-linux-gnu
              /usr/include/aarch64-linux-gnu)

    find_library(TRT_NVINFER_LIB
        NAMES nvinfer nvinfer_10
        HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/lib64 ${TRT_ROOT}/bin
              /usr/lib /usr/local/lib
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu)

    find_library(TRT_NVONNXPARSER_LIB
        NAMES nvonnxparser nvonnxparser_10
        HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/lib64 ${TRT_ROOT}/bin
              /usr/lib /usr/local/lib
              /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu)

    if(NOT TRT_INCLUDE_DIR OR NOT TRT_NVINFER_LIB)
        message(FATAL_ERROR
            "Не найден TensorRT. Передайте -DTRT_ROOT=/path или "
            "-DTRT_INCLUDE_DIR=... -DTRT_NVINFER_LIB=... (поставьте "
            "TensorRT 10+ из NVIDIA-репозитория или соберите без "
            "TRT: -DUSE_TENSORRT=OFF).")
    endif()

    message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
    message(STATUS "TensorRT lib:     ${TRT_NVINFER_LIB}")
    if(TRT_NVONNXPARSER_LIB)
        message(STATUS "TensorRT onnxparser: ${TRT_NVONNXPARSER_LIB}")
    else()
        message(STATUS "TensorRT onnxparser: НЕ найден — поддержка .onnx "
                       "отключена, доступны только .engine/.plan/.trt")
    endif()

    target_sources(ii PRIVATE inference_tensorrt.cpp)
    target_include_directories(ii PRIVATE ${TRT_INCLUDE_DIR})
    target_link_libraries(ii PRIVATE ${TRT_NVINFER_LIB})
    if(CUDAToolkit_FOUND)
        target_link_libraries(ii PRIVATE CUDA::cudart)
    else()
        target_include_directories(ii PRIVATE ${_INF_CUDA_INCLUDE_DIR})
        target_link_libraries(ii PRIVATE ${_INF_CUDART_LIB})
    endif()
    if(TRT_NVONNXPARSER_LIB)
        target_link_libraries(ii PRIVATE ${TRT_NVONNXPARSER_LIB})
        target_compile_definitions(ii PRIVATE INF_TRT_HAS_ONNX=1)
    endif()
    target_compile_definitions(ii PRIVATE INF_HAS_TENSORRT=1)
endif()

# ---- Бэкенд: DirectML (через ONNX Runtime) -------------------------------
# Бэкенд реализован поверх ONNX Runtime + DirectML execution provider —
# это и есть продакшн-стек Windows ML. Сборка требует поставки ORT с
# заголовками + onnxruntime.dll/lib. На Windows дополнительно ищем
# dml_provider_factory.h (поставляется в той же тарбе ORT) и включаем
# INF_DML_EP_AVAILABLE, без которого собирается только CPU EP-режим.
if(USE_DIRECTML)
    set(ORT_ROOT "" CACHE PATH
        "Корень ONNX Runtime (содержит include/ и lib/)")

    find_path(ORT_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS ${ORT_ROOT}/include
              ${ORT_ROOT}/include/onnxruntime
              ${ORT_ROOT}/include/onnxruntime/core/session
              /usr/include /usr/local/include
              /usr/include/onnxruntime)

    find_library(ORT_LIB
        NAMES onnxruntime
        HINTS ${ORT_ROOT}/lib ${ORT_ROOT}/lib64 ${ORT_ROOT}/bin
              /usr/lib /usr/local/lib
              /usr/lib/x86_64-linux-gnu)

    if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
        message(FATAL_ERROR
            "Не найден ONNX Runtime. Скачайте релиз с GitHub "
            "(microsoft/onnxruntime) и передайте -DORT_ROOT=/path. "
            "Для DirectML на Windows нужна сборка onnxruntime-win-x64-* "
            "(содержит dml_provider_factory.h).")
    endif()

    message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime lib:     ${ORT_LIB}")

    target_sources(ii PRIVATE inference_directml.cpp)
    target_include_directories(ii PRIVATE ${ORT_INCLUDE_DIR})
    target_link_libraries(ii PRIVATE ${ORT_LIB})
    target_compile_definitions(ii PRIVATE INF_HAS_DIRECTML=1)

    if(WIN32)
        # dml_provider_factory.h обычно лежит рядом с onnxruntime_cxx_api.h.
        find_path(ORT_DML_PROVIDER_HEADER
            NAMES dml_provider_factory.h
            HINTS ${ORT_INCLUDE_DIR}
                  ${ORT_ROOT}/include
                  ${ORT_ROOT}/include/onnxruntime)
        if(ORT_DML_PROVIDER_HEADER)
            target_compile_definitions(ii PRIVATE INF_DML_EP_AVAILABLE=1)
            message(STATUS "DirectML EP: подключён (header: "
                           "${ORT_DML_PROVIDER_HEADER})")
        else()
            message(WARNING
                "Не найден dml_provider_factory.h — DirectML EP не "
                "активирован. Поставьте onnxruntime-win-x64-* "
                "(или onnxruntime-directml). Бэкенд 'directml' "
                "соберётся, но будет работать только в CPU-EP режиме.")
        endif()
    else()
        message(STATUS "DirectML EP: недоступен на ${CMAKE_SYSTEM_NAME} — "
                       "бэкенд 'directml' собирается только с CPU EP "
                       "(полезно для регрессий ONNX vs TFLite).")
    endif()
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
message(STATUS "  DelegMod : ${USE_NEUTRON_DELEGATE}")
message(STATUS "  Display  : ${USE_DISPLAY}")
message(STATUS "  Camera   : ${USE_CAMERA}")
