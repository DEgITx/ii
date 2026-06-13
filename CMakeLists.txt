# Сборка C++ раннера `ii` и его ядра как переиспользуемой библиотеки.
#
# Архитектура сборки:
#   * ii_core  — СТАТИЧЕСКАЯ библиотека со всем, кроме CLI: движок `ii`,
#                бэкенды инференса (TFLite / TensorRT / DirectML), препроцесс,
#                tile/декод, YOLO, parallel, вывод (Wayland/D3D11), камера.
#                Это и есть «цельная библиотека» — её линкуют и бинарь, и
#                юнит-тесты, и любое встраивающее приложение.
#   * ii       — сам бинарь: только main + разбор аргументов (ii.cpp + cli.cpp),
#                слинкованный с ii_core. CLI намеренно не в библиотеке.
#
# Всё API живёт в едином пространстве имён `ii` (движок, контракт inf::Engine,
# хелперы раннера — больше нет раздельных inf/iirun/imgproc/tile/parallel).
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
#   * USE_II_ENGINE (ON  по умолчанию) — встроенный pure-C++ движок, без SDK;
#   * USE_TFLITE    (ON  по умолчанию) — TensorFlow Lite + опциональный
#                                        внешний делегат (NPU/GPU);
#   * USE_TENSORRT  (OFF по умолчанию) — NVIDIA TensorRT 10+, GPU;
#   * USE_DIRECTML  (OFF по умолчанию) — ONNX Runtime + DirectML EP.
#
# Пример (Linux):
#     cmake -S . -B build -DTFLITE_ROOT=/usr
#     cmake --build build -j
#
# Без графического вывода / без камеры:
#     cmake -S . -B build -DUSE_DISPLAY=OFF -DUSE_CAMERA=OFF
#
# Юнит-тесты без единого SDK (как в CI):
#     cmake -S . -B build -DBUILD_TESTS=ON -DUSE_TFLITE=OFF \
#         -DUSE_DISPLAY=OFF -DUSE_CAMERA=OFF
#     cmake --build build -j && ctest --test-dir build --output-on-failure

cmake_minimum_required(VERSION 3.16)
project(ii CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Опции по умолчанию ---------------------------------------------------
# Display и Camera — платформо-зависимые. Дисплей доступен на Linux
# (Wayland/EGL/GLES2) и на Windows (Win32/Direct3D 11), поэтому там ON.
# Камера: Linux — V4L2, Windows — Media Foundation; на остальных платформах
# OFF и сборка идёт со стабами, чтобы make_display()/make_camera()
# корректно вернули nullptr там, где бэкенда нет.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    option(USE_DISPLAY "Сборка с backend'ом для отображения (Wayland на Linux)" ON)
    option(USE_CAMERA  "Сборка с backend'ом для захвата с камеры (V4L2)"        ON)
elseif(WIN32)
    option(USE_DISPLAY "Сборка с backend'ом для отображения (Direct3D 11 на Windows)" ON)
    option(USE_CAMERA  "Сборка с backend'ом для захвата с камеры (Media Foundation)"  ON)
else()
    option(USE_DISPLAY "Сборка с backend'ом для отображения"                    OFF)
    option(USE_CAMERA  "Сборка с V4L2 backend'ом для захвата с камеры"          OFF)
endif()

# Бэкенды инференса. Все внешние бэкенды по умолчанию OFF: каждый требует
# своего SDK (TFLite — libtensorflowlite + заголовки, TensorRT — CUDA+nvinfer,
# DirectML — ONNX Runtime), и отсутствие любого из них не должно блокировать
# сборку «из коробки». Включаются опциями ниже + указанием путей через
# TFLITE_ROOT / TRT_ROOT / ORT_ROOT (см. блоки ниже).
option(USE_TFLITE    "Сборка с бэкендом TensorFlow Lite (требует libtensorflowlite)" OFF)
option(USE_TENSORRT  "Сборка с бэкендом NVIDIA TensorRT (требует CUDA+nvinfer)" OFF)
option(USE_DIRECTML  "Сборка с бэкендом DirectML/Windows ML (требует ORT)"      OFF)

# Встроенный движок `ii` — backend-нейтральная реализация inf::Engine на
# чистом C++20 без единого SDK. Доступен на любой платформе, поэтому ВСЕГДА
# ON по умолчанию: это единственный бэкенд, не требующий внешних зависимостей,
# так что бинарь собирается и работает сразу. Служит эталоном корректности для
# тяжёлых бэкендов и площадкой для новых слоёв (вплоть до трансформеров).
# Грузит ONNX (TFLite — позже).
option(USE_II_ENGINE "Сборка со встроенным движком инференса ii (pure C++)"    ON)

# Опциональный модуль внешнего NPU-делегата TFLite (см. delegate_neutron.cpp).
# Подставляет платформенный дефолтный путь к .so-плагину делегата на Linux.
# По умолчанию OFF — ядро раннера не зависит ни от какого конкретного
# ускорителя; путь к делегату всегда можно задать через --delegate.
option(USE_NEUTRON_DELEGATE
    "Подключить опциональный модуль внешнего NPU-делегата (Linux)"            OFF)

# Юнит-тесты (GoogleTest, подтягивается через FetchContent). По умолчанию
# OFF — тестируются только бэкенд-независимые модули, backend SDK не нужен.
option(BUILD_TESTS "Сборка юнит-тестов на GoogleTest" OFF)

# Исходники лежат в подкаталоге src/ (CMake в корне репозитория).
set(II_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# ---- Библиотека ii_core: всё, кроме CLI -----------------------------------
# Базовый (всегда собираемый, backend-нейтральный) набор. Опциональные части
# (движок ii, бэкенды, дисплей, камера) добавляются в эту же цель ниже через
# target_sources(ii_core ...) под своими опциями.
add_library(ii_core STATIC
    ${II_SRC_DIR}/preprocess.cpp
    ${II_SRC_DIR}/tensor_utils.cpp
    ${II_SRC_DIR}/yolo.cpp
    ${II_SRC_DIR}/yolo_render.cpp
    ${II_SRC_DIR}/inference.cpp
    ${II_SRC_DIR}/delegate.cpp
    ${II_SRC_DIR}/image_proc.cpp
    ${II_SRC_DIR}/tile.cpp
    ${II_SRC_DIR}/parallel.cpp)

# Заголовки лежат рядом с исходниками — src/ доступен и потребителям
# библиотеки (бинарь, тесты, встраивающее приложение), поэтому PUBLIC.
target_include_directories(ii_core PUBLIC ${II_SRC_DIR})

# ---- Бинарь: только main + CLI, всё остальное берётся из ii_core ----------
add_executable(ii
    ${II_SRC_DIR}/ii.cpp
    ${II_SRC_DIR}/cli.cpp)
target_link_libraries(ii PRIVATE ii_core)

# ---- Платформо-зависимые флаги компилятора (для библиотеки и бинаря) ------
# Применяем к обеим целям: оба TU содержат русские комментарии/литералы
# (нужен /utf-8 на MSVC) и должны собираться с одинаковыми предупреждениями.
foreach(_ii_tgt ii_core ii)
    if(MSVC)
        # /W4 ~ -Wall -Wextra; /utf-8 — из-за русских комментариев и литералов.
        # /O2 — только для не-Debug (в Debug CMake ставит /RTC1, несовместимый
        # с /O2). NOMINMAX гасит min/max-макросы из <windows.h>,
        # _CRT_SECURE_NO_WARNINGS — шум по fopen/strncpy.
        target_compile_options(${_ii_tgt} PRIVATE
            /W4 /utf-8 $<$<NOT:$<CONFIG:Debug>>:/O2>)
        target_compile_definitions(${_ii_tgt} PRIVATE
            NOMINMAX
            _CRT_SECURE_NO_WARNINGS
            WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(${_ii_tgt} PRIVATE -Wall -Wextra -O3)
    endif()
endforeach()

# ---- SIMD: AVX2/FMA на x86 ради векторизации горячих ядер -----------------
# Авто-векторизатор (свёртка/матумножение в engine/ops.cpp) на x86 по умолчанию
# ограничен SSE2 (особенно у MSVC) — это вдвое уже AVX2 и без FMA. Включаем
# AVX2+FMA, которые есть на любом x86-CPU примерно с 2013 г. На ARM (i.MX95)
# NEON — базовый набор, его -O3 задействует и без флагов. Отключите
# (-DUSE_NATIVE_SIMD=OFF) для совсем старого x86 без AVX2 или переносимого бинаря.
option(USE_NATIVE_SIMD "Включить AVX2/FMA на x86 для векторизации ядер" ON)
if(USE_NATIVE_SIMD)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _ii_arch)
    if(_ii_arch MATCHES "x86_64|amd64|i[0-9]86")
        if(MSVC)
            target_compile_options(ii_core PRIVATE /arch:AVX2)
        else()
            target_compile_options(ii_core PRIVATE -mavx2 -mfma)
        endif()
        message(STATUS "ii SIMD: AVX2+FMA (x86)")
    else()
        message(STATUS "ii SIMD: нативная для ${CMAKE_SYSTEM_PROCESSOR} (напр. NEON через -O3)")
    endif()
endif()

# dl/pthread — POSIX; на Windows их нет (а TFLite сам линкуется со своими
# системными зависимостями через .lib импорт-библиотеку). PUBLIC — чтобы
# зависимости унаследовал и бинарь, линкующийся со статической ii_core.
if(UNIX)
    target_link_libraries(ii_core PUBLIC dl pthread)
endif()

# std::thread / std::async (модуль parallel.cpp — пулы потоков). MSVC тащит
# поддержку потоков сам; glibc уже покрыт pthread выше, но Threads::Threads
# даёт правильный флаг (-pthread) и для MinGW, и для прочих тулчейнов.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ii_core PUBLIC Threads::Threads)

# ---- stb (одноголовники для image/resize, без внешних зависимостей) -------
include(FetchContent)
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(stb)
# PUBLIC — preprocess.cpp/image_proc.cpp в библиотеке тянут stb; тесты,
# собирающие те же модули через ii_core, наследуют путь.
target_include_directories(ii_core PUBLIC ${stb_SOURCE_DIR})

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

    # inference_tflite.cpp — единственный TU, тянущий заголовки TFLite,
    # поэтому include/lib приватны для ii_core (потребители идут через
    # inf::Engine и SDK не видят). INF_HAS_TFLITE нужен inference.cpp
    # (фабрика make_engine) — он тоже в ii_core, так что PRIVATE достаточно.
    target_sources(ii_core PRIVATE ${II_SRC_DIR}/inference_tflite.cpp)
    target_include_directories(ii_core PRIVATE ${TFLITE_INCLUDE_DIR})
    target_link_libraries(ii_core PUBLIC ${TFLITE_LIB})
    target_compile_definitions(ii_core PRIVATE INF_HAS_TFLITE=1)
endif()

# ---- Бэкенд: встроенный движок ii (pure C++, всегда доступен) -------------
# Никаких внешних зависимостей: ядро движка + мост inf::Engine компилируются
# прямо в библиотеку. INF_HAS_II включает фабричную ветку make_ii_engine() в
# inference.cpp. parallel.cpp (общий пул) уже в базовом наборе ii_core.
if(USE_II_ENGINE)
    target_sources(ii_core PRIVATE
        ${II_SRC_DIR}/engine/tensor.cpp
        ${II_SRC_DIR}/engine/ops.cpp
        ${II_SRC_DIR}/engine/graph.cpp
        ${II_SRC_DIR}/engine/kernels.cpp
        ${II_SRC_DIR}/engine/onnx.cpp
        ${II_SRC_DIR}/engine/backend.cpp)
    target_compile_definitions(ii_core PRIVATE INF_HAS_II=1)
    message(STATUS "ii engine: встроенный pure-C++ бэкенд (ON)")
endif()

# ---- Опциональный модуль внешнего NPU-делегата ---------------------------
# Платформенно-специфичная мелочь: дефолтный путь к .so-плагину делегата.
# Отделён от ядра, чтобы раннер не был привязан ни к одному ускорителю.
if(USE_NEUTRON_DELEGATE)
    target_sources(ii_core PRIVATE ${II_SRC_DIR}/delegate_neutron.cpp)
    target_compile_definitions(ii_core PRIVATE INF_HAS_DELEGATE_MODULE=1)
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

    target_sources(ii_core PRIVATE ${II_SRC_DIR}/inference_tensorrt.cpp)
    target_include_directories(ii_core PRIVATE ${TRT_INCLUDE_DIR})
    target_link_libraries(ii_core PUBLIC ${TRT_NVINFER_LIB})
    if(CUDAToolkit_FOUND)
        target_link_libraries(ii_core PUBLIC CUDA::cudart)
    else()
        target_include_directories(ii_core PRIVATE ${_INF_CUDA_INCLUDE_DIR})
        target_link_libraries(ii_core PUBLIC ${_INF_CUDART_LIB})
    endif()
    if(TRT_NVONNXPARSER_LIB)
        target_link_libraries(ii_core PUBLIC ${TRT_NVONNXPARSER_LIB})
        target_compile_definitions(ii_core PRIVATE INF_TRT_HAS_ONNX=1)
    endif()
    target_compile_definitions(ii_core PRIVATE INF_HAS_TENSORRT=1)
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

    target_sources(ii_core PRIVATE ${II_SRC_DIR}/inference_directml.cpp)
    target_include_directories(ii_core PRIVATE ${ORT_INCLUDE_DIR})
    target_link_libraries(ii_core PUBLIC ${ORT_LIB})
    target_compile_definitions(ii_core PRIVATE INF_HAS_DIRECTML=1)

    if(WIN32)
        # dml_provider_factory.h обычно лежит рядом с onnxruntime_cxx_api.h.
        find_path(ORT_DML_PROVIDER_HEADER
            NAMES dml_provider_factory.h
            HINTS ${ORT_INCLUDE_DIR}
                  ${ORT_ROOT}/include
                  ${ORT_ROOT}/include/onnxruntime)
        if(ORT_DML_PROVIDER_HEADER)
            target_compile_definitions(ii_core PRIVATE INF_DML_EP_AVAILABLE=1)
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

# ---- Backend для камеры --------------------------------------------------
# Linux -> V4L2 (camera_v4l2.cpp);
# Windows -> Media Foundation (camera_mediafoundation.cpp);
# прочее / USE_CAMERA=OFF -> заглушка (make_camera() возвращает nullptr).
if(USE_CAMERA AND WIN32)
    target_sources(ii_core PRIVATE ${II_SRC_DIR}/camera_mediafoundation.cpp)
    target_compile_definitions(ii_core PRIVATE NPU_HAS_CAMERA=1)
    # Media Foundation: mfplat/mf/mfreadwrite — сам reader и устройства;
    # mfuuid — GUID'ы форматов/атрибутов; ole32 — COM. Все есть в любом
    # Windows SDK, доп. путей не требуют.
    target_link_libraries(ii_core PUBLIC mfplat mf mfreadwrite mfuuid ole32)
elseif(USE_CAMERA)
    target_sources(ii_core PRIVATE ${II_SRC_DIR}/camera_v4l2.cpp)
    target_compile_definitions(ii_core PRIVATE NPU_HAS_CAMERA=1)
else()
    target_sources(ii_core PRIVATE ${II_SRC_DIR}/camera_stub.cpp)
endif()

# ---- Backend для отображения ---------------------------------------------
# Linux -> Wayland/EGL/GLES2 (display_wayland.cpp);
# Windows -> Win32/Direct3D 11 (display_d3d11.cpp);
# прочее / USE_DISPLAY=OFF -> заглушка (make_display() возвращает nullptr).
if(USE_DISPLAY AND WIN32)
    target_sources(ii_core PRIVATE ${II_SRC_DIR}/display_d3d11.cpp)
    # d3dcompiler — компиляция HLSL-шейдеров в рантайме (D3DCompile);
    # d3d11/dxgi — само устройство и swapchain. Все три есть в любом
    # Windows SDK, дополнительных путей не требуют.
    target_link_libraries(ii_core PUBLIC d3d11 dxgi d3dcompiler)
elseif(USE_DISPLAY)
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

    target_sources(ii_core PRIVATE
        ${II_SRC_DIR}/display_wayland.cpp
        ${XDG_SHELL_HEADER}
        ${XDG_SHELL_SOURCE})
    target_include_directories(ii_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(ii_core PUBLIC PkgConfig::WAYLAND)
else()
    target_sources(ii_core PRIVATE ${II_SRC_DIR}/display_stub.cpp)
endif()

# ---- Юнит-тесты (опционально) --------------------------------------------
# Подключается после FetchContent(stb), т.к. tests наследуют ${stb_SOURCE_DIR}
# и прочие пути от ii_core.
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Финальный summary ---------------------------------------------------
message(STATUS "ii build summary:")
message(STATUS "  platform : ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  ii engine: ${USE_II_ENGINE}")
message(STATUS "  TFLite   : ${USE_TFLITE}")
message(STATUS "  TensorRT : ${USE_TENSORRT}")
message(STATUS "  DirectML : ${USE_DIRECTML}")
message(STATUS "  DelegMod : ${USE_NEUTRON_DELEGATE}")
message(STATUS "  Display  : ${USE_DISPLAY}")
message(STATUS "  Camera   : ${USE_CAMERA}")
message(STATUS "  Tests    : ${BUILD_TESTS}")
