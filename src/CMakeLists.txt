# Сборка C++ раннера TFLite + внешний NPU-делегат делегата для NPU.
#
# Пример (на устройстве/в SDK BSP, где TFLite уже установлен):
#     cmake -S . -B build -DTFLITE_ROOT=/usr
#     cmake --build build -j
#
# Если TFLite собран отдельно — укажите пути явно:
#     cmake -S . -B build \
#         -DTFLITE_INCLUDE_DIR=/opt/tflite/include \
#         -DTFLITE_LIB=/opt/tflite/lib/libtensorflowlite.so

cmake_minimum_required(VERSION 3.16)
project(ii CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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
