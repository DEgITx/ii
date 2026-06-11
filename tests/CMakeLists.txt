# Юнит-тесты на GoogleTest. Тестируем только бэкенд-независимые модули
# (tensor_utils / yolo / preprocess + хелперы inference.h) — им не нужен
# ни один backend SDK, поэтому тесты собираются на любом dev-host.
#
# Включается опцией BUILD_TESTS (см. корневой CMakeLists). GoogleTest
# подтягивается через FetchContent — внешних зависимостей не требуется.

include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE)
# На Windows форсим общий CRT с тестируемым кодом, иначе линк падает.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Тестируемые исходники. Ровно те модули, что не тянут backend SDK:
#   * inference.cpp     — dtype_name/dtype_size (make_engine без INF_HAS_*
#                         компилируется в no-op fallback);
#   * tensor_utils.cpp  — numel/shape/dequantize;
#   * yolo.cpp          — decode/NMS/scale;
#   * preprocess.cpp    — letterbox/rgb_to_gray (тянет stb — заголовки ниже);
#   * image_proc.cpp    — декод выходного тензора в RGB (тянет stb_image_write);
#   * tile.cpp          — планирование сетки/extract/canvas (зависит от image_proc).
set(II_TEST_LIB_SOURCES
    ${II_SRC_DIR}/inference.cpp
    ${II_SRC_DIR}/tensor_utils.cpp
    ${II_SRC_DIR}/yolo.cpp
    ${II_SRC_DIR}/preprocess.cpp
    ${II_SRC_DIR}/image_proc.cpp
    ${II_SRC_DIR}/tile.cpp)

# Встроенный движок `ii` — backend-нейтральный, без SDK, поэтому его ядро
# (математика + граф) тестируется на любом dev-host. II_ENGINE_SOURCES
# задаётся в корневом CMakeLists.
add_executable(ii_tests
    ${II_TEST_LIB_SOURCES}
    ${II_ENGINE_SOURCES}
    ${II_SRC_DIR}/inference_ii.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_image_proc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ii_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ii_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ii_onnx.cpp)

# Заголовки раннера + stb (для preprocess.cpp). stb_SOURCE_DIR заполнен
# в корневом CMakeLists через FetchContent (tests подключается после).
target_include_directories(ii_tests PRIVATE
    ${II_SRC_DIR}
    ${stb_SOURCE_DIR})

target_link_libraries(ii_tests PRIVATE GTest::gtest_main)

if(MSVC)
    target_compile_options(ii_tests PRIVATE /utf-8)
    target_compile_definitions(ii_tests PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

include(GoogleTest)
gtest_discover_tests(ii_tests)
