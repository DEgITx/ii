# Юнит-тесты на GoogleTest.
#
# Тесты линкуются с цельной библиотекой ii_core (корневой CMakeLists), а не
# пересобирают её исходники по списку — это и есть смысл «единой библиотеки».
# Покрытие: бэкенд-нейтральные модули (tensor_utils / yolo / preprocess /
# image_proc / tile + хелперы inference.h) и весь движок `ii` (математика,
# граф, ONNX-загрузчик, NCHW-мост).
#
# Чтобы собирать без единого backend SDK (как в CI), конфигурируйте с
# -DUSE_TFLITE=OFF -DUSE_DISPLAY=OFF -DUSE_CAMERA=OFF: тогда ii_core состоит
# только из pure-C++ движка `ii` и нейтральных хелперов, и тестам хватает его.
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

add_executable(ii_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/test_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_preprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_image_proc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_engine_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_engine_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_engine_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_nchw_bridge.cpp)

# Вся тестируемая реализация — в ii_core. Заголовки раннера (src/) и stb
# наследуются транзитивно (они объявлены PUBLIC на ii_core).
target_link_libraries(ii_tests PRIVATE ii_core GTest::gtest_main)

if(MSVC)
    target_compile_options(ii_tests PRIVATE /utf-8)
    target_compile_definitions(ii_tests PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

include(GoogleTest)
gtest_discover_tests(ii_tests)
