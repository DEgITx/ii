// Юнит-тесты бэкенд-независимых хелперов tensor_utils.* :
// арифметика по shape, классификация «картиночного» входа и
// деквантование выходного тензора в float.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "inference.h"
#include "tensor_utils.h"

using ii::numel;
using ii::shape_to_str;
using ii::image_input_channels;
using ii::is_image_input;
using ii::dequantize_output;
using ii::DType;
using ii::TensorDesc;

// ---- numel ----------------------------------------------------------------

TEST(Numel, ProductOfPositiveDims) {
    EXPECT_EQ(numel({1, 3, 224, 224}), 150528u);
    EXPECT_EQ(numel({10}), 10u);
}

TEST(Numel, EmptyShapeIsOne) {
    // Пустой shape — скаляр, произведение пустого множества = 1.
    EXPECT_EQ(numel({}), 1u);
}

TEST(Numel, NonPositiveDimYieldsZero) {
    // Динамическая/нулевая ось -> 0 (не поддерживается).
    EXPECT_EQ(numel({1, -1, 224, 224}), 0u);
    EXPECT_EQ(numel({1, 0, 5}), 0u);
}

// ---- shape_to_str ---------------------------------------------------------

TEST(ShapeToStr, Formatting) {
    EXPECT_EQ(shape_to_str({1, 3, 224, 224}), "[1,3,224,224]");
    EXPECT_EQ(shape_to_str({}), "[]");
    EXPECT_EQ(shape_to_str({7}), "[7]");
}

// ---- image_input_channels / is_image_input -------------------------------

TEST(ImageInput, RgbNhwc) {
    EXPECT_EQ(image_input_channels({1, 224, 224, 3}), 3);
    EXPECT_TRUE(is_image_input({1, 224, 224, 3}));
}

TEST(ImageInput, GrayscaleNhwc) {
    EXPECT_EQ(image_input_channels({1, 96, 96, 1}), 1);
    EXPECT_TRUE(is_image_input({1, 96, 96, 1}));
}

TEST(ImageInput, RejectsNonImageShapes) {
    EXPECT_EQ(image_input_channels({1, 224, 224, 2}), 0);  // 2 канала
    EXPECT_EQ(image_input_channels({2, 224, 224, 3}), 0);  // batch != 1
    EXPECT_EQ(image_input_channels({224, 224, 3}), 0);     // не 4D
    EXPECT_FALSE(is_image_input({1, 1000}));
}

// ---- dequantize_output ----------------------------------------------------

namespace {
// Заполняет дескриптор так, чтобы bytes соответствовали числу элементов.
TensorDesc make_desc(DType dt, std::size_t count, float scale, int zp) {
    TensorDesc d;
    d.dtype = dt;
    d.scale = scale;
    d.zero_point = zp;
    d.bytes = count * ii::dtype_size(dt);
    return d;
}
}  // namespace

TEST(Dequantize, Float32IsCopied) {
    const float src[] = {-1.5f, 0.0f, 2.25f};
    auto d = make_desc(DType::Float32, 3, 0.0f, 0);
    std::vector<float> out;
    ASSERT_TRUE(dequantize_output(d, src, out));
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0], -1.5f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 2.25f);
}

TEST(Dequantize, Int8AffineFormula) {
    // (x - zp) * scale
    const std::int8_t src[] = {0, 10, -5};
    auto d = make_desc(DType::Int8, 3, 0.5f, 2);
    std::vector<float> out;
    ASSERT_TRUE(dequantize_output(d, src, out));
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0], (0 - 2) * 0.5f);
    EXPECT_FLOAT_EQ(out[1], (10 - 2) * 0.5f);
    EXPECT_FLOAT_EQ(out[2], (-5 - 2) * 0.5f);
}

TEST(Dequantize, UInt8AffineFormula) {
    const std::uint8_t src[] = {0, 128, 255};
    auto d = make_desc(DType::UInt8, 3, 1.0f / 255.0f, 0);
    std::vector<float> out;
    ASSERT_TRUE(dequantize_output(d, src, out));
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 1.0f);
}

TEST(Dequantize, ZeroScaleFallsBackToRaw) {
    // scale == 0 для int-тензора: берём сырое значение, иначе деление
    // на нулевую шкалу всё бы обнулило (s = 1.0f внутри).
    const std::int8_t src[] = {7, -3};
    auto d = make_desc(DType::Int8, 2, 0.0f, 0);
    std::vector<float> out;
    ASSERT_TRUE(dequantize_output(d, src, out));
    EXPECT_FLOAT_EQ(out[0], 7.0f);
    EXPECT_FLOAT_EQ(out[1], -3.0f);
}

TEST(Dequantize, UnknownDtypeFails) {
    const std::uint8_t src[] = {0};
    TensorDesc d;
    d.dtype = DType::Bool;  // не поддерживается деквантованием
    d.bytes = 1;
    std::vector<float> out;
    EXPECT_FALSE(dequantize_output(d, src, out));
}
