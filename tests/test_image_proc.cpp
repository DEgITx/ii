// Юнит-тесты декодера выходного тензора image-to-image моделей
// (image_proc.*): распознавание image-shaped выходов, парсинг
// OutputRange и декод тензора в RGB (float/quantized, LUT-кэш,
// grayscale-разворот, запись в произвольную область буфера).
//
// Зависит только от inference.h — backend SDK не требуется.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "image_proc.h"
#include "inference.h"

using ii::OutputRange;
using ii::parse_output_range;
using ii::is_image_output;
using ii::detect_image_output_index;
using ii::decode_image_output;
using ii::decode_image_output_to;
using ii::DecodeOptions;
using ii::OutputImage;
using ii::DecodeCache;
using ii::TensorDesc;
using ii::DType;

namespace {

// Удобный конструктор дескриптора выхода.
TensorDesc make_desc(std::vector<int> shape, DType dt,
                     float scale = 0.0f, int zp = 0) {
    TensorDesc d;
    d.shape = std::move(shape);
    d.dtype = dt;
    d.scale = scale;
    d.zero_point = zp;
    return d;
}

}  // namespace

// ---- parse_output_range ---------------------------------------------------

TEST(ParseOutputRange, KnownSynonyms) {
    OutputRange r = OutputRange::Byte;
    EXPECT_TRUE(parse_output_range("unit", r));
    EXPECT_EQ(r, OutputRange::Unit);
    EXPECT_TRUE(parse_output_range("0..1", r));
    EXPECT_EQ(r, OutputRange::Unit);

    EXPECT_TRUE(parse_output_range("signed", r));
    EXPECT_EQ(r, OutputRange::Signed);
    EXPECT_TRUE(parse_output_range("tanh", r));
    EXPECT_EQ(r, OutputRange::Signed);

    EXPECT_TRUE(parse_output_range("byte", r));
    EXPECT_EQ(r, OutputRange::Byte);
    EXPECT_TRUE(parse_output_range("0..255", r));
    EXPECT_EQ(r, OutputRange::Byte);
}

TEST(ParseOutputRange, UnknownLeavesOutputUntouched) {
    OutputRange r = OutputRange::Signed;
    EXPECT_FALSE(parse_output_range("nonsense", r));
    EXPECT_EQ(r, OutputRange::Signed);  // не тронут
}

// ---- is_image_output ------------------------------------------------------

TEST(IsImageOutput, AcceptsNhwc1Or3Channels) {
    EXPECT_TRUE(is_image_output(make_desc({1, 8, 8, 3}, DType::Float32)));
    EXPECT_TRUE(is_image_output(make_desc({1, 4, 16, 1}, DType::Float32)));
}

TEST(IsImageOutput, RejectsBadShapes) {
    EXPECT_FALSE(is_image_output(make_desc({8, 8, 3}, DType::Float32)));      // ранг 3
    EXPECT_FALSE(is_image_output(make_desc({2, 8, 8, 3}, DType::Float32)));   // N != 1
    EXPECT_FALSE(is_image_output(make_desc({1, 1, 8, 3}, DType::Float32)));   // H < 2
    EXPECT_FALSE(is_image_output(make_desc({1, 8, 8, 4}, DType::Float32)));   // C не 1/3
}

// ---- detect_image_output_index --------------------------------------------

TEST(DetectImageOutput, SingleCandidate) {
    std::vector<TensorDesc> outs = {
        make_desc({1, 10}, DType::Float32),          // не image
        make_desc({1, 8, 8, 3}, DType::Float32),     // image
    };
    EXPECT_EQ(detect_image_output_index(outs), 1);
}

TEST(DetectImageOutput, MultipleCandidatesAmbiguous) {
    std::vector<TensorDesc> outs = {
        make_desc({1, 8, 8, 3}, DType::Float32),
        make_desc({1, 4, 4, 1}, DType::Float32),
    };
    EXPECT_EQ(detect_image_output_index(outs), -1);  // неоднозначно
}

TEST(DetectImageOutput, NoCandidate) {
    std::vector<TensorDesc> outs = {
        make_desc({1, 1000}, DType::Float32),
    };
    EXPECT_EQ(detect_image_output_index(outs), -1);
}

// ---- decode_image_output: float ------------------------------------------

TEST(DecodeImageOutput, Float32UnitRange) {
    // [1,2,2,3]: проверяем порядок каналов RGB на пикселе (0,0); H,W>=2.
    TensorDesc d = make_desc({1, 2, 2, 3}, DType::Float32);
    std::vector<float> data(2 * 2 * 3, 0.0f);
    data[0] = 0.0f; data[1] = 0.5f; data[2] = 1.0f;  // пиксель (0,0)
    OutputImage out;
    DecodeOptions opt;  // Unit
    ASSERT_TRUE(decode_image_output(d, data.data(), opt, out));
    EXPECT_EQ(out.width, 2);
    EXPECT_EQ(out.height, 2);
    EXPECT_EQ(out.channels_src, 3);
    ASSERT_EQ(out.rgb.size(), 2u * 2u * 3u);
    EXPECT_EQ(out.rgb[0], 0);
    EXPECT_EQ(out.rgb[1], 128);  // lround(0.5*255) = 128
    EXPECT_EQ(out.rgb[2], 255);
}

TEST(DecodeImageOutput, Float32SignedRange) {
    TensorDesc d = make_desc({1, 2, 2, 3}, DType::Float32);
    std::vector<float> data(2 * 2 * 3, 0.0f);
    data[0] = -1.0f; data[1] = 0.0f; data[2] = 1.0f;
    OutputImage out;
    DecodeOptions opt;
    opt.range = OutputRange::Signed;  // v*0.5+0.5
    ASSERT_TRUE(decode_image_output(d, data.data(), opt, out));
    EXPECT_EQ(out.rgb[0], 0);    // -1 → 0.0
    EXPECT_EQ(out.rgb[1], 128);  //  0 → 0.5 → 128
    EXPECT_EQ(out.rgb[2], 255);  //  1 → 1.0
}

TEST(DecodeImageOutput, ClampsOutOfRange) {
    TensorDesc d = make_desc({1, 2, 2, 3}, DType::Float32);
    std::vector<float> data(2 * 2 * 3, 0.0f);
    data[0] = -5.0f; data[1] = 2.0f; data[2] = 0.5f;
    OutputImage out;
    DecodeOptions opt;  // Unit
    ASSERT_TRUE(decode_image_output(d, data.data(), opt, out));
    EXPECT_EQ(out.rgb[0], 0);    // < 0 клампится
    EXPECT_EQ(out.rgb[1], 255);  // > 1 клампится
    EXPECT_EQ(out.rgb[2], 128);  // 0.5 в пределах
}

TEST(DecodeImageOutput, GrayscaleExpandsToRgb) {
    // C=1 → R=G=B.
    TensorDesc d = make_desc({1, 2, 2, 1}, DType::Float32);
    std::vector<float> data = {0.0f, 1.0f,
                               0.5f, 0.0f};
    OutputImage out;
    DecodeOptions opt;
    ASSERT_TRUE(decode_image_output(d, data.data(), opt, out));
    EXPECT_EQ(out.channels_src, 1);
    ASSERT_EQ(out.rgb.size(), 2u * 2u * 3u);
    // пиксель (0,0) = 0 → (0,0,0)
    EXPECT_EQ(out.rgb[0], 0);
    EXPECT_EQ(out.rgb[1], 0);
    EXPECT_EQ(out.rgb[2], 0);
    // пиксель (1,0) = 1 → (255,255,255)
    EXPECT_EQ(out.rgb[3], 255);
    EXPECT_EQ(out.rgb[4], 255);
    EXPECT_EQ(out.rgb[5], 255);
}

// ---- decode_image_output: quantized + LUT-кэш -----------------------------

TEST(DecodeImageOutput, UInt8ByteRangeIdentity) {
    // UInt8, scale=1, zp=0, Byte: u → u*1 → /255 → *255 = u (identity).
    TensorDesc d = make_desc({1, 2, 2, 3}, DType::UInt8, /*scale=*/1.0f, 0);
    std::vector<std::uint8_t> data(2 * 2 * 3);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = (std::uint8_t)(i * 17);
    OutputImage out;
    DecodeOptions opt;
    opt.range = OutputRange::Byte;
    ASSERT_TRUE(decode_image_output(d, data.data(), opt, out));
    for (std::size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(out.rgb[i], data[i]) << "i=" << i;
}

TEST(DecodeImageOutput, Int8Dequant) {
    // Int8, scale=1/127, zp=0, Unit. Значение 127 → 1.0 → 255; 0 → 0;
    // -128 → <0 клампится. Пиксель (0,0) = [127,0,64], (0,1) = [-128,..].
    TensorDesc d = make_desc({1, 2, 2, 3}, DType::Int8, 1.0f / 127.0f, 0);
    std::vector<std::int8_t> data(2 * 2 * 3, 0);
    data[0] = 127; data[1] = 0; data[2] = 64;   // пиксель (0,0)
    data[3] = -128;                              // пиксель (0,1), канал R
    OutputImage out;
    DecodeOptions opt;  // Unit
    ASSERT_TRUE(decode_image_output(d, data.data(), opt, out));
    EXPECT_EQ(out.rgb[0], 255);                       // 127 → 1.0
    EXPECT_EQ(out.rgb[1], 0);                          // 0
    EXPECT_EQ(out.rgb[2], (std::uint8_t)std::lround(64.0 / 127.0 * 255.0));
    EXPECT_EQ(out.rgb[3], 0);                          // -128 → <0 клампится
}

TEST(DecodeImageOutput, CacheReuseGivesSameResult) {
    // Передаём DecodeCache — LUT строится один раз, повторный вызов с теми
    // же параметрами квантования даёт идентичный результат.
    TensorDesc d = make_desc({1, 2, 2, 3}, DType::UInt8, 1.0f / 255.0f, 0);
    std::vector<std::uint8_t> data(2 * 2 * 3);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = (std::uint8_t)(i * 21);
    DecodeCache cache;
    DecodeOptions opt;  // Unit: u/255*255 = u
    OutputImage a, b;
    ASSERT_TRUE(decode_image_output(d, data.data(), opt, a, &cache));
    ASSERT_TRUE(cache.lut_valid);
    ASSERT_TRUE(decode_image_output(d, data.data(), opt, b, &cache));
    EXPECT_EQ(a.rgb, b.rgb);
    for (std::size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(a.rgb[i], data[i]);
}

// ---- decode failure paths -------------------------------------------------

TEST(DecodeImageOutput, RejectsNonImageShape) {
    TensorDesc d = make_desc({1, 1000}, DType::Float32);
    std::vector<float> data(1000, 0.0f);
    OutputImage out;
    DecodeOptions opt;
    EXPECT_FALSE(decode_image_output(d, data.data(), opt, out));
}

TEST(DecodeImageOutput, RejectsUnsupportedDtype) {
    TensorDesc d = make_desc({1, 2, 2, 3}, DType::Int32);
    std::vector<std::int32_t> data(2 * 2 * 3, 0);
    OutputImage out;
    DecodeOptions opt;
    EXPECT_FALSE(decode_image_output(d, data.data(), opt, out));
}

// ---- decode_image_output_to: запись в под-область буфера ------------------

TEST(DecodeImageOutputTo, WritesIntoSubRegionWithStride) {
    // Буфер 4x4 RGB, инициализирован нулями. Декодируем 2x2 тайл в (1,1).
    const int W = 4, H = 4;
    std::vector<std::uint8_t> canvas((std::size_t)W * H * 3, 0);

    TensorDesc d = make_desc({1, 2, 2, 1}, DType::Float32);
    std::vector<float> data = {1.0f, 1.0f,
                               1.0f, 1.0f};  // всё белое
    DecodeOptions opt;
    ASSERT_TRUE(decode_image_output_to(d, data.data(), opt,
                                       canvas.data(),
                                       /*dst_stride=*/(std::size_t)W * 3,
                                       /*dst_x=*/1, /*dst_y=*/1));
    auto px = [&](int x, int y) { return canvas[(std::size_t)(y * W + x) * 3]; };
    // Тайл (1,1)..(2,2) = 255, остальное = 0.
    EXPECT_EQ(px(1, 1), 255);
    EXPECT_EQ(px(2, 2), 255);
    EXPECT_EQ(px(0, 0), 0);  // не тронут
    EXPECT_EQ(px(3, 3), 0);  // не тронут
    EXPECT_EQ(px(0, 1), 0);  // слева от тайла — не тронут
}

TEST(DecodeImageOutputTo, RejectsNullBuffer) {
    TensorDesc d = make_desc({1, 2, 2, 3}, DType::Float32);
    std::vector<float> data(2 * 2 * 3, 0.0f);
    DecodeOptions opt;
    EXPECT_FALSE(decode_image_output_to(d, data.data(), opt,
                                        nullptr, 12, 0, 0));
}
