// Юнит-тесты препроцессинга (preprocess.*): RGB->luma (BT.601) и
// letterbox-ресайз с паддингом. Загрузку файлов (load_image) не трогаем —
// она требует stb I/O и реального файла.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "preprocess.h"

using iirun::rgb_to_gray;
using iirun::letterbox;

// ---- rgb_to_gray ----------------------------------------------------------

TEST(RgbToGray, Grayscale) {
    // Y = (77*R + 150*G + 29*B) >> 8. Серый вход остаётся ~собой.
    std::vector<std::uint8_t> rgb = {0, 0, 0, 255, 255, 255, 128, 128, 128};
    std::vector<std::uint8_t> gray;
    rgb_to_gray(rgb.data(), 3, gray);
    ASSERT_EQ(gray.size(), 3u);
    EXPECT_EQ(gray[0], 0);
    EXPECT_EQ(gray[1], 255);   // (77+150+29)*255>>8 = 256*255>>8 = 255
    EXPECT_EQ(gray[2], 128);   // 256*128>>8 = 128
}

TEST(RgbToGray, PureChannels) {
    std::vector<std::uint8_t> rgb = {255, 0, 0, 0, 255, 0, 0, 0, 255};
    std::vector<std::uint8_t> gray;
    rgb_to_gray(rgb.data(), 3, gray);
    EXPECT_EQ(gray[0], (77u * 255) >> 8);   // только R
    EXPECT_EQ(gray[1], (150u * 255) >> 8);  // только G (самый яркий)
    EXPECT_EQ(gray[2], (29u * 255) >> 8);   // только B (самый тёмный)
    EXPECT_GT(gray[1], gray[0]);
    EXPECT_GT(gray[0], gray[2]);
}

// ---- letterbox ------------------------------------------------------------

TEST(Letterbox, OutputSizeAndPadding) {
    // Источник 2x1 (широкий), цель 4x4. r = min(4/2, 4/1) = 2,
    // new = 4x2, вертикальный паддинг сверху/снизу по 1 строке.
    std::vector<std::uint8_t> src = {10, 20, 30, 40, 50, 60};  // 2 пикселя RGB
    std::vector<std::uint8_t> dst;
    letterbox(src.data(), 2, 1, 4, 4, dst, /*pad=*/114);

    // Размер выхода = target_w * target_h * 3.
    ASSERT_EQ(dst.size(), 4u * 4u * 3u);

    // Верхняя строка — это паддинг (114,114,114,...).
    EXPECT_EQ(dst[0], 114);
    EXPECT_EQ(dst[1], 114);
    EXPECT_EQ(dst[2], 114);
    // Нижняя строка — тоже паддинг.
    std::size_t last_row = (std::size_t)(4 - 1) * 4 * 3;
    EXPECT_EQ(dst[last_row], 114);
}

TEST(Letterbox, SquareToSquareNoPadding) {
    // Источник и цель квадратные одного размера: r = 1, паддинга нет.
    std::vector<std::uint8_t> src(2 * 2 * 3, 200);
    std::vector<std::uint8_t> dst;
    letterbox(src.data(), 2, 2, 2, 2, dst, 114);
    ASSERT_EQ(dst.size(), 2u * 2u * 3u);
    // Никакого паддинга — все пиксели от источника (resize 1:1 сохраняет 200).
    for (auto v : dst) EXPECT_EQ(v, 200);
}

TEST(Letterbox, CustomPadValue) {
    std::vector<std::uint8_t> src = {0, 0, 0};  // 1x1
    std::vector<std::uint8_t> dst;
    letterbox(src.data(), 1, 1, 1, 3, dst, /*pad=*/255);
    ASSERT_EQ(dst.size(), 1u * 3u * 3u);
    // Центр (строка 1) — пиксель источника (0), края — паддинг (255).
    EXPECT_EQ(dst[0], 255);  // верхняя строка
    EXPECT_EQ(dst[6], 255);  // нижняя строка
}
