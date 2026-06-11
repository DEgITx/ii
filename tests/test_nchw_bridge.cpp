// Тесты «мостика раскладки» для ONNX-моделей: распознавание NHWC/NCHW
// (image_input_info) и channel-planar заливка входа (fill_input_nchw).

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "preprocess.h"
#include "tensor_utils.h"

using iirun::ImageLayout;
using iirun::image_input_info;
using iirun::fill_input_nchw;

TEST(ImageLayout, DetectNhwcVsNchw) {
    ImageLayout lay;
    int h = 0, w = 0;
    EXPECT_EQ(image_input_info({1, 3, 640, 640}, lay, h, w), 3);
    EXPECT_EQ(lay, ImageLayout::NCHW);
    EXPECT_EQ(h, 640);
    EXPECT_EQ(w, 640);

    EXPECT_EQ(image_input_info({1, 224, 224, 3}, lay, h, w), 3);
    EXPECT_EQ(lay, ImageLayout::NHWC);
    EXPECT_EQ(h, 224);

    EXPECT_EQ(image_input_info({1, 1, 8, 8}, lay, h, w), 1);
    EXPECT_EQ(lay, ImageLayout::NCHW);

    EXPECT_EQ(image_input_info({1, 8, 8, 1}, lay, h, w), 1);
    EXPECT_EQ(lay, ImageLayout::NHWC);

    EXPECT_EQ(image_input_info({1, 5, 5, 5}, lay, h, w), 0);   // не картинка
    EXPECT_EQ(lay, ImageLayout::None);
}

TEST(FillInputNchw, TransposesHwcToChw) {
    // 2x2 RGB, пиксели HWC.
    std::vector<uint8_t> rgb = {10, 20, 30,  40, 50, 60,
                                70, 80, 90,  100, 110, 120};
    inf::TensorDesc info;
    info.dtype = inf::DType::Float32;
    info.shape = {1, 3, 2, 2};

    std::vector<float> out(12, -1.0f);
    ASSERT_TRUE(fill_input_nchw(rgb, info, out.data(), 3, 2, 2));

    // Ожидаем планарную раскладку CHW, нормировка /255:
    //   R: 10,40,70,100 | G: 20,50,80,110 | B: 30,60,90,120
    std::vector<float> want = {10, 40, 70, 100, 20, 50, 80, 110, 30, 60, 90, 120};
    for (auto& v : want) v /= 255.0f;
    for (std::size_t i = 0; i < want.size(); ++i)
        EXPECT_NEAR(out[i], want[i], 1e-6f) << "i=" << i;
}
