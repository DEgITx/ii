// Юнит-тесты постобработки YOLOv8 (yolo.*): декодирование боксов,
// per-class NMS и обратный letterbox-маппинг scale_to_original.

#include <gtest/gtest.h>

#include <vector>

#include "yolo.h"

namespace {

// Собирает выход YOLO в layout [C, A] (channels_first) из списка
// anchor'ов {cx, cy, w, h, score_class0, score_class1, ...}.
std::vector<float> pack_channels_first(
    const std::vector<std::vector<float>>& anchors, int channels) {
    const int A = (int)anchors.size();
    std::vector<float> buf((std::size_t)channels * A, 0.0f);
    for (int a = 0; a < A; ++a)
        for (int c = 0; c < channels; ++c)
            buf[(std::size_t)c * A + a] = anchors[a][c];
    return buf;
}

}  // namespace

// ---- decode_yolov8 --------------------------------------------------------

TEST(DecodeYolo, SingleBoxPixelCoords) {
    // channels = 4 + nc, nc = 2. Один anchor, нормализация выключена
    // (координаты уже в пикселях входа).
    const int channels = 6, A = 1;
    // cx=100, cy=100, w=40, h=20, score c0=0.9, c1=0.1
    std::vector<std::vector<float>> anchors = {{100, 100, 40, 20, 0.9f, 0.1f}};
    auto buf = pack_channels_first(anchors, channels);

    YoloPostOptions opt;
    opt.normalized = false;
    opt.conf_thresh = 0.25f;

    auto dets = decode_yolov8(buf.data(), channels, A, /*channels_first=*/true,
                              640, 640, opt);
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 0);
    EXPECT_FLOAT_EQ(dets[0].score, 0.9f);
    // xyxy из центра: [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
    EXPECT_FLOAT_EQ(dets[0].x1, 80.0f);
    EXPECT_FLOAT_EQ(dets[0].y1, 90.0f);
    EXPECT_FLOAT_EQ(dets[0].x2, 120.0f);
    EXPECT_FLOAT_EQ(dets[0].y2, 110.0f);
}

TEST(DecodeYolo, NormalizedScalesToInput) {
    const int channels = 5, A = 1;  // nc = 1
    // Нормализованные координаты: центр кадра, половина размера.
    std::vector<std::vector<float>> anchors = {{0.5f, 0.5f, 0.5f, 0.5f, 0.8f}};
    auto buf = pack_channels_first(anchors, channels);

    YoloPostOptions opt;
    opt.normalized = true;

    auto dets = decode_yolov8(buf.data(), channels, A, true, 640, 480, opt);
    ASSERT_EQ(dets.size(), 1u);
    // cx=320, cy=240, w=320, h=240 -> [160,120,480,360]
    EXPECT_FLOAT_EQ(dets[0].x1, 160.0f);
    EXPECT_FLOAT_EQ(dets[0].y1, 120.0f);
    EXPECT_FLOAT_EQ(dets[0].x2, 480.0f);
    EXPECT_FLOAT_EQ(dets[0].y2, 360.0f);
}

TEST(DecodeYolo, BelowThresholdDropped) {
    const int channels = 5, A = 2;
    std::vector<std::vector<float>> anchors = {
        {0.5f, 0.5f, 0.2f, 0.2f, 0.10f},  // ниже порога
        {0.5f, 0.5f, 0.2f, 0.2f, 0.50f},  // выше порога
    };
    auto buf = pack_channels_first(anchors, channels);

    YoloPostOptions opt;
    opt.normalized = true;
    opt.conf_thresh = 0.25f;

    auto dets = decode_yolov8(buf.data(), channels, A, true, 100, 100, opt);
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_FLOAT_EQ(dets[0].score, 0.50f);
}

TEST(DecodeYolo, NmsSuppressesOverlapSameClass) {
    const int channels = 5, A = 2;
    // Два почти совпадающих бокса одного класса -> NMS оставит 1.
    std::vector<std::vector<float>> anchors = {
        {50, 50, 40, 40, 0.9f},
        {51, 51, 40, 40, 0.8f},
    };
    auto buf = pack_channels_first(anchors, channels);

    YoloPostOptions opt;
    opt.normalized = false;
    opt.iou_thresh = 0.45f;

    auto dets = decode_yolov8(buf.data(), channels, A, true, 200, 200, opt);
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_FLOAT_EQ(dets[0].score, 0.9f);  // оставлен бокс с большим score
}

TEST(DecodeYolo, NmsKeepsDifferentClasses) {
    const int channels = 6, A = 2;  // nc = 2
    // Совпадающие боксы, но разных классов -> оба остаются (per-class NMS).
    std::vector<std::vector<float>> anchors = {
        {50, 50, 40, 40, 0.9f, 0.1f},  // класс 0
        {50, 50, 40, 40, 0.1f, 0.8f},  // класс 1
    };
    auto buf = pack_channels_first(anchors, channels);

    YoloPostOptions opt;
    opt.normalized = false;
    opt.iou_thresh = 0.45f;

    auto dets = decode_yolov8(buf.data(), channels, A, true, 200, 200, opt);
    EXPECT_EQ(dets.size(), 2u);
}

TEST(DecodeYolo, EmptyOnDegenerateShape) {
    std::vector<float> buf(10, 0.0f);
    YoloPostOptions opt;
    // channels = 4 -> nc = 0, должно вернуть пусто.
    EXPECT_TRUE(decode_yolov8(buf.data(), 4, 1, true, 100, 100, opt).empty());
    // num_anchors = 0.
    EXPECT_TRUE(decode_yolov8(buf.data(), 6, 0, true, 100, 100, opt).empty());
}

// ---- scale_to_original ----------------------------------------------------

TEST(ScaleToOriginal, UndoesLetterboxPadding) {
    // Оригинал 320x160 в letterbox 640x640: r = min(640/320, 640/160) = 2,
    // new = 640x320, паддинг сверху/снизу dy = (640-320)/2 = 160, dx = 0.
    std::vector<Detection> dets(1);
    dets[0].x1 = 0;    dets[0].y1 = 160;   // соответствует (0,0) оригинала
    dets[0].x2 = 640;  dets[0].y2 = 480;   // соответствует (320,160) оригинала

    scale_to_original(dets, 640, 640, 320, 160);

    EXPECT_FLOAT_EQ(dets[0].x1, 0.0f);
    EXPECT_FLOAT_EQ(dets[0].y1, 0.0f);
    EXPECT_FLOAT_EQ(dets[0].x2, 320.0f);
    EXPECT_FLOAT_EQ(dets[0].y2, 160.0f);
}

TEST(ScaleToOriginal, ClipsToImageBounds) {
    std::vector<Detection> dets(1);
    dets[0].x1 = -50;  dets[0].y1 = -50;
    dets[0].x2 = 9999; dets[0].y2 = 9999;

    scale_to_original(dets, 640, 640, 320, 160);

    EXPECT_GE(dets[0].x1, 0.0f);
    EXPECT_GE(dets[0].y1, 0.0f);
    EXPECT_LE(dets[0].x2, 320.0f);
    EXPECT_LE(dets[0].y2, 160.0f);
}

TEST(ScaleToOriginal, NoOpForInvalidOriginalSize) {
    std::vector<Detection> dets(1);
    dets[0].x1 = 1; dets[0].y1 = 2; dets[0].x2 = 3; dets[0].y2 = 4;
    scale_to_original(dets, 640, 640, 0, 0);  // невалидный размер -> no-op
    EXPECT_FLOAT_EQ(dets[0].x1, 1.0f);
    EXPECT_FLOAT_EQ(dets[0].x2, 3.0f);
}

// ---- class names ----------------------------------------------------------

TEST(ClassNames, CustomTableAndReset) {
    set_class_names({"cat", "dog"});
    EXPECT_EQ(yolo_class_name(0), "cat");
    EXPECT_EQ(yolo_class_name(1), "dog");
    EXPECT_EQ(yolo_class_name(99), "");  // вне диапазона
    EXPECT_EQ(yolo_class_name(-1), "");

    set_class_names({});  // пусто -> восстановление COCO
    EXPECT_EQ(yolo_class_name(0), "person");
}
