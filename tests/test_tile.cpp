// Юнит-тесты тайл-арифметики (tile.*): планирование сетки (plan_tiles),
// извлечение тайла с replicate-edge (extract_tile) и сборка результата
// в canvas — over-composite с линейной растушёвкой по ведущим краям
// (TileCanvas).
//
// Модуль чисто «пиксельный», backend SDK не требуется.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "tile.h"

using ii::plan_tiles;
using ii::extract_tile;
using ii::TileLayout;
using ii::TileCanvas;
using ii::OutputImage;

// ---- plan_tiles -----------------------------------------------------------

TEST(PlanTiles, SourceSmallerThanTile) {
    // src <= tile по обеим осям → ровно одна позиция (0,0).
    TileLayout l = plan_tiles(/*src_w=*/50, /*src_h=*/40,
                              /*tile_w=*/96, /*tile_h=*/96, /*overlap=*/0);
    EXPECT_EQ(l.nx(), 1);
    EXPECT_EQ(l.ny(), 1);
    EXPECT_EQ(l.count(), 1);
    EXPECT_EQ(l.x0[0], 0);
    EXPECT_EQ(l.y0[0], 0);
}

TEST(PlanTiles, SourceEqualToTile) {
    // src == tile → тоже одна позиция (ветка src <= tile).
    TileLayout l = plan_tiles(96, 96, 96, 96, 0);
    EXPECT_EQ(l.nx(), 1);
    EXPECT_EQ(l.ny(), 1);
}

TEST(PlanTiles, NoOverlapExactMultiple) {
    // 200 / 100 без overlap: stride=100. Позиции 0 и 100 (последняя
    // прижата к краю, но совпадает с шагом — без дубля).
    TileLayout l = plan_tiles(200, 100, 100, 100, 0);
    ASSERT_EQ(l.nx(), 2);
    EXPECT_EQ(l.x0[0], 0);
    EXPECT_EQ(l.x0[1], 100);
    EXPECT_EQ(l.ny(), 1);  // 100x100 источник по высоте == tile
}

TEST(PlanTiles, SnapLastTileToEdge) {
    // 250 / 100 без overlap: 0, 100, затем остаток 50 — последний тайл
    // прижимается к правому краю на x=150 (покрывает 150..250).
    TileLayout l = plan_tiles(250, 100, 100, 100, 0);
    ASSERT_EQ(l.nx(), 3);
    EXPECT_EQ(l.x0[0], 0);
    EXPECT_EQ(l.x0[1], 100);
    EXPECT_EQ(l.x0[2], 150);  // src_w - tile_w
}

TEST(PlanTiles, WithOverlapStride) {
    // overlap=20 → stride=80. 180 / 100: x=0 (0+100<180), затем 80
    // (80+100=180 не < 180 — стоп), последний прижат к краю = 80.
    TileLayout l = plan_tiles(180, 100, 100, 100, 20);
    ASSERT_EQ(l.nx(), 2);
    EXPECT_EQ(l.x0[0], 0);
    EXPECT_EQ(l.x0[1], 80);  // src_w - tile_w, совпало со stride
}

TEST(PlanTiles, OverlapGreaterEqualTileClampsStride) {
    // Патология: overlap >= tile. stride зажимается в 1, не зацикливаемся.
    TileLayout l = plan_tiles(105, 10, 100, 100, /*overlap=*/200);
    // По X src(105) > tile(100): шагаем по 1 от 0, пока 0+100<105 (x<5),
    // т.е. 0..4, затем прижатый край 5 → 6 позиций.
    EXPECT_EQ(l.nx(), 6);
    EXPECT_EQ(l.x0.front(), 0);
    EXPECT_EQ(l.x0.back(), 5);  // src_w - tile_w
    EXPECT_EQ(l.ny(), 1);       // src_h(10) <= tile_h(100)
}

// ---- extract_tile ---------------------------------------------------------

TEST(ExtractTile, FullyInsideCopiesExact) {
    // 3x2 RGB источник, вырезаем 2x2 из (1,0).
    // Пиксели = (10*idx) для наглядности.
    std::vector<std::uint8_t> src(3 * 2 * 3);
    for (int i = 0; i < 3 * 2; ++i) {
        src[i * 3 + 0] = (std::uint8_t)(i * 10);
        src[i * 3 + 1] = (std::uint8_t)(i * 10 + 1);
        src[i * 3 + 2] = (std::uint8_t)(i * 10 + 2);
    }
    std::vector<std::uint8_t> dst;
    extract_tile(src.data(), 3, 2, /*x0=*/1, /*y0=*/0, 2, 2, dst);
    ASSERT_EQ(dst.size(), 2u * 2u * 3u);
    // Верхне-левый пиксель тайла = src(1,0) = индекс 1.
    EXPECT_EQ(dst[0], 10);
    EXPECT_EQ(dst[1], 11);
    EXPECT_EQ(dst[2], 12);
    // Верхне-правый = src(2,0) = индекс 2.
    EXPECT_EQ(dst[3], 20);
    // Нижне-левый = src(1,1) = индекс 4.
    EXPECT_EQ(dst[6], 40);
}

TEST(ExtractTile, ReplicateEdgeBeyondBounds) {
    // Источник 2x2, тайл 3x3 из (0,0): правый/нижний край повторяется.
    std::vector<std::uint8_t> src = {
        1, 1, 1,   2, 2, 2,    // строка 0
        3, 3, 3,   4, 4, 4,    // строка 1
    };
    std::vector<std::uint8_t> dst;
    extract_tile(src.data(), 2, 2, 0, 0, 3, 3, dst);
    ASSERT_EQ(dst.size(), 3u * 3u * 3u);
    auto px = [&](int x, int y) { return dst[(y * 3 + x) * 3]; };
    EXPECT_EQ(px(0, 0), 1);
    EXPECT_EQ(px(1, 0), 2);
    EXPECT_EQ(px(2, 0), 2);  // x=2 за краем → повтор x=1
    EXPECT_EQ(px(0, 2), 3);  // y=2 за краем → повтор y=1
    EXPECT_EQ(px(2, 2), 4);  // оба за краем → правый-нижний
}

TEST(ExtractTile, SingleChannelGray) {
    // 3x2 grayscale (1 канал) источник, вырезаем 2x2 из (1,0).
    std::vector<std::uint8_t> src = {
        10, 20, 30,   // строка 0
        40, 50, 60,   // строка 1
    };
    std::vector<std::uint8_t> dst;
    extract_tile(src.data(), 3, 2, /*x0=*/1, /*y0=*/0, 2, 2, dst,
                 /*channels=*/1);
    ASSERT_EQ(dst.size(), 2u * 2u * 1u);
    EXPECT_EQ(dst[0], 20);  // src(1,0)
    EXPECT_EQ(dst[1], 30);  // src(2,0)
    EXPECT_EQ(dst[2], 50);  // src(1,1)
    EXPECT_EQ(dst[3], 60);  // src(2,1)
}

TEST(ExtractTile, SingleChannelReplicateEdge) {
    // 2x2 gray, тайл 3x3 из (0,0): правый/нижний край повторяется.
    std::vector<std::uint8_t> src = {1, 2,
                                     3, 4};
    std::vector<std::uint8_t> dst;
    extract_tile(src.data(), 2, 2, 0, 0, 3, 3, dst, /*channels=*/1);
    ASSERT_EQ(dst.size(), 3u * 3u * 1u);
    auto px = [&](int x, int y) { return dst[y * 3 + x]; };
    EXPECT_EQ(px(2, 0), 2);  // за правым краем → повтор x=1
    EXPECT_EQ(px(0, 2), 3);  // за нижним краем → повтор y=1
    EXPECT_EQ(px(2, 2), 4);  // оба за краем
}

// ---- TileCanvas: composite (overwrite, ramp=0) ----------------------------

TEST(TileCanvas, OverwritePaste) {
    // ramp_x=ramp_y=0 → α=1 везде → простой overwrite.
    TileCanvas c;
    c.reset(/*w=*/4, /*h=*/4);

    OutputImage t;
    t.width = 2;
    t.height = 2;
    t.channels_src = 3;
    t.rgb.assign(2 * 2 * 3, 77);

    c.composite(t, /*dst_x=*/1, /*dst_y=*/1, /*ramp_x=*/0, /*ramp_y=*/0);

    ASSERT_EQ(c.rgb.size(), 4u * 4u * 3u);
    // Внутри пасты (1,1) — значение тайла.
    EXPECT_EQ(c.rgb[(1 * 4 + 1) * 3], 77);
    EXPECT_EQ(c.rgb[(2 * 4 + 2) * 3], 77);
}

TEST(TileCanvas, OverwriteClipsAtCanvasEdge) {
    // Тайл частично за правым/нижним краем canvas — лишнее клиппится,
    // без выхода за буфер.
    TileCanvas c;
    c.reset(3, 3);
    OutputImage t;
    t.width = 2;
    t.height = 2;
    t.channels_src = 3;
    t.rgb.assign(2 * 2 * 3, 200);
    c.composite(t, /*dst_x=*/2, /*dst_y=*/2, 0, 0);  // влезает только (2,2)
    EXPECT_EQ(c.rgb[(2 * 3 + 2) * 3], 200);
}

// ---- TileCanvas: composite (feathering по ведущим краям) ------------------

TEST(TileCanvas, CompositeNoRampIsOverwrite) {
    // ramp=0 → тайл кладётся как есть, без подмешивания старого canvas’а.
    TileCanvas c;
    c.reset(2, 2);
    OutputImage t;
    t.width = 2;
    t.height = 2;
    t.channels_src = 3;
    t.rgb.assign(2 * 2 * 3, 100);
    c.composite(t, 0, 0, 0, 0);
    for (auto v : c.rgb) EXPECT_EQ(v, 100);
}

TEST(TileCanvas, CompositeEqualTilesPreserveValueInSeam) {
    // Два горизонтальных соседа одинакового значения с перекрытием:
    // cross-fade двух равных значений = то же значение (без осветления/
    // затемнения шва). Левый тайл непрозрачен (первый столбец, ramp_x=0),
    // правый ramp’ится в перекрытии.
    TileCanvas c;
    c.reset(3, 2);
    OutputImage a, b;
    a.width = b.width = 2;
    a.height = b.height = 2;
    a.channels_src = b.channels_src = 3;
    a.rgb.assign(2 * 2 * 3, 150);
    b.rgb.assign(2 * 2 * 3, 150);
    c.composite(a, 0, 0, /*ramp_x=*/0, /*ramp_y=*/0);  // первый столбец
    c.composite(b, 1, 0, /*ramp_x=*/1, /*ramp_y=*/0);  // x=1 перекрыт
    for (auto v : c.rgb) EXPECT_EQ(v, 150);
}

TEST(TileCanvas, CompositeBlendsTowardNewTileInSeam) {
    // Перекрытие тайлов с разными значениями: в зоне шва результат лежит
    // строго между старым и новым значением (плавный переход, без скачка
    // к чистому новому или чистому старому на кромке).
    TileCanvas c;
    c.reset(4, 1);
    OutputImage a, b;
    a.width = b.width = 3;
    a.height = b.height = 1;
    a.channels_src = b.channels_src = 3;
    a.rgb.assign(3 * 1 * 3, 0);    // левый тайл — чёрный
    b.rgb.assign(3 * 1 * 3, 240);  // правый тайл — светлый
    c.composite(a, 0, 0, /*ramp_x=*/0, 0);           // x: 0,1,2 ← 0
    c.composite(b, 1, 0, /*ramp_x=*/2, 0);           // x: 1,2,3, ramp на 1,2
    // x=0 — только a → 0.
    EXPECT_EQ(c.rgb[(0) * 3], 0);
    // x=1,2 — зона шва: между 0 и 240, монотонно растёт слева направо.
    const int s1 = c.rgb[(1) * 3];
    const int s2 = c.rgb[(2) * 3];
    EXPECT_GT(s1, 0);
    EXPECT_LT(s1, 240);
    EXPECT_GT(s2, s1);
    EXPECT_LT(s2, 240);
    // x=3 — вне ramp’а правого тайла (d>=ramp) → чистый b.
    EXPECT_EQ(c.rgb[(3) * 3], 240);
}

TEST(TileCanvas, CompositeReuseSameSize) {
    // Повторный reset того же размера и новый проход дают корректный
    // результат без «протёкших» значений предыдущего кадра.
    TileCanvas c;
    c.reset(2, 2);
    OutputImage t;
    t.width = 2;
    t.height = 2;
    t.channels_src = 3;
    t.rgb.assign(2 * 2 * 3, 60);
    c.composite(t, 0, 0, 0, 0);

    c.reset(2, 2);  // тот же размер — без realloc
    t.rgb.assign(2 * 2 * 3, 200);
    c.composite(t, 0, 0, 0, 0);
    for (auto v : c.rgb) EXPECT_EQ(v, 200);
}
