// Юнит-тесты тайл-арифметики (tile.*): планирование сетки (plan_tiles),
// извлечение тайла с replicate-edge (extract_tile) и сборка результата
// в canvas — overwrite-paste и линейный feathering (TileCanvas).
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

// ---- TileCanvas: overwrite (blend=false) ----------------------------------

TEST(TileCanvas, OverwritePaste) {
    TileCanvas c;
    c.reset(/*w=*/4, /*h=*/4, /*overlap_out=*/0, /*blend=*/false);

    OutputImage t;
    t.width = 2;
    t.height = 2;
    t.channels_src = 3;
    t.rgb.assign(2 * 2 * 3, 77);

    c.paste(t, /*dst_x=*/1, /*dst_y=*/1);
    c.finalize();  // no-op в overwrite-режиме

    ASSERT_EQ(c.rgb.size(), 4u * 4u * 3u);
    // Внутри пасты (1,1) — значение тайла.
    EXPECT_EQ(c.rgb[(1 * 4 + 1) * 3], 77);
    EXPECT_EQ(c.rgb[(2 * 4 + 2) * 3], 77);
}

TEST(TileCanvas, OverwriteClipsAtCanvasEdge) {
    // Тайл частично за правым/нижним краем canvas — лишнее клиппится,
    // без выхода за буфер.
    TileCanvas c;
    c.reset(3, 3, 0, false);
    OutputImage t;
    t.width = 2;
    t.height = 2;
    t.channels_src = 3;
    t.rgb.assign(2 * 2 * 3, 200);
    c.paste(t, /*dst_x=*/2, /*dst_y=*/2);  // влезает только пиксель (2,2)
    c.finalize();
    EXPECT_EQ(c.rgb[(2 * 3 + 2) * 3], 200);
}

// ---- TileCanvas: blend (feathering) ---------------------------------------

TEST(TileCanvas, BlendSingleTileNoOverlapIsIdentity) {
    // overlap_out=0 → axis_w=1 везде, вес=1, finalize = acc/1 = тайл.
    TileCanvas c;
    c.reset(2, 2, /*overlap_out=*/0, /*blend=*/true);
    OutputImage t;
    t.width = 2;
    t.height = 2;
    t.channels_src = 3;
    t.rgb.assign(2 * 2 * 3, 100);
    c.paste(t, 0, 0);
    c.finalize();
    for (auto v : c.rgb) EXPECT_EQ(v, 100);
}

TEST(TileCanvas, BlendTwoTilesAverageInOverlap) {
    // Два тайла одинакового значения, перекрытие → взвешенное среднее
    // того же значения = само значение (ни осветления, ни затемнения).
    TileCanvas c;
    c.reset(3, 2, /*overlap_out=*/0, /*blend=*/true);
    OutputImage a, b;
    a.width = b.width = 2;
    a.height = b.height = 2;
    a.channels_src = b.channels_src = 3;
    a.rgb.assign(2 * 2 * 3, 150);
    b.rgb.assign(2 * 2 * 3, 150);
    c.paste(a, 0, 0);
    c.paste(b, 1, 0);  // колонка x=1 перекрыта обоими
    c.finalize();
    // Везде должно остаться 150 (среднее равных значений).
    for (auto v : c.rgb) EXPECT_EQ(v, 150);
}

TEST(TileCanvas, BlendCleanInvariantAllowsReuse) {
    // После finalize() буферы чисты (blend_clean) — повторный reset того
    // же размера и новый проход дают корректный результат.
    TileCanvas c;
    c.reset(2, 2, 0, true);
    OutputImage t;
    t.width = 2;
    t.height = 2;
    t.channels_src = 3;
    t.rgb.assign(2 * 2 * 3, 60);
    c.paste(t, 0, 0);
    c.finalize();

    c.reset(2, 2, 0, true);  // тот же размер — без полного memset
    t.rgb.assign(2 * 2 * 3, 200);
    c.paste(t, 0, 0);
    c.finalize();
    for (auto v : c.rgb) EXPECT_EQ(v, 200);  // нет «протёкших» 60
}
