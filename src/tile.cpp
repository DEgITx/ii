// Реализация tile.h: планирование сетки, извлечение тайла с replicate-edge
// и сборка результата в canvas (с опциональным линейным feathering’ом).
//
// Никаких бэкенд-специфичных хедеров — модуль чисто «пиксельный».

#include "tile.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tile {

namespace {

// Подбор позиций тайлов вдоль одной оси. Возвращает массив x0:
//   * если src <= tile — одна позиция (0); extract_tile сделает паддинг;
//   * иначе шагаем (tile - overlap) пикселей, пока полный тайл влезает,
//     последний тайл прижимаем к правому краю — даже если он
//     перекрывается сильнее, чем overlap, на стыке это даст разумный
//     результат (в blend-режиме веса плавно сольются, в overwrite —
//     просто перекроется).
void plan_axis(int src, int tile, int stride, std::vector<int>& positions) {
    positions.clear();
    if (src <= tile) {
        positions.push_back(0);
        return;
    }
    int x = 0;
    while (x + tile < src) {
        positions.push_back(x);
        x += stride;
    }
    int last = src - tile;
    // Если последняя посчитанная позиция и так совпадает с правым
    // краем — не дублируем; иначе добавляем «прижатую» позицию.
    if (positions.empty() || positions.back() != last) {
        positions.push_back(last);
    }
}

}  // namespace

TileLayout plan_tiles(int src_w, int src_h, int tile_w, int tile_h,
                      int overlap) {
    TileLayout l;
    l.src_w   = src_w;
    l.src_h   = src_h;
    l.tile_w  = tile_w;
    l.tile_h  = tile_h;
    l.overlap = overlap;

    // Stride < 1 — патологический случай (overlap >= tile). Зажимаем в 1,
    // чтобы не зациклиться в plan_axis; вызывающая сторона должна
    // проверить разумность overlap’а заранее.
    int sx = tile_w - overlap;
    int sy = tile_h - overlap;
    if (sx < 1) sx = 1;
    if (sy < 1) sy = 1;

    plan_axis(src_w, tile_w, sx, l.x0);
    plan_axis(src_h, tile_h, sy, l.y0);
    return l;
}

void extract_tile(const uint8_t* src_rgb, int src_w, int src_h,
                  int x0, int y0, int tile_w, int tile_h,
                  std::vector<uint8_t>& dst) {
    dst.resize((size_t)tile_w * tile_h * 3);

    // Быстрый путь: тайл целиком внутри source — копируем построчно
    // через memcpy. Это самый частый случай (внутренние тайлы) и он
    // должен быть максимально дешёвым: для 96×96 одна полная сетка
    // 1920×1080 — это ~250 тайлов, и аллокаций / клампов мы себе
    // позволить не можем.
    if (x0 >= 0 && y0 >= 0
        && x0 + tile_w <= src_w && y0 + tile_h <= src_h) {
        const size_t row_bytes = (size_t)tile_w * 3;
        for (int y = 0; y < tile_h; ++y) {
            const uint8_t* sp =
                &src_rgb[(size_t)((y0 + y) * src_w + x0) * 3];
            uint8_t* dp = &dst[(size_t)y * row_bytes];
            std::memcpy(dp, sp, row_bytes);
        }
        return;
    }

    // Медленный путь: тайл частично за границей. Replicate-edge —
    // ближайший пиксель в исходнике. Для рандомных тайлов вдоль края
    // это даёт визуально нейтральное продолжение (без чёрных полос).
    for (int y = 0; y < tile_h; ++y) {
        int sy = y0 + y;
        if (sy < 0) sy = 0;
        else if (sy >= src_h) sy = src_h - 1;
        for (int x = 0; x < tile_w; ++x) {
            int sx = x0 + x;
            if (sx < 0) sx = 0;
            else if (sx >= src_w) sx = src_w - 1;
            const uint8_t* sp = &src_rgb[(size_t)(sy * src_w + sx) * 3];
            uint8_t* dp = &dst[(size_t)(y * tile_w + x) * 3];
            dp[0] = sp[0];
            dp[1] = sp[1];
            dp[2] = sp[2];
        }
    }
}

void TileCanvas::reset(int w, int h, int overlap_out_, bool blend_) {
    width       = w;
    height      = h;
    overlap_out = overlap_out_;
    blend       = blend_;

    const size_t pixels    = (size_t)w * h;
    const size_t rgb_bytes = pixels * 3;
    rgb.assign(rgb_bytes, 0);
    if (blend) {
        acc.assign(rgb_bytes, 0.0f);
        weight.assign(pixels, 0.0f);
    } else {
        acc.clear();
        weight.clear();
    }
}

void TileCanvas::paste(const imgproc::OutputImage& tile, int dst_x, int dst_y) {
    if (tile.rgb.empty() || tile.width <= 0 || tile.height <= 0) return;
    if (width <= 0 || height <= 0) return;

    if (!blend) {
        // Простой overwrite + клиппинг в границы canvas. Цикл по
        // валидному диапазону — без проверок на каждый пиксель.
        const int y_beg = std::max(0, -dst_y);
        const int y_end = std::min(tile.height, height - dst_y);
        const int x_beg = std::max(0, -dst_x);
        const int x_end = std::min(tile.width,  width  - dst_x);
        if (y_beg >= y_end || x_beg >= x_end) return;
        const size_t row_bytes = (size_t)(x_end - x_beg) * 3;
        for (int y = y_beg; y < y_end; ++y) {
            const uint8_t* sp =
                &tile.rgb[(size_t)(y * tile.width + x_beg) * 3];
            uint8_t* dp =
                &rgb[(size_t)((dst_y + y) * width + (dst_x + x_beg)) * 3];
            std::memcpy(dp, sp, row_bytes);
        }
        return;
    }

    // Blend-режим. Весовая функция = произведение линейных ramp’ов
    // вдоль каждой оси: вес пикселя растёт от 0 на самом краю тайла
    // до 1 на расстоянии overlap_out пикселей внутрь, и так же спадает
    // у противоположного края. Это даёт мягкие швы шириной overlap_out
    // даже без хитрых cosine-весов. Если overlap_out <= 0, ramp
    // вырождается в постоянную 1 — blend становится «среднее
    // арифметическое в зоне пересечения», что обычно тоже OK.
    auto axis_w = [](int idx, int dim, int ov) -> float {
        if (ov <= 0) return 1.0f;
        // расстояние до ближайшего края тайла (в пикселях)
        const float d_left  = (float)(idx + 1);
        const float d_right = (float)(dim - idx);
        const float d       = std::min(d_left, d_right);
        const float w       = d / (float)ov;
        if (w < 0.0f) return 0.0f;
        if (w > 1.0f) return 1.0f;
        return w;
    };

    const int y_beg = std::max(0, -dst_y);
    const int y_end = std::min(tile.height, height - dst_y);
    const int x_beg = std::max(0, -dst_x);
    const int x_end = std::min(tile.width,  width  - dst_x);
    if (y_beg >= y_end || x_beg >= x_end) return;

    for (int y = y_beg; y < y_end; ++y) {
        const float wy = axis_w(y, tile.height, overlap_out);
        const uint8_t* sp_row =
            &tile.rgb[(size_t)(y * tile.width) * 3];
        for (int x = x_beg; x < x_end; ++x) {
            const float wx = axis_w(x, tile.width, overlap_out);
            const float w  = wx * wy;
            if (w <= 0.0f) continue;
            const uint8_t* sp = &sp_row[(size_t)x * 3];
            const size_t  k   = (size_t)((dst_y + y) * width + (dst_x + x));
            acc[k * 3 + 0] += w * (float)sp[0];
            acc[k * 3 + 1] += w * (float)sp[1];
            acc[k * 3 + 2] += w * (float)sp[2];
            weight[k]      += w;
        }
    }
}

void TileCanvas::finalize() {
    if (!blend) return;
    const size_t pixels = (size_t)width * height;
    for (size_t k = 0; k < pixels; ++k) {
        const float w = weight[k];
        if (w <= 0.0f) continue;  // пиксель не покрыт ни одним тайлом
        const float inv = 1.0f / w;
        for (int c = 0; c < 3; ++c) {
            float v = acc[k * 3 + c] * inv;
            if (v < 0.0f)   v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            rgb[k * 3 + c] = (uint8_t)std::lround(v);
        }
    }
}

}  // namespace tile
