// Реализация tile.h: планирование сетки, извлечение тайла с replicate-edge
// и сборка результата в canvas (incremental over-composite с линейным
// feathering’ом по ведущим краям).
//
// Никаких бэкенд-специфичных хедеров — модуль чисто «пиксельный».

#include "tile.h"

#include <algorithm>
#include <cstring>

#include "parallel.h"

namespace ii {

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

void extract_tile(const uint8_t* src, int src_w, int src_h,
                  int x0, int y0, int tile_w, int tile_h,
                  std::vector<uint8_t>& dst, int channels) {
    const int ch = channels > 0 ? channels : 1;
    dst.resize((size_t)tile_w * tile_h * ch);

    // Быстрый путь: тайл целиком внутри source — копируем построчно
    // через memcpy. Это самый частый случай (внутренние тайлы) и он
    // должен быть максимально дешёвым: для 96×96 одна полная сетка
    // 1920×1080 — это ~250 тайлов, и аллокаций / клампов мы себе
    // позволить не можем.
    if (x0 >= 0 && y0 >= 0
        && x0 + tile_w <= src_w && y0 + tile_h <= src_h) {
        const size_t row_bytes = (size_t)tile_w * ch;
        for (int y = 0; y < tile_h; ++y) {
            const uint8_t* sp =
                &src[(size_t)((y0 + y) * src_w + x0) * ch];
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
            const uint8_t* sp = &src[(size_t)(sy * src_w + sx) * ch];
            uint8_t* dp = &dst[(size_t)(y * tile_w + x) * ch];
            for (int k = 0; k < ch; ++k) dp[k] = sp[k];
        }
    }
}

void TileCanvas::reset(int w, int h) {
    width  = w;
    height = h;
    // plan_tiles снэпает последний тайл к краю, так что canvas покрывается
    // тайлами целиком — каждый байт rgb гарантированно перезаписан
    // composite/decode. resize без zero-init: один memset при (ре)аллокации,
    // дальше — no-op (ii.cpp ping-pong’ает буфер с out_img того же размера).
    const size_t rgb_bytes = (size_t)w * h * 3;
    if (rgb.size() != rgb_bytes) rgb.resize(rgb_bytes);
}

namespace {

// α тайла (fixed-point 0..256) на смещении d от ведущего края при ширине
// растушёвки r: вне полосы (d >= r) — 256 (полностью непрозрачно), внутри —
// линейный рост, центрированный так, что концы не вырождаются в 0/256
// (нет «шва» из чистого фона/тайла на самой кромке).
inline int lead_alpha(int d, int r) {
    if (r <= 0 || d >= r) return 256;
    const int a = (2 * d + 1) * 128 / r;
    return a < 0 ? 0 : (a > 256 ? 256 : a);
}

}  // namespace

void TileCanvas::composite(const ii::OutputImage& tile, int dst_x, int dst_y,
                           int ramp_x, int ramp_y) {
    if (tile.rgb.empty() || tile.width <= 0 || tile.height <= 0) return;
    if (width <= 0 || height <= 0) return;

    // Клиппинг тайла в границы canvas — дальше цикл без проверок на пиксель.
    const int y_beg = std::max(0, -dst_y);
    const int y_end = std::min(tile.height, height - dst_y);
    const int x_beg = std::max(0, -dst_x);
    const int x_end = std::min(tile.width,  width  - dst_x);
    if (y_beg >= y_end || x_beg >= x_end) return;

    // Строки независимы (каждая читает свою строку canvas+тайла и пишет свою),
    // поэтому делим [y_beg, y_end) между ядрами. Зависимость «тайл N+1 читает
    // запись тайла N в зоне перекрытия» не нарушается: тайлы по-прежнему идут
    // по очереди, параллелится только нутро одного composite.
    const std::int64_t grain =
        std::max<std::int64_t>(1, 16384 / (width > 0 ? width : 1));
    ii::parallel_for(y_end - y_beg, grain, [&](std::int64_t rb, std::int64_t re) {
        for (int y = y_beg + (int)rb; y < y_beg + (int)re; ++y) {
            const int ay = lead_alpha(y, ramp_y);
            const uint8_t* sp =
                &tile.rgb[(size_t)(y * tile.width + x_beg) * 3];
            uint8_t* dp =
                &rgb[(size_t)((dst_y + y) * width + (dst_x + x_beg)) * 3];

            // Строка целиком непрозрачна (ни верхней, ни левой растушёвки) —
            // простой byte-копир, как в overwrite-режиме.
            if (ay == 256 && ramp_x <= 0) {
                std::memcpy(dp, sp, (size_t)(x_end - x_beg) * 3);
                continue;
            }
            for (int x = x_beg; x < x_end; ++x, sp += 3, dp += 3) {
                const int a = (lead_alpha(x, ramp_x) * ay) >> 8;  // 0..256
                if (a >= 256) {                                    // непрозрачно — копия
                    dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
                } else {
                    const int ia = 256 - a;
                    dp[0] = (uint8_t)((a * sp[0] + ia * dp[0] + 128) >> 8);
                    dp[1] = (uint8_t)((a * sp[1] + ia * dp[1] + 128) >> 8);
                    dp[2] = (uint8_t)((a * sp[2] + ia * dp[2] + 128) >> 8);
                }
            }
        }
    });
}

} // namespace ii
