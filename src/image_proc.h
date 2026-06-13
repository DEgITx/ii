// Декодирование выходного тензора image-to-image моделей в RGB-кадр.
//
// Сценарии использования:
//   * Super-resolution (FSRCNN, ESPCN, EDSR, ...) — модель отдаёт
//     апскейл-картинку NHWC [1,H_out,W_out,3] (или [1,H_out,W_out,1]
//     для Y-канала);
//   * Low-light enhancement / denoise (LiteEnhanceNet, DnCNN с
//     глобальным residual) — модель отдаёт уже скорректированную
//     картинку того же размера, что и вход NHWC [1,H,W,3];
//   * любой другой image-to-image NPU-пайплайн с похожим выходом.
//
// Сам раннер (ii.cpp) намеренно не знает деталей этого декодирования —
// вся «как из тензора получить пиксели» логика собрана здесь, чтобы
// добавление нового семейства моделей не раздувало ii.cpp.
//
// Алгоритм (повторяет dequantize_output + clamp/scale в uint8):
//   1. Проверяем, что shape тензора — NHWC [1,H,W,C], C ∈ {1, 3}.
//   2. Каждый элемент дéквантуем в float: для float32/float16 — as is,
//      для int8/uint8/int16/uint16 — по формуле (x - zp) * scale из
//      самой модели (TFLite кладёт это в TensorDesc::scale/zero_point).
//   3. Нормируем по `OutputRange` в [0..1]:
//        Unit   — модель уже отдаёт [0..1] (типичный image-to-image);
//        Signed — [-1..1] (нормализованный residual / output);
//        Byte   — [0..255] (модели, обученные на сыром uint8 без /255).
//   4. Клампим в [0..1] и умножаем на 255 → uint8 RGB HWC.
//   5. Grayscale (C=1) разворачиваем в R=G=B, чтобы Display::show_rgb
//      и stb_image_write принимали единый формат.
//
// Поведение по «странным» dtype: warn-once в stderr и возврат false —
// это критично для видео-цикла, где иначе stderr залило бы поверх
// FPS-лога на каждый кадр.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "inference.h"

namespace ii {

// Диапазон значений выхода модели до нормировки в uint8 [0..255].
// Подбирается одним флагом --output-range у ii.cpp.
enum class OutputRange {
    Unit,      // [0..1]   — default, подходит и для FSRCNN, и для LiteEnhanceNet
    Signed,    // [-1..1]  — нормализованный residual / некоторые tanh-выходы
    Byte,      // [0..255] — сети, обученные на сыром uint8 без /255
};

// Парсер строкового представления (для CLI). Возвращает true, если имя
// узнано; иначе *out не трогается и можно использовать дефолт.
bool parse_output_range(const std::string& name, OutputRange& out);

struct DecodeOptions {
    OutputRange range = OutputRange::Unit;
};

// Декодированный кадр-изображение. rgb всегда HWC RGB uint8 (3 канала),
// channels_src сообщает, сколько каналов было в исходном тензоре (1
// разворачивается в R=G=B).
struct OutputImage {
    int width = 0;
    int height = 0;
    int channels_src = 0;          // 1 или 3
    std::vector<uint8_t> rgb;      // HWC RGB, размер = w*h*3
};

// Кэш квантования для горячего декода. Для квантованных INT8/UInt8 выходов
// каждый возможный байт даёт ровно один результат — заранее посчитанная
// 256-байтная таблица «q → uint8 в [0..255]» полностью убирает std::lround,
// fmul, clamp из горячего цикла. Это критично в tile-режиме, где decode
// вызывается N раз за кадр (для 96×96 → 384×384 это 147 К пикселей × 24
// тайла = 3.5 М вызовов lround на кадр без кэша).
//
// Cache key = (dtype, scale, zp, range). Пересчёт срабатывает, только если
// что-то реально поменялось — в стабильном видео-цикле это сравнение
// четырёх полей за вызов decode.
struct DecodeCache {
    ii::DType  dtype = ii::DType::Unknown;
    float       scale = 0.0f;
    int         zero_point = 0;
    OutputRange range = OutputRange::Unit;
    bool        lut_valid = false;
    uint8_t     lut[256]{};  // используется только для Int8/UInt8
};

// Похож ли тензор на изображение: NHWC [1,H,W,1] или [1,H,W,3] с H,W >= 2.
bool is_image_output(const ii::TensorDesc& info);

// Автодетект единственного «image-shaped» выхода среди списка.
// Возвращает:
//   * индекс — если ровно один кандидат подходит под is_image_output;
//   * -1     — если кандидатов несколько (нужен явный --output-index)
//              или ни одного.
int detect_image_output_index(const std::vector<ii::TensorDesc>& outs);

// Декодировать тензор в RGB-кадр. Печатает причину отказа в stderr
// (один раз) для нештатных dtype/shape, что важно в видео-цикле.
//
// Если передан cache, для INT8/UInt8 выходов используется 256-байтная
// LUT (см. DecodeCache). nullptr — каждый вызов сам пересчитывает
// таблицу (для не-горячих путей это копейки, и старый код вызовов
// продолжает компилироваться без изменений).
bool decode_image_output(const ii::TensorDesc& info, const void* data,
                         const DecodeOptions& opt, OutputImage& out,
                         DecodeCache* cache = nullptr);

// Декодировать тензор напрямую в (под-)область внешнего RGB-буфера,
// без промежуточной аллокации. Используется tile-режимом: каждый
// тайл пишется сразу в нужную позицию canvas, что убирает один
// memcpy и один memset на тайл (~10 МБ DRAM-трафика на кадр для
// 1996×1332 × 24 тайла).
//
// dst_rgb       — указатель на начало буфера (0,0 верхний-левый);
// dst_stride    — байты между началами соседних строк (>= dst_w * 3);
// dst_x, dst_y  — позиция верхнего-левого угла декодированного тайла.
//
// Caller отвечает за то, что dst_x + W и dst_y + H влезают в буфер
// (W, H — размеры выхода тензора). Для tile.plan_tiles это гарантировано
// «snap-to-edge» логикой.
bool decode_image_output_to(const ii::TensorDesc& info, const void* data,
                            const DecodeOptions& opt,
                            uint8_t* dst_rgb, std::size_t dst_stride,
                            int dst_x, int dst_y,
                            DecodeCache* cache = nullptr);

// Сохранить кадр в PNG через stb_image_write. Возвращает false и
// печатает причину в stderr при ошибке записи.
bool save_png(const OutputImage& img, const std::string& path);

} // namespace ii
