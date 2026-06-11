// Препроцессинг входа: загрузка изображения, letterbox-ресайз (стандарт
// YOLO), конвертация RGB → luma (BT.601) и заливка входного тензора
// модели с квантованием (идентично quantize_to_input).
//
// Этот модуль — единственный TU, который тянет stb_image / stb_image_resize2
// реализации (load_image / letterbox). Вынесено из ii.cpp, чтобы основной
// раннер не зависел от stb напрямую.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "inference.h"

namespace iirun {

// Загруженное изображение: RGB HWC, 3 канала, 0..255.
struct Image {
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
};

// Загрузка изображения (любой формат, поддерживаемый stb) в RGB888.
bool load_image(const std::string& path, Image& out);

// Ресайз с сохранением пропорций + паддинг до (target_h x target_w).
// Для YOLO стандарт — заполнение серым (114).
//
// Перегрузка с сырым указателем удобна для видео-источников (камера),
// чтобы не копировать кадр в промежуточный std::vector.
void letterbox(const uint8_t* src_rgb, int src_w, int src_h,
               int target_w, int target_h,
               std::vector<uint8_t>& dst, uint8_t pad = 114);

inline void letterbox(const Image& src, int target_w, int target_h,
                      std::vector<uint8_t>& dst, uint8_t pad = 114) {
    letterbox(src.rgb.data(), src.w, src.h, target_w, target_h, dst, pad);
}

// RGB HWC → grayscale HW (BT.601 luma). Интовая аппроксимация:
// Y = (77*R + 150*G + 29*B) / 256.
void rgb_to_gray(const uint8_t* rgb, std::size_t pixels,
                 std::vector<uint8_t>& gray);

// Заполнение входного тензора. На вход — RGB (или grayscale) HWC uint8
// [0..255]. Нормализуется в [0,1] и квантуется по scale/zero_point модели
// для целочисленных типов (идентично quantize_to_input).
//
// data — сырой указатель на память входа модели (Engine::input_data).
// Бэкенд гарантирует, что буфер достаточен для info.bytes. Для горячего
// пути (tile-режим) квантование идёт через 256-элементную LUT, кэшируемую
// на тред по ключу (dtype, scale, zp).
bool fill_input(const std::vector<uint8_t>& rgb, const inf::TensorDesc& info,
                void* data);

// Заливка входа в NCHW-раскладке (channel-planar) — для ONNX-моделей
// [1,C,H,W]. На вход тот же HWC-буфер, что и у fill_input (как из letterbox),
// плюс геометрия C/H/W; пиксели транспонируются HWC -> CHW. Нормализация и
// квантование идентичны fill_input.
bool fill_input_nchw(const std::vector<uint8_t>& rgb, const inf::TensorDesc& info,
                     void* data, int C, int H, int W);

}  // namespace iirun
