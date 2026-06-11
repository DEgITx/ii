// Мелкие универсальные помощники над тензорами (inf::TensorDesc):
// арифметика по shape, печать сводки/головы выхода, деквантование
// произвольного выходного тензора в float.
//
// Вынесено из ii.cpp, чтобы основной раннер не тонул в утилитарном
// коде. Ничего бэкенд-специфичного здесь нет — только inference.h
// (типы тензоров) + стандартная библиотека.

#pragma once

#include <string>
#include <vector>

#include "inference.h"

namespace iirun {

// Историческое имя — алиас, чтобы не плодить inf::TensorDesc в каждой
// строчке. Всё остальное (DType, half_to_float, dtype_name, dtype_size)
// приходит из inference.h.
using TensorInfo = inf::TensorDesc;

// Число элементов по shape; 0 для динамического/нулевого размера
// (отрицательная/нулевая ось не поддерживается).
std::size_t numel(const std::vector<int>& shape);

// "[1,3,224,224]" — компактная строка из shape для логов/ошибок.
std::string shape_to_str(const std::vector<int>& shape);

// Число каналов «картиночного» входа модели по shape:
//   * 3 — обычная RGB-модель NHWC [1,H,W,3];
//   * 1 — grayscale/Y-модель NHWC [1,H,W,1] (FSRCNN-Y, denoise по
//         яркости и т.п.) — на вход подаём luma (BT.601) от RGB-кадра;
//   * 0 — не «картинка» в этом смысле, обычные пайплайны letterbox/
//         display/yolo неприменимы.
int image_input_channels(const std::vector<int>& s);

inline bool is_image_input(const std::vector<int>& s) {
    return image_input_channels(s) > 0;
}

// Раскладка «картиночного» входа: NHWC (типично для TFLite) или NCHW
// (типично для ONNX). None — вход не похож на изображение.
enum class ImageLayout { None, NHWC, NCHW };

// Распознать картиночный вход и его геометрию независимо от раскладки.
// Возвращает число каналов (1|3) или 0; при ненулевом результате
// заполняет layout/h/w. NHWC [1,H,W,1|3] предпочитается (обратная
// совместимость), иначе NCHW [1,1|3,H,W].
int image_input_info(const std::vector<int>& s, ImageLayout& layout,
                     int& h, int& w);

// Печать строки-сводки по тензору: имя/shape/dtype (+ quant, если есть).
void print_tensor(const char* prefix, const TensorInfo& t);

// Печать первых n_show элементов выхода в float (с деквантованием).
// data — Engine::output_data(i).
void print_output_head(const TensorInfo& info, const void* data,
                       int n_show = 10);

// Монотонные миллисекунды (steady_clock) для замеров.
double now_ms();

// Деквантование произвольного выходного тензора в float-буфер.
//
// Поддерживаемые типы:
//   * float32 / float16      — копируем (с конвертацией для half);
//   * int8 / uint8 / int16 / uint16 — аффинная формула (x - zp) * scale;
//   * int32 — то же (некоторые экспортёры так дают «головы» с большим
//     динамическим диапазоном).
//
// Если scale == 0 для целочисленного тензора (модель не квантована
// в полноценном смысле) — берём сырое значение как float, иначе деление
// на нулевую шкалу всё бы обнулило. На неизвестном dtype — warn-once
// в stderr (важно для видео-цикла) и возврат false.
bool dequantize_output(const TensorInfo& info, const void* data,
                       std::vector<float>& out);

}  // namespace iirun
