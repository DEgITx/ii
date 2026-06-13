// «Связующий» слой YOLO для раннера: автодетект YOLO-головы среди выходов
// модели и конвертация декодированных детекций в DisplayBox’ы (с цветом по
// классу) для отрисовки. Само декодирование боксов/NMS живёт в yolo.h.
//
// Вынесено из ii.cpp, чтобы YOLO-специфика не размазывалась по основному
// раннеру.

#pragma once

#include <vector>

#include "display.h"
#include "tensor_utils.h"
#include "yolo.h"

namespace ii {

// Описание найденной YOLO-головы: индекс выхода и его раскладка.
struct YoloHead {
    int  output_index   = -1;     // индекс в out_info
    int  channels       = 0;      // 4 + nc
    int  num_anchors    = 0;
    bool channels_first = true;   // true: [1, C, A]; false: [1, A, C]
};

// Авто-определение YOLO-выхода и его layout’а среди выходов модели.
// Ищем 3D-тензор вида [1, A, B] или [1, B, A], где меньший из {A, B}
// = 4 + nc (типично 84 для COCO), а больший — число anchor’ов.
// forced >= 0 — взять именно этот выход (с проверкой формы); -1 —
// автовыбор (ошибка, если кандидат не один). Результат в h.
bool detect_yolo_head(const std::vector<TensorInfo>& outs, int forced,
                      YoloHead& h);

// Конвертация детекций в DisplayBox’ы: подпись (имя класса или clsN),
// score и стабильный по классу цвет.
std::vector<DisplayBox> detections_to_boxes(const std::vector<Detection>& dets);

} // namespace ii
