// Постобработка выхода YOLOv8 (декодирование боксов + NMS).
//
// Сознательно отделено от ii.cpp — основной раннер остаётся
// «универсальным движком инференса», а YOLO-специфика подключается
// сверху через флаг --yolo.
//
// Поддерживается стандартный head YOLOv8:
//   * выход формы [1, 4 + nc, A]   (channels-first, классический Ultralytics)
//   * выход формы [1, A, 4 + nc]   (channels-last, появляется при некоторых
//     TFLite-экспортах после транспонирования)
//   где A — число anchor’ов (8400 для 640x640), nc — число классов.
//
// Декодеру всё равно, в каком виде пришли координаты:
//   * normalized = true  — xy/wh лежат в [0..1] относительно входа модели
//     (Ultralytics по умолчанию для TFLite экспорта),
//   * normalized = false — xy/wh уже в пикселях входа модели.
//
// На выходе — список Detection в координатах исходного изображения
// (после обратного letterbox-маппинга), готовых к отрисовке.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Detection {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;  // в пикселях изображения
    int   class_id = 0;
    float score = 0.0f;
};

struct YoloPostOptions {
    float conf_thresh = 0.25f;   // порог уверенности класса
    float iou_thresh  = 0.45f;   // порог IoU для NMS
    int   max_dets    = 300;     // ограничение результата (после NMS)
    bool  normalized  = true;    // координаты модели в [0..1] vs пикселях
};

// Декодирует выход YOLOv8 в детекции в координатах входа модели
// (input_w x input_h, т.е. в letterbox-пространстве).
//
// data         — деквантованный float-буфер выходного тензора.
// channels     — 4 + nc (например 84 для COCO).
// num_anchors  — A (например 8400).
// channels_first — true: layout [C, A]; false: [A, C].
std::vector<Detection> decode_yolov8(const float* data,
                                     int channels,
                                     int num_anchors,
                                     bool channels_first,
                                     int input_w,
                                     int input_h,
                                     const YoloPostOptions& opt);

// Преобразует координаты детекций из letterbox-пространства
// (input_w x input_h) в координаты исходного изображения (orig_w x orig_h).
// Используется когда на экране показывается оригинал, а не препроцесс.
void scale_to_original(std::vector<Detection>& dets,
                       int input_w, int input_h,
                       int orig_w,  int orig_h);

// Имя класса COCO (0..79). Возвращает "" если индекс вне диапазона
// или если пользователь не подменил таблицу через set_class_names().
const std::string& yolo_class_name(int idx);

// Подменить таблицу имён классов (например, прочитать из файла,
// одно имя на строку). Если массив пустой — восстановит COCO.
void set_class_names(const std::vector<std::string>& names);

// Прочитать имена классов из текстового файла (по строке на класс).
// Возвращает false при ошибке открытия. Пустые строки игнорируются.
bool load_class_names(const std::string& path);
