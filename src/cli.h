// Разбор командной строки раннера: структура Args со всеми режимами/
// флагами, печать --help и parse_args. Вынесено из ii.cpp — основной
// файл оставляет себе только main() и логику прогона.

#pragma once

#include <cstdint>
#include <string>

#include "inference.h"

namespace iirun {

struct Args {
    std::string model;
    std::string image;
    // Имя бэкенда инференса: "ii" / "tflite" / "tensorrt" / "directml".
    // Пустая строка (дефолт) -> авто: TFLite, если собран, иначе встроенный
    // движок `ii`. См. inf::make_engine.
    std::string backend;
    std::string delegate = inf::default_delegate_path();
    bool no_delegate = false;
    bool benchmark = false;
    int runs = 100;
    int warmup = 3;
    int threads = 0;
    bool display = false;       // открыть окно Wayland/EGL
    bool show_input = false;    // выводить letterbox-вход вместо оригинала
    int  win_w = 960;
    int  win_h = 720;
    bool loop = false;          // непрерывный цикл инференса (демо для видео)
    bool stats = false;         // FPS/jitter оверлей + лог в stdout
    int  log_interval_ms = 1000;
    bool vsync = true;          // ограничение FPS частотой обновления экрана
    // ---- YOLO-постобработка ----
    bool        yolo = false;        // включить декодирование боксов
    float       conf = 0.25f;
    float       iou  = 0.45f;
    int         max_dets = 300;
    bool        yolo_pixel_coords = false;  // выход уже в пикселях (не 0..1)
    int         yolo_output = -1;    // -1 = автовыбор (самый «толстый» выход)
    std::string classes_path;        // файл с именами классов (по строке)
    // ---- Проверка точности относительно эталона ----
    bool        compare_cpu = false; // ту же модель прогнать на CPU и сравнить
    std::string compare_model;       // путь к эталонной .tflite (CPU)
    // ---- Декодирование выхода как изображения (FSRCNN / enhance / denoise) ----
    // Сценарий: модель делает image-to-image — апскейл (FSRCNN), осветление
    // (LiteEnhanceNet) или фильтр шума (DnCNN). Выход имеет shape
    // NHWC [1,H_out,W_out,1|3] и значения в диапазоне (см. output_range).
    // С --show-output показываем декодированный кадр модели вместо
    // оригинала / препроцесса; алгоритм декодирования живёт в image_proc.h,
    // ii.cpp лишь вызывает его и подсовывает буфер в Display::show_rgb.
    // Несовместимо с --yolo (там выход — это головы детекций, а не картинка).
    bool         show_output  = false;
    int          output_index = -1;             // -1 = автодетект единственного
    std::string  output_range_str = "unit";     // unit | signed | byte
    std::string  save_output_path;              // PNG-путь для одиночного прогона
    // ---- Tiling-режим (sliding window) для image-to-image моделей --------
    // Когда модель имеет маленькое окно входа (FSRCNN 96×96 и т.п.), вместо
    // того чтобы letterbox’ить весь кадр в 96×96 (и терять детали), мы
    // пробегаем исходным разрешением сеткой тайлов размером с окно модели,
    // прогоняем каждый тайл независимо и сшиваем выходы в полноразмерный
    // canvas. На дисплее и в FPS-счётчике «один полный кадр = 1 такт» —
    // т.е. фреймрейт показывает реальную пропускную способность на
    // исходном разрешении (а не на сжатом до 96×96).
    //
    // Требования: image-shaped выход (NHWC [1,Hout,Wout,1|3]) с целочисленным
    // масштабом относительно входа (для SR ×2/×3/×4, для denoise/enhance ×1).
    // Несовместимо с --yolo, --compare*, --random-input (нужен реальный
    // источник кадров: image либо --camera).
    bool         tile_mode    = false;
    // Перекрытие тайлов в пикселях input-space. 0 — стык в стык
    // (быстро, могут быть видны швы у SR-моделей). 4..16 — линейное
    // feathering в зоне overlap, швы становятся незаметны ценой
    // ~16 байт на пиксель canvas’а (float-аккумуляторы).
    int          tile_overlap = 0;
    // ---- Захват с камеры (V4L2) ----
    // Когда задан --camera, источник кадров — V4L2-устройство, а не
    // картинка/random. Включает свой собственный цикл инференса (с
    // --display и без), несовместимый с --benchmark/--loop/--compare*.
    // image и --random-input при заданной камере игнорируются.
    std::string camera_device;        // путь к /dev/videoN; "" = камера выкл.
    int         camera_w   = 640;     // желаемое разрешение, драйвер может
    int         camera_h   = 480;     // выбрать ближайшее (см. лог Camera)
    int         camera_fps = 30;      // желаемая частота кадров (best effort)
    // ---- Случайный вход вместо изображения ----
    // Когда включено, входной тензор главной (и при сравнении — эталонной)
    // модели заполняется случайным uint8-буфером. Картинка не нужна —
    // позиционный аргумент image можно опустить. Применимо ко всем режимам:
    //   * одиночный инференс / --benchmark / --loop — фаззер для замера
    //     скорости и стабильности на произвольных данных;
    //   * --compare / --compare-cpu — фаззер расхождений NPU vs эталон без
    //     зависимости от датасета (оба входа заполняются одним буфером,
    //     метрики усредняются по random_runs прогонам).
    // Старое имя флага --compare-random остаётся как алиас.
    bool        random_input = false;
    int         random_runs  = 1;     // используется в режиме сравнения
    uint32_t    random_seed  = 0;     // 0 = взять из std::random_device
    // ---- Экспорт замеров в CSV ----
    // Префикс пути для CSV-файлов с замерами. Если задан, в зависимости
    // от активных режимов создаются:
    //   <prefix>.bench.csv    — пер-итерационная латентность бенчмарка;
    //   <prefix>.fps.csv      — FPS-семплы видео-цикла (раз в log-interval);
    //   <prefix>.compare.csv  — пер-выходные метрики режима compare.
    // Каждый файл само-документируем: в шапке `# key=value` лежат
    // модель, делегат, threads, runs, timestamp и т.п.
    std::string export_prefix;
    // ---- Мониторинг CPU/памяти ----
    // --sysmon включает периодический замер потребления процессом RAM
    // (RSS / VmHWM / VmSwap), CPU (% одного ядра, multi-thread даёт >100)
    // и системного CPU. Печатается сводка в конце прогона + (если задан
    // --export) пишется <prefix>.sysmon.csv с per-сэмплом и
    // <prefix>.sysmon.summary.csv с агрегатами. Для бенчмарка снимаем
    // baseline до прогрева и финальный замер после; в видео-цикле семплим
    // раз в --sysmon-interval мс. Работает только на Linux (на dev-хосте
    // под Windows/macOS init() вернёт false и блок будет пропущен).
    bool sysmon = false;
    int  sysmon_interval_ms = 1000;
};

// Печать справки по использованию (список собранных бэкендов берётся из
// inf::available_backends()).
void print_usage(const char* prog);

// Разбор argv в Args. Возвращает false при ошибке или -h/--help
// (в обоих случаях вызывающая сторона должна завершиться).
bool parse_args(int argc, char** argv, Args& a);

}  // namespace iirun
