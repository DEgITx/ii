// Абстрактный интерфейс окна для вывода RGB-кадров.
//
// Реализация по умолчанию — Wayland + EGL + GLES2 (display_wayland.cpp).
// Если USE_DISPLAY=OFF в CMake, собирается заглушка (display_stub.cpp),
// и make_display() возвращает nullptr — клиентский код просто пропускает
// вывод.
//
// Дизайн рассчитан на видео-пайплайн: один Display создаётся один раз,
// затем show_rgb() вызывается на каждый кадр. Между кадрами одинакового
// размера повторного выделения текстуры не происходит — используется
// glTexSubImage2D.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Бокс для отрисовки поверх кадра (например, результат YOLO-детекции).
// Координаты в пикселях ИСХОДНОЙ текстуры (того кадра, который ушёл в
// show_rgb): Display сам отмаппит их в окно с учётом letterbox-viewport.
struct DisplayBox {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
    std::string label;          // подпись (можно пустую)
    float score = -1.0f;        // 0..1, отрицательное => в подпись не идёт
    uint8_t r = 0, g = 255, b = 0;  // цвет рамки (по умолчанию ярко-зелёный)
};

// dtype-коды для TileFrameDesc::dtype. Подмножество ii::DType, которое
// умеет собирать GPU-путь (квантованные выходы NPU). Типы вне списка →
// раннер откатывается на CPU-decode. Держим как plain int, чтобы display.h
// не тянул inference.h (он backend-нейтрален).
enum TileDType {
    TILE_DT_INT8  = 0,   // bit-pattern в R8 UNORM, в шейдере → signed
    TILE_DT_UINT8 = 1,
};

// Описание GPU-сборки кадра из сырых квантованных тайлов (tile-режим).
// POD без зависимостей от ii::-типов: раннер транслирует ii::DType /
// ii::OutputRange в эти числовые поля. canvas_w/h задаются на каждый кадр
// (зависят от разрешения источника), остальное стабильно за прогон.
//
// Семантика деквантизации повторяет image_proc.cpp:decode_kernel:
//   val = scale * (q - zero_point);   q — целое из тайла (int8/uint8)
//   range: 0=unit [0..1], 1=signed [-1..1]→[0..1], 2=byte [0..255]→[0..1]
//   clamp в [0..1]; C=1 разворачивается в R=G=B.
struct TileFrameDesc {
    int   canvas_w = 0, canvas_h = 0;  // итоговый кадр, output-space px
    int   tile_w   = 0, tile_h   = 0;  // выход одного тайла, px
    int   channels = 1;                // 1 (Y→broadcast) | 3 (RGB)
    int   dtype    = TILE_DT_INT8;     // см. TileDType
    float scale    = 1.0f;             // квант-scale модели (> 0)
    int   zero_point = 0;
    int   range    = 0;                // 0=unit 1=signed 2=byte (OutputRange)
    int   blend    = 0;                // 1 — alpha-blend швов (overlap>0);
                                       // 0 — pure overwrite (без чтения dst
                                       //     блендером на каждый пиксель)
};

struct Display {
    virtual ~Display() = default;

    // Создать окно. Размеры — стартовые, окно может быть отресайжено
    // пользователем; show_rgb сам подгонит viewport.
    // vsync = true ограничивает FPS частотой обновления экрана (обычно
    // 60 Гц), false — отдаёт максимум того, что выдаёт пайплайн (но
    // возможен tearing на динамичной картинке). Для замеров производительности
    // выключайте: тайминги дисплея перестанут квантоваться по vblank.
    virtual bool init(int w, int h, const char* title, bool vsync = true) = 0;

    // Залить и отрисовать кадр. RGB HWC, 8 бит/канал, без выравнивания строк.
    // Возвращает false, если окно было закрыто пользователем.
    virtual bool show_rgb(const uint8_t* rgb, int w, int h) = 0;

    // Прокачать события неблокирующе (resize, close, ping/pong).
    // Возвращает false, когда окно закрыто. Зовите между кадрами видео.
    virtual bool poll() = 0;

    // Блокирующее ожидание следующего события (использовать после
    // одиночного show_rgb, чтобы окно не закрывалось мгновенно).
    // Возвращает false, когда окно закрыто.
    virtual bool wait() = 0;

    // Установить текст оверлея (рисуется поверх кадра в левом верхнем углу).
    // Передайте nullptr или пустую строку, чтобы скрыть. Текст копируется
    // внутрь Display, можно сразу освобождать буфер.
    virtual void set_overlay_text(const char* text) = 0;

    // Установить набор боксов для отрисовки поверх кадра. Передайте пустой
    // вектор, чтобы убрать все. Координаты в пикселях текущей текстуры
    // (см. DisplayBox); при изменении размера окна Display сам пересчитает
    // их положение. Боксы копируются внутрь — буфер можно сразу освобождать.
    virtual void set_boxes(const std::vector<DisplayBox>& boxes) = 0;

    // ---- GPU-сборка кадра из сырых квантованных тайлов (tile-режим) ------
    //
    // Быстрый путь для image-to-image tiling: вместо CPU-decode каждого
    // тайла в RGB-canvas и заливки его текстурой, сырые int8/uint8 выходы
    // тайлов заливаются маленькими текстурами, а деквантизация + broadcast
    // + over-composite делаются фрагментным шейдером прямо в offscreen-цель
    // размером canvas. Это снимает основной CPU-bottleneck tile-режима
    // (decode/fill ~10 МБ/кадр) и режет трафик заливки.
    //
    // Протокол одного кадра:
    //   if (disp->supports_tile_frame()) {
    //     disp->tile_frame_begin(desc);                  // подготовить FBO
    //     for each tile: disp->tile_frame_add(raw,...);  // upload + квад
    //     // (set_boxes / set_overlay_text как обычно)
    //     disp->tile_frame_present();                     // compose + show
    //   }
    //
    // Все вызовы — на потоке, владеющем GL/D3D-контекстом (тот же, что
    // зовёт show_rgb). Бэкенды без поддержки оставляют дефолты (false/no-op),
    // и раннер использует CPU-путь.

    // Умеет ли бэкенд собирать кадр из тайлов на GPU. false → CPU-путь.
    virtual bool supports_tile_frame() const { return false; }

    // Начать кадр: подготовить offscreen-цель canvas_w×canvas_h и стейт
    // деквантизации (desc стабилен, кроме canvas_*). Возвращает false при
    // неудаче — раннер откатится на CPU-путь для этого кадра.
    virtual bool tile_frame_begin(const TileFrameDesc& desc) {
        (void)desc; return false;
    }

    // Залить и скомпоновать один тайл. raw — байты выходного тензора тайла
    // (tile_w*tile_h*channels элементов dtype); (dst_x,dst_y) — позиция
    // верхнего-левого угла в canvas (output-space); ramp_x/ramp_y — ширина
    // линейной растушёвки по ЛЕВОМУ/ВЕРХНЕМУ краю (0 → непрозрачно). Тайлы
    // подаются в raster-порядке (composite опирается на уже нарисованных
    // соседей слева/сверху).
    virtual void tile_frame_add(const void* raw, int dst_x, int dst_y,
                                int ramp_x, int ramp_y) {
        (void)raw; (void)dst_x; (void)dst_y; (void)ramp_x; (void)ramp_y;
    }

    // Завершить кадр: отрисовать собранную цель в окно (letterbox + боксы +
    // оверлей) и present. Аналог show_rgb для tile-режима. Возвращает false,
    // если окно закрыто.
    virtual bool tile_frame_present() { return false; }

    // Прочитать собранный кадр обратно в CPU (для --save-output): rgb — HWC
    // RGB uint8 (w*h*3), w/h = размеры canvas. Возвращает false, если читать
    // нечего. Вызывать после tile_frame_add всех тайлов (до или после
    // present). Дорогая операция (GPU→CPU стол) — только для одиночного
    // сохранения, не для hot-path.
    virtual bool tile_frame_readback(std::vector<uint8_t>& rgb,
                                     int& w, int& h) {
        (void)rgb; (void)w; (void)h; return false;
    }
};

// Возвращает реализацию Display для текущей платформы или nullptr,
// если поддержка дисплея не собрана.
std::unique_ptr<Display> make_display();
