// Источник кадров для унифицированного видео-цикла раннера.
//
// Один интерфейс — две реализации (камера / статический буфер) — один
// общий цикл инференса. Источник отвечает на два вопроса:
//   1) какой буфер RGB подавать на этой итерации (вместе с его исходными
//      размерами, нужными YOLO для обратного letterbox-маппинга);
//   2) надо ли его препроцессить + заливать в тензор. Для камеры — да
//      на каждом кадре; для статической картинки — нет (вход модели уже
//      заполнен один раз до входа в цикл).
//
// Возврат nullptr из next() означает «пропусти эту итерацию» (например,
// таймаут камеры). Цикл при этом проверит флаг прерывания и пойдёт
// ждать следующий кадр.
//
// Классы тривиальны (тонкие обёртки) — модуль header-only.

#pragma once

#include <cstdint>

#include "camera.h"

namespace ii {

struct FrameSource {
    virtual ~FrameSource() = default;
    virtual const uint8_t* next(int& out_w, int& out_h,
                                bool& out_needs_preprocess) = 0;
};

class CameraFrameSource : public FrameSource {
public:
    CameraFrameSource(Camera& cam, int timeout_ms = 1000)
        : cam_(cam), timeout_ms_(timeout_ms) {}
    const uint8_t* next(int& w, int& h, bool& needs_preprocess) override {
        w = cam_.width();
        h = cam_.height();
        needs_preprocess = true;       // живой источник: новый кадр каждый раз
        return cam_.grab(timeout_ms_);
    }
private:
    Camera& cam_;
    int     timeout_ms_;
};

class StaticFrameSource : public FrameSource {
public:
    StaticFrameSource(const uint8_t* rgb, int w, int h)
        : rgb_(rgb), w_(w), h_(h) {}
    const uint8_t* next(int& w, int& h, bool& needs_preprocess) override {
        w = w_;
        h = h_;
        needs_preprocess = false;      // вход модели уже заполнен снаружи
        return rgb_;
    }
private:
    const uint8_t* rgb_;
    int            w_, h_;
};

} // namespace ii
