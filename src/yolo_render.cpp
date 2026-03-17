// Реализация yolo_render.h. См. заголовок для назначения.

#include "yolo_render.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace iirun {

bool detect_yolo_head(const std::vector<TensorInfo>& outs, int forced,
                      YoloHead& h) {
    // try_pick пишет результат в out, никаких побочных эффектов на h —
    // это важно при автодетекте, где мы пробуем все выходы по очереди и
    // не должны затирать ранее найденного кандидата мусором от неудачной
    // попытки.
    auto try_pick = [&](int i, YoloHead& out) -> bool {
        const auto& s = outs[i].shape;
        if (s.size() != 3 || s[0] != 1) return false;
        int a = s[1];
        int b = s[2];
        // Эвристика: «каналы» = меньшая ось, «anchor’ы» = большая.
        // На YOLOv8 это всегда верно (84 ≪ 8400). Если у пользователя
        // экзотическая модель — пусть передаёт --yolo-output явно и
        // понимает, что делает.
        int channels = std::min(a, b);
        int anchors  = std::max(a, b);
        if (channels < 5 || anchors < channels * 4) return false;
        out.output_index   = i;
        out.channels       = channels;
        out.num_anchors    = anchors;
        out.channels_first = (channels == a);
        return true;
    };

    if (forced >= 0) {
        if (forced >= (int)outs.size()) {
            std::fprintf(stderr,
                "--yolo-output %d вне диапазона (0..%zu).\n",
                forced, outs.size() - 1);
            return false;
        }
        if (!try_pick(forced, h)) {
            std::fprintf(stderr,
                "Выход %d не похож на YOLO-голову (ожидаю [1, C, A] или "
                "[1, A, C]).\n", forced);
            return false;
        }
        return true;
    }
    int found = -1;
    YoloHead picked;
    for (int i = 0; i < (int)outs.size(); ++i) {
        YoloHead tmp;
        if (try_pick(i, tmp)) {
            if (found >= 0) {
                std::fprintf(stderr,
                    "Найдено несколько подходящих YOLO-выходов; уточните "
                    "--yolo-output.\n");
                return false;
            }
            found  = i;
            picked = tmp;
        }
    }
    if (found < 0) {
        std::fprintf(stderr,
            "Не нашёл YOLO-голову среди выходов модели.\n");
        return false;
    }
    h = picked;
    return true;
}

namespace {

// Стабильный цвет под класс — вращение по hue с фиксированным шагом.
// Один и тот же class_id даст одинаковый цвет от кадра к кадру.
void class_color(int class_id, uint8_t& r, uint8_t& g, uint8_t& b) {
    // Золотое сечение ≈ 0.618; даёт хорошо различимые цвета даже для
    // соседних индексов.
    float h = std::fmod(class_id * 0.61803398875f, 1.0f) * 6.0f;
    float c = 1.0f;          // saturation = 1, value = 1
    float x = c * (1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f));
    float r1 = 0, g1 = 0, b1 = 0;
    if      (h < 1) { r1 = c; g1 = x; }
    else if (h < 2) { r1 = x; g1 = c; }
    else if (h < 3) { g1 = c; b1 = x; }
    else if (h < 4) { g1 = x; b1 = c; }
    else if (h < 5) { r1 = x; b1 = c; }
    else            { r1 = c; b1 = x; }
    r = (uint8_t)std::lround(r1 * 255.0f);
    g = (uint8_t)std::lround(g1 * 255.0f);
    b = (uint8_t)std::lround(b1 * 255.0f);
}

}  // namespace

std::vector<DisplayBox> detections_to_boxes(
        const std::vector<Detection>& dets) {
    std::vector<DisplayBox> out;
    out.reserve(dets.size());
    for (const auto& d : dets) {
        DisplayBox b;
        b.x1 = d.x1; b.y1 = d.y1; b.x2 = d.x2; b.y2 = d.y2;
        b.label = yolo_class_name(d.class_id);
        if (b.label.empty()) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "cls%d", d.class_id);
            b.label = buf;
        }
        b.score = d.score;
        class_color(d.class_id, b.r, b.g, b.b);
        out.push_back(b);
    }
    return out;
}

}  // namespace iirun
