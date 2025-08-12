// Реализация постобработки YOLOv8 (см. yolo.h).

#include "yolo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>

namespace {

// COCO 80 — стандартный набор классов, на котором обучен публичный
// YOLOv8. При желании пользователь может подменить через
// set_class_names() / load_class_names().
const std::vector<std::string> kCocoNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
};

std::vector<std::string> g_class_names = kCocoNames;
const std::string         g_empty_name;

float iou_xyxy(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, inter_x2 - inter_x1);
    float ih = std::max(0.0f, inter_y2 - inter_y1);
    float inter = iw * ih;
    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// Per-class NMS: сортируем кандидатов внутри каждого класса по score
// и жадно отбрасываем пересекающиеся. Между разными классами боксы
// конкурировать не должны (один и тот же объект может легитимно
// попасть в две категории — отдадим решение пользователю).
std::vector<Detection> nms_per_class(std::vector<Detection> dets,
                                     float iou_thresh,
                                     int max_dets) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  if (a.class_id != b.class_id) return a.class_id < b.class_id;
                  return a.score > b.score;
              });

    std::vector<Detection> kept;
    kept.reserve(dets.size());

    std::size_t i = 0;
    while (i < dets.size()) {
        std::size_t j = i;
        while (j < dets.size() && dets[j].class_id == dets[i].class_id) ++j;
        // Диапазон [i, j) — все кандидаты одного класса, уже отсортированы
        // по убыванию score.
        std::vector<bool> suppressed(j - i, false);
        for (std::size_t k = i; k < j; ++k) {
            if (suppressed[k - i]) continue;
            kept.push_back(dets[k]);
            if ((int)kept.size() >= max_dets) return kept;
            for (std::size_t m = k + 1; m < j; ++m) {
                if (suppressed[m - i]) continue;
                if (iou_xyxy(dets[k], dets[m]) > iou_thresh)
                    suppressed[m - i] = true;
            }
        }
        i = j;
    }
    return kept;
}

}  // namespace

const std::string& yolo_class_name(int idx) {
    if (idx < 0 || idx >= (int)g_class_names.size()) return g_empty_name;
    return g_class_names[idx];
}

void set_class_names(const std::vector<std::string>& names) {
    g_class_names = names.empty() ? kCocoNames : names;
}

bool load_class_names(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "Не удалось открыть файл классов: %s\n",
                     path.c_str());
        return false;
    }
    std::vector<std::string> names;
    std::string line;
    while (std::getline(f, line)) {
        // Удаляем хвостовой \r (CRLF из Windows-файлов).
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.empty()) continue;
        names.push_back(line);
    }
    if (names.empty()) {
        std::fprintf(stderr, "В %s не найдено имён классов.\n", path.c_str());
        return false;
    }
    set_class_names(names);
    std::printf("Загружено имён классов: %zu (%s)\n", names.size(),
                path.c_str());
    return true;
}

std::vector<Detection> decode_yolov8(const float* data,
                                     int channels,
                                     int num_anchors,
                                     bool channels_first,
                                     int input_w,
                                     int input_h,
                                     const YoloPostOptions& opt) {
    std::vector<Detection> raw;
    raw.reserve(256);

    const int nc = channels - 4;
    if (nc <= 0 || num_anchors <= 0) return raw;

    // Доступ к (channel c, anchor a) с учётом обоих layout’ов:
    // channels_first  — выход [C, A], шаг по C = num_anchors;
    // channels_last   — выход [A, C], шаг по A = channels.
    auto at = [&](int c, int a) -> float {
        return channels_first ? data[(std::size_t)c * num_anchors + a]
                              : data[(std::size_t)a * channels + c];
    };

    const float scale_x = opt.normalized ? (float)input_w : 1.0f;
    const float scale_y = opt.normalized ? (float)input_h : 1.0f;

    for (int a = 0; a < num_anchors; ++a) {
        // Берём argmax по классам. Это эквивалент per-anchor лучшего класса
        // и быстрее, чем материализовать nc детекций на anchor (nms сделал
        // бы то же самое).
        int   best_c = 0;
        float best_s = at(4, a);
        for (int c = 1; c < nc; ++c) {
            float s = at(4 + c, a);
            if (s > best_s) { best_s = s; best_c = c; }
        }
        if (best_s < opt.conf_thresh) continue;

        float cx = at(0, a) * scale_x;
        float cy = at(1, a) * scale_y;
        float w  = at(2, a) * scale_x;
        float h  = at(3, a) * scale_y;

        Detection d;
        d.x1 = cx - 0.5f * w;
        d.y1 = cy - 0.5f * h;
        d.x2 = cx + 0.5f * w;
        d.y2 = cy + 0.5f * h;
        d.class_id = best_c;
        d.score    = best_s;
        raw.push_back(d);
    }

    return nms_per_class(std::move(raw), opt.iou_thresh, opt.max_dets);
}

void scale_to_original(std::vector<Detection>& dets,
                       int input_w, int input_h,
                       int orig_w,  int orig_h) {
    if (orig_w <= 0 || orig_h <= 0) return;
    // Те же r/dx/dy, что в letterbox(): прямой ход — оригинал → letterbox,
    // обратный — вычитаем паддинг и делим на масштаб.
    float r = std::min((float)input_w / orig_w, (float)input_h / orig_h);
    int new_w = (int)std::round(orig_w * r);
    int new_h = (int)std::round(orig_h * r);
    float dx = (input_w - new_w) * 0.5f;
    float dy = (input_h - new_h) * 0.5f;

    for (auto& d : dets) {
        d.x1 = (d.x1 - dx) / r;
        d.y1 = (d.y1 - dy) / r;
        d.x2 = (d.x2 - dx) / r;
        d.y2 = (d.y2 - dy) / r;
        // Клипуем в границы исходной картинки — иначе бокс,
        // вылезающий за край, рисуется некорректно.
        d.x1 = std::clamp(d.x1, 0.0f, (float)orig_w);
        d.y1 = std::clamp(d.y1, 0.0f, (float)orig_h);
        d.x2 = std::clamp(d.x2, 0.0f, (float)orig_w);
        d.y2 = std::clamp(d.y2, 0.0f, (float)orig_h);
    }
}
