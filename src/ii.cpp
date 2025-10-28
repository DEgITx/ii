// Универсальный раннер TFLite-моделей на NPU (внешний делегат).
//
// Цель: пропустить изображение через INT8-квантованную модель
// (например yolov8m_int8.tflite) и получить «сырые» выходы.
//
// Поддерживаются модели:
//   * один вход; для путей с картинкой (загрузка image, --display, --yolo,
//     image-режим --compare) ожидается NHWC [1,H,W,3] (uint8/int8/float32);
//     в режиме --random-input shape произвольный (например [1,9,9,1] для
//     табличных/регрессионных сетей);
//   * любое число выходов произвольных типов;
//   * квантование scale/zero_point читается из самой модели.
//
// YOLO-постобработка (декодирование боксов, NMS) подключается сверху
// через флаг --yolo (см. yolo.h). Сам движок остаётся универсальным —
// без флага модель прогоняется как «чёрный ящик», на экран идёт сырое
// изображение.
//
// Запуск (на устройстве):
//   ./ii yolov8m_int8.tflite image.jpg
//   ./ii model.tflite image.jpg --no-delegate
//   ./ii model.tflite image.jpg --benchmark --runs 100
//   ./ii model.tflite image.jpg --display              # окно Wayland
//   ./ii model.tflite image.jpg --display --show-input # показать препроц.
//   ./ii yolov8m_int8.tflite img.jpg --display --yolo  # боксы поверх
//   ./ii yolov8m_int8.tflite img.jpg --display --yolo --conf 0.4 --iou 0.5
//   ./ii model.tflite --random-input --benchmark --runs 100   # без картинки
//   ./ii model.tflite --random-input --compare-cpu --random-runs 50
//   ./ii yolov8m_int8.tflite --camera --display --yolo  # видео-поток
//   ./ii yolov8m_int8.tflite --camera /dev/video2 --camera-size 1280x720 \
//       --camera-fps 30 --display --yolo --stats        # с замером FPS

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "camera.h"
#include "csv_export.h"
#include "display.h"
#include "stats.h"
#include "sysmon.h"
#include "yolo.h"

#include <atomic>
#include <csignal>

namespace {
std::atomic<bool> g_interrupted{false};
void on_sigint(int) { g_interrupted = true; }
}  // namespace

namespace {

constexpr const char* kDefaultDelegate = "/lib/libdelegate.so";

// ---------------------------------------------------------------------------
// Обёртка над интерпретатором TFLite + внешним делегатом.
// Делегат живёт дольше интерпретатора: сначала разрушаем interpreter,
// затем сам делегат, иначе TFLite крашится в деструкторе.
// ---------------------------------------------------------------------------
struct Engine {
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteDelegate* delegate = nullptr;

    ~Engine() {
        interpreter.reset();
        if (delegate) TfLiteExternalDelegateDelete(delegate);
    }

    bool load(const std::string& model_path,
              const std::string& delegate_path,
              int num_threads) {
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model) {
            std::fprintf(stderr, "Не удалось загрузить модель: %s\n",
                         model_path.c_str());
            return false;
        }
        if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk
            || !interpreter) {
            std::fprintf(stderr, "InterpreterBuilder упал.\n");
            return false;
        }
        if (num_threads > 0) interpreter->SetNumThreads(num_threads);

        if (!delegate_path.empty()) {
            auto opts = TfLiteExternalDelegateOptionsDefault(delegate_path.c_str());
            delegate = TfLiteExternalDelegateCreate(&opts);
            if (!delegate) {
                std::fprintf(stderr,
                    "Не удалось создать external delegate: %s\n",
                    delegate_path.c_str());
                return false;
            }
            if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
                std::fprintf(stderr, "ModifyGraphWithDelegate упал.\n");
                return false;
            }
            std::printf("Делегат загружен: %s\n", delegate_path.c_str());
        } else {
            std::printf("Делегат не используется — CPU.\n");
        }

        if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::fprintf(stderr, "AllocateTensors упал.\n");
            return false;
        }
        return true;
    }

    bool invoke() { return interpreter->Invoke() == kTfLiteOk; }
};

// ---------------------------------------------------------------------------
// Описание тензора (срез TfLiteTensor для удобной печати/обработки)
// ---------------------------------------------------------------------------
struct TensorInfo {
    int index = 0;
    std::string name;
    std::vector<int> shape;
    TfLiteType dtype = kTfLiteNoType;
    float scale = 0.0f;
    int32_t zero_point = 0;
    size_t bytes = 0;
};

TensorInfo describe(const TfLiteTensor* t, int index) {
    TensorInfo info;
    info.index = index;
    info.name = t->name ? t->name : "";
    info.shape.assign(t->dims->data, t->dims->data + t->dims->size);
    info.dtype = t->type;
    info.scale = t->params.scale;
    info.zero_point = t->params.zero_point;
    info.bytes = t->bytes;
    return info;
}

const char* dtype_name(TfLiteType t) {
    switch (t) {
        case kTfLiteFloat32: return "float32";
        case kTfLiteFloat16: return "float16";
        case kTfLiteInt8:    return "int8";
        case kTfLiteUInt8:   return "uint8";
        case kTfLiteInt16:   return "int16";
        case kTfLiteUInt16:  return "uint16";
        case kTfLiteInt32:   return "int32";
        case kTfLiteUInt32:  return "uint32";
        case kTfLiteInt64:   return "int64";
        case kTfLiteBool:    return "bool";
        default:             return "?";
    }
}

size_t dtype_size(TfLiteType t) {
    switch (t) {
        case kTfLiteFloat32:
        case kTfLiteInt32:
        case kTfLiteUInt32:  return 4;
        case kTfLiteInt64:   return 8;
        case kTfLiteFloat16:
        case kTfLiteInt16:
        case kTfLiteUInt16:  return 2;
        case kTfLiteInt8:
        case kTfLiteUInt8:
        case kTfLiteBool:    return 1;
        default:             return 1;
    }
}

// IEEE 754 binary16 -> binary32 (без зависимости от __fp16 / F16C).
// Корректно обрабатывает denormals, ±0, ±inf и NaN.
inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t       mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;                       // ±0
        } else {
            // denormal: нормализуем, сдвигая мантиссу влево.
            int e = -1;
            do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = (sign << 31) | ((127u - 15u - (uint32_t)e) << 23)
                 | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);  // inf / NaN
    } else {
        bits = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

size_t numel(const std::vector<int>& shape) {
    size_t n = 1;
    for (int d : shape) {
        if (d <= 0) return 0;  // динамический/нулевой размер не поддерживаем
        n *= (size_t)d;
    }
    return n;
}

std::string shape_to_str(const std::vector<int>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

// Является ли shape «картинкой» NHWC [1,H,W,3] — единственный формат, под
// который заточены letterbox/изображение/--display/--yolo.
bool is_nhwc3(const std::vector<int>& s) {
    return s.size() == 4 && s[0] == 1 && s[3] == 3;
}

void print_tensor(const char* prefix, const TensorInfo& t) {
    std::printf("%s %-32s shape=[", prefix, t.name.c_str());
    for (size_t i = 0; i < t.shape.size(); ++i)
        std::printf("%s%d", i ? "," : "", t.shape[i]);
    std::printf("] dtype=%s", dtype_name(t.dtype));
    if (t.scale != 0.0f)
        std::printf(" quant=(scale=%.6g, zp=%d)", t.scale, t.zero_point);
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// Загрузка изображения и letterbox-препроцессинг (стандарт YOLO).
// ---------------------------------------------------------------------------
struct Image {
    std::vector<uint8_t> rgb;   // HWC, 3 канала, 0..255
    int w = 0, h = 0;
};

bool load_image(const std::string& path, Image& out) {
    int w = 0, h = 0, c = 0;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        std::fprintf(stderr, "Не удалось загрузить изображение: %s (%s)\n",
                     path.c_str(), stbi_failure_reason());
        return false;
    }
    out.rgb.assign(data, data + (size_t)w * h * 3);
    out.w = w;
    out.h = h;
    stbi_image_free(data);
    return true;
}

// Ресайз с сохранением пропорций + паддинг до (target_h x target_w).
// Для YOLO стандарт — заполнение серым (114).
//
// Перегрузка с сырым указателем удобна для видео-источников (камера),
// чтобы не копировать кадр в промежуточный std::vector. Реальная
// арифметика — здесь, обёртка над Image ниже.
void letterbox(const uint8_t* src_rgb, int src_w, int src_h,
               int target_w, int target_h,
               std::vector<uint8_t>& dst, uint8_t pad = 114) {
    float r = std::min((float)target_w / src_w, (float)target_h / src_h);
    int new_w = (int)std::round(src_w * r);
    int new_h = (int)std::round(src_h * r);
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;

    // resized живёт между вызовами, но переалоцируется только при смене
    // размера ресайза (например, при ресайзе камеры на лету). Для
    // стационарного видео-цикла память аллоцируется один раз.
    static thread_local std::vector<uint8_t> resized;
    resized.resize((size_t)new_w * new_h * 3);
    stbir_resize_uint8_linear(src_rgb, src_w, src_h, 0,
                              resized.data(), new_w, new_h, 0,
                              STBIR_RGB);

    dst.assign((size_t)target_w * target_h * 3, pad);
    int dx = (target_w - new_w) / 2;
    int dy = (target_h - new_h) / 2;
    for (int y = 0; y < new_h; ++y) {
        std::memcpy(&dst[((y + dy) * target_w + dx) * 3],
                    &resized[(size_t)y * new_w * 3],
                    (size_t)new_w * 3);
    }
}

inline void letterbox(const Image& src, int target_w, int target_h,
                      std::vector<uint8_t>& dst, uint8_t pad = 114) {
    letterbox(src.rgb.data(), src.w, src.h, target_w, target_h, dst, pad);
}

// ---------------------------------------------------------------------------
// Заполнение входного тензора. На вход — RGB HWC uint8 [0..255].
// Нормализуется в [0,1] и квантуется по scale/zero_point модели для
// целочисленных типов. (Логика идентична quantize_to_input.)
// ---------------------------------------------------------------------------
bool fill_input(const std::vector<uint8_t>& rgb, const TensorInfo& info,
                TfLiteTensor* tensor) {
    const size_t n = rgb.size();

    // Универсальная квантователь под произвольный целочисленный тип.
    auto quant_int = [&](auto* p, int lo, int hi) {
        using T = std::decay_t<decltype(*p)>;
        const float s  = info.scale ? info.scale : 1.0f;
        const int   zp = info.zero_point;
        for (size_t i = 0; i < n; ++i) {
            int q = (int)std::lround((rgb[i] / 255.0f) / s + zp);
            if (q < lo) q = lo;
            if (q > hi) q = hi;
            p[i] = (T)q;
        }
    };

    switch (info.dtype) {
        case kTfLiteFloat32: {
            float* p = reinterpret_cast<float*>(tensor->data.data);
            for (size_t i = 0; i < n; ++i) p[i] = rgb[i] / 255.0f;
            return true;
        }
        case kTfLiteInt8:
            quant_int(reinterpret_cast<int8_t*>(tensor->data.data), -128, 127);
            return true;
        case kTfLiteUInt8:
            quant_int(reinterpret_cast<uint8_t*>(tensor->data.data), 0, 255);
            return true;
        case kTfLiteInt16:
            quant_int(reinterpret_cast<int16_t*>(tensor->data.data),
                      -32768, 32767);
            return true;
        case kTfLiteUInt16:
            quant_int(reinterpret_cast<uint16_t*>(tensor->data.data),
                      0, 65535);
            return true;
        default:
            std::fprintf(stderr, "Неподдерживаемый dtype входа: %s (код %d)\n",
                         dtype_name(info.dtype), (int)info.dtype);
            return false;
    }
}

// ---------------------------------------------------------------------------
// Печать первых n_show элементов выхода в float (с деквантованием).
// ---------------------------------------------------------------------------
void print_output_head(const TensorInfo& info, const TfLiteTensor* tensor,
                       int n_show = 10) {
    size_t total = info.bytes / dtype_size(info.dtype);
    n_show = (int)std::min<size_t>((size_t)n_show, total);

    const float s  = info.scale ? info.scale : 1.0f;
    const int   zp = info.zero_point;
    std::printf("  output %-32s first %d:", info.name.c_str(), n_show);
    for (int i = 0; i < n_show; ++i) {
        float v = 0.0f;
        switch (info.dtype) {
            case kTfLiteFloat32:
                v = reinterpret_cast<const float*>(tensor->data.data)[i];
                break;
            case kTfLiteFloat16:
                v = half_to_float(
                    reinterpret_cast<const uint16_t*>(tensor->data.data)[i]);
                break;
            case kTfLiteInt8:
                v = (reinterpret_cast<const int8_t*>(tensor->data.data)[i] - zp) * s;
                break;
            case kTfLiteUInt8:
                v = (reinterpret_cast<const uint8_t*>(tensor->data.data)[i] - zp) * s;
                break;
            case kTfLiteInt16:
                v = (reinterpret_cast<const int16_t*>(tensor->data.data)[i] - zp) * s;
                break;
            case kTfLiteUInt16:
                v = ((int)reinterpret_cast<const uint16_t*>(tensor->data.data)[i] - zp) * s;
                break;
            case kTfLiteInt32:
                v = ((float)reinterpret_cast<const int32_t*>(tensor->data.data)[i]
                     - (float)zp) * s;
                break;
            default: break;
        }
        std::printf(" %.4f", v);
    }
    std::printf("\n");
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
               steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Деквантование произвольного выходного тензора в float-буфер.
//
// Поддерживаемые типы:
//   * float32 / float16      — копируем (с конвертацией для half);
//   * int8 / uint8 / int16 / uint16 — аффинная формула (x - zp) * scale;
//   * int32 — то же (некоторые экспортёры так дают «головы» с большим
//     динамическим диапазоном).
//
// Если scale == 0 для целочисленного тензора (т.е. модель не квантована
// в полноценном смысле) — берём сырое значение как float, иначе деление
// на ноль / нулевая шкала всё бы обнулило.
// ---------------------------------------------------------------------------
bool dequantize_output(const TensorInfo& info, const TfLiteTensor* tensor,
                       std::vector<float>& out) {
    const std::size_t ds = dtype_size(info.dtype);
    const std::size_t total = info.bytes / ds;
    out.resize(total);

    const float s  = info.scale ? info.scale : 1.0f;
    const int   zp = info.zero_point;

    switch (info.dtype) {
        case kTfLiteFloat32: {
            std::memcpy(out.data(), tensor->data.data,
                        total * sizeof(float));
            return true;
        }
        case kTfLiteFloat16: {
            const uint16_t* p =
                reinterpret_cast<const uint16_t*>(tensor->data.data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = half_to_float(p[i]);
            return true;
        }
        case kTfLiteInt8: {
            const int8_t* p =
                reinterpret_cast<const int8_t*>(tensor->data.data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case kTfLiteUInt8: {
            const uint8_t* p =
                reinterpret_cast<const uint8_t*>(tensor->data.data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case kTfLiteInt16: {
            const int16_t* p =
                reinterpret_cast<const int16_t*>(tensor->data.data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case kTfLiteUInt16: {
            const uint16_t* p =
                reinterpret_cast<const uint16_t*>(tensor->data.data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = ((int)p[i] - zp) * s;
            return true;
        }
        case kTfLiteInt32: {
            const int32_t* p =
                reinterpret_cast<const int32_t*>(tensor->data.data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = ((float)p[i] - (float)zp) * s;
            return true;
        }
        default: {
            // В видео-цикле эта ошибка иначе зальёт stderr на каждый кадр —
            // печатаем один раз за всё время жизни процесса.
            static bool warned = false;
            if (!warned) {
                std::fprintf(stderr,
                    "Деквантование не поддерживает dtype=%s (код %d). "
                    "Дальнейшие подобные сообщения подавлены.\n",
                    dtype_name(info.dtype), (int)info.dtype);
                warned = true;
            }
            return false;
        }
    }
}

// ---------------------------------------------------------------------------
// Авто-определение YOLO-выхода и его layout’а.
// Ищем 3D-тензор вида [1, A, B] или [1, B, A], где меньший из {A, B}
// = 4 + nc (типично 84 для COCO), а больший — число anchor’ов.
// Если кандидат не найден или их несколько — лучше явно указать
// --yolo-output <i>.
// ---------------------------------------------------------------------------
struct YoloHead {
    int  output_index   = -1;     // индекс в out_info
    int  channels       = 0;      // 4 + nc
    int  num_anchors    = 0;
    bool channels_first = true;   // true: [1, C, A]; false: [1, A, C]
};

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

// ---------------------------------------------------------------------------
// Источник кадров для видео-цикла.
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
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Args {
    std::string model;
    std::string image;
    std::string delegate = kDefaultDelegate;
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

void print_usage(const char* prog) {
    std::printf(
        "Usage: %s <model.tflite> [image] [options]\n"
        "  image не обязателен в режиме --random-input.\n"
        "Options:\n"
        "  --delegate <path>   путь к внешний делегату (по умолчанию %s)\n"
        "  --no-delegate       запустить на CPU (без делегата)\n"
        "  --benchmark         прогрев + замер скорости\n"
        "  --runs <N>          число итераций бенчмарка (по умолчанию 100)\n"
        "  --warmup <N>        число итераций прогрева (по умолчанию 3)\n"
        "  --threads <N>       число CPU-потоков интерпретатора\n"
        "  --display           открыть окно (Wayland/EGL/GLES2)\n"
        "  --show-input        выводить препроцесированный (letterbox) вход\n"
        "  --win <WxH>         стартовый размер окна (по умолчанию 960x720)\n"
        "  --loop              непрерывный цикл инференс+отрисовка (Ctrl+C)\n"
        "  --stats             счётчик FPS/jitter (оверлей + лог в stdout)\n"
        "  --log-interval <ms> период лога FPS, мс (по умолчанию 1000)\n"
        "  --no-vsync          отключить vsync (макс. FPS, возможен tearing)\n"
        "  --yolo              декодировать выход как YOLOv8 и рисовать боксы\n"
        "  --conf <p>          порог уверенности (по умолчанию 0.25)\n"
        "  --iou <p>           порог IoU для NMS (по умолчанию 0.45)\n"
        "  --max-dets <N>      макс. число детекций (по умолчанию 300)\n"
        "  --yolo-pixel-coords координаты модели в пикселях, а не 0..1\n"
        "  --yolo-output <i>   индекс выхода с боксами (по умолчанию авто)\n"
        "  --classes <path>    файл имён классов (по строке; default = COCO80)\n"
        "  --compare-cpu       прогнать ту же модель на CPU и сравнить выходы\n"
        "  --compare <path>    прогнать эталонную .tflite на CPU и сравнить\n"
        "                      (например, исходный float32 vs INT8 на NPU)\n"
        "  --random-input      использовать случайный входной буфер вместо\n"
        "                      изображения; работает с одиночным инференсом,\n"
        "                      --benchmark, --loop, --compare* (картинку можно\n"
        "                      не указывать). Алиас: --compare-random\n"
        "  --random-runs <N>   в --compare* число случайных прогонов для\n"
        "                      агрегации метрик (def 1). Алиас: --compare-runs\n"
        "  --random-seed <N>   seed RNG для воспроизводимости (def — случайный).\n"
        "                      Алиас: --compare-seed\n"
        "  --camera [dev]      захват с V4L2-камеры (по умолчанию /dev/video0).\n"
        "                      Включает собственный цикл инференса; image и\n"
        "                      --random-input игнорируются. Совместимо с\n"
        "                      --display, --yolo, --stats, --show-input.\n"
        "  --camera-size WxH   запрашиваемое разрешение камеры (def 640x480)\n"
        "  --camera-fps <N>    запрашиваемая частота кадров камеры (def 30)\n"
        "  --export <prefix>   писать замеры в CSV: <prefix>.bench.csv (для\n"
        "                      --benchmark), <prefix>.fps.csv (для видео-\n"
        "                      цикла со --stats), <prefix>.compare.csv (для\n"
        "                      --compare/--compare-cpu). Пути с / создают\n"
        "                      нужные директории заранее: --export results/run01\n"
        "  --sysmon            мониторинг CPU/памяти процесса (RSS, VmHWM,\n"
        "                      потоки, %% ядра) и системного CPU. Сводка\n"
        "                      в stdout + при --export пишется sysmon.csv\n"
        "  --sysmon-interval <ms> период семплирования sysmon в видео-цикле\n"
        "                      (по умолчанию 1000 мс)\n",
        prog, kDefaultDelegate);
}

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 2) { print_usage(argv[0]); return false; }
    a.model = argv[1];
    // image — опциональный позиционный аргумент. Если argv[2] начинается
    // с '-' (или его нет) — считаем, что картинку не передавали и весь
    // остаток разбираем как флаги. Это нужно, чтобы --compare-random мог
    // работать без изображения вообще.
    int start = 2;
    if (argc >= 3 && argv[2][0] != '-') {
        a.image = argv[2];
        start   = 3;
    }
    for (int i = start; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--delegate"    && i + 1 < argc) a.delegate    = argv[++i];
        else if (s == "--no-delegate")                 a.no_delegate = true;
        else if (s == "--benchmark")                   a.benchmark   = true;
        else if (s == "--runs"        && i + 1 < argc) a.runs        = std::atoi(argv[++i]);
        else if (s == "--warmup"      && i + 1 < argc) a.warmup      = std::atoi(argv[++i]);
        else if (s == "--threads"     && i + 1 < argc) a.threads     = std::atoi(argv[++i]);
        else if (s == "--display")                     a.display     = true;
        else if (s == "--show-input")                  a.show_input  = true;
        else if (s == "--loop")                        a.loop        = true;
        else if (s == "--stats")                       a.stats       = true;
        else if (s == "--log-interval"&& i + 1 < argc) a.log_interval_ms = std::atoi(argv[++i]);
        else if (s == "--no-vsync")                    a.vsync       = false;
        else if (s == "--yolo")                        a.yolo        = true;
        else if (s == "--conf"        && i + 1 < argc) a.conf        = (float)std::atof(argv[++i]);
        else if (s == "--iou"         && i + 1 < argc) a.iou         = (float)std::atof(argv[++i]);
        else if (s == "--max-dets"    && i + 1 < argc) a.max_dets    = std::atoi(argv[++i]);
        else if (s == "--yolo-pixel-coords")           a.yolo_pixel_coords = true;
        else if (s == "--yolo-output" && i + 1 < argc) a.yolo_output = std::atoi(argv[++i]);
        else if (s == "--classes"     && i + 1 < argc) a.classes_path = argv[++i];
        else if (s == "--compare-cpu")                 a.compare_cpu = true;
        else if (s == "--compare"     && i + 1 < argc) a.compare_model = argv[++i];
        else if (s == "--random-input"
              || s == "--compare-random")              a.random_input = true;
        else if ((s == "--random-runs" || s == "--compare-runs")
                 && i + 1 < argc)                      a.random_runs  = std::atoi(argv[++i]);
        else if ((s == "--random-seed" || s == "--compare-seed")
                 && i + 1 < argc)                      a.random_seed  = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (s == "--camera") {
            // Опциональный позиционный параметр: путь к устройству.
            // Любая следующая опция (начинается с '-') означает, что
            // путь не задан и берётся дефолт /dev/video0.
            a.camera_device = "/dev/video0";
            if (i + 1 < argc && argv[i + 1][0] != '-')
                a.camera_device = argv[++i];
        }
        else if (s == "--camera-size" && i + 1 < argc) {
            std::string v = argv[++i];
            auto x = v.find('x');
            if (x == std::string::npos) {
                std::fprintf(stderr,
                    "--camera-size ожидает WxH, получено: %s\n", v.c_str());
                return false;
            }
            a.camera_w = std::atoi(v.substr(0, x).c_str());
            a.camera_h = std::atoi(v.substr(x + 1).c_str());
        }
        else if (s == "--camera-fps"  && i + 1 < argc) a.camera_fps  = std::atoi(argv[++i]);
        else if (s == "--export"      && i + 1 < argc) a.export_prefix = argv[++i];
        else if (s == "--sysmon")                      a.sysmon       = true;
        else if (s == "--sysmon-interval" && i + 1 < argc)
                                                      a.sysmon_interval_ms = std::atoi(argv[++i]);
        else if (s == "--win"         && i + 1 < argc) {
            // Принимаем формат "WxH" (например 1280x720).
            std::string v = argv[++i];
            auto x = v.find('x');
            if (x == std::string::npos) {
                std::fprintf(stderr, "--win ожидает WxH, получено: %s\n", v.c_str());
                return false;
            }
            a.win_w = std::atoi(v.substr(0, x).c_str());
            a.win_h = std::atoi(v.substr(x + 1).c_str());
        }
        else if (s == "-h" || s == "--help") { print_usage(argv[0]); return false; }
        else {
            std::fprintf(stderr, "Неизвестный аргумент: %s\n", s.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    // Источников входа три: картинка, --random-input, --camera. Хотя бы
    // один обязателен. --camera (если задана) перебивает остальные:
    // получает контроль над основным циклом и игнорирует image/random.
    const bool has_camera = !args.camera_device.empty();
    const bool has_image  = !args.image.empty() && !has_camera;
    if (!has_image && !args.random_input && !has_camera) {
        std::fprintf(stderr,
            "Не указан источник входа. Передайте image, либо "
            "--random-input, либо --camera.\n");
        print_usage(argv[0]);
        return 1;
    }
    if (args.random_runs < 1) {
        std::fprintf(stderr, "--random-runs должен быть >= 1.\n");
        return 1;
    }
    if (args.random_input && args.display && !has_image && !has_camera) {
        std::fprintf(stderr,
            "--display требует источник кадров (image или --camera); "
            "при чистом --random-input нечего показывать.\n");
        return 1;
    }
    if (has_camera && !args.image.empty()) {
        std::fprintf(stderr,
            "Внимание: image=%s проигнорирован, активен --camera %s.\n",
            args.image.c_str(), args.camera_device.c_str());
    }

    Engine eng;
    if (!eng.load(args.model,
                  args.no_delegate ? std::string{} : args.delegate,
                  args.threads)) return 2;

    // Тег для логов: где именно крутится инференс.
    const char* label_main = args.no_delegate ? "CPU" : "NPU";

    // ---- Мониторинг CPU/памяти ----
    // Инициализируем заранее, чтобы baseline-семпл захватил состояние ещё
    // до загрузки тензоров и первого invoke. На не-Linux init() вернёт
    // false; SysAccum останется пустым, а вызовы sample() — безвредными.
    SysMonitor sysmon;
    SysAccum   sysmon_acc;
    if (args.sysmon) {
        if (!sysmon.init()) {
            std::fprintf(stderr,
                "Внимание: --sysmon недоступен (не Linux или /proc нечитаем) — "
                "пропускаем замеры.\n");
        } else {
            std::printf("Мониторинг CPU/памяти включён "
                        "(ядер в системе: %d, интервал %d мс)\n",
                        sysmon.num_cpus(), args.sysmon_interval_ms);
        }
    }
    // Открываем оба CSV (per-sample и summary) лениво по факту первого
    // использования, чтобы не плодить пустые файлы для путей, в которых
    // sysmon не активирован (например, --camera при --no-export).
    CsvExport sysmon_csv;
    bool      sysmon_csv_opened = false;
    const double sysmon_t0 = now_ms();

    // ---- Сводка по тензорам ----
    std::vector<TensorInfo> in_info, out_info;
    for (int idx : eng.interpreter->inputs())
        in_info.push_back(describe(eng.interpreter->tensor(idx), idx));
    for (int idx : eng.interpreter->outputs())
        out_info.push_back(describe(eng.interpreter->tensor(idx), idx));

    std::printf("Входы:\n");
    for (auto& t : in_info)  print_tensor(" -", t);
    std::printf("Выходы:\n");
    for (auto& t : out_info) print_tensor(" -", t);

    if (in_info.size() != 1) {
        std::fprintf(stderr,
            "Эта реализация рассчитана на 1 вход, у модели %zu.\n",
            in_info.size());
        return 3;
    }

    // ---- Экспорт замеров в CSV ----
    // Префикс из --export. Конкретные файлы открываются лениво в
    // соответствующих режимах (бенчмарк / видео-цикл / compare),
    // чтобы не плодить пустые файлы для неактивных путей. Шапка
    // самодокументируема: модель, делегат, threads, shape входа,
    // timestamp — этого хватает, чтобы потом склеивать прогоны
    // разных моделей в одну таблицу.
    auto open_export = [&](CsvExport& e, const char* suffix,
                           const char* kind) -> bool {
        if (args.export_prefix.empty()) return false;
        std::string path = args.export_prefix + "." + suffix + ".csv";
        if (!e.open(path)) return false;
        e.meta("kind",         "%s", kind);
        e.meta("started",      "%s", iso_timestamp_now().c_str());
        e.meta("model",        "%s", csv_escape(args.model).c_str());
        e.meta("delegate",     "%s", args.no_delegate ? "cpu" : "npu");
        if (!args.no_delegate)
            e.meta("delegate_path", "%s",
                   csv_escape(args.delegate).c_str());
        e.meta("threads",      "%d", args.threads);
        e.meta("input_shape",  "%s",
               shape_to_str(in_info[0].shape).c_str());
        e.meta("input_dtype",  "%s", dtype_name(in_info[0].dtype));
        std::printf("CSV экспорт (%s): %s\n", kind, e.path().c_str());
        return true;
    };

    // Открыть (один раз) <prefix>.sysmon.csv с шапкой колонок per-sample.
    // Возвращает true, если файл готов к writef. Если --export не задан
    // или sysmon не инициализирован, экспорт молча отключён.
    auto ensure_sysmon_csv = [&]() -> bool {
        if (!args.sysmon || !sysmon.initialized()) return false;
        if (sysmon_csv_opened) return sysmon_csv.is_open();
        sysmon_csv_opened = true;
        if (!open_export(sysmon_csv, "sysmon", "sysmon")) return false;
        sysmon_csv.meta("interval_ms", "%d", args.sysmon_interval_ms);
        sysmon_csv.meta("num_cpus",    "%d", sysmon.num_cpus());
        sysmon_csv.header(
            "t_ms,phase,cpu_proc_pct,sys_cpu_pct,"
            "rss_kb,vsz_kb,peak_rss_kb,swap_kb,threads,"
            "mem_total_kb,mem_avail_kb,wall_ms");
        return true;
    };

    // Финальная сводка sysmon: один последний семпл + строка-агрегат в
    // stdout и в <prefix>.sysmon.summary.csv. Идемпотентен — повторные
    // вызовы перезатрут .summary тем же содержимым; на практике каждая
    // ветка main вызывает её ровно один раз перед своим return.
    bool sysmon_finalized = false;
    auto sysmon_finalize = [&]() {
        if (sysmon_finalized) return;
        sysmon_finalized = true;
        if (!args.sysmon || !sysmon.initialized()) return;
        // последний семпл захватываем тут же, чтобы попал в агрегат
        SysSample s = sysmon.sample();
        if (s.ok) {
            sysmon_acc.add(s);
            std::printf("[sysmon final   ] %s\n", sysmon_format(s).c_str());
            if (sysmon_csv.is_open()) {
                sysmon_csv.writef(
                    "%.3f,%s,%.3f,%.3f,"
                    "%ld,%ld,%ld,%ld,%d,"
                    "%ld,%ld,%.3f",
                    now_ms() - sysmon_t0, "final",
                    s.cpu_proc_pct, s.sys_cpu_pct,
                    s.rss_kb, s.vsz_kb, s.peak_rss_kb, s.swap_kb, s.threads,
                    s.mem_total_kb, s.mem_avail_kb, s.wall_ms);
            }
        }
        if (sysmon_acc.empty()) return;
        std::printf(
            "[sysmon summary] samples=%d  CPU proc avg=%.1f%% "
            "max=%.1f%%  sys avg=%.1f%% max=%.1f%%  RSS max=%ld kB  "
            "VmHWM=%ld kB  threads max=%d\n",
            sysmon_acc.n,
            sysmon_acc.cpu_proc_avg(), sysmon_acc.cpu_proc_max,
            sysmon_acc.sys_cpu_avg(),  sysmon_acc.sys_cpu_max,
            sysmon_acc.rss_max, sysmon_acc.peak_rss,
            sysmon_acc.threads_max);
        if (!args.export_prefix.empty()) {
            CsvExport ss;
            if (open_export(ss, "sysmon.summary", "sysmon_summary")) {
                ss.meta("interval_ms", "%d", args.sysmon_interval_ms);
                ss.meta("num_cpus",    "%d", sysmon.num_cpus());
                ss.header(
                    "samples,cpu_proc_avg_pct,cpu_proc_max_pct,"
                    "sys_cpu_avg_pct,sys_cpu_max_pct,"
                    "rss_max_kb,vsz_max_kb,peak_rss_kb,threads_max");
                ss.writef("%d,%.3f,%.3f,%.3f,%.3f,%ld,%ld,%ld,%d",
                          sysmon_acc.n,
                          sysmon_acc.cpu_proc_avg(), sysmon_acc.cpu_proc_max,
                          sysmon_acc.sys_cpu_avg(),  sysmon_acc.sys_cpu_max,
                          sysmon_acc.rss_max, sysmon_acc.vsz_max,
                          sysmon_acc.peak_rss, sysmon_acc.threads_max);
            }
        }
        if (sysmon_csv.is_open()) sysmon_csv.flush();
    };

    // Один комплексный шаг: снять семпл, добавить в агрегат, напечатать
    // строку в stdout и (если включён --export) — в sysmon.csv.
    // phase — короткая метка фазы прогона (init/warmup/bench/video/final).
    auto sysmon_log = [&](const char* phase) {
        if (!args.sysmon || !sysmon.initialized()) return;
        SysSample s = sysmon.sample();
        if (!s.ok) return;
        sysmon_acc.add(s);
        std::printf("[sysmon %-8s] %s\n", phase, sysmon_format(s).c_str());
        std::fflush(stdout);
        if (ensure_sysmon_csv()) {
            sysmon_csv.writef(
                "%.3f,%s,%.3f,%.3f,"
                "%ld,%ld,%ld,%ld,%d,"
                "%ld,%ld,%.3f",
                now_ms() - sysmon_t0, phase,
                s.cpu_proc_pct, s.sys_cpu_pct,
                s.rss_kb, s.vsz_kb, s.peak_rss_kb, s.swap_kb, s.threads,
                s.mem_total_kb, s.mem_avail_kb, s.wall_ms);
        }
    };
    // NHWC [1,H,W,3] — типовой формат для путей с изображением. Для чисто
    // рандомного --compare-random разрешаем любой shape (например [1,9,9,1]).
    const auto& s = in_info[0].shape;
    const bool nhwc3 = is_nhwc3(s);
    int in_h = 0, in_w = 0;
    if (nhwc3) { in_h = s[1]; in_w = s[2]; }
    if (has_image && !nhwc3) {
        std::fprintf(stderr,
            "Для путей с изображением поддерживается только NHWC [1,H,W,3]. "
            "Получено shape=%s.\n", shape_to_str(s).c_str());
        return 3;
    }

    // ---- Загрузка и preprocess изображения / генерация случайного входа ----
    // Возможные сценарии:
    //   * --camera                  — вход формируется в цикле захвата,
    //                                 здесь только проверяем параметры;
    //   * есть картинка             — letterbox + fill_input;
    //   * --random-input без картинки — генерируем случайный uint8-буфер
    //     по shape входа и заливаем его (один раз; benchmark/loop не
    //     перегенерируют — для замера скорости содержимое не важно).
    Image img;
    std::vector<uint8_t> input_rgb;
    TfLiteTensor* in_t = eng.interpreter->tensor(in_info[0].index);
    double t0 = 0.0, t1 = 0.0;
    if (has_camera) {
        // Для камеры ждём первый кадр уже в основном цикле — здесь лишь
        // проверяем, что вход модели совместим (NHWC [1,H,W,3]).
        if (!nhwc3) {
            std::fprintf(stderr,
                "Для --camera нужен вход NHWC [1,H,W,3], получено %s.\n",
                shape_to_str(s).c_str());
            return 4;
        }
        std::printf("Камера: устройство=%s, запрошено %dx%d @ %d fps\n",
                    args.camera_device.c_str(),
                    args.camera_w, args.camera_h, args.camera_fps);
    } else if (has_image) {
        if (!load_image(args.image, img)) return 4;
        std::printf("Изображение: %s  %dx%d -> letterbox %dx%d\n",
                    args.image.c_str(), img.w, img.h, in_w, in_h);

        letterbox(img, in_w, in_h, input_rgb);
        if (!fill_input(input_rgb, in_info[0], in_t)) return 5;
    } else {
        // --random-input без картинки: формируем случайный буфер длиной
        // numel(shape). fill_input трактует его плоско, так что для любого
        // ранга/раскладки результат корректен.
        const size_t n_in = numel(in_info[0].shape);
        if (n_in == 0) {
            std::fprintf(stderr,
                "Динамический shape входа не поддерживается в --random-input.\n");
            return 4;
        }
        const uint32_t seed = args.random_seed
            ? args.random_seed
            : (uint32_t)std::random_device{}();
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 255);
        input_rgb.resize(n_in);
        for (auto& v : input_rgb) v = (uint8_t)dist(rng);
        if (!fill_input(input_rgb, in_info[0], in_t)) return 5;
        std::printf("Случайный вход: shape=%s, seed=%u\n",
                    shape_to_str(in_info[0].shape).c_str(), seed);
    }

    // Первый замер: фиксируем «старт» — память сразу после загрузки модели,
    // делегата и заливки входа. CPU-проценты будут посчитаны от init() до
    // этой точки, поэтому отражают активность во время AllocateTensors /
    // подготовки делегата.
    sysmon_log("init");

    // ---- YOLO: предварительная подготовка ----
    // Детектируем голову один раз (формат [1, C, A] vs [1, A, C]) до
    // первого invoke, чтобы в видео-цикле не делать это каждый кадр.
    YoloHead yolo_head;
    YoloPostOptions yolo_opts;
    if (args.yolo) {
        if (!has_image && !has_camera) {
            std::fprintf(stderr,
                "--yolo требует источник кадров (image или --camera).\n");
            return 6;
        }
        if (!detect_yolo_head(out_info, args.yolo_output, yolo_head)) return 6;
        yolo_opts.conf_thresh = args.conf;
        yolo_opts.iou_thresh  = args.iou;
        yolo_opts.max_dets    = args.max_dets;
        yolo_opts.normalized  = !args.yolo_pixel_coords;
        if (!args.classes_path.empty()) load_class_names(args.classes_path);
        std::printf("YOLO: выход[%d] layout=%s C=%d A=%d, conf=%.2f iou=%.2f%s\n",
                    yolo_head.output_index,
                    yolo_head.channels_first ? "[1,C,A]" : "[1,A,C]",
                    yolo_head.channels, yolo_head.num_anchors,
                    yolo_opts.conf_thresh, yolo_opts.iou_thresh,
                    yolo_opts.normalized ? " (норм. координаты)"
                                         : " (координаты в пикселях)");
    }

    // ---- Общие буферы постобработки YOLO ----
    // Один раз выделили — переиспользуется во всех путях (одиночный
    // инференс, видео-цикл, камера). Никаких per-frame аллокаций.
    std::vector<float>      yolo_dequant;
    std::vector<Detection>  dets;
    std::vector<DisplayBox> boxes;

    // Декодирование выхода YOLO для последнего инференса. Вызывается
    // из run_video_loop и из одиночного display-кадра. orig_w/orig_h —
    // размеры исходного кадра (для обратного letterbox-маппинга).
    auto run_yolo_postproc = [&](int orig_w, int orig_h) {
        if (!args.yolo) return;
        const TfLiteTensor* t = eng.interpreter->tensor(
            out_info[yolo_head.output_index].index);
        if (!dequantize_output(out_info[yolo_head.output_index], t,
                               yolo_dequant)) return;
        dets = decode_yolov8(yolo_dequant.data(),
                             yolo_head.channels, yolo_head.num_anchors,
                             yolo_head.channels_first,
                             in_w, in_h, yolo_opts);
        if (!args.show_input)
            scale_to_original(dets, in_w, in_h, orig_w, orig_h);
        boxes = detections_to_boxes(dets);
    };

    // ---- Унифицированный видео-цикл ----
    // Один цикл на все три сценария:
    //   * --camera [+ --display]       — src=CameraFrameSource;
    //   * image + --display + --loop   — src=StaticFrameSource;
    //   * --loop без --display и без камеры — src=nullptr (просто
    //     гоняем invoke, чтобы померить throughput NPU).
    // disp=nullptr, когда окно не нужно.
    //
    // Шаги одной итерации (всегда одинаковы, отсутствующие просто
    // пропускаются):
    //   1. poll() окна          — выйти, если пользователь закрыл;
    //   2. next() источника     — получить кадр и его размеры;
    //   3. (опц.) препроцесс    — letterbox + fill_input для живого
    //                              источника; для статического вход
    //                              уже заполнен снаружи;
    //   4. invoke               — инференс;
    //   5. fps.tick();
    //   6. (опц.) YOLO postproc — декодирование боксов в координатах
    //                              текущего кадра;
    //   7. (опц.) stats         — оверлей FPS на окно + периодический
    //                              лог в stdout;
    //   8. (опц.) show_rgb      — отрисовка кадра в окне (через
    //                              glTexSubImage2D, без realloc).
    auto run_video_loop = [&](FrameSource* src, Display* disp) -> int {
        std::signal(SIGINT, on_sigint);
        FpsCounter fps;
        std::printf("Видео-цикл запущен. Ctrl+C%s для выхода.\n",
                    disp ? " или закрытие окна" : "");

        // CSV-экспорт FPS-семплов. Открывается только при --export.
        // Пишем по одной строке за каждый интервал log-interval_ms,
        // независимо от --stats (stats влияет только на оверлей и
        // лог в stdout). Колонки — те же агрегаты, что и в format(),
        // плюс счётчик кадров для удобной перепроверки.
        CsvExport fps_csv;
        const bool dump_fps = open_export(fps_csv, "fps", "video_loop");
        if (dump_fps) {
            fps_csv.meta("log_interval_ms", "%d", args.log_interval_ms);
            fps_csv.meta("vsync",           "%s",
                         args.vsync ? "on" : "off");
            fps_csv.meta("source",          "%s",
                         src ? (has_camera ? "camera" : "static")
                             : "none");
            fps_csv.header("t_ms,frame,fps,dt_avg_ms,dt_min_ms,"
                           "dt_max_ms,jitter_ms");
        }
        const double loop_t0 = now_ms();
        std::size_t  frame_n = 0;
        // Отдельный таймер семплирования sysmon — независим от FPS-лога,
        // чтобы пользователь мог поставить, например, 5 с интервал sysmon
        // при ежесекундном FPS-логе. last_sysmon_t = -inf при старте,
        // чтобы первый семпл сработал почти сразу (после первого кадра).
        double last_sysmon_t = -1e18;

        // orig_w/orig_h — размеры «исходного» кадра. Если источника
        // нет, для YOLO принимаем их равными размерам входа модели —
        // scale_to_original в этом случае становится тождественным.
        int orig_w = in_w, orig_h = in_h;

        while (!g_interrupted && (!disp || disp->poll())) {
            const uint8_t* frame = nullptr;
            if (src) {
                bool needs_preprocess = false;
                int  fw = 0, fh = 0;
                frame = src->next(fw, fh, needs_preprocess);
                if (!frame) {
                    // Источник не отдал кадр (таймаут камеры / skip) —
                    // даём циклу шанс заметить SIGINT и идём дальше.
                    if (g_interrupted) break;
                    continue;
                }
                orig_w = fw;
                orig_h = fh;
                if (needs_preprocess) {
                    letterbox(frame, fw, fh, in_w, in_h, input_rgb);
                    if (!fill_input(input_rgb, in_info[0], in_t)) break;
                }
            }

            if (!eng.invoke()) {
                std::fprintf(stderr, "Invoke упал.\n");
                break;
            }
            fps.tick();
            ++frame_n;

            if (args.yolo) {
                run_yolo_postproc(orig_w, orig_h);
                if (disp) disp->set_boxes(boxes);
            }

            // log_due() надо звать всегда, когда есть хоть один потребитель
            // периодического тика (stdout-лог под --stats или CSV-экспорт),
            // иначе таймер просто не сработает.
            const bool want_periodic = args.stats || dump_fps;
            const bool periodic = want_periodic
                && fps.log_due(args.log_interval_ms);
            if (args.stats) {
                std::string overlay = fps.format();
                if (disp) disp->set_overlay_text(overlay.c_str());
                if (periodic) {
                    std::printf("[%s] %s\n", label_main, overlay.c_str());
                    std::fflush(stdout);
                }
            }
            if (dump_fps && periodic && !fps.empty()) {
                fps_csv.writef("%.3f,%zu,%.3f,%.4f,%.4f,%.4f,%.4f",
                               now_ms() - loop_t0, frame_n,
                               fps.fps(), fps.avg_ms(),
                               fps.min_ms(), fps.max_ms(),
                               fps.jitter_ms());
            }
            // Семплирование sysmon в видео-цикле: раз в sysmon_interval_ms.
            // Семплим даже без --stats — нагрузка на CPU/память актуальна
            // и для чистого --camera/--loop без оверлея.
            if (args.sysmon && sysmon.initialized()) {
                const double now = now_ms();
                if (now - last_sysmon_t >= (double)args.sysmon_interval_ms) {
                    sysmon_log("video");
                    last_sysmon_t = now;
                }
            }

            if (disp && frame) {
                const uint8_t* show = args.show_input ? input_rgb.data()
                                                      : frame;
                const int sw = args.show_input ? in_w : orig_w;
                const int sh = args.show_input ? in_h : orig_h;
                if (!disp->show_rgb(show, sw, sh)) break;
            }
        }
        if (dump_fps) fps_csv.flush();

        // ---- FPS summary ----
        // Одна строка с агрегатами за весь видео-цикл:
        //   * frames / wall_s / avg_fps — lifetime (точно по счётчику
        //     кадров и стенным часам, не по окну);
        //   * dt_*  / jitter           — текущее состояние FpsCounter
        //     (скользящее окно 120 кадров) — характеризует стабильность
        //     к моменту остановки. Для коротких прогонов окно = весь
        //     прогон, разницы нет.
        // Файл пишется только если был хоть один кадр — иначе таблица
        // была бы из NaN’ов.
        if (args.export_prefix.size() && frame_n > 0) {
            CsvExport fs;
            if (open_export(fs, "fps.summary", "video_loop_summary")) {
                fs.meta("source", "%s",
                        src ? (has_camera ? "camera" : "static") : "none");
                fs.meta("vsync",  "%s", args.vsync ? "on" : "off");
                fs.header("frames,wall_s,avg_fps,"
                          "dt_avg_ms,dt_min_ms,dt_max_ms,jitter_ms");
                const double wall_s = (now_ms() - loop_t0) / 1000.0;
                const double avg_fps = wall_s > 0.0
                    ? (double)frame_n / wall_s : 0.0;
                fs.writef("%zu,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f",
                          frame_n, wall_s, avg_fps,
                          fps.avg_ms(), fps.min_ms(), fps.max_ms(),
                          fps.jitter_ms());
            }
        }
        std::printf("\nОстановлено.\n");
        return 0;
    };

    // ---- Камера ----
    // Открывает V4L2-камеру и (опционально) окно, после чего отдаёт
    // управление общему run_video_loop. Заменяет собой одиночный
    // инференс / --benchmark / --loop / --compare* при наличии --camera.
    if (has_camera) {
        if (args.benchmark || args.loop || args.compare_cpu
            || !args.compare_model.empty() || args.random_input) {
            std::fprintf(stderr,
                "Внимание: --benchmark/--loop/--compare*/--random-input "
                "игнорируются в режиме --camera (используется общий "
                "видео-цикл).\n");
        }

        auto cam = make_camera();
        if (!cam) {
            std::fprintf(stderr,
                "Поддержка камеры не собрана (USE_CAMERA=OFF) или "
                "недоступна на этой платформе.\n");
            return 7;
        }
        if (!cam->open(args.camera_device,
                       args.camera_w, args.camera_h, args.camera_fps)) {
            return 7;
        }

        std::unique_ptr<Display> disp;
        if (args.display) {
            disp = make_display();
            if (!disp) {
                std::fprintf(stderr,
                    "Поддержка дисплея не собрана (USE_DISPLAY=OFF).\n");
                return 7;
            }
            if (!disp->init(args.win_w, args.win_h, "npu", args.vsync))
                return 7;
        }

        CameraFrameSource src(*cam, /*timeout_ms=*/1000);
        int rc = run_video_loop(&src, disp.get());
        sysmon_finalize();
        return rc;
    }

    // ---- Один прогон ----
    // Делаем как только в input_t уже что-то залито (картинка или random).
    // В чистом --compare-random режиме вход для сравнения зальётся внутри
    // run_compare, поэтому одиночный прогон тут пропускаем. С камерой
    // одиночного прогона нет вовсе — кадры идут в собственном цикле ниже.
    const bool do_single_invoke = !has_camera && (has_image
        || (args.random_input && !args.compare_cpu && args.compare_model.empty()));
    if (do_single_invoke) {
        t0 = now_ms();
        if (!eng.invoke()) { std::fprintf(stderr, "Invoke упал.\n"); return 6; }
        t1 = now_ms();
        std::printf("Инференс: %.3f мс\n", t1 - t0);

        for (size_t i = 0; i < out_info.size(); ++i) {
            const TfLiteTensor* t = eng.interpreter->tensor(out_info[i].index);
            print_output_head(out_info[i], t);
        }
        sysmon_log("invoke");
    }

    // ---- Сравнение точности с эталоном ----
    // Сравниваем выходы текущей модели с выходами эталонной (на CPU):
    //   * --compare-cpu       — та же .tflite, но без делегата;
    //   * --compare <path>    — указанная .tflite (например, исходная
    //                          float32 версия той же сети).
    // Метрики (MAE / RMSE / max|diff|) считаются после деквантования в
    // float, поэлементно. Если у выходов разные размеры — сравнивается
    // общий префикс длиной min(size_a, size_b) и печатается пометка.
    //
    // Дополнительно, если основной выход квантован (scale > 0), печатаем
    // те же метрики «в квантах INT8» — Δ/scale. Это даёт интуитивную
    // шкалу: 1 квант = минимально различимое значение этой модели,
    // т.е. предел разрешения INT8. MAE < 1·q считается идеальным
    // совпадением (лучше уже не передать восемью битами).
    // CSV-экспорт результатов сравнения. Один файл на оба возможных
    // вызова (--compare и --compare-cpu) — различаем их по колонке `ref`.
    // Открываем заранее, чтобы run_compare мог свободно писать.
    //
    // Пара файлов:
    //   * compare.csv         — per-run × per-output diff (сырые данные);
    //   * compare.summary.csv — агрегаты по всем прогонам (одна строка
    //                            на пару (ref, output) с MAE/RMSE/max|diff|
    //                            и теми же метриками в «квантах» INT8).
    CsvExport compare_csv;
    CsvExport compare_summary_csv;
    const bool will_compare = !args.compare_model.empty() || args.compare_cpu;
    const bool dump_compare = will_compare
        && open_export(compare_csv, "compare", "compare");
    if (dump_compare) {
        compare_csv.meta("random_input", "%s",
                         args.random_input ? "1" : "0");
        compare_csv.meta("random_runs",  "%d", args.random_runs);
        compare_csv.header(
            "ref,run,output_idx,output_name,elems,"
            "mae,rmse,max_abs_diff,scale,mae_q,rmse_q,maxd_q,trimmed");
    }
    const bool dump_compare_sum = will_compare
        && open_export(compare_summary_csv, "compare.summary",
                       "compare_summary");
    if (dump_compare_sum) {
        compare_summary_csv.meta("random_input", "%s",
                                 args.random_input ? "1" : "0");
        compare_summary_csv.meta("random_runs",  "%d", args.random_runs);
        compare_summary_csv.header(
            "ref,output_idx,output_name,runs,elems,"
            "mae,rmse,max_abs_diff,scale,mae_q,rmse_q,maxd_q,trimmed");
    }

    auto run_compare = [&](const std::string& ref_model_path,
                           const char* ref_label) {
        std::printf("\n=== Сравнение %s vs %s (%s) ===\n",
                    label_main, ref_label, ref_model_path.c_str());
        Engine ref;
        if (!ref.load(ref_model_path, std::string{}, args.threads)) return;

        std::vector<TensorInfo> ref_in, ref_out;
        for (int idx : ref.interpreter->inputs())
            ref_in.push_back(describe(ref.interpreter->tensor(idx), idx));
        for (int idx : ref.interpreter->outputs())
            ref_out.push_back(describe(ref.interpreter->tensor(idx), idx));

        if (ref_in.size() != 1) {
            std::fprintf(stderr,
                "[%s] ожидаю единственный вход, у модели %zu; пропуск.\n",
                ref_label, ref_in.size());
            return;
        }
        const auto& rs = ref_in[0].shape;
        const bool ref_nhwc3 = is_nhwc3(rs);
        int ref_h = 0, ref_w = 0;
        if (ref_nhwc3) { ref_h = rs[1]; ref_w = rs[2]; }
        // Жёстко требуем [1,H,W,3] только в image-режиме compare: там мы
        // letterbox’им реальную картинку в ref. В --compare-random никаких
        // картинок нет — сравниваем тензор-в-тензор любого ранга.
        if (has_image && !ref_nhwc3) {
            std::fprintf(stderr,
                "[%s] для сравнения по изображению ожидаю вход NHWC [1,H,W,3]; "
                "получено shape=%s; пропуск.\n",
                ref_label, shape_to_str(rs).c_str());
            return;
        }
        TfLiteTensor* main_in_t = eng.interpreter->tensor(in_info[0].index);
        TfLiteTensor* ref_in_t  = ref.interpreter->tensor(ref_in[0].index);

        const size_t n_out = std::min(out_info.size(), ref_out.size());
        if (out_info.size() != ref_out.size()) {
            std::fprintf(stderr,
                "[%s] число выходов различается (%zu vs %zu); "
                "сравниваю первые %zu по индексу.\n",
                ref_label, out_info.size(), ref_out.size(), n_out);
        }

        // Аккумуляторы метрик по каждому выходу, агрегированные по всем
        // прогонам. В режиме одиночного запуска цифры совпадают со старым
        // поведением; в режиме --random-input N просто усредняются по
        // большему числу элементов (все runs × elems_per_run).
        struct Acc { double sum = 0, sse = 0, maxd = 0; size_t n = 0;
                     bool   trimmed = false; };
        std::vector<Acc> acc(n_out);

        // Одна итерация сравнения: оба интерпретатора уже должны быть
        // заполнены входами. Делает invoke обоих и аккумулирует поэлементные
        // расхождения. Возвращает false при ошибке инференса (фатально).
        // run_idx используется только для CSV-экспорта (per-run строки),
        // на агрегатную статистику не влияет.
        auto compare_once = [&](int run_idx) -> bool {
            if (!eng.invoke()) {
                std::fprintf(stderr, "[main] Invoke упал.\n");
                return false;
            }
            if (!ref.invoke()) {
                std::fprintf(stderr, "[%s] Invoke упал.\n", ref_label);
                return false;
            }
            std::vector<float> a, b;
            for (size_t i = 0; i < n_out; ++i) {
                const TfLiteTensor* ta =
                    eng.interpreter->tensor(out_info[i].index);
                const TfLiteTensor* tb =
                    ref.interpreter->tensor(ref_out[i].index);
                if (!dequantize_output(out_info[i], ta, a)) continue;
                if (!dequantize_output(ref_out[i], tb, b)) continue;
                const size_t m = std::min(a.size(), b.size());
                if (m == 0) continue;
                const bool trimmed_now = (a.size() != b.size());
                if (trimmed_now) acc[i].trimmed = true;
                // Локальные аккумуляторы для CSV-строки этой итерации.
                // Глобальный acc[i] обновляем тут же — двойной проход
                // по данным был бы лишним.
                double r_sum = 0.0, r_sse = 0.0, r_max = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    double d = std::fabs((double)a[j] - (double)b[j]);
                    r_sum += d;
                    r_sse += d * d;
                    if (d > r_max) r_max = d;
                }
                acc[i].sum  += r_sum;
                acc[i].sse  += r_sse;
                if (r_max > acc[i].maxd) acc[i].maxd = r_max;
                acc[i].n    += m;

                if (dump_compare) {
                    const double mae  = r_sum / (double)m;
                    const double rmse = std::sqrt(r_sse / (double)m);
                    const float  s    = out_info[i].scale;
                    const double inv_s = (s > 0.0f)
                        ? 1.0 / (double)s : 0.0;
                    compare_csv.writef(
                        "%s,%d,%zu,%s,%zu,"
                        "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d",
                        ref_label, run_idx, i,
                        csv_escape(out_info[i].name).c_str(), m,
                        mae, rmse, r_max, (double)s,
                        mae  * inv_s,
                        rmse * inv_s,
                        r_max * inv_s,
                        trimmed_now ? 1 : 0);
                }
            }
            return true;
        };

        int runs_done = 0;
        if (args.random_input) {
            // ---- Режим случайных входов ----
            // Каждая итерация: один и тот же случайный буфер uint8 [0..255]
            // заливается в обе сети через fill_input (нормализация в [0,1]
            // и квантование берутся из самой модели — каждая сеть видит
            // свою «честную» подачу). Работает для произвольных шейпов:
            // картинок [1,H,W,3], таблиц [1,9,9,1], векторов [1,N] и т.д.
            const uint32_t seed = args.random_seed
                ? args.random_seed
                : (uint32_t)std::random_device{}();
            std::mt19937 rng(seed);
            std::uniform_int_distribution<int> dist(0, 255);

            const size_t main_n   = numel(in_info[0].shape);
            const size_t ref_n    = numel(rs);
            const bool same_shape = (in_info[0].shape == rs);
            const bool same_numel = (main_n == ref_n);
            const bool both_img   = nhwc3 && ref_nhwc3;

            const char* mode_note;
            if      (same_shape) mode_note = " (общий буфер)";
            else if (same_numel) mode_note = " (разный shape, тот же numel — общий буфер)";
            else if (both_img)   mode_note = " (letterbox под ref)";
            else                 mode_note = " (независимые случайные входы — diff не сопоставим)";

            std::printf("Случайные входы: runs=%d, seed=%u, main=%s, ref=%s%s\n",
                        args.random_runs, seed,
                        shape_to_str(in_info[0].shape).c_str(),
                        shape_to_str(rs).c_str(), mode_note);

            if (main_n == 0 || ref_n == 0) {
                std::fprintf(stderr,
                    "[%s] неподдерживаемый динамический shape входа; пропуск.\n",
                    ref_label);
                return;
            }

            std::vector<uint8_t> rnd_main(main_n);
            std::vector<uint8_t> rnd_ref;
            for (int r = 0; r < args.random_runs && !g_interrupted; ++r) {
                for (auto& v : rnd_main) v = (uint8_t)dist(rng);
                if (!fill_input(rnd_main, in_info[0], main_in_t)) return;
                if (same_shape || same_numel) {
                    // fill_input трактует входной буфер плоско — при равном
                    // числе элементов раскладка по осям не важна.
                    if (!fill_input(rnd_main, ref_in[0], ref_in_t)) return;
                } else if (both_img) {
                    Image tmp; tmp.rgb = rnd_main; tmp.w = in_w; tmp.h = in_h;
                    letterbox(tmp, ref_w, ref_h, rnd_ref);
                    if (!fill_input(rnd_ref, ref_in[0], ref_in_t)) return;
                } else {
                    // Шейпы и numel разные, картинками не являются — других
                    // разумных вариантов нет. Льём независимый рандом, метрики
                    // diff’а интерпретировать осторожно (пометили в логе).
                    rnd_ref.resize(ref_n);
                    for (auto& v : rnd_ref) v = (uint8_t)dist(rng);
                    if (!fill_input(rnd_ref, ref_in[0], ref_in_t)) return;
                }
                if (!compare_once(r)) return;
                ++runs_done;
            }
        } else {
            // ---- Режим одного изображения (старое поведение) ----
            if (!has_image) {
                std::fprintf(stderr,
                    "[%s] изображения нет; используйте --random-input.\n",
                    ref_label);
                return;
            }
            // main_in_t уже заполнен исходным letterbox’ом во внешнем коде —
            // не перезаливаем, чтобы избежать лишней работы. Заливаем только
            // ref (его размер может отличаться).
            std::vector<uint8_t> ref_rgb;
            letterbox(img, ref_w, ref_h, ref_rgb);
            if (!fill_input(ref_rgb, ref_in[0], ref_in_t)) return;
            if (!compare_once(0)) return;
            ++runs_done;
        }

        if (runs_done == 0) {
            std::fprintf(stderr, "[%s] не выполнено ни одного прогона.\n",
                         ref_label);
            return;
        }
        if (runs_done > 1) {
            std::printf("Прогонов выполнено: %d (метрики усреднены по всем)\n",
                        runs_done);
        }

        double agg_mae_sum = 0.0, agg_max = 0.0;
        size_t agg_n = 0;
        for (size_t i = 0; i < n_out; ++i) {
            if (acc[i].n == 0) continue;
            const double mae  = acc[i].sum / (double)acc[i].n;
            const double rmse = std::sqrt(acc[i].sse / (double)acc[i].n);
            const char* note  = acc[i].trimmed ? "  (срез по min длине)" : "";
            std::printf("  output[%zu] %-32s elems=%zu  "
                        "MAE=%.6f  RMSE=%.6f  max|diff|=%.6f%s\n",
                        i, out_info[i].name.c_str(), acc[i].n,
                        mae, rmse, acc[i].maxd, note);
            const float s_main = out_info[i].scale;
            if (s_main > 0.0f) {
                std::printf("                                              "
                            "scale=%.6g  ->  MAE=%.2f*q  RMSE=%.2f*q  "
                            "max|diff|=%.2f*q\n",
                            s_main,
                            mae        / s_main,
                            rmse       / s_main,
                            acc[i].maxd/ s_main);
            }
            if (dump_compare_sum) {
                const double inv_s = (s_main > 0.0f)
                    ? 1.0 / (double)s_main : 0.0;
                compare_summary_csv.writef(
                    "%s,%zu,%s,%d,%zu,"
                    "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d",
                    ref_label, i,
                    csv_escape(out_info[i].name).c_str(),
                    runs_done, acc[i].n,
                    mae, rmse, acc[i].maxd, (double)s_main,
                    mae         * inv_s,
                    rmse        * inv_s,
                    acc[i].maxd * inv_s,
                    acc[i].trimmed ? 1 : 0);
            }
            agg_mae_sum += acc[i].sum;
            if (acc[i].maxd > agg_max) agg_max = acc[i].maxd;
            agg_n       += acc[i].n;
        }
        if (agg_n > 0) {
            std::printf("  -> суммарно: MAE=%.6f  max|diff|=%.6f  (%zu элементов)\n",
                        agg_mae_sum / (double)agg_n, agg_max, agg_n);
        }
        if (dump_compare)     compare_csv.flush();
        if (dump_compare_sum) compare_summary_csv.flush();
    };

    if (!args.compare_model.empty()) {
        run_compare(args.compare_model, "REF");
    }
    if (args.compare_cpu) {
        if (args.no_delegate) {
            std::fprintf(stderr,
                "--compare-cpu проигнорирован: основной запуск уже на CPU.\n");
        } else {
            run_compare(args.model, "CPU");
        }
    }

    // Если входа нет вообще — дальше идти нечего. В режиме --random-input
    // вход уже залит случайным буфером выше, так что --benchmark / --loop
    // (без --display) могут отработать. --display и YOLO-постпроцессинг
    // требуют реальной картинки и тут пропускаются.
    if (!has_image) {
        if (args.yolo) {
            std::fprintf(stderr,
                "--yolo требует входное изображение, пропуск постобработки.\n");
        }
        if (args.display) {
            // Сюда не должны попасть из-за проверки в начале main, но на
            // всякий случай.
            return 0;
        }
    }

    // ---- Бенчмарк ----
    // Без --export меряем только агрегат (один вызов now_ms() на N
    // итераций — минимум накладных). С --export таймим каждую
    // итерацию отдельно, чтобы получить распределение латентности
    // (для гистограмм / квантилей p50/p95/p99 / поиска выбросов).
    // Накладные ~50–100 нс на now_ms() ничтожны на фоне инференса.
    if (args.benchmark) {
        for (int i = 0; i < args.warmup; ++i) eng.invoke();
        // После прогрева — фиксируем CPU/память: на этом интервале
        // прогревочные invoke’ы уже отработали, кэши/JIT делегата
        // прогрелись, а сам бенчмарк ещё не начался.
        sysmon_log("warmup");

        CsvExport bench_csv;
        const bool dump_bench = open_export(bench_csv, "bench", "benchmark");
        if (dump_bench) {
            bench_csv.meta("warmup", "%d", args.warmup);
            bench_csv.meta("runs",   "%d", args.runs);
            bench_csv.header("run,ms");
        }

        double sum = 0.0, mn = 1e18, mx = 0.0;
        // При активном экспорте дополнительно копим полный массив
        // латентностей — нужен для перцентилей в summary. 1000 double’ов
        // = 8 КБ, на пайплайн не влияет.
        std::vector<double> lat;
        if (dump_bench) {
            lat.reserve((size_t)args.runs);
            for (int i = 0; i < args.runs; ++i) {
                double t0b = now_ms();
                eng.invoke();
                double dt = now_ms() - t0b;
                sum += dt;
                if (dt < mn) mn = dt;
                if (dt > mx) mx = dt;
                lat.push_back(dt);
                bench_csv.writef("%d,%.6f", i, dt);
            }
            bench_csv.flush();
            double avg = sum / args.runs;
            std::printf("Бенчмарк: %d итераций, среднее %.3f мс "
                        "(min %.3f, max %.3f), %.1f инф/с\n",
                        args.runs, avg, mn, mx, 1000.0 / avg);
        } else {
            double s0 = now_ms();
            for (int i = 0; i < args.runs; ++i) eng.invoke();
            double s1 = now_ms();
            double avg = (s1 - s0) / args.runs;
            std::printf("Бенчмарк: %d итераций, среднее %.3f мс, %.1f инф/с\n",
                        args.runs, avg, 1000.0 / avg);
        }
        // CPU/память «во время бенчмарка»: дельта от warmup-семпла, что
        // соответствует ровно периоду runs итераций.
        sysmon_log("bench");

        // ---- Bench summary ----
        // Одна строка с агрегатами. Файл само-достаточен: все
        // мета-поля (модель, делегат, threads, runs) уже есть в шапке
        // через open_export, плюс перцентили p50/p95/p99 — для оценки
        // хвоста распределения (важнее среднего, когда есть выбросы).
        // Несколько прогонов разных моделей собираются в одну таблицу
        // тривиально:  cat results/*.bench.summary.csv | grep -v '^#'
        if (dump_bench && !lat.empty()) {
            CsvExport bs;
            if (open_export(bs, "bench.summary", "benchmark_summary")) {
                bs.meta("warmup", "%d", args.warmup);
                bs.meta("runs",   "%d", args.runs);
                bs.header(
                    "runs,avg_ms,min_ms,max_ms,"
                    "p50_ms,p95_ms,p99_ms,std_ms,throughput_ips");
                std::sort(lat.begin(), lat.end());
                auto pct = [&](double q) {
                    size_t k = (size_t)std::lround(
                        q * (double)(lat.size() - 1));
                    if (k >= lat.size()) k = lat.size() - 1;
                    return lat[k];
                };
                const double avg = sum / (double)lat.size();
                double sse = 0.0;
                for (double v : lat) sse += (v - avg) * (v - avg);
                const double stddev = std::sqrt(sse / (double)lat.size());
                bs.writef("%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f",
                          lat.size(), avg, mn, mx,
                          pct(0.50), pct(0.95), pct(0.99),
                          stddev, 1000.0 / avg);
            }
        }
    }

    // ---- Видео-цикл без окна ----
    // Источника кадров нет — общий цикл просто крутит invoke + FPS.
    // Полезно, чтобы померить чистую пропускную способность NPU.
    if (args.loop && !args.display) {
        int rc = run_video_loop(/*src=*/nullptr, /*disp=*/nullptr);
        sysmon_finalize();
        return rc;
    }

    // ---- Графический вывод (Wayland/EGL) ----
    // Показываем либо оригинальное изображение, либо то, что реально
    // ушло в сеть после letterbox’а (полезно для отладки препроцессинга).
    if (args.display) {
        auto disp = make_display();
        if (!disp) {
            std::fprintf(stderr,
                "Поддержка дисплея не собрана (USE_DISPLAY=OFF).\n");
            return 7;
        }
        if (!disp->init(args.win_w, args.win_h, "npu", args.vsync)) return 7;

        // Что показывать: оригинал картинки или препроцесс.
        const uint8_t* frame = args.show_input ? input_rgb.data()
                                               : img.rgb.data();
        const int fw = args.show_input ? in_w : img.w;
        const int fh = args.show_input ? in_h : img.h;

        // ---- Цикл с окном ----
        // Источник статический: вход модели уже залит в image-блоке
        // выше, поэтому StaticFrameSource просит цикл ничего не
        // препроцессить — только invoke + show_rgb на каждой итерации.
        if (args.loop) {
            StaticFrameSource src(frame, fw, fh);
            int rc = run_video_loop(&src, disp.get());
            sysmon_finalize();
            return rc;
        }

        // ---- Одиночный кадр ----
        if (args.stats) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "inference %.2f ms", t1 - t0);
            disp->set_overlay_text(buf);
        }
        if (args.yolo) {
            run_yolo_postproc(img.w, img.h);
            disp->set_boxes(boxes);
        }
        if (!disp->show_rgb(frame, fw, fh)) { sysmon_finalize(); return 0; }
        std::printf("Окно открыто. Закройте его, чтобы выйти.\n");
        // Один статический кадр: блокируемся на событиях Wayland до закрытия.
        while (disp->wait()) {}
    }

    sysmon_finalize();
    return 0;
}
