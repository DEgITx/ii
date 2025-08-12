// Универсальный раннер TFLite-моделей на NPU (внешний делегат).
//
// Цель: пропустить изображение через INT8-квантованную модель
// (например yolov8m_int8.tflite) и получить «сырые» выходы.
//
// Поддерживаются модели:
//   * один вход вида NHWC [1,H,W,3] (uint8 / int8 / float32);
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
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

#include "display.h"
#include "stats.h"
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
void letterbox(const Image& src, int target_w, int target_h,
               std::vector<uint8_t>& dst, uint8_t pad = 114) {
    float r = std::min((float)target_w / src.w, (float)target_h / src.h);
    int new_w = (int)std::round(src.w * r);
    int new_h = (int)std::round(src.h * r);
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;

    std::vector<uint8_t> resized((size_t)new_w * new_h * 3);
    stbir_resize_uint8_linear(src.rgb.data(), src.w, src.h, 0,
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
};

void print_usage(const char* prog) {
    std::printf(
        "Usage: %s <model.tflite> <image> [options]\n"
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
        "  --classes <path>    файл имён классов (по строке; default = COCO80)\n",
        prog, kDefaultDelegate);
}

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 3) { print_usage(argv[0]); return false; }
    a.model = argv[1];
    a.image = argv[2];
    for (int i = 3; i < argc; ++i) {
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

    Engine eng;
    if (!eng.load(args.model,
                  args.no_delegate ? std::string{} : args.delegate,
                  args.threads)) return 2;

    // Тег для логов: где именно крутится инференс.
    const char* label_main = args.no_delegate ? "CPU" : "NPU";

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
            "Эта реализация рассчитана на 1 вход-изображение, у модели %zu.\n",
            in_info.size());
        return 3;
    }
    // Ожидаем NHWC [1,H,W,3] — типовой формат TFLite-конвертированных моделей.
    const auto& s = in_info[0].shape;
    if (s.size() != 4 || s[0] != 1 || s[3] != 3) {
        std::fprintf(stderr,
            "Поддерживается только NHWC [1,H,W,3]. Получено shape ранг=%zu.\n",
            s.size());
        return 3;
    }
    const int in_h = s[1];
    const int in_w = s[2];

    // ---- Загрузка и preprocess изображения ----
    Image img;
    if (!load_image(args.image, img)) return 4;
    std::printf("Изображение: %s  %dx%d -> letterbox %dx%d\n",
                args.image.c_str(), img.w, img.h, in_w, in_h);

    std::vector<uint8_t> input_rgb;
    letterbox(img, in_w, in_h, input_rgb);

    TfLiteTensor* in_t = eng.interpreter->tensor(in_info[0].index);
    if (!fill_input(input_rgb, in_info[0], in_t)) return 5;

    // ---- YOLO: предварительная подготовка ----
    // Детектируем голову один раз (формат [1, C, A] vs [1, A, C]) до
    // первого invoke, чтобы в видео-цикле не делать это каждый кадр.
    YoloHead yolo_head;
    YoloPostOptions yolo_opts;
    if (args.yolo) {
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

    // ---- Один прогон ----
    double t0 = now_ms();
    if (!eng.invoke()) { std::fprintf(stderr, "Invoke упал.\n"); return 6; }
    double t1 = now_ms();
    std::printf("Инференс: %.3f мс\n", t1 - t0);

    // ---- Сводка по выходам ----
    for (size_t i = 0; i < out_info.size(); ++i) {
        const TfLiteTensor* t = eng.interpreter->tensor(out_info[i].index);
        print_output_head(out_info[i], t);
    }

    // ---- YOLO: декодирование боксов после первого инференса ----
    // Локальные буферы переиспользуются в видео-цикле ниже, поэтому
    // объявляем их здесь (за условием на yolo, чтобы не платить за
    // ненужные аллокации, когда YOLO выключен).
    std::vector<float> yolo_dequant;
    std::vector<Detection> dets;
    std::vector<DisplayBox> boxes;
    auto run_yolo_postproc = [&]() {
        const TfLiteTensor* t =
            eng.interpreter->tensor(out_info[yolo_head.output_index].index);
        if (!dequantize_output(out_info[yolo_head.output_index], t,
                               yolo_dequant)) return;
        dets = decode_yolov8(yolo_dequant.data(),
                             yolo_head.channels, yolo_head.num_anchors,
                             yolo_head.channels_first,
                             in_w, in_h, yolo_opts);
        // Координаты возвращаются в letterbox-пространстве. Если на экран
        // идёт оригинал — отмаппим назад в его пиксели, иначе оставим как есть.
        if (!args.show_input)
            scale_to_original(dets, in_w, in_h, img.w, img.h);
        boxes = detections_to_boxes(dets);
    };

    // ---- Бенчмарк ----
    if (args.benchmark) {
        for (int i = 0; i < args.warmup; ++i) eng.invoke();
        double s0 = now_ms();
        for (int i = 0; i < args.runs; ++i) eng.invoke();
        double s1 = now_ms();
        double avg = (s1 - s0) / args.runs;
        std::printf("Бенчмарк: %d итераций, среднее %.3f мс, %.1f инф/с\n",
                    args.runs, avg, 1000.0 / avg);
    }

    // ---- Видео-цикл без окна (только инференс + статистика) ----
    // Полезно, чтобы померить FPS NPU без накладных расходов на отрисовку.
    if (args.loop && !args.display) {
        std::signal(SIGINT, on_sigint);
        FpsCounter fps;
        std::printf("Цикл инференса запущен. Ctrl+C для выхода.\n");
        while (!g_interrupted) {
            eng.invoke();
            fps.tick();
            if (args.stats && fps.log_due(args.log_interval_ms)) {
                std::printf("[%s] %s\n", label_main, fps.format().c_str());
                std::fflush(stdout);
            }
        }
        std::printf("\nОстановлено.\n");
        return 0;
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

        const uint8_t* frame = args.show_input ? input_rgb.data()
                                               : img.rgb.data();
        const int fw = args.show_input ? in_w : img.w;
        const int fh = args.show_input ? in_h : img.h;

        // ---- Видео-цикл с окном ----
        // Каждая итерация: инференс -> show_rgb -> tick FPS -> опц. лог.
        // Для реального видео замените блок eng.invoke() на захват кадра
        // + препроцессинг + invoke; show_rgb() оставьте — текстура будет
        // обновляться через glTexSubImage2D (без перевыделения).
        if (args.loop) {
            std::signal(SIGINT, on_sigint);
            FpsCounter fps;
            std::printf("Видео-цикл запущен. Ctrl+C или закрытие окна = выход.\n");
            while (!g_interrupted && disp->poll()) {
                eng.invoke();
                fps.tick();

                std::string overlay;
                if (args.stats) {
                    overlay = fps.format();
                    disp->set_overlay_text(overlay.c_str());
                    if (fps.log_due(args.log_interval_ms)) {
                        std::printf("[%s] %s\n", label_main, overlay.c_str());
                        std::fflush(stdout);
                    }
                }
                // YOLO постобработка на каждом кадре. Сам кадр (frame)
                // в --loop у нас статический, но детекции пересчитываем —
                // когда сюда добавите захват видео, координаты будут
                // обновляться вместе с входом без правок дисплея.
                if (args.yolo) {
                    run_yolo_postproc();
                    disp->set_boxes(boxes);
                }
                if (!disp->show_rgb(frame, fw, fh)) break;
            }
            std::printf("\nОстановлено.\n");
            return 0;
        }

        // ---- Одиночный кадр ----
        if (args.stats) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "inference %.2f ms", t1 - t0);
            disp->set_overlay_text(buf);
        }
        if (args.yolo) disp->set_boxes(boxes);
        if (!disp->show_rgb(frame, fw, fh)) return 0;
        std::printf("Окно открыто. Закройте его, чтобы выйти.\n");
        // Один статический кадр: блокируемся на событиях Wayland до закрытия.
        while (disp->wait()) {}
    }

    return 0;
}
