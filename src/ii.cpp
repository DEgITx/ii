// Универсальный раннер TFLite-моделей на NPU (внешний делегат).
//
// Цель: пропустить изображение через INT8-квантованную модель
// (например yolov8m_int8.tflite) и получить «сырые» выходы.
// YOLO-постобработка (NMS, декодирование боксов) сюда сознательно не
// включена — это базовый движок инференса, поверх которого можно
// строить любую сеть.
//
// Поддерживаются модели:
//   * один вход вида NHWC [1,H,W,3] (uint8 / int8 / float32);
//   * любое число выходов произвольных типов;
//   * квантование scale/zero_point читается из самой модели.
//
// Запуск (на устройстве):
//   ./ii yolov8m_int8.tflite image.jpg
//   ./ii model.tflite image.jpg --no-delegate
//   ./ii model.tflite image.jpg --benchmark --runs 100

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
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

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
        case kTfLiteInt8:    return "int8";
        case kTfLiteUInt8:   return "uint8";
        case kTfLiteInt32:   return "int32";
        case kTfLiteInt64:   return "int64";
        default:             return "?";
    }
}

size_t dtype_size(TfLiteType t) {
    switch (t) {
        case kTfLiteFloat32: return 4;
        case kTfLiteInt32:   return 4;
        case kTfLiteInt64:   return 8;
        case kTfLiteInt8:
        case kTfLiteUInt8:   return 1;
        default:             return 1;
    }
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
// Нормализуется в [0,1] и квантуется по scale/zero_point модели для INT8/UINT8.
// (Логика идентична quantize_to_input.)
// ---------------------------------------------------------------------------
bool fill_input(const std::vector<uint8_t>& rgb, const TensorInfo& info,
                TfLiteTensor* tensor) {
    const size_t n = rgb.size();
    switch (info.dtype) {
        case kTfLiteFloat32: {
            float* p = reinterpret_cast<float*>(tensor->data.data);
            for (size_t i = 0; i < n; ++i) p[i] = rgb[i] / 255.0f;
            return true;
        }
        case kTfLiteInt8: {
            int8_t* p = reinterpret_cast<int8_t*>(tensor->data.data);
            const float s = info.scale ? info.scale : 1.0f;
            const int zp = info.zero_point;
            for (size_t i = 0; i < n; ++i) {
                int q = (int)std::lround((rgb[i] / 255.0f) / s + zp);
                if (q < -128) q = -128;
                if (q >  127) q =  127;
                p[i] = (int8_t)q;
            }
            return true;
        }
        case kTfLiteUInt8: {
            uint8_t* p = reinterpret_cast<uint8_t*>(tensor->data.data);
            const float s = info.scale ? info.scale : 1.0f;
            const int zp = info.zero_point;
            for (size_t i = 0; i < n; ++i) {
                int q = (int)std::lround((rgb[i] / 255.0f) / s + zp);
                if (q <   0) q =   0;
                if (q > 255) q = 255;
                p[i] = (uint8_t)q;
            }
            return true;
        }
        default:
            std::fprintf(stderr, "Неподдерживаемый dtype входа: %s\n",
                         dtype_name(info.dtype));
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

    std::printf("  output %-32s first %d:", info.name.c_str(), n_show);
    for (int i = 0; i < n_show; ++i) {
        float v = 0.0f;
        switch (info.dtype) {
            case kTfLiteFloat32:
                v = reinterpret_cast<const float*>(tensor->data.data)[i];
                break;
            case kTfLiteInt8: {
                int8_t q = reinterpret_cast<const int8_t*>(tensor->data.data)[i];
                v = (q - info.zero_point) * info.scale;
                break;
            }
            case kTfLiteUInt8: {
                uint8_t q = reinterpret_cast<const uint8_t*>(tensor->data.data)[i];
                v = (q - info.zero_point) * info.scale;
                break;
            }
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
        "  --threads <N>       число CPU-потоков интерпретатора\n",
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

    return 0;
}
