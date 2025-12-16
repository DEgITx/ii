// Реализация декодирования выходного тензора image-to-image моделей
// в RGB-кадр (+ PNG-запись).
//
// Все «как из тензора получить пиксели» собрано здесь и нигде больше,
// чтобы добавление нового семейства моделей (FSRCNN-варианты, разные
// enhance/denoise сети) не раздувало универсальный раннер ii.cpp.
//
// Зависимости: только inference.h (для dtype/half_to_float) и
// stb_image_write.h (FetchContent'ом тянется в CMakeLists.txt
// вместе с остальным stb). Никаких бэкенд-специфичных хедеров.

#include "image_proc.h"

#include <cmath>
#include <cstdio>

// STB_IMAGE_WRITE_IMPLEMENTATION должен быть ровно в одной TU; здесь —
// единственное место, где он подключается. ii.cpp подключает только
// stb_image.h и stb_image_resize2.h.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace imgproc {

namespace {

inline uint8_t to_byte(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return (uint8_t)std::lround(v * 255.0f);
}

// Маппинг float-значения тензора в [0..1] согласно объявленному
// «диапазону модели». Дальше всё одинаково — clamp + *255.
inline float to_unit(float v, OutputRange r) {
    switch (r) {
        case OutputRange::Unit:   return v;
        case OutputRange::Signed: return v * 0.5f + 0.5f;
        case OutputRange::Byte:   return v * (1.0f / 255.0f);
    }
    return v;
}

}  // namespace

bool parse_output_range(const std::string& name, OutputRange& out) {
    // Принимаем компактные синонимы — это CLI-флаг, пусть будет дружелюбно.
    if (name == "unit" || name == "0..1" || name == "0:1") {
        out = OutputRange::Unit;   return true;
    }
    if (name == "signed" || name == "-1..1" || name == "tanh") {
        out = OutputRange::Signed; return true;
    }
    if (name == "byte" || name == "0..255" || name == "255") {
        out = OutputRange::Byte;   return true;
    }
    return false;
}

bool is_image_output(const inf::TensorDesc& info) {
    const auto& s = info.shape;
    if (s.size() != 4) return false;
    if (s[0] != 1) return false;
    if (s[1] < 2 || s[2] < 2) return false;
    if (s[3] != 1 && s[3] != 3) return false;
    return true;
}

int detect_image_output_index(const std::vector<inf::TensorDesc>& outs) {
    int found = -1;
    for (size_t i = 0; i < outs.size(); ++i) {
        if (is_image_output(outs[i])) {
            // Несколько image-shaped выходов — пусть пользователь
            // выберет явно (--output-index): мы не угадаем, какой из
            // них «главный» (думаем про сети с несколькими SR-головами).
            if (found >= 0) return -1;
            found = (int)i;
        }
    }
    return found;
}

bool decode_image_output(const inf::TensorDesc& info, const void* data,
                         const DecodeOptions& opt, OutputImage& out) {
    if (!is_image_output(info)) {
        std::fprintf(stderr,
            "image_proc: выход '%s' не похож на изображение "
            "(ожидаю NHWC [1,H,W,1|3], получил shape ранга %zu).\n",
            info.name.c_str(), info.shape.size());
        return false;
    }

    const int h = info.shape[1];
    const int w = info.shape[2];
    const int c = info.shape[3];

    // Per-tensor квантование: scale==0 трактуем как «не квантован» —
    // оставляем как float, иначе на выходе был бы ноль.
    const float scale = info.scale != 0.0f ? info.scale : 1.0f;
    const int   zp    = info.zero_point;

    const size_t pixels = (size_t)w * h;
    const size_t n      = pixels * (size_t)c;

    // Заранее проверяем dtype, чтобы при отказе оставить out.rgb пустым —
    // ii.cpp использует empty() как индикатор fail-on-frame и фолбэкается
    // на оригинальный кадр в видео-цикле.
    switch (info.dtype) {
        case inf::DType::Float32:
        case inf::DType::Float16:
        case inf::DType::Int8:
        case inf::DType::UInt8:
        case inf::DType::Int16:
        case inf::DType::UInt16:
            break;
        default: {
            static bool warned = false;
            if (!warned) {
                std::fprintf(stderr,
                    "image_proc: dtype выхода=%s не поддержан для "
                    "image-decode. Последующие подобные сообщения "
                    "подавлены.\n", inf::dtype_name(info.dtype));
                warned = true;
            }
            out.rgb.clear();
            return false;
        }
    }

    out.width  = w;
    out.height = h;
    out.channels_src = c;
    out.rgb.assign((size_t)w * h * 3, 0);

    // Записать один пиксель в out.rgb. Для C=3 индекс i — линейный по
    // элементам тензора; для C=1 — индекс пикселя, значение льётся в
    // R=G=B. Это два разных tight-цикла, чтобы не плодить условий в
    // горячем пути.
    auto put_3ch = [&](size_t i, float v) {
        out.rgb[i] = to_byte(to_unit(v, opt.range));
    };
    auto put_1ch = [&](size_t k, float v) {
        const uint8_t b = to_byte(to_unit(v, opt.range));
        out.rgb[k * 3 + 0] = b;
        out.rgb[k * 3 + 1] = b;
        out.rgb[k * 3 + 2] = b;
    };

    switch (info.dtype) {
        case inf::DType::Float32: {
            const float* p = reinterpret_cast<const float*>(data);
            if (c == 3) { for (size_t i = 0; i < n; ++i)      put_3ch(i, p[i]); }
            else        { for (size_t k = 0; k < pixels; ++k) put_1ch(k, p[k]); }
            return true;
        }
        case inf::DType::Float16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            if (c == 3) {
                for (size_t i = 0; i < n; ++i)
                    put_3ch(i, inf::half_to_float(p[i]));
            } else {
                for (size_t k = 0; k < pixels; ++k)
                    put_1ch(k, inf::half_to_float(p[k]));
            }
            return true;
        }
        case inf::DType::Int8: {
            const int8_t* p = reinterpret_cast<const int8_t*>(data);
            if (c == 3) {
                for (size_t i = 0; i < n; ++i)
                    put_3ch(i, ((int)p[i] - zp) * scale);
            } else {
                for (size_t k = 0; k < pixels; ++k)
                    put_1ch(k, ((int)p[k] - zp) * scale);
            }
            return true;
        }
        case inf::DType::UInt8: {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
            if (c == 3) {
                for (size_t i = 0; i < n; ++i)
                    put_3ch(i, ((int)p[i] - zp) * scale);
            } else {
                for (size_t k = 0; k < pixels; ++k)
                    put_1ch(k, ((int)p[k] - zp) * scale);
            }
            return true;
        }
        case inf::DType::Int16: {
            const int16_t* p = reinterpret_cast<const int16_t*>(data);
            if (c == 3) {
                for (size_t i = 0; i < n; ++i)
                    put_3ch(i, ((int)p[i] - zp) * scale);
            } else {
                for (size_t k = 0; k < pixels; ++k)
                    put_1ch(k, ((int)p[k] - zp) * scale);
            }
            return true;
        }
        case inf::DType::UInt16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            if (c == 3) {
                for (size_t i = 0; i < n; ++i)
                    put_3ch(i, ((int)p[i] - zp) * scale);
            } else {
                for (size_t k = 0; k < pixels; ++k)
                    put_1ch(k, ((int)p[k] - zp) * scale);
            }
            return true;
        }
        default:
            // Не достижимо — guard выше уже отсеял неподдерживаемые dtype.
            // Оставляем явный return для строгих компиляторов.
            out.rgb.clear();
            return false;
    }
}

bool save_png(const OutputImage& img, const std::string& path) {
    if (img.rgb.empty() || img.width <= 0 || img.height <= 0) {
        std::fprintf(stderr,
            "image_proc: пустое изображение, пропускаю запись %s\n",
            path.c_str());
        return false;
    }
    const int stride = img.width * 3;
    if (!stbi_write_png(path.c_str(), img.width, img.height, 3,
                        img.rgb.data(), stride)) {
        std::fprintf(stderr,
            "image_proc: не удалось записать %s\n", path.c_str());
        return false;
    }
    return true;
}

}  // namespace imgproc
