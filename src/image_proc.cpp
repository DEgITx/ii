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

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "parallel.h"

// STB_IMAGE_WRITE_IMPLEMENTATION должен быть ровно в одной TU; здесь —
// единственное место, где он подключается. ii.cpp подключает только
// stb_image.h и stb_image_resize2.h.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace ii {

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

// Пересчитать LUT кэша под текущие параметры квантования, если они
// изменились. Возвращает true, если LUT валиден и применим для
// горячего цикла (т.е. dtype ∈ {Int8, UInt8} и cache != nullptr).
bool ensure_decode_lut(DecodeCache* cache, const ii::TensorDesc& info,
                       OutputRange range) {
    if (!cache) return false;
    if (info.dtype != ii::DType::Int8 && info.dtype != ii::DType::UInt8) {
        cache->lut_valid = false;
        return false;
    }
    if (cache->lut_valid
        && cache->dtype == info.dtype
        && cache->scale == info.scale
        && cache->zero_point == info.zero_point
        && cache->range == range) {
        return true;  // кэш актуален — горячий путь
    }

    const float scale = info.scale != 0.0f ? info.scale : 1.0f;
    const int   zp    = info.zero_point;
    if (info.dtype == ii::DType::Int8) {
        // Индекс LUT = uint8-перепрочтение int8: u ∈ [0..127] → int8 0..127,
        // u ∈ [128..255] → int8 -128..-1. Так в горячем цикле достаточно
        // привести int8 к uint8 и сразу обратиться в таблицу.
        for (int u = 0; u < 256; ++u) {
            const int q = (int)(int8_t)u;
            cache->lut[u] = to_byte(to_unit((q - zp) * scale, range));
        }
    } else {
        for (int u = 0; u < 256; ++u) {
            cache->lut[u] = to_byte(to_unit((u - zp) * scale, range));
        }
    }
    cache->dtype      = info.dtype;
    cache->scale      = info.scale;
    cache->zero_point = info.zero_point;
    cache->range      = range;
    cache->lut_valid  = true;
    return true;
}

// Минимальное число пикселей на параллельный кусок: ниже него дробить не
// окупается (накладные расходы волны > выгоды). Грейн по строкам выводим
// как kDecodeGrainPx / w, чтобы кусок содержал ~столько пикселей вне
// зависимости от ширины выхода. Мелкий выход (single-image) останется
// последовательным сам собой (см. parallel_for / plan_chunks).
constexpr std::int64_t kDecodeGrainPx = 16384;

// Тайтовое ядро декодера: проходит по тензору и пишет в произвольный
// уголок буфера с заданным шагом строки. Параметризован функтором
// conv (TSrc → uint8_t), что позволяет одним шаблоном обслуживать
// и LUT-ветку для int8/uint8 и общую float-ветку.
//
// dst_base — адрес «логического» (dst_x, dst_y) пикселя в буфере, т.е.
// caller уже сдвинул его на (dst_y * stride + dst_x * 3).
//
// Строки выхода независимы (каждая пишет в свой диапазон [y*stride ..]),
// поэтому делим их между ядрами через parallel_for — результат бит-в-бит
// совпадает с однопоточным. conv захватывает const-LUT/параметры, чтение
// из нескольких потоков безопасно.
template <typename TSrc, typename FConv>
void decode_kernel(const TSrc* p, int w, int h, int c,
                   uint8_t* dst_base, std::size_t stride, FConv conv) {
    const std::int64_t grain =
        std::max<std::int64_t>(1, kDecodeGrainPx / (w > 0 ? w : 1));
    if (c == 3) {
        ii::parallel_for(h, grain, [&](std::int64_t yb, std::int64_t ye) {
            for (int y = (int)yb; y < (int)ye; ++y) {
                uint8_t*    dp = dst_base + (std::size_t)y * stride;
                const TSrc* sp = p + (std::size_t)y * w * 3;
                for (int x = 0; x < w; ++x) {
                    dp[0] = conv(sp[0]);
                    dp[1] = conv(sp[1]);
                    dp[2] = conv(sp[2]);
                    dp += 3;
                    sp += 3;
                }
            }
        });
    } else {
        ii::parallel_for(h, grain, [&](std::int64_t yb, std::int64_t ye) {
            for (int y = (int)yb; y < (int)ye; ++y) {
                uint8_t*    dp = dst_base + (std::size_t)y * stride;
                const TSrc* sp = p + (std::size_t)y * w;
                for (int x = 0; x < w; ++x) {
                    const uint8_t b = conv(*sp++);
                    dp[0] = b;
                    dp[1] = b;
                    dp[2] = b;
                    dp += 3;
                }
            }
        });
    }
}

// Общая «развилка по dtype» для декода: и `decode_image_output`, и
// `decode_image_output_to` сходятся сюда. dst_base уже сдвинут на
// (dst_x, dst_y) caller’ом; stride — байты между строками назначения.
bool decode_dispatch(const ii::TensorDesc& info, const void* data,
                     const DecodeOptions& opt,
                     uint8_t* dst_base, std::size_t stride,
                     DecodeCache* cache) {
    const int   w     = info.shape[2];
    const int   h     = info.shape[1];
    const int   c     = info.shape[3];
    const float scale = info.scale != 0.0f ? info.scale : 1.0f;
    const int   zp    = info.zero_point;
    const auto  range = opt.range;

    // Локальный кэш — если caller не передал свой. Тогда LUT строится
    // на каждый вызов, но это всё ещё в 30+ раз быстрее, чем lround
    // на каждый пиксель: 256-итераций инициализации против H·W·C
    // итераций горячего цикла.
    DecodeCache  local;
    DecodeCache* cc = cache ? cache : &local;

    switch (info.dtype) {
        case ii::DType::Int8: {
            ensure_decode_lut(cc, info, range);
            const uint8_t* lut = cc->lut;
            const int8_t*  p   = reinterpret_cast<const int8_t*>(data);
            decode_kernel(p, w, h, c, dst_base, stride,
                          [lut](int8_t v) { return lut[(uint8_t)v]; });
            return true;
        }
        case ii::DType::UInt8: {
            ensure_decode_lut(cc, info, range);
            const uint8_t* lut = cc->lut;
            const uint8_t* p   = reinterpret_cast<const uint8_t*>(data);
            decode_kernel(p, w, h, c, dst_base, stride,
                          [lut](uint8_t v) { return lut[v]; });
            return true;
        }
        case ii::DType::Float32: {
            const float* p = reinterpret_cast<const float*>(data);
            decode_kernel(p, w, h, c, dst_base, stride,
                          [range](float v) {
                              return to_byte(to_unit(v, range));
                          });
            return true;
        }
        case ii::DType::Float16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            decode_kernel(p, w, h, c, dst_base, stride,
                          [range](uint16_t v) {
                              return to_byte(
                                  to_unit(ii::half_to_float(v), range));
                          });
            return true;
        }
        case ii::DType::Int16: {
            const int16_t* p = reinterpret_cast<const int16_t*>(data);
            decode_kernel(p, w, h, c, dst_base, stride,
                          [range, scale, zp](int16_t v) {
                              return to_byte(
                                  to_unit(((int)v - zp) * scale, range));
                          });
            return true;
        }
        case ii::DType::UInt16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            decode_kernel(p, w, h, c, dst_base, stride,
                          [range, scale, zp](uint16_t v) {
                              return to_byte(
                                  to_unit(((int)v - zp) * scale, range));
                          });
            return true;
        }
        default:
            return false;
    }
}

// Проверка dtype с warn-once для нештатных типов (общая часть для
// обеих публичных функций).
bool check_decode_dtype(const ii::TensorDesc& info) {
    switch (info.dtype) {
        case ii::DType::Float32:
        case ii::DType::Float16:
        case ii::DType::Int8:
        case ii::DType::UInt8:
        case ii::DType::Int16:
        case ii::DType::UInt16:
            return true;
        default: {
            static bool warned = false;
            if (!warned) {
                std::fprintf(stderr,
                    "image_proc: dtype выхода=%s не поддержан для "
                    "image-decode. Последующие подобные сообщения "
                    "подавлены.\n", ii::dtype_name(info.dtype));
                warned = true;
            }
            return false;
        }
    }
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

bool is_image_output(const ii::TensorDesc& info) {
    const auto& s = info.shape;
    if (s.size() != 4) return false;
    if (s[0] != 1) return false;
    if (s[1] < 2 || s[2] < 2) return false;
    if (s[3] != 1 && s[3] != 3) return false;
    return true;
}

int detect_image_output_index(const std::vector<ii::TensorDesc>& outs) {
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

bool decode_image_output(const ii::TensorDesc& info, const void* data,
                         const DecodeOptions& opt, OutputImage& out,
                         DecodeCache* cache) {
    if (!is_image_output(info)) {
        std::fprintf(stderr,
            "image_proc: выход '%s' не похож на изображение "
            "(ожидаю NHWC [1,H,W,1|3], получил shape ранга %zu).\n",
            info.name.c_str(), info.shape.size());
        return false;
    }
    if (!check_decode_dtype(info)) {
        out.rgb.clear();
        return false;
    }

    const int h = info.shape[1];
    const int w = info.shape[2];
    const int c = info.shape[3];

    out.width        = w;
    out.height       = h;
    out.channels_src = c;

    // resize вместо assign(N, 0): после первого кадра размер не меняется
    // и resize становится no-op (никаких memset’ов). Если буфер меньше
    // нужного — резервируем без zero-init «хвоста», т.к. весь объём
    // сразу же перезаписывается decode_kernel’ом.
    const std::size_t need = (std::size_t)w * h * 3;
    if (out.rgb.size() != need) out.rgb.resize(need);

    return decode_dispatch(info, data, opt,
                           out.rgb.data(), (std::size_t)w * 3, cache);
}

bool decode_image_output_to(const ii::TensorDesc& info, const void* data,
                            const DecodeOptions& opt,
                            uint8_t* dst_rgb, std::size_t dst_stride,
                            int dst_x, int dst_y,
                            DecodeCache* cache) {
    if (!is_image_output(info)) {
        std::fprintf(stderr,
            "image_proc: выход '%s' не похож на изображение "
            "(ожидаю NHWC [1,H,W,1|3], получил shape ранга %zu).\n",
            info.name.c_str(), info.shape.size());
        return false;
    }
    if (!check_decode_dtype(info)) return false;
    if (!dst_rgb || dst_stride == 0) return false;

    uint8_t* base = dst_rgb
                  + (std::size_t)dst_y * dst_stride
                  + (std::size_t)dst_x * 3;
    return decode_dispatch(info, data, opt, base, dst_stride, cache);
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

} // namespace ii
