// Реализация preprocess.h. Единственный TU с реализациями stb_image
// и stb_image_resize2 (load_image / letterbox).

#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

namespace iirun {

bool load_image(const std::string& path, Image& out) {
    int w = 0, h = 0, c = 0;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        std::fprintf(stderr, "Не удалось загрузить изображение: %s (%s)\n",
                     path.c_str(), stbi_failure_reason());
        return false;
    }
    out.rgb.assign(data, data + (std::size_t)w * h * 3);
    out.w = w;
    out.h = h;
    stbi_image_free(data);
    return true;
}

void letterbox(const uint8_t* src_rgb, int src_w, int src_h,
               int target_w, int target_h,
               std::vector<uint8_t>& dst, uint8_t pad) {
    float r = std::min((float)target_w / src_w, (float)target_h / src_h);
    int new_w = (int)std::round(src_w * r);
    int new_h = (int)std::round(src_h * r);
    if (new_w < 1) new_w = 1;
    if (new_h < 1) new_h = 1;

    // resized живёт между вызовами, но переалоцируется только при смене
    // размера ресайза (например, при ресайзе камеры на лету). Для
    // стационарного видео-цикла память аллоцируется один раз.
    static thread_local std::vector<uint8_t> resized;
    resized.resize((std::size_t)new_w * new_h * 3);
    stbir_resize_uint8_linear(src_rgb, src_w, src_h, 0,
                              resized.data(), new_w, new_h, 0,
                              STBIR_RGB);

    dst.assign((std::size_t)target_w * target_h * 3, pad);
    int dx = (target_w - new_w) / 2;
    int dy = (target_h - new_h) / 2;
    for (int y = 0; y < new_h; ++y) {
        std::memcpy(&dst[((y + dy) * target_w + dx) * 3],
                    &resized[(std::size_t)y * new_w * 3],
                    (std::size_t)new_w * 3);
    }
}

// Y = (77*R + 150*G + 29*B) / 256. Сумма коэффициентов = 256 → сдвиг
// вправо на 8 ровно делит на 256, без округления (отклонение от точной
// BT.601 ≤ 1 на старшем бите — для квантования к INT8 это шум модели).
void rgb_to_gray(const uint8_t* rgb, std::size_t pixels,
                 std::vector<uint8_t>& gray) {
    gray.resize(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        const unsigned r = rgb[i * 3 + 0];
        const unsigned g = rgb[i * 3 + 1];
        const unsigned b = rgb[i * 3 + 2];
        gray[i] = (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
    }
}

namespace {

// Кэш квантования входа. Подкладываем 256-элементную LUT (вход uint8
// имеет ровно 256 возможных значений), что убирает std::lround / fmul /
// clamp из горячего цикла. В tile-режиме это в районе 9216 элементов
// квантования на тайл × 24 тайла = 220K вызовов lround на кадр без
// кэша; с LUT — одна индексация в L1.
//
// Ключ кэша = (dtype, scale, zp); в --compare-режиме два разных info
// будут пинг-понговать ключ и пересчитывать LUT, но это копейки
// (256 итераций) на фоне даже одного инвока.
struct InputQuantCache {
    inf::DType dtype = inf::DType::Unknown;
    float      scale = 0.0f;
    int        zero_point = 0;
    bool       valid = false;
    uint8_t    lut8[256]{};      // для Int8 (bit-pattern) / UInt8
    uint16_t   lut16[256]{};     // для Int16 (bit-pattern) / UInt16
};

bool ensure_input_quant_lut(InputQuantCache& c, const inf::TensorDesc& info) {
    if (c.valid
        && c.dtype == info.dtype
        && c.scale == info.scale
        && c.zero_point == info.zero_point) {
        return true;
    }
    const float s  = info.scale ? info.scale : 1.0f;
    const int   zp = info.zero_point;
    int lo = 0, hi = 0;
    switch (info.dtype) {
        case inf::DType::Int8:   lo = -128;   hi = 127;   break;
        case inf::DType::UInt8:  lo = 0;      hi = 255;   break;
        case inf::DType::Int16:  lo = -32768; hi = 32767; break;
        case inf::DType::UInt16: lo = 0;      hi = 65535; break;
        default:
            c.valid = false;
            return false;
    }
    for (int u = 0; u < 256; ++u) {
        int q = (int)std::lround((u / 255.0f) / s + zp);
        if (q < lo) q = lo;
        if (q > hi) q = hi;
        // Сохраняем bit-pattern целевого типа в беззнаковом контейнере —
        // в горячем цикле просто кастуем к int8_t/int16_t и пишем в
        // буфер модели.
        if (info.dtype == inf::DType::Int8 || info.dtype == inf::DType::UInt8) {
            c.lut8[u] = (uint8_t)q;
        } else {
            c.lut16[u] = (uint16_t)q;
        }
    }
    c.dtype      = info.dtype;
    c.scale      = info.scale;
    c.zero_point = info.zero_point;
    c.valid      = true;
    return true;
}

}  // namespace

bool fill_input(const std::vector<uint8_t>& rgb, const inf::TensorDesc& info,
                void* data) {
    const std::size_t n = rgb.size();

    // Один кэш на тред — параметры info стабильны за прогон, переключение
    // между двумя info (compare-mode) переинициализирует таблицу за 256
    // итераций, что несущественно.
    thread_local InputQuantCache cache;

    switch (info.dtype) {
        case inf::DType::Float32: {
            float* p = reinterpret_cast<float*>(data);
            for (std::size_t i = 0; i < n; ++i) p[i] = rgb[i] / 255.0f;
            return true;
        }
        case inf::DType::Int8: {
            ensure_input_quant_lut(cache, info);
            int8_t* p = reinterpret_cast<int8_t*>(data);
            const uint8_t* lut = cache.lut8;
            for (std::size_t i = 0; i < n; ++i) p[i] = (int8_t)lut[rgb[i]];
            return true;
        }
        case inf::DType::UInt8: {
            ensure_input_quant_lut(cache, info);
            uint8_t* p = reinterpret_cast<uint8_t*>(data);
            const uint8_t* lut = cache.lut8;
            for (std::size_t i = 0; i < n; ++i) p[i] = lut[rgb[i]];
            return true;
        }
        case inf::DType::Int16: {
            ensure_input_quant_lut(cache, info);
            int16_t* p = reinterpret_cast<int16_t*>(data);
            const uint16_t* lut = cache.lut16;
            for (std::size_t i = 0; i < n; ++i) p[i] = (int16_t)lut[rgb[i]];
            return true;
        }
        case inf::DType::UInt16: {
            ensure_input_quant_lut(cache, info);
            uint16_t* p = reinterpret_cast<uint16_t*>(data);
            const uint16_t* lut = cache.lut16;
            for (std::size_t i = 0; i < n; ++i) p[i] = lut[rgb[i]];
            return true;
        }
        default:
            std::fprintf(stderr, "Неподдерживаемый dtype входа: %s (код %d)\n",
                         inf::dtype_name(info.dtype), (int)info.dtype);
            return false;
    }
}

}  // namespace iirun
