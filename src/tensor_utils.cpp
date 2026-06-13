// Реализация tensor_utils.h. См. заголовок для назначения.

#include "tensor_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace ii {

std::size_t numel(const std::vector<int>& shape) {
    std::size_t n = 1;
    for (int d : shape) {
        if (d <= 0) return 0;  // динамический/нулевой размер не поддерживаем
        n *= (std::size_t)d;
    }
    return n;
}

std::string shape_to_str(const std::vector<int>& shape) {
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

int image_input_channels(const std::vector<int>& s) {
    if (s.size() != 4 || s[0] != 1) return 0;
    if (s[3] == 1 || s[3] == 3) return s[3];
    return 0;
}

int image_input_info(const std::vector<int>& s, ImageLayout& layout,
                     int& h, int& w) {
    layout = ImageLayout::None;
    h = 0;
    w = 0;
    if (s.size() != 4 || s[0] != 1) return 0;
    if (s[3] == 1 || s[3] == 3) {            // NHWC [1,H,W,C]
        layout = ImageLayout::NHWC;
        h = s[1];
        w = s[2];
        return s[3];
    }
    if (s[1] == 1 || s[1] == 3) {            // NCHW [1,C,H,W] (ONNX)
        layout = ImageLayout::NCHW;
        h = s[2];
        w = s[3];
        return s[1];
    }
    return 0;
}

void print_tensor(const char* prefix, const TensorInfo& t) {
    std::printf("%s %-32s shape=[", prefix, t.name.c_str());
    for (std::size_t i = 0; i < t.shape.size(); ++i)
        std::printf("%s%d", i ? "," : "", t.shape[i]);
    std::printf("] dtype=%s", ii::dtype_name(t.dtype));
    if (t.scale != 0.0f)
        std::printf(" quant=(scale=%.6g, zp=%d)", t.scale, t.zero_point);
    std::printf("\n");
}

void print_output_head(const TensorInfo& info, const void* data, int n_show) {
    std::size_t total = info.bytes / ii::dtype_size(info.dtype);
    n_show = (int)std::min<std::size_t>((std::size_t)n_show, total);

    const float s  = info.scale ? info.scale : 1.0f;
    const int   zp = info.zero_point;
    std::printf("  output %-32s first %d:", info.name.c_str(), n_show);
    for (int i = 0; i < n_show; ++i) {
        float v = 0.0f;
        switch (info.dtype) {
            case ii::DType::Float32:
                v = reinterpret_cast<const float*>(data)[i];
                break;
            case ii::DType::Float16:
                v = ii::half_to_float(
                    reinterpret_cast<const uint16_t*>(data)[i]);
                break;
            case ii::DType::Int8:
                v = (reinterpret_cast<const int8_t*>(data)[i] - zp) * s;
                break;
            case ii::DType::UInt8:
                v = (reinterpret_cast<const uint8_t*>(data)[i] - zp) * s;
                break;
            case ii::DType::Int16:
                v = (reinterpret_cast<const int16_t*>(data)[i] - zp) * s;
                break;
            case ii::DType::UInt16:
                v = ((int)reinterpret_cast<const uint16_t*>(data)[i] - zp) * s;
                break;
            case ii::DType::Int32:
                v = ((float)reinterpret_cast<const int32_t*>(data)[i]
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

bool dequantize_output(const TensorInfo& info, const void* data,
                       std::vector<float>& out) {
    const std::size_t ds = ii::dtype_size(info.dtype);
    const std::size_t total = info.bytes / ds;
    out.resize(total);

    const float s  = info.scale ? info.scale : 1.0f;
    const int   zp = info.zero_point;

    switch (info.dtype) {
        case ii::DType::Float32: {
            std::memcpy(out.data(), data, total * sizeof(float));
            return true;
        }
        case ii::DType::Float16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = ii::half_to_float(p[i]);
            return true;
        }
        case ii::DType::Int8: {
            const int8_t* p = reinterpret_cast<const int8_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case ii::DType::UInt8: {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case ii::DType::Int16: {
            const int16_t* p = reinterpret_cast<const int16_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case ii::DType::UInt16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = ((int)p[i] - zp) * s;
            return true;
        }
        case ii::DType::Int32: {
            const int32_t* p = reinterpret_cast<const int32_t*>(data);
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
                    ii::dtype_name(info.dtype), (int)info.dtype);
                warned = true;
            }
            return false;
        }
    }
}

} // namespace ii
