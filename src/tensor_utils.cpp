// Реализация tensor_utils.h. См. заголовок для назначения.

#include "tensor_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace iirun {

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

void print_tensor(const char* prefix, const TensorInfo& t) {
    std::printf("%s %-32s shape=[", prefix, t.name.c_str());
    for (std::size_t i = 0; i < t.shape.size(); ++i)
        std::printf("%s%d", i ? "," : "", t.shape[i]);
    std::printf("] dtype=%s", inf::dtype_name(t.dtype));
    if (t.scale != 0.0f)
        std::printf(" quant=(scale=%.6g, zp=%d)", t.scale, t.zero_point);
    std::printf("\n");
}

void print_output_head(const TensorInfo& info, const void* data, int n_show) {
    std::size_t total = info.bytes / inf::dtype_size(info.dtype);
    n_show = (int)std::min<std::size_t>((std::size_t)n_show, total);

    const float s  = info.scale ? info.scale : 1.0f;
    const int   zp = info.zero_point;
    std::printf("  output %-32s first %d:", info.name.c_str(), n_show);
    for (int i = 0; i < n_show; ++i) {
        float v = 0.0f;
        switch (info.dtype) {
            case inf::DType::Float32:
                v = reinterpret_cast<const float*>(data)[i];
                break;
            case inf::DType::Float16:
                v = inf::half_to_float(
                    reinterpret_cast<const uint16_t*>(data)[i]);
                break;
            case inf::DType::Int8:
                v = (reinterpret_cast<const int8_t*>(data)[i] - zp) * s;
                break;
            case inf::DType::UInt8:
                v = (reinterpret_cast<const uint8_t*>(data)[i] - zp) * s;
                break;
            case inf::DType::Int16:
                v = (reinterpret_cast<const int16_t*>(data)[i] - zp) * s;
                break;
            case inf::DType::UInt16:
                v = ((int)reinterpret_cast<const uint16_t*>(data)[i] - zp) * s;
                break;
            case inf::DType::Int32:
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
    const std::size_t ds = inf::dtype_size(info.dtype);
    const std::size_t total = info.bytes / ds;
    out.resize(total);

    const float s  = info.scale ? info.scale : 1.0f;
    const int   zp = info.zero_point;

    switch (info.dtype) {
        case inf::DType::Float32: {
            std::memcpy(out.data(), data, total * sizeof(float));
            return true;
        }
        case inf::DType::Float16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = inf::half_to_float(p[i]);
            return true;
        }
        case inf::DType::Int8: {
            const int8_t* p = reinterpret_cast<const int8_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case inf::DType::UInt8: {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case inf::DType::Int16: {
            const int16_t* p = reinterpret_cast<const int16_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = (p[i] - zp) * s;
            return true;
        }
        case inf::DType::UInt16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            for (std::size_t i = 0; i < total; ++i)
                out[i] = ((int)p[i] - zp) * s;
            return true;
        }
        case inf::DType::Int32: {
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
                    inf::dtype_name(info.dtype), (int)info.dtype);
                warned = true;
            }
            return false;
        }
    }
}

}  // namespace iirun
