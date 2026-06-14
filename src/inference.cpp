// Общая (backend-нейтральная) часть слоя инференса: маленькие
// хелперы по dtype и фабрика make_engine() с дисптачом по имени
// бэкенда. Все TFLite-зависимости живут в inference_tflite.cpp;
// будущие TensorRT/DirectML — в своих файлах. Здесь только их
// внешние объявления.

#include "inference.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace ii {

const char* dtype_name(DType t) {
    switch (t) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::Int8:    return "int8";
        case DType::UInt8:   return "uint8";
        case DType::Int16:   return "int16";
        case DType::UInt16:  return "uint16";
        case DType::Int32:   return "int32";
        case DType::UInt32:  return "uint32";
        case DType::Int64:   return "int64";
        case DType::Bool:    return "bool";
        case DType::Unknown:
        default:             return "?";
    }
}

std::size_t dtype_size(DType t) {
    switch (t) {
        case DType::Float32:
        case DType::Int32:
        case DType::UInt32:  return 4;
        case DType::Int64:   return 8;
        case DType::Float16:
        case DType::Int16:
        case DType::UInt16:  return 2;
        case DType::Int8:
        case DType::UInt8:
        case DType::Bool:    return 1;
        case DType::Unknown:
        default:             return 1;
    }
}

// Объявления фабрик каждого бэкенда — определяются в его .cpp.
// Защищены #ifdef’ами здесь, чтобы при USE_TFLITE=OFF (когда-нибудь)
// линковщик не пытался их разрешить.
#if defined(INF_HAS_TFLITE)
std::unique_ptr<Engine> make_tflite_engine();
#endif
#if defined(INF_HAS_TENSORRT)
std::unique_ptr<Engine> make_tensorrt_engine();
#endif
#if defined(INF_HAS_DIRECTML)
std::unique_ptr<Engine> make_directml_engine();
#endif
#if defined(INF_HAS_II)
std::unique_ptr<Engine> make_ii_engine();
#endif

namespace {
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}
}  // namespace

std::unique_ptr<Engine> make_engine(const std::string& backend) {
    const std::string name = to_lower(backend);

    // Дефолтный бэкенд (пустое имя): TFLite, если собран, иначе встроенный
    // `ii` (он доступен всегда). Так бинарь без backend-SDK остаётся
    // работоспособным «из коробки».
    if (name.empty()) {
#if defined(INF_HAS_TFLITE)
        return make_tflite_engine();
#elif defined(INF_HAS_II)
        return make_ii_engine();
#else
        std::fprintf(stderr, "Не собран ни один бэкенд инференса.\n");
        return nullptr;
#endif
    }
    if (name == "ii") {
#if defined(INF_HAS_II)
        return make_ii_engine();
#else
        std::fprintf(stderr,
            "Бэкенд 'ii' не собран (USE_II_ENGINE=OFF).\n");
        return nullptr;
#endif
    }
    if (name == "tflite" || name == "tensorflow-lite") {
#if defined(INF_HAS_TFLITE)
        return make_tflite_engine();
#else
        std::fprintf(stderr,
            "Бэкенд 'tflite' не собран в этом бинаре. "
            "Доступные: см. --backends.\n");
        return nullptr;
#endif
    }
    if (name == "tensorrt" || name == "trt") {
#if defined(INF_HAS_TENSORRT)
        return make_tensorrt_engine();
#else
        std::fprintf(stderr,
            "Бэкенд 'tensorrt' пока не реализован/не собран.\n");
        return nullptr;
#endif
    }
    if (name == "directml" || name == "dml") {
#if defined(INF_HAS_DIRECTML)
        return make_directml_engine();
#else
        std::fprintf(stderr,
            "Бэкенд 'directml' пока не реализован/не собран.\n");
        return nullptr;
#endif
    }

    std::fprintf(stderr, "Неизвестный бэкенд: %s\n", backend.c_str());
    return nullptr;
}

std::vector<std::string> available_backends() {
    std::vector<std::string> out;
#if defined(INF_HAS_II)
    out.emplace_back("ii");
#endif
#if defined(INF_HAS_TFLITE)
    out.emplace_back("tflite");
#endif
#if defined(INF_HAS_TENSORRT)
    out.emplace_back("tensorrt");
#endif
#if defined(INF_HAS_DIRECTML)
    out.emplace_back("directml");
#endif
    return out;
}

std::vector<std::string> backend_priority() {
    // Порядок «best → worst»: сперва аппаратно-ускоренные GPU-бэкенды
    // (tensorrt самый быстрый там, где есть NVIDIA; directml — GPU на
    // Windows), затем tflite (CPU + опциональный делегат/NPU), и в самом
    // конце встроенный `ii` — pure-CPU, всегда работоспособен, служит
    // гарантированным fallback и численным оракулом. Пересекаем со
    // списком собранных бэкендов, сохраняя этот приоритет.
    static const char* order[] = {"tensorrt", "directml", "tflite", "ii"};
    const auto built = available_backends();
    std::vector<std::string> out;
    for (const char* name : order) {
        for (const auto& b : built) {
            if (b == name) { out.emplace_back(name); break; }
        }
    }
    return out;
}

std::unique_ptr<Engine> load_best_engine(const std::string& model_path,
                                         const Engine::Options& opts,
                                         std::string* chosen_backend) {
    const auto order = backend_priority();
    if (order.empty()) {
        std::fprintf(stderr, "Не собран ни один бэкенд инференса.\n");
        return nullptr;
    }
    // Перебираем кандидатов по приоритету; берём первый, который РЕАЛЬНО
    // загрузил модель этими opts. load() сам печатает диагностику — при
    // неудаче поясняем, что пробуем следующего кандидата, чтобы вывод не
    // выглядел как фатальная ошибка.
    for (std::size_t i = 0; i < order.size(); ++i) {
        const std::string& name = order[i];
        std::printf("Авто-выбор бэкенда: пробую '%s'...\n", name.c_str());
        auto eng = make_engine(name);
        if (!eng) continue;  // на всякий случай: имя из собранного списка
        if (eng->load(model_path, opts)) {
            std::printf("Авто-выбор бэкенда: выбран '%s'.\n", name.c_str());
            if (chosen_backend) *chosen_backend = name;
            return eng;
        }
        if (i + 1 < order.size()) {
            std::fprintf(stderr,
                "Бэкенд '%s' не смог загрузить модель — пробую следующий.\n",
                name.c_str());
        }
    }
    std::fprintf(stderr,
        "Авто-выбор: ни один из собранных бэкендов не смог загрузить '%s'.\n",
        model_path.c_str());
    return nullptr;
}

// default_delegate_path() вынесена в delegate.cpp (платформенно-нейтральная
// часть) + опциональные модули делегатов — чтобы ядро инференса не зависело
// от конкретного ускорителя.

} // namespace ii
