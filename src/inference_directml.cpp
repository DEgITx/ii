// Реализация inf::Engine поверх ONNX Runtime с execution provider
// DirectML (фактически — стек Windows ML: ONNX-граф крутится поверх
// DirectX 12 на любом D3D12-совместимом GPU/iGPU/NPU под Windows).
//
// Почему ORT, а не «голый» DirectML? У DirectML нет встроенного
// загрузчика моделей: API оперирует уже построенным графом из
// DML-операторов. Все продакшн-пути (Windows ML, ORT) и без того
// собирают DirectML-граф из ONNX через ORT EP. Поэтому реализуем
// именно ORT-сессию с подключённым DML EP — это и есть то, что
// называют "DirectML backend" в продуктовом смысле.
//
// Поддерживаемый формат модели — .onnx (любая ONNX-Quant-схема: float32,
// float16, INT8 dynamic/static). .tflite ORT не открывает; для INT8 c
// квантизацией нужно либо конвертировать TFLite→ONNX
// (например, tf2onnx или прямой экспорт из Keras в onnx), либо
// собирать модель напрямую в ONNX.
//
// Хост-контракт совпадает с TFLite/TRT: input_data()/output_data()
// возвращают CPU-память; перед invoke() пользователь её заполняет,
// после — читает. ORT-сессия оборачивает эти буферы в Ort::Value через
// CreateTensor (нулевая копия), поэтому накладные расходы на ввод/
// вывод минимальны — все нужные перегоны на GPU делает сам DML EP.
//
// delegate_path:
//   * пустая строка — DML EP активен (NPU/GPU); это и есть «делегат»;
//   * "cpu" — DML EP не подключается, используется ORT CPU EP (это
//     соответствует --no-delegate в ii.cpp и нужно для эталонных
//     прогонов в --compare/--compare-cpu).
// num_threads — прокидывается в SetIntraOpNumThreads (актуально только
// для CPU EP; DML EP сам разруливает воркеры).

#include "inference.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#if defined(_WIN32) && defined(INF_DML_EP_AVAILABLE)
#include <dml_provider_factory.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

namespace inf {

namespace {

DType from_ort(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return DType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:return DType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:   return DType::Int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  return DType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  return DType::Int16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return DType::UInt16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return DType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return DType::UInt32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return DType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:   return DType::Bool;
        default:                                   return DType::Unknown;
    }
}

ONNXTensorElementDataType to_ort(DType t) {
    switch (t) {
        case DType::Float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DType::Float16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case DType::Int8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case DType::UInt8:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case DType::Int16:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case DType::UInt16:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        case DType::Int32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case DType::UInt32:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
        case DType::Int64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case DType::Bool:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        case DType::Unknown:
        default:             return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

// ORT-среда — синглтон процесса. Создавать её на каждый Engine
// разрешено, но: логгер у ORT глобальный, и второй Env переинициализирует
// его уровень логирования (а с DML EP это критично — мы выставляем
// WARNING, чтобы не утопиться в info-спаме инициализации D3D).
// Поэтому держим один Env на процесс.
Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ii");
    return env;
}

#if defined(_WIN32)
// Конвертация UTF-8 → UTF-16. На Windows API ORT принимает wchar_t*.
// Для ASCII-путей подошло бы и std::wstring(path.begin(), path.end()),
// но если в пути появится кириллица (типичная ситуация на dev-машине),
// неконвертированная версия даст крах. Используем настоящий
// MultiByteToWideChar.
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = ::MultiByteToWideChar(CP_UTF8, 0,
        s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0,
        s.c_str(), (int)s.size(), w.data(), n);
    return w;
}
#endif

class DirectMLEngine final : public Engine {
public:
    ~DirectMLEngine() override {
        // Сначала рушим сессию (держит ссылки на буферы через Ort::Value
        // в Run-сценарии, но между Run-ами ничего не удерживает —
        // поэтому порядок здесь не критичен, но детерминированный).
        session_.reset();
        for (void* p : in_bufs_)  std::free(p);
        for (void* p : out_bufs_) std::free(p);
    }

    bool load(const std::string& model_path, const Options& opts) override {
        try {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (opts.num_threads > 0) {
                so.SetIntraOpNumThreads(opts.num_threads);
            }

            const bool want_dml =
                (opts.delegate_path != "cpu") && (!force_cpu_(opts));
            if (want_dml) {
#if defined(_WIN32) && defined(INF_DML_EP_AVAILABLE)
                // DML EP несовместим с memory pattern (он выделяет
                // буферы на GPU и менеджит их сам). ORT-документация
                // явно требует выключить mem pattern + установить
                // sequential execution mode.
                so.DisableMemPattern();
                so.SetExecutionMode(ORT_SEQUENTIAL);
                int device_id = 0;
                OrtStatus* st =
                    OrtSessionOptionsAppendExecutionProvider_DML(so, device_id);
                if (st) {
                    const char* msg = Ort::GetApi().GetErrorMessage(st);
                    std::fprintf(stderr,
                        "[DirectML] EP не подключился: %s\n",
                        msg ? msg : "?");
                    Ort::GetApi().ReleaseStatus(st);
                    return false;
                }
                std::printf("[DirectML] EP подключён (device=%d)\n",
                            device_id);
                using_dml_ = true;
#else
                std::fprintf(stderr,
                    "[DirectML] EP недоступен в этой сборке "
                    "(Windows + INF_DML_EP_AVAILABLE требуются). "
                    "Откатываемся на CPU EP.\n");
                using_dml_ = false;
#endif
            } else {
                std::printf("[DirectML] EP отключён (delegate_path='%s'), "
                            "используем CPU EP.\n",
                            opts.delegate_path.c_str());
                using_dml_ = false;
            }

#if defined(_WIN32)
            const std::wstring wpath = widen(model_path);
            session_ = std::make_unique<Ort::Session>(
                ort_env(), wpath.c_str(), so);
#else
            session_ = std::make_unique<Ort::Session>(
                ort_env(), model_path.c_str(), so);
#endif

            mem_info_ = std::make_unique<Ort::MemoryInfo>(
                Ort::MemoryInfo::CreateCpu(
                    OrtArenaAllocator, OrtMemTypeDefault));

            if (!build_descs_and_alloc_()) return false;

            std::printf("[DirectML] модель загружена: %s (in=%zu, out=%zu)\n",
                        model_path.c_str(),
                        in_descs_.size(), out_descs_.size());
            return true;
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[DirectML] ORT-исключение: %s\n", e.what());
            return false;
        }
    }

    bool invoke() override {
        if (!session_) return false;
        try {
            // Каждый Run заворачиваем буферы в Ort::Value заново —
            // shape/dtype зафиксированы, аллокация Ort::Value лёгкая
            // (она по сути просто хранит указатель + метаданные).
            // Дороже было бы держать массив пред-созданных Value, но
            // ORT не разрешает повторное использование одного и того же
            // Value в нескольких Run без явной reset, поэтому
            // создание-на-лету проще.
            std::vector<Ort::Value> in_vals;
            std::vector<Ort::Value> out_vals;
            in_vals.reserve(in_descs_.size());
            out_vals.reserve(out_descs_.size());

            for (std::size_t i = 0; i < in_descs_.size(); ++i) {
                in_vals.push_back(Ort::Value::CreateTensor(
                    *mem_info_,
                    in_bufs_[i], in_descs_[i].bytes,
                    in_shape64_[i].data(), in_shape64_[i].size(),
                    to_ort(in_descs_[i].dtype)));
            }
            for (std::size_t i = 0; i < out_descs_.size(); ++i) {
                out_vals.push_back(Ort::Value::CreateTensor(
                    *mem_info_,
                    out_bufs_[i], out_descs_[i].bytes,
                    out_shape64_[i].data(), out_shape64_[i].size(),
                    to_ort(out_descs_[i].dtype)));
            }

            session_->Run(Ort::RunOptions{nullptr},
                in_names_c_.data(),  in_vals.data(),  in_vals.size(),
                out_names_c_.data(), out_vals.data(), out_vals.size());
            return true;
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[DirectML] Run упал: %s\n", e.what());
            return false;
        }
    }

    const std::vector<TensorDesc>& inputs()  const override { return in_descs_;  }
    const std::vector<TensorDesc>& outputs() const override { return out_descs_; }

    void* input_data(int idx) override {
        if (idx < 0 || idx >= (int)in_bufs_.size()) return nullptr;
        return in_bufs_[idx];
    }
    const void* output_data(int idx) const override {
        if (idx < 0 || idx >= (int)out_bufs_.size()) return nullptr;
        return out_bufs_[idx];
    }

    const char* backend_name() const override {
        // Имя бэкенда фиксированное — пользователь выбирал именно
        // 'directml'. Где именно крутится исполнение (DML EP vs CPU
        // EP), сообщается в логах при load.
        return "directml";
    }

private:
    // На случай, если delegate_path задан непустой, но не "cpu" —
    // ORT не различает понятие «делегат», поэтому любой другой
    // непустой путь трактуем буквально: «пользователь хочет DML EP».
    static bool force_cpu_(const Options& opts) {
        if (opts.delegate_path == "cpu" ||
            opts.delegate_path == "CPU") return true;
        return false;
    }

    bool build_descs_and_alloc_() {
        if (!session_) return false;
        Ort::AllocatorWithDefaultOptions alloc;

        const std::size_t n_in  = session_->GetInputCount();
        const std::size_t n_out = session_->GetOutputCount();

        in_descs_.clear();
        out_descs_.clear();
        in_names_.clear();
        out_names_.clear();
        in_names_c_.clear();
        out_names_c_.clear();
        in_shape64_.clear();
        out_shape64_.clear();
        in_bufs_.clear();
        out_bufs_.clear();

        in_descs_.reserve(n_in);
        out_descs_.reserve(n_out);

        for (std::size_t i = 0; i < n_in; ++i) {
            auto name_alloc = session_->GetInputNameAllocated(i, alloc);
            in_names_.emplace_back(name_alloc.get());

            Ort::TypeInfo ti = session_->GetInputTypeInfo(i);
            auto tsi = ti.GetTensorTypeAndShapeInfo();

            TensorDesc d;
            d.index = (int)i;
            d.name  = in_names_.back();
            d.dtype = from_ort(tsi.GetElementType());
            auto dims64 = tsi.GetShape();
            // Динамические оси в ONNX (-1) фиксируем на 1 — иначе
            // невозможно посчитать размер буфера. Если у модели
            // настоящий dynamic batch, пользователь должен
            // ре-экспортировать её со статическими формами (как и
            // в случае TFLite).
            for (auto& v : dims64) if (v < 0) v = 1;
            d.shape.assign(dims64.begin(), dims64.end());

            std::size_t n = 1;
            for (auto v : dims64) n *= (std::size_t)(v > 0 ? v : 1);
            d.bytes = n * dtype_size(d.dtype);

            in_shape64_.push_back(std::move(dims64));
            in_descs_.push_back(std::move(d));

            void* buf = std::malloc(in_descs_.back().bytes);
            if (!buf) {
                std::fprintf(stderr,
                    "[DirectML] malloc input '%s' (%zu байт) упал.\n",
                    in_descs_.back().name.c_str(),
                    in_descs_.back().bytes);
                return false;
            }
            in_bufs_.push_back(buf);
        }

        for (std::size_t i = 0; i < n_out; ++i) {
            auto name_alloc = session_->GetOutputNameAllocated(i, alloc);
            out_names_.emplace_back(name_alloc.get());

            Ort::TypeInfo ti = session_->GetOutputTypeInfo(i);
            auto tsi = ti.GetTensorTypeAndShapeInfo();

            TensorDesc d;
            d.index = (int)i;
            d.name  = out_names_.back();
            d.dtype = from_ort(tsi.GetElementType());
            auto dims64 = tsi.GetShape();
            for (auto& v : dims64) if (v < 0) v = 1;
            d.shape.assign(dims64.begin(), dims64.end());

            std::size_t n = 1;
            for (auto v : dims64) n *= (std::size_t)(v > 0 ? v : 1);
            d.bytes = n * dtype_size(d.dtype);

            out_shape64_.push_back(std::move(dims64));
            out_descs_.push_back(std::move(d));

            void* buf = std::malloc(out_descs_.back().bytes);
            if (!buf) {
                std::fprintf(stderr,
                    "[DirectML] malloc output '%s' (%zu байт) упал.\n",
                    out_descs_.back().name.c_str(),
                    out_descs_.back().bytes);
                return false;
            }
            out_bufs_.push_back(buf);
        }

        // Заранее формируем массив `const char*` — Run() требует
        // плоский массив сишных строк, а GetInputNameAllocated отдаёт
        // unique_ptr с кастомным делитером (одну за вызов). Чтобы не
        // дёргать аллокатор на каждом Invoke, фиксируем строки в
        // std::string и подсовываем их .c_str().
        in_names_c_.reserve(in_names_.size());
        for (const auto& s : in_names_)  in_names_c_.push_back(s.c_str());
        out_names_c_.reserve(out_names_.size());
        for (const auto& s : out_names_) out_names_c_.push_back(s.c_str());

        return true;
    }

    std::unique_ptr<Ort::Session>     session_;
    std::unique_ptr<Ort::MemoryInfo>  mem_info_;
    bool                              using_dml_ = false;

    std::vector<TensorDesc>           in_descs_;
    std::vector<TensorDesc>           out_descs_;
    std::vector<std::string>          in_names_;
    std::vector<std::string>          out_names_;
    std::vector<const char*>          in_names_c_;
    std::vector<const char*>          out_names_c_;
    // Те же shape, что и в TensorDesc, но в int64 (формат ORT).
    // Дублируем, чтобы не конвертировать на каждый Run.
    std::vector<std::vector<std::int64_t>> in_shape64_;
    std::vector<std::vector<std::int64_t>> out_shape64_;
    std::vector<void*>                in_bufs_;
    std::vector<void*>                out_bufs_;
};

}  // namespace

std::unique_ptr<Engine> make_directml_engine() {
    return std::unique_ptr<Engine>(new DirectMLEngine());
}

}  // namespace inf
