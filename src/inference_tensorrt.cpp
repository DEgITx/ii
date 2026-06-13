// Реализация ii::Engine поверх NVIDIA TensorRT.
//
// Поддерживаемые форматы модели:
//   * сериализованный движок (.engine / .plan / .trt) — десериализуется
//     напрямую через IRuntime::deserializeCudaEngine();
//   * .onnx — собирается через nvonnxparser в момент load(). Билд может
//     занимать секунды/минуты, но в I/O-контракте это не отражается —
//     просто долгая load(). Для продакшна предполагается прекомпиляция
//     .engine на той же GPU/архитектуре и подача его сюда.
//
// Хост-контракт совпадает с TFLite-бэкендом: input_data()/output_data()
// возвращают указатели на CPU-память. Бэкенд сам копирует их через
// cudaMemcpyAsync H2D перед enqueueV3() и D2H после, а затем
// синхронизирует stream — так вызовы fill_input/dequantize_output из
// ii.cpp работают без изменений.
//
// Версия TensorRT: 10.x (enqueueV3 + setTensorAddress). На 8.x API
// отличается (executeV2 + bindings[]), но 10.x — текущая основная
// линейка с долгосрочной поддержкой, и переносить раннер обратно
// смысла нет.
//
// Поведение совместимо с inference.h:
//   * delegate_path игнорируется (TensorRT — это и есть «делегат» к
//     GPU; CPU-режима нет, пользователь увидит warning при --no-delegate);
//   * num_threads игнорируется (управление потоками выполняется
//     драйвером CUDA, TRT внутри сам разруливает поток).

#include "inference.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#if defined(INF_TRT_HAS_ONNX)
#include <NvOnnxParser.h>
#endif
#include <cuda_runtime_api.h>

namespace ii {

namespace {

// ---- Маленькие хелперы ---------------------------------------------------

DType from_trt(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return DType::Float32;
        case nvinfer1::DataType::kHALF:  return DType::Float16;
        case nvinfer1::DataType::kINT8:  return DType::Int8;
        case nvinfer1::DataType::kINT32: return DType::Int32;
        case nvinfer1::DataType::kBOOL:  return DType::Bool;
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kUINT8: return DType::UInt8;
#endif
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: return DType::Int64;
#endif
        default:                         return DType::Unknown;
    }
}

bool ends_with(const std::string& s, const char* suffix) {
    const std::size_t n = std::strlen(suffix);
    if (s.size() < n) return false;
    // Сравнение без учёта регистра — пользователь мог дать ".ENGINE".
    for (std::size_t i = 0; i < n; ++i) {
        char a = s[s.size() - n + i];
        char b = suffix[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf((std::size_t)sz);
    if (!f.read(buf.data(), sz)) return {};
    return buf;
}

// TensorRT-объекты в 10.x уничтожаются обычным `delete`. На 8.x был
// `destroy()`, но мы не таргетим эту ветку. Кастомный делитер всё же
// удобен — он молча проглатывает nullptr и совместим с RAII в
// случае ранних returns.
struct TrtDeleter {
    template <typename T>
    void operator()(T* p) const noexcept { if (p) delete p; }
};
template <typename T> using TrtPtr = std::unique_ptr<T, TrtDeleter>;

// Логгер TensorRT: пропускаем только ворнинги и ошибки, info-спам от
// билда (количество слоёв, выбор тактики и т.п.) глушим — это сотни
// строк, в отчётах не нужны.
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

// Глобальный логгер: TRT держит его указателем и читает в любых тредах
// (включая background-билд CUDA-ядер). Делать его per-Engine можно, но
// тогда переживший Engine билд (если он завершится позже) уронит
// процесс. Статический — самый безопасный вариант.
TrtLogger& trt_logger() {
    static TrtLogger l;
    return l;
}

#define INF_CUDA_CHECK(call)                                              \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            std::fprintf(stderr, "[TensorRT] CUDA error в %s:%d: %s\n",   \
                         __FILE__, __LINE__, cudaGetErrorString(_e));    \
            return false;                                                 \
        }                                                                 \
    } while (0)

class TensorRTEngine final : public Engine {
public:
    ~TensorRTEngine() override {
        // Сначала освобождаем CUDA-ресурсы (буферы и stream привязаны
        // к контексту), затем сами TRT-объекты в порядке "context →
        // engine → runtime". TRT 10 это терпит, но детерминированный
        // порядок упрощает диагностику если что-то крашится.
        free_buffers_();
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        context_.reset();
        engine_.reset();
        runtime_.reset();
    }

    bool load(const std::string& model_path, const Options& opts) override {
        if (!opts.delegate_path.empty()) {
            // delegate_path семантически не применим к TRT, но если
            // пользователь его всё-таки задал — не молчим, чтобы он
            // не подумал, что путь к плагину был учтён.
            std::printf("[TensorRT] delegate_path '%s' игнорируется "
                        "(TRT работает только на GPU)\n",
                        opts.delegate_path.c_str());
        }
        (void)opts.num_threads;  // см. шапку файла.

        runtime_.reset(nvinfer1::createInferRuntime(trt_logger()));
        if (!runtime_) {
            std::fprintf(stderr,
                "[TensorRT] createInferRuntime упал.\n");
            return false;
        }

        std::vector<char> engine_blob;
        if (ends_with(model_path, ".onnx")) {
#if defined(INF_TRT_HAS_ONNX)
            engine_blob = build_from_onnx_(model_path);
            if (engine_blob.empty()) return false;
#else
            std::fprintf(stderr,
                "[TensorRT] сборка из .onnx требует nvonnxparser, "
                "но он не подключён при компиляции.\n");
            return false;
#endif
        } else {
            engine_blob = read_file(model_path);
            if (engine_blob.empty()) {
                std::fprintf(stderr,
                    "[TensorRT] не открыть/пустой движок: %s\n",
                    model_path.c_str());
                return false;
            }
        }

        engine_.reset(runtime_->deserializeCudaEngine(
            engine_blob.data(), engine_blob.size()));
        if (!engine_) {
            std::fprintf(stderr,
                "[TensorRT] deserializeCudaEngine упал. "
                "Возможно, движок собран под другую GPU/TRT-версию.\n");
            return false;
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            std::fprintf(stderr,
                "[TensorRT] createExecutionContext упал.\n");
            return false;
        }

        INF_CUDA_CHECK(cudaStreamCreate(&stream_));

        if (!build_descs_and_bind_()) {
            return false;
        }

        std::printf("[TensorRT] движок загружен: %s "
                    "(in=%zu, out=%zu)\n",
                    model_path.c_str(),
                    in_descs_.size(), out_descs_.size());
        return true;
    }

    bool invoke() override {
        if (!context_ || !stream_) return false;
        // H2D копии всех входов.
        for (std::size_t i = 0; i < in_descs_.size(); ++i) {
            const int bi = in_buf_idx_[i];
            INF_CUDA_CHECK(cudaMemcpyAsync(
                dev_bufs_[bi], host_bufs_[bi], in_descs_[i].bytes,
                cudaMemcpyHostToDevice, stream_));
        }
        if (!context_->enqueueV3(stream_)) {
            std::fprintf(stderr, "[TensorRT] enqueueV3 упал.\n");
            return false;
        }
        // D2H копии всех выходов. Стрим сериализует H2D/exec/D2H, так
        // что результат гарантированно доступен после
        // cudaStreamSynchronize.
        for (std::size_t i = 0; i < out_descs_.size(); ++i) {
            const int bi = out_buf_idx_[i];
            INF_CUDA_CHECK(cudaMemcpyAsync(
                host_bufs_[bi], dev_bufs_[bi], out_descs_[i].bytes,
                cudaMemcpyDeviceToHost, stream_));
        }
        INF_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }

    const std::vector<TensorDesc>& inputs()  const override { return in_descs_;  }
    const std::vector<TensorDesc>& outputs() const override { return out_descs_; }

    void* input_data(int idx) override {
        if (idx < 0 || idx >= (int)in_buf_idx_.size()) return nullptr;
        return host_bufs_[in_buf_idx_[idx]];
    }
    const void* output_data(int idx) const override {
        if (idx < 0 || idx >= (int)out_buf_idx_.size()) return nullptr;
        return host_bufs_[out_buf_idx_[idx]];
    }

    const char* backend_name() const override { return "tensorrt"; }

private:
    // Считаем количество элементов в shape, трактуя -1 (dynamic) как 1.
    // Для статических моделей (типичный экспорт ultralytics export
    // format=engine) это даёт корректный размер.
    static std::size_t numel_(const nvinfer1::Dims& d) {
        std::size_t n = 1;
        for (int i = 0; i < d.nbDims; ++i) {
            const int v = d.d[i];
            n *= (std::size_t)(v > 0 ? v : 1);
        }
        return n;
    }

    // После createExecutionContext():
    //   1) пробегаем все IO-тензоры,
    //   2) для каждого аллоцируем host + device буфер ровно по байту
    //      на элемент * dtype_size(),
    //   3) привязываем device-указатель к имени через setTensorAddress.
    // TRT 10 проверяет, что все адреса заданы перед enqueueV3, иначе
    // он молча упадёт на null pointer.
    bool build_descs_and_bind_() {
        in_descs_.clear();
        out_descs_.clear();
        in_buf_idx_.clear();
        out_buf_idx_.clear();
        host_bufs_.clear();
        dev_bufs_.clear();

        const int nbio = engine_->getNbIOTensors();
        for (int i = 0; i < nbio; ++i) {
            const char* name = engine_->getIOTensorName(i);
            const auto mode  = engine_->getTensorIOMode(name);
            const auto trtdt = engine_->getTensorDataType(name);

            // getTensorShape у context_ корректно учитывает динамику,
            // если её зафиксировали setInputShape. Для статических
            // моделей это идентично engine_->getTensorShape, но мы
            // ходим через контекст на случай, когда движок собран с
            // optimization profile (динамические оси заполняются
            // дефолтным профилем automatically).
            auto dims = context_->getTensorShape(name);

            TensorDesc d;
            d.index      = i;
            d.name       = name ? name : "";
            d.dtype      = from_trt(trtdt);
            d.shape.assign(dims.d, dims.d + dims.nbDims);
            d.scale      = 0.0f;
            d.zero_point = 0;
            d.bytes      = numel_(dims) * dtype_size(d.dtype);

            // Аллокация host + device. host — обычный malloc (ii.cpp
            // обращается к буферу как к простому массиву); device —
            // cudaMalloc.
            void* host = std::malloc(d.bytes);
            void* dev  = nullptr;
            cudaError_t e = cudaMalloc(&dev, d.bytes);
            if (!host || e != cudaSuccess) {
                std::fprintf(stderr,
                    "[TensorRT] alloc упал для тензора '%s' (%zu байт).\n",
                    d.name.c_str(), d.bytes);
                if (host) std::free(host);
                free_buffers_();
                return false;
            }
            host_bufs_.push_back(host);
            dev_bufs_.push_back(dev);

            if (!context_->setTensorAddress(name, dev)) {
                std::fprintf(stderr,
                    "[TensorRT] setTensorAddress упал для '%s'.\n", name);
                free_buffers_();
                return false;
            }

            const int buf_idx = (int)host_bufs_.size() - 1;
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                in_descs_.push_back(std::move(d));
                in_buf_idx_.push_back(buf_idx);
            } else {
                out_descs_.push_back(std::move(d));
                out_buf_idx_.push_back(buf_idx);
            }
        }
        return true;
    }

    void free_buffers_() {
        for (void* h : host_bufs_) std::free(h);
        for (void* d : dev_bufs_)  if (d) cudaFree(d);
        host_bufs_.clear();
        dev_bufs_.clear();
        in_buf_idx_.clear();
        out_buf_idx_.clear();
    }

#if defined(INF_TRT_HAS_ONNX)
    // Сборка движка из .onnx. Долгая операция (десятки секунд для
    // YOLOv8), но удобна для CI и быстрого прототипирования. В
    // продакшне предполагается вызывать `trtexec` отдельно и подавать
    // сюда уже готовый .engine.
    std::vector<char> build_from_onnx_(const std::string& path) {
        TrtPtr<nvinfer1::IBuilder> builder(
            nvinfer1::createInferBuilder(trt_logger()));
        if (!builder) {
            std::fprintf(stderr, "[TensorRT] createInferBuilder упал.\n");
            return {};
        }
        // EXPLICIT_BATCH обязателен в TRT 8+: все измерения, включая
        // batch, описываются явно в shape тензоров. Старый implicit
        // batch удалён в 10.x.
        const auto explicit_batch =
            1U << (std::uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH;
        TrtPtr<nvinfer1::INetworkDefinition> network(
            builder->createNetworkV2(explicit_batch));
        if (!network) {
            std::fprintf(stderr, "[TensorRT] createNetworkV2 упал.\n");
            return {};
        }
        TrtPtr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, trt_logger()));
        if (!parser->parseFromFile(path.c_str(),
            (int)nvinfer1::ILogger::Severity::kWARNING)) {
            // Парсер уже вылил детали в логгер; здесь просто констатируем.
            std::fprintf(stderr,
                "[TensorRT] парсинг ONNX упал: %s\n", path.c_str());
            return {};
        }
        TrtPtr<nvinfer1::IBuilderConfig> config(
            builder->createBuilderConfig());
        // FP16 включаем только если железо поддерживает (на современных
        // GPU это безопасно ускоряет YOLO/SR-сети ~в 2x без заметной
        // потери качества). INT8-калибровка требует датасета —
        // оставляем её для отдельного этапа на хосте (trtexec --int8).
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::printf("[TensorRT] FP16 включён (поддерживается железом)\n");
        }
        std::printf("[TensorRT] сборка движка из ONNX... "
                    "(может занять до минуты)\n");
        TrtPtr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized) {
            std::fprintf(stderr,
                "[TensorRT] buildSerializedNetwork упал.\n");
            return {};
        }
        std::vector<char> blob(serialized->size());
        std::memcpy(blob.data(), serialized->data(), serialized->size());
        std::printf("[TensorRT] движок собран (%zu байт)\n", blob.size());
        return blob;
    }
#endif

    TrtPtr<nvinfer1::IRuntime>          runtime_;
    TrtPtr<nvinfer1::ICudaEngine>       engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    cudaStream_t                        stream_ = nullptr;

    std::vector<TensorDesc> in_descs_;
    std::vector<TensorDesc> out_descs_;
    // host_bufs_ и dev_bufs_ упорядочены в порядке итерации getNbIOTensors().
    // Маппинги *_buf_idx_ дают индекс в этих массивах для inputs()[i] /
    // outputs()[i].
    std::vector<void*>      host_bufs_;
    std::vector<void*>      dev_bufs_;
    std::vector<int>        in_buf_idx_;
    std::vector<int>        out_buf_idx_;
};

}  // namespace

// Фабрика бэкенда — регистрируется в inference.cpp через external
// объявление под #ifdef INF_HAS_TENSORRT.
std::unique_ptr<Engine> make_tensorrt_engine() {
    return std::unique_ptr<Engine>(new TensorRTEngine());
}

} // namespace ii
