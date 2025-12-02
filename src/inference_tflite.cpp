// Реализация inf::Engine поверх TensorFlow Lite + external delegate.
//
// Все включения tflite/* — здесь и только здесь. Остальные модули
// раннера (ii.cpp, yolo.cpp, и т.п.) видят только inference.h и могут
// собираться без TFLite-зависимостей, когда в будущем подключим
// TensorRT/DirectML.
//
// Поведение специально совместимо со старым инлайн-движком ii.cpp:
//   * порядок разрушения (interpreter до delegate) — TFLite иначе
//     крашится в деструкторе;
//   * AllocateTensors после ModifyGraphWithDelegate — иначе делегат
//     не получит шанс заменить ноды;
//   * SetNumThreads только при threads > 0 (иначе бэкенд возьмёт свой
//     дефолт).

#include "inference.h"

#include <cstdio>
#include <utility>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"

namespace inf {

namespace {

// Маппинг TfLiteType -> inf::DType. Покрывает всё, что встречается у
// нас в моделях (включая float16 для голов «лёгких» экспортов).
DType from_tflite(TfLiteType t) {
    switch (t) {
        case kTfLiteFloat32: return DType::Float32;
        case kTfLiteFloat16: return DType::Float16;
        case kTfLiteInt8:    return DType::Int8;
        case kTfLiteUInt8:   return DType::UInt8;
        case kTfLiteInt16:   return DType::Int16;
        case kTfLiteUInt16:  return DType::UInt16;
        case kTfLiteInt32:   return DType::Int32;
        case kTfLiteUInt32:  return DType::UInt32;
        case kTfLiteInt64:   return DType::Int64;
        case kTfLiteBool:    return DType::Bool;
        default:             return DType::Unknown;
    }
}

class TfliteEngine final : public Engine {
public:
    ~TfliteEngine() override {
        // Порядок важен: сначала рушим интерпретатор (он удерживает
        // ссылки на ноды делегата), потом сам делегат. Иначе TFLite
        // дереференсит уже освобождённую память в своём деструкторе.
        interpreter_.reset();
        if (delegate_) {
            TfLiteExternalDelegateDelete(delegate_);
            delegate_ = nullptr;
        }
    }

    bool load(const std::string& model_path, const Options& opts) override {
        model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model_) {
            std::fprintf(stderr, "Не удалось загрузить модель: %s\n",
                         model_path.c_str());
            return false;
        }
        tflite::InterpreterBuilder builder(*model_, resolver_);
        if (builder(&interpreter_) != kTfLiteOk || !interpreter_) {
            std::fprintf(stderr, "InterpreterBuilder упал.\n");
            return false;
        }
        if (opts.num_threads > 0) {
            interpreter_->SetNumThreads(opts.num_threads);
        }

        if (!opts.delegate_path.empty()) {
            auto dopts =
                TfLiteExternalDelegateOptionsDefault(opts.delegate_path.c_str());
            delegate_ = TfLiteExternalDelegateCreate(&dopts);
            if (!delegate_) {
                std::fprintf(stderr,
                    "Не удалось создать external delegate: %s\n",
                    opts.delegate_path.c_str());
                return false;
            }
            if (interpreter_->ModifyGraphWithDelegate(delegate_) != kTfLiteOk) {
                std::fprintf(stderr, "ModifyGraphWithDelegate упал.\n");
                return false;
            }
            std::printf("Делегат загружен: %s\n", opts.delegate_path.c_str());
        } else {
            std::printf("Делегат не используется — CPU.\n");
        }

        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            std::fprintf(stderr, "AllocateTensors упал.\n");
            return false;
        }

        // Кэшируем дескрипторы — после AllocateTensors раскладка
        // тензоров фиксирована, повторно лезть в TfLiteTensor на
        // каждый input_data() / output_data() смысла нет.
        build_descs_();
        return true;
    }

    bool invoke() override {
        return interpreter_ && interpreter_->Invoke() == kTfLiteOk;
    }

    const std::vector<TensorDesc>& inputs()  const override { return in_descs_;  }
    const std::vector<TensorDesc>& outputs() const override { return out_descs_; }

    void* input_data(int idx) override {
        if (!interpreter_ || idx < 0 || idx >= (int)in_descs_.size())
            return nullptr;
        return interpreter_->tensor(in_descs_[idx].index)->data.data;
    }
    const void* output_data(int idx) const override {
        if (!interpreter_ || idx < 0 || idx >= (int)out_descs_.size())
            return nullptr;
        return interpreter_->tensor(out_descs_[idx].index)->data.data;
    }

    const char* backend_name() const override { return "tflite"; }

private:
    void build_descs_() {
        in_descs_.clear();
        out_descs_.clear();
        for (int idx : interpreter_->inputs()) {
            in_descs_.push_back(describe_(interpreter_->tensor(idx), idx));
        }
        for (int idx : interpreter_->outputs()) {
            out_descs_.push_back(describe_(interpreter_->tensor(idx), idx));
        }
    }

    static TensorDesc describe_(const TfLiteTensor* t, int idx) {
        TensorDesc d;
        d.index = idx;
        d.name  = t->name ? t->name : "";
        d.shape.assign(t->dims->data, t->dims->data + t->dims->size);
        d.dtype = from_tflite(t->type);
        d.scale = t->params.scale;
        d.zero_point = t->params.zero_point;
        d.bytes = t->bytes;
        return d;
    }

    std::unique_ptr<tflite::FlatBufferModel>  model_;
    tflite::ops::builtin::BuiltinOpResolver   resolver_;
    std::unique_ptr<tflite::Interpreter>      interpreter_;
    TfLiteDelegate*                           delegate_ = nullptr;
    std::vector<TensorDesc>                   in_descs_;
    std::vector<TensorDesc>                   out_descs_;
};

}  // namespace

// Фабричная функция бэкенда. Регистрируется в inference.cpp через
// внешнее объявление — так в CMake мы можем линковать TFLite-бэкенд
// опционально (USE_TFLITE=ON/OFF). Если файл не включён в сборку,
// inference.cpp получит unresolved symbol и make_engine("tflite")
// вернёт nullptr — это и есть желаемое поведение.
std::unique_ptr<Engine> make_tflite_engine() {
    return std::unique_ptr<Engine>(new TfliteEngine());
}

}  // namespace inf
