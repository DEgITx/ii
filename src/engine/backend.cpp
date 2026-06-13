// Реализация inf::Engine поверх встроенного движка `ii`.
//
// Это мост между backend-нейтральным контрактом раннера (inference.h) и
// собственным графовым движком (engine/graph.* / engine/ops.*). Зависимостей от
// SDK здесь нет — бэкенд `ii` собирается всегда и на любой платформе, что
// делает его эталоном корректности для TFLite/TensorRT/DirectML и площадкой
// разработки новых слоёв.
//
// Поток работы:
//   load()  -> загрузить модель в ii::Graph, выделить входные буферы прямо в
//              исполнителе, один «прогрев» с нулями ради вывода форм выходов;
//   host    -> input_data(i) отдаёт указатель во входной буфер исполнителя,
//              препроцесс пишет туда напрямую (zero-copy);
//   invoke()-> гоняет граф на уже заполненных входах и забирает выходы;
//   output_data(i) -> указатель в выходной буфер.
//
// Веса модели исполнитель держит по указателю и не копирует их при прогоне, а
// промежуточные тензоры освобождает по мере прохождения графа — поэтому даже
// в живом видеоцикле invoke() не плодит копий и держит минимальный пик памяти.

#include "inference.h"

#include <cstdio>
#include <memory>

#include "engine/graph.h"
#include "engine/loader.h"
#include "engine/parallel.h"

namespace inf {

namespace {

// Форма ii::Shape -> вектор int, как в TensorDesc.
std::vector<int> to_int_shape(const ii::Shape& s) {
    std::vector<int> out(s.size());
    for (std::size_t i = 0; i < s.size(); ++i)
        out[i] = static_cast<int>(s[i]);
    return out;
}

class IiEngine final : public Engine {
public:
    bool load(const std::string& model_path, const Options& opts) override {
        // Число потоков для параллельных ядер движка. 0 -> по числу ядер,
        // 1 -> строго последовательно (см. engine/parallel.h). Берём из
        // Options::num_threads, как и прочие бэкенды интерпретируют --threads.
        ii::set_num_threads(opts.num_threads);

        std::string err;
        if (!ii::load_onnx(model_path, graph_, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return false;
        }
        exec_ = std::make_unique<ii::Executor>(graph_);

        // Входные дескрипторы + буферы. Форму берём из объявленных входов
        // графа (loader проставляет конкретные размеры, динамический батч
        // фиксируем в 1) и выделяем буфер прямо в исполнителе — препроцесс
        // затем пишет в него через input_data() без промежуточных копий.
        in_descs_.clear();
        for (std::size_t i = 0; i < graph_.input_names.size(); ++i) {
            const std::string& name = graph_.input_names[i];
            ii::Shape shp = resolve_shape_(name);
            exec_->input(i) = ii::Tensor(shp);  // нули, нужного размера
            TensorDesc d;
            d.name = name;
            d.shape = to_int_shape(shp);
            d.dtype = DType::Float32;
            d.bytes = static_cast<std::size_t>(ii::numel(shp)) * sizeof(float);
            in_descs_.push_back(d);
        }

        // «Прогрев» с нулями: гоняем граф один раз, чтобы получить реальные
        // формы выходов (надёжнее, чем доверять value_info из модели).
        if (!prime_()) return false;
        return true;
    }

    bool invoke() override {
        if (!exec_) return false;
        if (!exec_->run()) return false;
        // Копируем выходы в собственные буферы: указатель output_data() должен
        // оставаться стабильным между прогонами (контракт inf::Engine).
        for (std::size_t i = 0; i < out_buf_.size(); ++i) {
            const ii::Tensor* o = exec_->output(i);
            if (!o) return false;
            out_buf_[i] = *o;
        }
        return true;
    }

    const std::vector<TensorDesc>& inputs()  const override { return in_descs_; }
    const std::vector<TensorDesc>& outputs() const override { return out_descs_; }

    void* input_data(int idx) override {
        if (idx < 0 || idx >= (int)in_descs_.size()) return nullptr;
        return exec_->input(static_cast<std::size_t>(idx)).data.data();
    }
    const void* output_data(int idx) const override {
        if (idx < 0 || idx >= (int)out_buf_.size()) return nullptr;
        return out_buf_[idx].data.data();
    }

    const char* backend_name() const override { return "ii"; }

private:
    // Конкретная форма входа: объявленная в графе, с заменой неположительных
    // (динамических) осей на 1.
    ii::Shape resolve_shape_(const std::string& name) const {
        auto it = graph_.input_shapes.find(name);
        ii::Shape s = (it == graph_.input_shapes.end()) ? ii::Shape{} : it->second;
        for (auto& d : s) if (d <= 0) d = 1;
        return s;
    }

    bool prime_() {
        if (!exec_->run()) {  // входы уже заполнены нулями в load()
            std::fprintf(stderr, "ii: прогрев графа не удался.\n");
            return false;
        }
        out_descs_.clear();
        out_buf_.clear();
        for (const std::string& name : graph_.output_names) {
            const ii::Tensor* o = exec_->value(name);
            if (!o) {
                std::fprintf(stderr, "ii: выход '%s' не вычислен.\n",
                             name.c_str());
                return false;
            }
            TensorDesc d;
            d.name = name;
            d.shape = to_int_shape(o->shape);
            d.dtype = DType::Float32;
            d.bytes = static_cast<std::size_t>(o->numel()) * sizeof(float);
            out_descs_.push_back(d);
            out_buf_.push_back(*o);
        }
        return true;
    }

    ii::Graph                     graph_;
    std::unique_ptr<ii::Executor> exec_;
    std::vector<TensorDesc>       in_descs_, out_descs_;
    std::vector<ii::Tensor>       out_buf_;  // стабильные буферы выходов
};

}  // namespace

// Фабрика бэкенда `ii`. Регистрируется в inference.cpp через внешнее
// объявление (как и прочие бэкенды), под guard'ом INF_HAS_II.
std::unique_ptr<Engine> make_ii_engine() {
    return std::unique_ptr<Engine>(new IiEngine());
}

}  // namespace inf
