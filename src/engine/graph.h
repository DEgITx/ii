// Граф вычислений движка `ii` + реестр ядер + исполнитель.
//
// Идея расширяемости: граф хранит ноды в backend-нейтральном виде
// (тип операции строкой в стиле ONNX — "Conv", "MatMul", "Add", ...),
// а исполнитель для каждой ноды ищет ядро в OpRegistry по этому типу.
// Чтобы добавить новый слой (вплоть до блоков трансформера), достаточно
// зарегистрировать ядро через register_op(...) — трогать исполнитель,
// загрузчик ONNX и остальной движок не нужно.
//
// Загрузчики моделей (ONNX — engine/onnx.cpp; TFLite — позже) наполняют
// Graph; мост к inf::Engine (engine/backend.cpp) гоняет его через Executor.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/tensor.h"

namespace ii {

// Атрибут ноды. ONNX-атрибуты бывают целочисленные/вещественные/строковые
// и их списки; здесь храним все варианты разом — что заполнено, то и
// читается через типизированные геттеры Node.
struct Attribute {
    std::vector<std::int64_t> ints;
    std::vector<float>        floats;
    std::string               s;
};

// Нода графа: тип операции + имена входных/выходных значений + атрибуты.
// Имена значений — ключи в общей таблице тензоров исполнителя.
struct Node {
    std::string                               op_type;
    std::string                               name;
    std::vector<std::string>                  inputs;
    std::vector<std::string>                  outputs;
    std::unordered_map<std::string, Attribute> attrs;

    bool has(const std::string& k) const { return attrs.count(k) != 0; }

    std::vector<std::int64_t> ints(const std::string& k,
                                   std::vector<std::int64_t> def = {}) const {
        auto it = attrs.find(k);
        return it == attrs.end() ? def : it->second.ints;
    }
    std::int64_t i(const std::string& k, std::int64_t def) const {
        auto it = attrs.find(k);
        return (it == attrs.end() || it->second.ints.empty())
                   ? def : it->second.ints[0];
    }
    float f(const std::string& k, float def) const {
        auto it = attrs.find(k);
        return (it == attrs.end() || it->second.floats.empty())
                   ? def : it->second.floats[0];
    }
    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = attrs.find(k);
        return it == attrs.end() ? def : it->second.s;
    }
};

// Граф: топологически упорядоченные ноды + веса-константы (initializers) +
// имена входов/выходов всего графа. shape входов может быть неполной
// (динамический батч) — реальная форма приходит при подаче данных.
struct Graph {
    std::vector<Node>                       nodes;
    std::unordered_map<std::string, Tensor> initializers;
    std::vector<std::string>                input_names;
    std::vector<std::string>                output_names;
    // Объявленные формы входов (для дескрипторов inf::Engine; -1 == динам.).
    std::unordered_map<std::string, Shape>  input_shapes;
};

// Сигнатура ядра: по входным тензорам (указатели на значения в таблице
// исполнителя) и самой ноде (атрибуты) вернуть выходные тензоры по порядку.
// Бросать std::runtime_error при ошибке — исполнитель добавит контекст.
using Kernel = std::function<std::vector<Tensor>(
    const std::vector<const Tensor*>&, const Node&)>;

// Глобальный реестр ядер: тип операции -> ядро. Заполняется встроенными
// ядрами (engine/kernels.cpp) и может дополняться пользователем движка.
class OpRegistry {
public:
    static OpRegistry& instance();
    void          add(const std::string& op_type, Kernel k);
    const Kernel* find(const std::string& op_type) const;
    std::vector<std::string> registered() const;

private:
    std::unordered_map<std::string, Kernel> ops_;
};

// Сахар для регистрации ядра из любого модуля.
inline void register_op(const std::string& op_type, Kernel k) {
    OpRegistry::instance().add(op_type, std::move(k));
}

// Гарантирует, что встроенные ядра (engine/kernels.cpp) зарегистрированы.
// Вызывается исполнителем; идемпотентна и потокобезопасна на первом вызове.
void ensure_builtin_kernels();

// Исполнитель графа: одна загруженная модель = один Executor. Не
// потокобезопасен (как и весь inf::Engine) — load/run последовательны.
class Executor {
public:
    explicit Executor(const Graph& g);

    // Прогнать граф на заданных входах (имя -> тензор). Возвращает false
    // и пишет причину в stderr при ошибке (неизвестная нода, несовпадение
    // форм и т.п.). Результаты доступны через output()/value().
    bool run(const std::unordered_map<std::string, Tensor>& inputs);

    // Тензор по имени значения (вход, выход или промежуточный) после run().
    const Tensor* value(const std::string& name) const;
    // Тензор i-го выхода графа (по порядку output_names).
    const Tensor* output(std::size_t i) const;

private:
    const Graph&                            g_;
    std::unordered_map<std::string, Tensor> vals_;
};

}  // namespace ii
