// Реестр ядер + исполнитель графа движка `ii`. Сами ядра (привязка
// ONNX-типов к engine/ops.*) живут в engine/kernels.cpp; здесь — механизм.

#include "engine/graph.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ii {

OpRegistry& OpRegistry::instance() {
    static OpRegistry reg;
    return reg;
}

void OpRegistry::add(const std::string& op_type, Kernel k) {
    ops_[op_type] = std::move(k);
}

const Kernel* OpRegistry::find(const std::string& op_type) const {
    auto it = ops_.find(op_type);
    return it == ops_.end() ? nullptr : &it->second;
}

std::vector<std::string> OpRegistry::registered() const {
    std::vector<std::string> out;
    out.reserve(ops_.size());
    for (const auto& kv : ops_) out.push_back(kv.first);
    return out;
}

// ---- план исполнения + таблица значений ------------------------------------
//
// Всё состояние исполнителя спрятано в State (pimpl): план, построенный один
// раз при конструировании, и таблица слотов, которую переиспользует каждый
// run(). Вспомогательные типы — члены State, поэтому делят его линковку и не
// «протекают» в заголовок.
struct Executor::State {
    using ValueId = int;
    static constexpr ValueId kOmitted = -1;   // опущенный опциональный вход ноды

    // Откуда значение берёт данные и кто владеет его памятью.
    enum class Kind { Constant, Input, Computed, Unbound };

    // Ячейка таблицы значений. Либо смотрит на внешний неизменяемый тензор
    // (вес — без копии и без владения), либо владеет собственным буфером
    // (вход / выход / промежуток). Адрес ячейки стабилен весь срок жизни
    // Executor: вектор слотов выделяется один раз и больше не растёт.
    struct Slot {
        const Tensor* constant = nullptr;     // != null => вес, read-only
        Tensor        owned;                  // собственный буфер
        bool          filled = false;         // owned валиден?
        const Tensor* read() const {
            return constant ? constant : (filled ? &owned : nullptr);
        }
    };

    // Нода, подготовленная к исполнению: разрешённые слоты входов/выходов,
    // закэшированное ядро и промежутки, чья жизнь кончается на этом шаге.
    struct Step {
        const Node*          node = nullptr;
        const Kernel*        kernel = nullptr; // null => добор в execute()
        std::vector<ValueId> inputs;
        std::vector<ValueId> outputs;
        std::vector<ValueId> free_after;       // освободить ПОСЛЕ шага
    };

    explicit State(const Graph& g) : graph(g) { build(); }

    void    build();        // однократно: спланировать исполнение
    bool    execute();      // прогнать план над таблицей слотов
    ValueId id_for(const std::string& nm) const {
        auto it = id_of.find(nm);
        return it == id_of.end() ? kOmitted : it->second;
    }

    const Graph&                             graph;
    std::vector<Step>                        steps;
    std::vector<Kind>                        kind;        // по ValueId
    std::vector<std::string>                 name;        // ValueId -> имя
    std::unordered_map<std::string, ValueId> id_of;       // имя -> ValueId
    std::vector<ValueId>                     input_ids;   // входы графа, по порядку
    std::vector<ValueId>                     output_ids;  // выходы графа, по порядку
    std::vector<Slot>                        slots;       // таблица значений
    std::vector<const Tensor*>               call_inputs; // переиспользуемый буфер
};

void Executor::State::build() {
    ensure_builtin_kernels();
    const OpRegistry& reg = OpRegistry::instance();

    // Завести (или найти) ValueId по имени. Пустое имя — опущенный вход ноды.
    auto intern = [&](const std::string& nm) -> ValueId {
        if (nm.empty()) return kOmitted;
        auto it = id_of.find(nm);
        if (it != id_of.end()) return it->second;
        ValueId id = static_cast<ValueId>(name.size());
        id_of.emplace(nm, id);
        name.push_back(nm);
        kind.push_back(Kind::Unbound);
        return id;
    };

    // Веса и входы графа (вход с именем веса перекрывает его — редкий случай).
    for (const auto& kv : graph.initializers) kind[intern(kv.first)] = Kind::Constant;
    for (const auto& nm : graph.input_names) {
        ValueId id = intern(nm);
        kind[id] = Kind::Input;
        input_ids.push_back(id);
    }

    // Ноды: входы/выходы -> ValueId, ядро кэшируем. Заодно копим продюсеров.
    std::vector<int> producer;        // ValueId -> индекс шага-продюсера (-1)
    std::vector<int> producer_count;  // ValueId -> сколько раз порождён
    auto ensure = [](std::vector<int>& v, std::size_t n, int def) {
        if (v.size() < n) v.resize(n, def);
    };

    steps.reserve(graph.nodes.size());
    for (std::size_t si = 0; si < graph.nodes.size(); ++si) {
        const Node& node = graph.nodes[si];
        Step step;
        step.node = &node;
        step.kernel = reg.find(node.op_type);   // может быть null -> добор в execute()
        for (const auto& in : node.inputs) step.inputs.push_back(intern(in));
        for (const auto& out : node.outputs) {
            ValueId id = intern(out);
            if (kind[id] != Kind::Input) kind[id] = Kind::Computed;
            ensure(producer, static_cast<std::size_t>(id) + 1, -1);
            ensure(producer_count, static_cast<std::size_t>(id) + 1, 0);
            if (producer[id] < 0) producer[id] = static_cast<int>(si);
            ++producer_count[id];
            step.outputs.push_back(id);
        }
        steps.push_back(std::move(step));
    }

    for (const auto& nm : graph.output_names) output_ids.push_back(intern(nm));

    // Время жизни: последний шаг-потребитель каждого значения.
    const std::size_t n = name.size();
    ensure(producer, n, -1);
    ensure(producer_count, n, 0);
    std::vector<int> last_use(n, -1);
    for (std::size_t si = 0; si < steps.size(); ++si)
        for (ValueId id : steps[si].inputs)
            if (id != kOmitted) last_use[id] = static_cast<int>(si);

    std::vector<char> is_output(n, 0);
    for (ValueId id : output_ids)
        if (id != kOmitted) is_output[static_cast<std::size_t>(id)] = 1;

    // Освобождаем только промежутки — не веса, не входы, не выходы графа — и
    // только при единственном продюсере (страховка против не-SSA графов,
    // собранных вручную): иначе можно было бы освободить ещё нужное значение.
    for (ValueId id = 0; id < static_cast<ValueId>(n); ++id) {
        if (kind[id] != Kind::Computed || is_output[id] || producer_count[id] != 1)
            continue;
        int free_at = last_use[id] >= 0 ? last_use[id] : producer[id];
        if (free_at >= 0) steps[static_cast<std::size_t>(free_at)].free_after.push_back(id);
    }

    // Таблица значений: один раз выделяем и сразу привязываем веса/входы.
    slots.resize(n);
    for (ValueId id = 0; id < static_cast<ValueId>(n); ++id) {
        Slot& slot = slots[static_cast<std::size_t>(id)];
        if (kind[id] == Kind::Constant) {
            auto it = graph.initializers.find(name[id]);
            if (it != graph.initializers.end()) slot.constant = &it->second;
        } else if (kind[id] == Kind::Input) {
            slot.filled = true;   // вход всегда владеет буфером; хост заполняет его
        }
    }
}

bool Executor::State::execute() {
    const OpRegistry& reg = OpRegistry::instance();

    for (Step& step : steps) {
        const Kernel* kernel =
            step.kernel ? step.kernel : reg.find(step.node->op_type);
        if (!kernel) {
            std::fprintf(stderr, "ii: нет ядра для операции '%s' (нода '%s')\n",
                         step.node->op_type.c_str(), step.node->name.c_str());
            return false;
        }

        // Собрать входы. kOmitted (пустое имя в ONNX) -> nullptr: ядро само
        // решает, опционален ли вход.
        call_inputs.clear();
        for (ValueId id : step.inputs) {
            if (id == kOmitted) { call_inputs.push_back(nullptr); continue; }
            const Tensor* t = slots[static_cast<std::size_t>(id)].read();
            if (!t) {
                std::fprintf(stderr,
                    "ii: нода '%s' (%s) требует значение '%s', которого нет\n",
                    step.node->name.c_str(), step.node->op_type.c_str(),
                    name[id].c_str());
                return false;
            }
            call_inputs.push_back(t);
        }

        std::vector<Tensor> outs;
        try {
            outs = (*kernel)(call_inputs, *step.node);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ii: ошибка в ноде '%s' (%s): %s\n",
                         step.node->name.c_str(), step.node->op_type.c_str(), e.what());
            return false;
        }
        if (outs.size() != step.outputs.size()) {
            std::fprintf(stderr,
                "ii: нода '%s' (%s) вернула %zu выходов, ожидалось %zu\n",
                step.node->name.c_str(), step.node->op_type.c_str(),
                outs.size(), step.outputs.size());
            return false;
        }

        // Записать выходы в их слоты (становятся владеющими).
        for (std::size_t j = 0; j < outs.size(); ++j) {
            Slot& slot = slots[static_cast<std::size_t>(step.outputs[j])];
            slot.owned    = std::move(outs[j]);
            slot.filled   = true;
            slot.constant = nullptr;
        }
        // Освободить промежутки, чья жизнь закончилась на этом шаге.
        for (ValueId id : step.free_after) {
            Slot& slot = slots[static_cast<std::size_t>(id)];
            slot.owned  = Tensor{};
            slot.filled = false;
        }
    }
    return true;
}

// ---- публичный фасад -------------------------------------------------------
Executor::Executor(const Graph& graph) : state_(std::make_unique<State>(graph)) {}
Executor::~Executor() = default;

std::size_t Executor::input_count() const { return state_->input_ids.size(); }

int Executor::input_index(const std::string& name) const {
    const auto& ids = state_->input_ids;
    for (std::size_t i = 0; i < ids.size(); ++i)
        if (state_->name[static_cast<std::size_t>(ids[i])] == name)
            return static_cast<int>(i);
    return -1;
}

Tensor& Executor::input(std::size_t i) {
    if (i >= state_->input_ids.size())
        throw std::out_of_range("Executor::input: индекс входа вне диапазона");
    return state_->slots[static_cast<std::size_t>(state_->input_ids[i])].owned;
}

bool Executor::run() { return state_->execute(); }

bool Executor::run(const std::unordered_map<std::string, Tensor>& feed) {
    for (const auto& kv : feed) {
        int idx = input_index(kv.first);
        if (idx < 0) {
            std::fprintf(stderr, "ii: '%s' не является входом графа\n",
                         kv.first.c_str());
            return false;
        }
        input(static_cast<std::size_t>(idx)) = kv.second;
    }
    return run();
}

const Tensor* Executor::value(const std::string& name) const {
    State::ValueId id = state_->id_for(name);
    return id == State::kOmitted
               ? nullptr
               : state_->slots[static_cast<std::size_t>(id)].read();
}

const Tensor* Executor::output(std::size_t i) const {
    if (i >= state_->output_ids.size()) return nullptr;
    State::ValueId id = state_->output_ids[i];
    return id == State::kOmitted
               ? nullptr
               : state_->slots[static_cast<std::size_t>(id)].read();
}

}  // namespace ii
