// Реестр ядер + исполнитель графа движка `ii`. Сами ядра (привязка
// ONNX-типов к ii_ops.*) живут в ii_kernels.cpp; здесь — механизм.

#include "ii_graph.h"

#include <cstdio>

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

Executor::Executor(const Graph& g) : g_(g) {}

bool Executor::run(const std::unordered_map<std::string, Tensor>& inputs) {
    ensure_builtin_kernels();
    vals_.clear();

    // Затравка таблицы значений: сначала веса-константы, затем поданные
    // входы (вход с тем же именем перекрывает константу — это норм).
    for (const auto& kv : g_.initializers) vals_[kv.first] = kv.second;
    for (const auto& kv : inputs)           vals_[kv.first] = kv.second;

    const OpRegistry& reg = OpRegistry::instance();

    for (const Node& node : g_.nodes) {
        const Kernel* k = reg.find(node.op_type);
        if (!k) {
            std::fprintf(stderr,
                "ii: нет ядра для операции '%s' (нода '%s')\n",
                node.op_type.c_str(), node.name.c_str());
            return false;
        }
        // Собрать входы. Пустое имя входа в ONNX == «вход опущен» —
        // передаём nullptr, ядро решает, опционален ли он.
        std::vector<const Tensor*> ins;
        ins.reserve(node.inputs.size());
        for (const std::string& in : node.inputs) {
            if (in.empty()) { ins.push_back(nullptr); continue; }
            auto it = vals_.find(in);
            if (it == vals_.end()) {
                std::fprintf(stderr,
                    "ii: нода '%s' (%s) требует значение '%s', которого нет\n",
                    node.name.c_str(), node.op_type.c_str(), in.c_str());
                return false;
            }
            ins.push_back(&it->second);
        }

        std::vector<Tensor> outs;
        try {
            outs = (*k)(ins, node);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ii: ошибка в ноде '%s' (%s): %s\n",
                         node.name.c_str(), node.op_type.c_str(), e.what());
            return false;
        }
        if (outs.size() != node.outputs.size()) {
            std::fprintf(stderr,
                "ii: нода '%s' (%s) вернула %zu выходов, ожидалось %zu\n",
                node.name.c_str(), node.op_type.c_str(),
                outs.size(), node.outputs.size());
            return false;
        }
        for (std::size_t i = 0; i < outs.size(); ++i)
            vals_[node.outputs[i]] = std::move(outs[i]);
    }
    return true;
}

const Tensor* Executor::value(const std::string& name) const {
    auto it = vals_.find(name);
    return it == vals_.end() ? nullptr : &it->second;
}

const Tensor* Executor::output(std::size_t i) const {
    if (i >= g_.output_names.size()) return nullptr;
    return value(g_.output_names[i]);
}

}  // namespace ii
