// Встроенные ядра движка `ii`: привязка операций ONNX-стиля к чистым
// функциям из ii_ops.*. Это «таблица слоёв» движка — чтобы добавить новый
// слой, достаточно дописать сюда один register_op(...) (или вызвать его из
// своего модуля); граф и исполнитель при этом не меняются.
//
// Покрыты обе ветки: CNN-детекторы (Conv/Pool/Concat/Upsample — YOLOv8) и
// строительные блоки трансформеров (MatMul/Gemm/Softmax/LayerNorm/Gelu).

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "ii_graph.h"
#include "ii_ops.h"

namespace ii {

namespace {

// Обязательный вход ноды: бросает, если ONNX опустил его (nullptr).
const Tensor& need(const std::vector<const Tensor*>& in, std::size_t i,
                   const char* what) {
    if (i >= in.size() || in[i] == nullptr)
        throw std::runtime_error(std::string("отсутствует вход: ") + what);
    return *in[i];
}

// Опциональный вход: nullptr, если отсутствует.
const Tensor* opt(const std::vector<const Tensor*>& in, std::size_t i) {
    return i < in.size() ? in[i] : nullptr;
}

const Tensor kEmpty;  // пустой тензор для «нет bias/нет C»

// Развернуть ONNX-атрибуты в Conv2DParams (по умолчанию — без паддинга,
// шаг/дилатация 1, одна группа).
Conv2DParams conv_params(const Node& n) {
    Conv2DParams p;
    auto s = n.ints("strides");
    if (s.size() == 2) { p.stride = {s[0], s[1]}; }
    auto d = n.ints("dilations");
    if (d.size() == 2) { p.dilation = {d[0], d[1]}; }
    auto pad = n.ints("pads");  // [top,left,bottom,right]
    if (pad.size() == 4) { p.pad = {pad[0], pad[1], pad[2], pad[3]}; }
    p.groups = n.i("group", 1);
    return p;
}

PoolParams pool_params(const Node& n) {
    PoolParams p;
    auto k = n.ints("kernel_shape");
    if (k.size() == 2) { p.kernel = {k[0], k[1]}; }
    auto s = n.ints("strides");
    p.stride = (s.size() == 2) ? std::array<std::int64_t, 2>{s[0], s[1]}
                               : p.kernel;  // ONNX: по умолчанию шаг = окно
    auto pad = n.ints("pads");
    if (pad.size() == 4) { p.pad = {pad[0], pad[1], pad[2], pad[3]}; }
    p.count_include_pad = n.i("count_include_pad", 0) != 0;
    return p;
}

// Тензор-форма (ONNX хранит её int64; у нас буфер float — округляем).
Shape to_shape(const Tensor& t) {
    Shape s(t.data.size());
    for (std::size_t i = 0; i < t.data.size(); ++i)
        s[i] = static_cast<std::int64_t>(std::llround(t.data[i]));
    return s;
}

std::vector<Tensor> one(Tensor t) { return std::vector<Tensor>{std::move(t)}; }

void register_all() {
    // --- поэлементная арифметика ---
    register_op("Add", [](auto& in, const Node&) {
        return one(add(need(in, 0, "A"), need(in, 1, "B")));
    });
    register_op("Sub", [](auto& in, const Node&) {
        return one(sub(need(in, 0, "A"), need(in, 1, "B")));
    });
    register_op("Mul", [](auto& in, const Node&) {
        return one(mul(need(in, 0, "A"), need(in, 1, "B")));
    });
    register_op("Div", [](auto& in, const Node&) {
        return one(div(need(in, 0, "A"), need(in, 1, "B")));
    });

    // --- активации ---
    register_op("Relu", [](auto& in, const Node&) {
        return one(relu(need(in, 0, "X")));
    });
    register_op("Sigmoid", [](auto& in, const Node&) {
        return one(sigmoid(need(in, 0, "X")));
    });
    register_op("Tanh", [](auto& in, const Node&) {
        return one(tanh_(need(in, 0, "X")));
    });
    // SiLU экспортируется и как "Silu", и как HardSwish-сосед; держим явное имя.
    register_op("Silu", [](auto& in, const Node&) {
        return one(silu(need(in, 0, "X")));
    });
    register_op("LeakyRelu", [](auto& in, const Node& n) {
        return one(leaky_relu(need(in, 0, "X"), n.f("alpha", 0.01f)));
    });
    register_op("Gelu", [](auto& in, const Node&) {
        return one(gelu(need(in, 0, "X")));
    });
    register_op("Erf", [](auto& in, const Node&) {
        // На случай, если GELU экспортирован как явная формула с Erf.
        const Tensor& x = need(in, 0, "X");
        Tensor o(x.shape);
        for (std::size_t i = 0; i < x.data.size(); ++i)
            o.data[i] = std::erf(x.data[i]);
        return one(std::move(o));
    });
    register_op("Clip", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "X");
        const Tensor* lo = opt(in, 1);
        const Tensor* hi = opt(in, 2);
        float lov = lo && !lo->empty() ? lo->data[0] : n.f("min", -INFINITY);
        float hiv = hi && !hi->empty() ? hi->data[0] : n.f("max", INFINITY);
        return one(clip(x, lov, hiv));
    });

    // --- softmax / нормировки ---
    register_op("Softmax", [](auto& in, const Node& n) {
        return one(softmax(need(in, 0, "X"), n.i("axis", -1)));
    });
    register_op("LayerNormalization", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "X");
        const Tensor& w = need(in, 1, "Scale");
        const Tensor* b = opt(in, 2);
        return one(layer_norm(x, w, b ? *b : kEmpty,
                              n.i("axis", -1), n.f("epsilon", 1e-5f)));
    });

    // --- матумножение ---
    register_op("MatMul", [](auto& in, const Node&) {
        return one(matmul(need(in, 0, "A"), need(in, 1, "B")));
    });
    register_op("Gemm", [](auto& in, const Node& n) {
        const Tensor& a = need(in, 0, "A");
        const Tensor& b = need(in, 1, "B");
        const Tensor* c = opt(in, 2);
        return one(gemm(a, b, c ? *c : kEmpty,
                        n.f("alpha", 1.0f), n.f("beta", 1.0f),
                        n.i("transA", 0) != 0, n.i("transB", 0) != 0));
    });

    // --- свёртка / пулинг ---
    register_op("Conv", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "X");
        const Tensor& w = need(in, 1, "W");
        const Tensor* b = opt(in, 2);
        return one(conv2d(x, w, b ? *b : kEmpty, conv_params(n)));
    });
    register_op("MaxPool", [](auto& in, const Node& n) {
        return one(maxpool2d(need(in, 0, "X"), pool_params(n)));
    });
    register_op("AveragePool", [](auto& in, const Node& n) {
        return one(avgpool2d(need(in, 0, "X"), pool_params(n)));
    });
    register_op("GlobalAveragePool", [](auto& in, const Node&) {
        return one(global_average_pool(need(in, 0, "X")));
    });

    // --- перестановки формы ---
    register_op("Concat", [](auto& in, const Node& n) {
        std::vector<const Tensor*> xs;
        for (const Tensor* t : in) if (t) xs.push_back(t);
        return one(concat(xs, n.i("axis", 0)));
    });
    register_op("Reshape", [](auto& in, const Node&) {
        const Tensor& x = need(in, 0, "data");
        const Tensor& s = need(in, 1, "shape");
        return one(reshape(x, to_shape(s)));
    });
    register_op("Flatten", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "X");
        int ax = norm_axis(n.i("axis", 1), x.ndim());
        std::int64_t outer = 1, inner = 1;
        for (int i = 0; i < ax; ++i) outer *= x.shape[i];
        for (int i = ax; i < x.ndim(); ++i) inner *= x.shape[i];
        return one(reshape(x, Shape{outer, inner}));
    });
    register_op("Transpose", [](auto& in, const Node& n) {
        return one(transpose(need(in, 0, "X"), n.ints("perm")));
    });
}

}  // namespace

void ensure_builtin_kernels() {
    static std::once_flag once;
    std::call_once(once, register_all);
}

}  // namespace ii
