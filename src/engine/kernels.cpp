// Встроенные ядра движка `ii`: привязка операций ONNX-стиля к чистым
// функциям из engine/ops.*. Это «таблица слоёв» движка — чтобы добавить новый
// слой, достаточно дописать сюда один register_op(...) (или вызвать его из
// своего модуля); граф и исполнитель при этом не меняются.
//
// Покрыты обе ветки: CNN-детекторы (Conv/Pool/Concat/Upsample — YOLOv8) и
// строительные блоки трансформеров (MatMul/Gemm/Softmax/LayerNorm/Gelu).

#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/graph.h"
#include "engine/ops.h"

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

// Паддинг ONNX auto_pad=SAME_* по одной пространственной оси: {begin, end}.
// Гарантирует out = ceil(in / stride); лишний пиксель при нечётной сумме идёт
// в конец (SAME_UPPER) или в начало (SAME_LOWER).
std::array<std::int64_t, 2> same_pad(std::int64_t in, std::int64_t k,
                                     std::int64_t stride, std::int64_t dilation,
                                     bool upper) {
    std::int64_t eff = dilation * (k - 1) + 1;          // эффективный размер окна
    std::int64_t out = (in + stride - 1) / stride;       // ceil(in/stride)
    std::int64_t total = (out - 1) * stride + eff - in;
    if (total < 0) total = 0;
    std::int64_t half = total / 2;
    return upper ? std::array<std::int64_t, 2>{half, total - half}
                 : std::array<std::int64_t, 2>{total - half, half};
}

// Разложить auto_pad/pads в [top,left,bottom,right]. spatial — {H,W} входа,
// win — {kH,kW} окна (для Conv — из весов, для пулинга — из kernel_shape).
// stride/dilation нужны только для SAME. Если auto_pad нет/NOTSET — берём pads.
std::array<std::int64_t, 4> resolve_pads(
    const Node& n, std::array<std::int64_t, 2> spatial,
    std::array<std::int64_t, 2> win, std::array<std::int64_t, 2> stride,
    std::array<std::int64_t, 2> dilation, bool have_spatial) {
    std::string ap = n.str("auto_pad", "NOTSET");
    if (!ap.empty() && ap != "NOTSET") {
        if (ap == "VALID") return {0, 0, 0, 0};
        if ((ap == "SAME_UPPER" || ap == "SAME_LOWER") && have_spatial) {
            bool up = (ap == "SAME_UPPER");
            auto h = same_pad(spatial[0], win[0], stride[0], dilation[0], up);
            auto w = same_pad(spatial[1], win[1], stride[1], dilation[1], up);
            return {h[0], w[0], h[1], w[1]};  // top,left,bottom,right
        }
        throw std::runtime_error("неподдержанный auto_pad='" + ap + "'");
    }
    auto pad = n.ints("pads");  // [top,left,bottom,right]
    if (pad.size() == 4) return {pad[0], pad[1], pad[2], pad[3]};
    return {0, 0, 0, 0};
}

// Развернуть ONNX-атрибуты в Conv2DParams (по умолчанию — без паддинга,
// шаг/дилатация 1, одна группа). x/w нужны для auto_pad=SAME_*.
Conv2DParams conv_params(const Node& n, const Tensor& x, const Tensor& w) {
    Conv2DParams p;
    auto s = n.ints("strides");
    if (s.size() == 2) { p.stride = {s[0], s[1]}; }
    auto d = n.ints("dilations");
    if (d.size() == 2) { p.dilation = {d[0], d[1]}; }
    p.groups = n.i("group", 1);
    bool ok = x.ndim() == 4 && w.ndim() == 4;
    p.pad = resolve_pads(
        n, {ok ? x.shape[2] : 0, ok ? x.shape[3] : 0},
        {ok ? w.shape[2] : 0, ok ? w.shape[3] : 0}, p.stride, p.dilation, ok);
    return p;
}

PoolParams pool_params(const Node& n, const Tensor& x) {
    PoolParams p;
    auto k = n.ints("kernel_shape");
    if (k.size() == 2) { p.kernel = {k[0], k[1]}; }
    auto s = n.ints("strides");
    p.stride = (s.size() == 2) ? std::array<std::int64_t, 2>{s[0], s[1]}
                               : p.kernel;  // ONNX: по умолчанию шаг = окно
    p.count_include_pad = n.i("count_include_pad", 0) != 0;
    bool ok = x.ndim() == 4;
    p.pad = resolve_pads(n, {ok ? x.shape[2] : 0, ok ? x.shape[3] : 0}, p.kernel,
                         p.stride, {1, 1}, ok);  // пулинг без дилатации
    return p;
}

// Разбор режимов ONNX Resize из атрибутов ноды.
ResizeMode resize_mode(const Node& n) {
    std::string m = n.str("mode", "nearest");
    if (m == "nearest") return ResizeMode::Nearest;
    if (m == "linear" || m == "bilinear") return ResizeMode::Linear;
    throw std::runtime_error("Resize: неподдержанный mode='" + m +
                             "' (есть nearest, linear)");
}
ResizeCoord resize_coord(const Node& n) {
    std::string c = n.str("coordinate_transformation_mode", "half_pixel");
    if (c == "half_pixel")         return ResizeCoord::HalfPixel;
    if (c == "pytorch_half_pixel") return ResizeCoord::PytorchHalfPixel;
    if (c == "align_corners")      return ResizeCoord::AlignCorners;
    if (c == "asymmetric")         return ResizeCoord::Asymmetric;
    throw std::runtime_error(
        "Resize: неподдержанный coordinate_transformation_mode='" + c + "'");
}
ResizeNearest resize_nmode(const Node& n) {
    std::string m = n.str("nearest_mode", "round_prefer_floor");
    if (m == "round_prefer_floor") return ResizeNearest::RoundPreferFloor;
    if (m == "round_prefer_ceil")  return ResizeNearest::RoundPreferCeil;
    if (m == "floor")              return ResizeNearest::Floor;
    if (m == "ceil")               return ResizeNearest::Ceil;
    throw std::runtime_error("Resize: неподдержанный nearest_mode='" + m + "'");
}

// Тензор-форма (ONNX хранит её int64; у нас буфер float — округляем).
Shape to_shape(const Tensor& t) {
    Shape s(t.data.size());
    for (std::size_t i = 0; i < t.data.size(); ++i)
        s[i] = static_cast<std::int64_t>(std::llround(t.data[i]));
    return s;
}

// То же, но как «список целых» (для starts/ends/axes/scales-индексов и пр.).
std::vector<std::int64_t> ints_of(const Tensor& t) {
    std::vector<std::int64_t> v(t.data.size());
    for (std::size_t i = 0; i < t.data.size(); ++i)
        v[i] = static_cast<std::int64_t>(std::llround(t.data[i]));
    return v;
}

// Список int из входа i (opset≥13/≥10/≥18) или из атрибута name (старые
// opset'ы) — многие ONNX-операции мигрировали параметры из атрибутов во входы.
std::vector<std::int64_t> ints_in_or_attr(const std::vector<const Tensor*>& in,
                                          std::size_t i, const Node& n,
                                          const char* attr) {
    if (i < in.size() && in[i] && !in[i]->empty()) return ints_of(*in[i]);
    return n.ints(attr);
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
        return one(conv2d(x, w, b ? *b : kEmpty, conv_params(n, x, w)));
    });
    register_op("MaxPool", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "X");
        return one(maxpool2d(x, pool_params(n, x)));
    });
    register_op("AveragePool", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "X");
        return one(avgpool2d(x, pool_params(n, x)));
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

    // --- доп. поэлементные (YOLOv8-головы, трансформеры) ---
    register_op("Exp",        [](auto& in, const Node&) { return one(exp_(need(in, 0, "X"))); });
    register_op("Sqrt",       [](auto& in, const Node&) { return one(sqrt_(need(in, 0, "X"))); });
    register_op("Abs",        [](auto& in, const Node&) { return one(abs_(need(in, 0, "X"))); });
    register_op("Neg",        [](auto& in, const Node&) { return one(neg(need(in, 0, "X"))); });
    register_op("Reciprocal", [](auto& in, const Node&) { return one(reciprocal(need(in, 0, "X"))); });
    register_op("Pow", [](auto& in, const Node&) { return one(pow_(need(in, 0, "A"), need(in, 1, "B"))); });
    register_op("Min", [](auto& in, const Node&) { return one(min_(need(in, 0, "A"), need(in, 1, "B"))); });
    register_op("Max", [](auto& in, const Node&) { return one(max_(need(in, 0, "A"), need(in, 1, "B"))); });

    // --- resize (neck upsample, image-to-image) ---
    register_op("Resize", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "X");
        const Tensor* sc = opt(in, 2);   // scales (вход 2)
        const Tensor* sz = opt(in, 3);   // sizes  (вход 3)
        std::vector<float> scales;
        if (sc && !sc->empty()) {
            for (float v : sc->data) scales.push_back(v);
        } else if (sz && !sz->empty()) {
            if (sz->numel() != x.ndim())
                throw std::runtime_error("Resize: длина sizes != ndim входа");
            for (int i = 0; i < x.ndim(); ++i)
                scales.push_back(sz->data[(std::size_t)i] / (float)x.shape[i]);
        } else {
            throw std::runtime_error("Resize: нет ни scales, ни sizes");
        }
        return one(resize(x, scales, resize_mode(n), resize_coord(n),
                          resize_nmode(n)));
    });

    // --- split (C2f) ---
    register_op("Split", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "X");
        std::int64_t axis = n.i("axis", 0);
        std::vector<std::int64_t> sizes = ints_in_or_attr(in, 1, n, "split");
        if (sizes.empty()) {  // поровну на число выходов
            std::int64_t parts = (std::int64_t)n.outputs.size();
            std::int64_t dim = x.dim((int)axis);
            for (std::int64_t k = 0; k < parts; ++k)
                sizes.push_back(dim / parts + (k == parts - 1 ? dim % parts : 0));
        }
        return split(x, axis, sizes);
    });

    // --- slice ---
    register_op("Slice", [](auto& in, const Node& n) {
        const Tensor& x = need(in, 0, "data");
        auto starts = ints_in_or_attr(in, 1, n, "starts");
        auto ends   = ints_in_or_attr(in, 2, n, "ends");
        auto axes   = ints_in_or_attr(in, 3, n, "axes");
        auto steps  = ints_in_or_attr(in, 4, n, "steps");
        return one(slice(x, starts, ends, axes, steps));
    });

    // --- gather / shape / (un)squeeze ---
    register_op("Gather", [](auto& in, const Node& n) {
        return one(gather(need(in, 0, "data"), need(in, 1, "indices"), n.i("axis", 0)));
    });
    register_op("Shape", [](auto& in, const Node&) {
        return one(shape_of(need(in, 0, "X")));
    });
    register_op("Unsqueeze", [](auto& in, const Node& n) {
        return one(unsqueeze(need(in, 0, "X"), ints_in_or_attr(in, 1, n, "axes")));
    });
    register_op("Squeeze", [](auto& in, const Node& n) {
        return one(squeeze(need(in, 0, "X"), ints_in_or_attr(in, 1, n, "axes")));
    });

    // --- редукции (axes из входа или атрибута, keepdims по умолчанию 1) ---
    struct RedReg {
        static Kernel make(int kind) {
            return [kind](const std::vector<const Tensor*>& in, const Node& n) {
                auto axes = ints_in_or_attr(in, 1, n, "axes");
                bool keep = n.i("keepdims", 1) != 0;
                return one(reduce(need(in, 0, "X"), axes, keep, kind));
            };
        }
    };
    register_op("ReduceSum",  RedReg::make(0));
    register_op("ReduceMean", RedReg::make(1));
    register_op("ReduceMax",  RedReg::make(2));
    register_op("ReduceMin",  RedReg::make(3));

    // --- «прозрачные» операции, частые в shape-подграфах экспортов ---
    register_op("Identity", [](auto& in, const Node&) {
        return one(need(in, 0, "X"));
    });
    register_op("Cast", [](auto& in, const Node& n) {
        // Движок считает всё во float32. Cast к целому типу усекаем к нулю
        // (ONNX-семантика float->int), к вещественному — проброс как есть.
        // ONNX DataType: 6=int32,7=int64,9=bool,12=uint32,13=uint64,5=int16,
        // 3=int8,2=uint8.
        const Tensor& x = need(in, 0, "X");
        std::int64_t to = n.i("to", 1);
        bool to_int = (to == 2 || to == 3 || to == 5 || to == 6 ||
                       to == 7 || to == 9 || to == 12 || to == 13);
        if (!to_int) return one(x);
        Tensor o(x.shape);
        for (std::size_t i = 0; i < x.data.size(); ++i)
            o.data[i] = std::trunc(x.data[i]);
        return one(std::move(o));
    });
}

}  // namespace

void ensure_builtin_kernels() {
    static std::once_flag once;
    std::call_once(once, register_all);
}

}  // namespace ii
