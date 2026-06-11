// Математические ядра движка `ii`.
//
// Чистые функции над ii::Tensor: каждая принимает входы по const-ссылке и
// возвращает новый тензор. Никакого внутреннего состояния, никакой
// зависимости от графа — ровно это делает их легко и подробно тестируемыми
// (см. tests/test_ii_ops.cpp). Слой графа (ii_graph.*) лишь раскладывает
// атрибуты ONNX-нод в эти вызовы.
//
// Соглашения:
//   * счёт в float32, раскладка row-major (C-order);
//   * NCHW для свёрток/пулинга (как в ONNX);
//   * broadcasting — по правилам NumPy/ONNX (выравнивание форм справа);
//   * при несовместимых формах бросается std::runtime_error (executor
//     ловит и сообщает, какая нода упала).
//
// Набор намеренно покрывает обе ветки: CNN (Conv/Pool/Concat — YOLOv8) и
// строительные блоки трансформеров (MatMul/Softmax/LayerNorm/Gelu).

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "ii_tensor.h"

namespace ii {

// ---- Поэлементные бинарные операции (с broadcasting) ----------------------
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// Итоговая форма broadcasting двух форм (или исключение при несовместимости).
// Вынесена в API: полезна и в тестах, и в проверках графа.
Shape broadcast_shape(const Shape& a, const Shape& b);

// ---- Активации (поэлементно) ----------------------------------------------
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor silu(const Tensor& x);                       // x * sigmoid(x), aka Swish
Tensor tanh_(const Tensor& x);
Tensor leaky_relu(const Tensor& x, float alpha = 0.01f);
Tensor clip(const Tensor& x, float lo, float hi);
Tensor gelu(const Tensor& x);                       // точный, через erf
Tensor add_scalar(const Tensor& x, float v);
Tensor mul_scalar(const Tensor& x, float v);

// ---- Softmax / нормировки -------------------------------------------------
// Softmax вдоль оси axis (по умолчанию последняя). Численно устойчив:
// вычитает максимум по оси перед exp.
Tensor softmax(const Tensor& x, std::int64_t axis = -1);

// LayerNorm по последним осям, начиная с axis: нормирует среднее/дисперсию,
// затем масштабирует на weight и сдвигает на bias (оба формы нормируемого
// хвоста, либо bias может быть пустым). eps — стабилизатор дисперсии.
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  std::int64_t axis = -1, float eps = 1e-5f);

// ---- Матричные операции ---------------------------------------------------
// Батч-матумножение по семантике ONNX MatMul: последние две оси —
// матрица (M,K)x(K,N)->(M,N); ведущие оси («батч») broadcast'ятся.
// 1-D входы трактуются как в NumPy (вектор слева — (1,K) с последующим
// схлопыванием, справа — (K,1)).
Tensor matmul(const Tensor& a, const Tensor& b);

// ONNX Gemm: Y = alpha*op(A)*op(B) + beta*C, A/B двумерны, op — опц.
// транспонирование, C broadcast'ится к (M,N) (может быть пустым).
Tensor gemm(const Tensor& a, const Tensor& b, const Tensor& c,
            float alpha = 1.0f, float beta = 1.0f,
            bool trans_a = false, bool trans_b = false);

// ---- Свёртка / пулинг (NCHW) ----------------------------------------------
struct Conv2DParams {
    std::array<std::int64_t, 2> stride{1, 1};      // {sH, sW}
    std::array<std::int64_t, 4> pad{0, 0, 0, 0};   // {top, left, bottom, right}
    std::array<std::int64_t, 2> dilation{1, 1};    // {dH, dW}
    std::int64_t                groups = 1;
};

// x:(N,C,H,W), w:(M, C/groups, kH, kW), bias:(M) или пустой.
Tensor conv2d(const Tensor& x, const Tensor& w, const Tensor& bias,
              const Conv2DParams& p);

struct PoolParams {
    std::array<std::int64_t, 2> kernel{1, 1};      // {kH, kW}
    std::array<std::int64_t, 2> stride{1, 1};
    std::array<std::int64_t, 4> pad{0, 0, 0, 0};   // {top, left, bottom, right}
    bool count_include_pad = false;                // только для avgpool
};

Tensor maxpool2d(const Tensor& x, const PoolParams& p);
Tensor avgpool2d(const Tensor& x, const PoolParams& p);
Tensor global_average_pool(const Tensor& x);       // (N,C,H,W)->(N,C,1,1)

// ---- Перестановки формы ---------------------------------------------------
// Конкатенация по оси axis (формы совпадают по всем прочим осям).
Tensor concat(const std::vector<const Tensor*>& xs, std::int64_t axis);

// Reshape с поддержкой -1 (вывести из общего числа) и 0 (скопировать
// размерность входа на этой позиции, ONNX-семантика allowzero=0).
Tensor reshape(const Tensor& x, const Shape& new_shape);

// Перестановка осей по perm (длина perm == ndim). Пустой perm == реверс осей.
Tensor transpose(const Tensor& x, const std::vector<std::int64_t>& perm);

// Resize ближайшим соседом по пространственным осям H,W (NCHW),
// целочисленные масштабы sh,sw (типовой upsample в детекторах).
Tensor upsample_nearest(const Tensor& x, std::int64_t sh, std::int64_t sw);

}  // namespace ii
