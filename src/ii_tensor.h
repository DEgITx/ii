// Базовый тип тензора для собственного движка инференса `ii`.
//
// Движок `ii` — это backend-нейтральная реализация inf::Engine, написанная
// на чистом C++20 без единого SDK. Он всегда доступен в любой сборке на
// любой платформе (см. опцию USE_II_ENGINE), служит эталоном корректности
// для проверки тяжёлых бэкендов (TFLite/TensorRT/DirectML) и площадкой,
// куда удобно дописывать новые слои — вплоть до трансформеров.
//
// Весь счёт внутри `ii` идёт в float32: эталону важнее простота и
// предсказуемость, чем экономия памяти. Квантованные веса конвертируются
// в float на этапе загрузки модели. Раскладка — строго row-major (C-order),
// как в ONNX/NumPy.
//
// Заголовок не тянет ничего из inference.h — типы `ii` самостоятельны;
// мост к inf::Engine живёт отдельно в inference_ii.cpp.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ii {

// Форма тензора — список размерностей, row-major. int64_t, чтобы без потерь
// принимать ONNX-данные (там размерности — int64).
using Shape = std::vector<std::int64_t>;

// Число элементов = произведение размерностей. Для скаляра (пустой shape)
// произведение пустого множества == 1.
inline std::int64_t numel(const Shape& s) {
    std::int64_t n = 1;
    for (std::int64_t d : s) n *= d;
    return n;
}

// Нормализация оси: ONNX допускает отрицательные индексы осей
// (-1 == последняя). Приводим к диапазону [0, ndim).
inline int norm_axis(std::int64_t axis, int ndim) {
    if (axis < 0) axis += ndim;
    return static_cast<int>(axis);
}

// "[1,3,224,224]" — для логов/сообщений об ошибках.
std::string shape_to_string(const Shape& s);

// Плотный тензор float32 в row-major раскладке.
//
// data.size() всегда равно numel(shape) — инвариант поддерживается
// конструкторами и reshape-операциями. Прямой доступ к data разрешён:
// ядра математики работают с буфером напрямую ради скорости.
struct Tensor {
    Shape              shape;
    std::vector<float> data;

    Tensor() = default;

    // Тензор заданной формы, заполненный fill (по умолчанию нулями).
    explicit Tensor(Shape s, float fill = 0.0f)
        : shape(std::move(s)),
          data(static_cast<std::size_t>(ii::numel(shape)), fill) {}

    // Тензор из готового буфера. Вызывающий гарантирует
    // d.size() == numel(s) (проверяется в debug-сборке).
    Tensor(Shape s, std::vector<float> d)
        : shape(std::move(s)), data(std::move(d)) {}

    std::int64_t numel() const { return ii::numel(shape); }
    int          ndim()  const { return static_cast<int>(shape.size()); }

    // Размер оси с поддержкой отрицательного индекса (-1 == последняя).
    std::int64_t dim(int axis) const {
        return shape[static_cast<std::size_t>(norm_axis(axis, ndim()))];
    }

    float&       operator[](std::size_t i)       { return data[i]; }
    const float& operator[](std::size_t i) const { return data[i]; }

    bool empty() const { return data.empty(); }
};

// Строчные (row-major) шаги по элементам для формы s: stride[i] —
// сколько элементов буфера занимает один шаг по оси i. Последняя ось — 1.
// Используется ядрами для индексации многомерных тензоров.
std::vector<std::int64_t> row_major_strides(const Shape& s);

}  // namespace ii
