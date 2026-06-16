// Юнит-тесты математических ядер движка `ii` (engine/ops.*).
//
// Это основной щит корректности эталонного движка: значения сверяются с
// посчитанными вручную, а не «сами с собой». Покрыты broadcasting,
// активации, softmax/layernorm, matmul/gemm (вкл. батч и транспонирование),
// свёртка (шаг/паддинг/группы), пулинг, конкатенация, reshape/transpose.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "engine/ops.h"
#include "engine/tensor.h"

using ii::Shape;
using ii::Tensor;

namespace {

// Сравнение содержимого тензора с эталонным буфером (порядок row-major).
void expect_data(const Tensor& t, const std::vector<float>& want,
                 float eps = 1e-4f) {
    ASSERT_EQ(t.data.size(), want.size());
    for (std::size_t i = 0; i < want.size(); ++i)
        EXPECT_NEAR(t.data[i], want[i], eps) << "по индексу " << i;
}

}  // namespace

// ---- broadcasting ----------------------------------------------------------

TEST(Broadcast, ShapeRules) {
    EXPECT_EQ(ii::broadcast_shape({1, 3, 1}, {2, 1, 4}), (Shape{2, 3, 4}));
    EXPECT_EQ(ii::broadcast_shape({5}, {3, 1}), (Shape{3, 5}));
    EXPECT_EQ(ii::broadcast_shape({}, {2, 2}), (Shape{2, 2}));
}

TEST(Broadcast, Incompatible) {
    EXPECT_THROW(ii::broadcast_shape({3}, {4}), std::runtime_error);
}

TEST(Ewise, AddScalarBroadcast) {
    Tensor a(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor b(Shape{1}, std::vector<float>{10});
    expect_data(ii::add(a, b), {11, 12, 13, 14});
}

TEST(Ewise, MulRowBroadcast) {
    // (2,2) * (2,) -> столбцовое умножение по последней оси.
    Tensor a(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor b(Shape{2}, std::vector<float>{10, 100});
    expect_data(ii::mul(a, b), {10, 200, 30, 400});
}

TEST(Ewise, SubDiv) {
    Tensor a(Shape{3}, std::vector<float>{10, 20, 30});
    Tensor b(Shape{3}, std::vector<float>{1, 2, 3});
    expect_data(ii::sub(a, b), {9, 18, 27});
    expect_data(ii::div(a, b), {10, 10, 10});
}

TEST(Ewise, SameShapeFastPath) {
    // Совпадающие многомерные формы идут по быстрому пути (без broadcasting):
    // проверяем все четыре операции поэлементно и сохранение формы.
    Tensor a(Shape{2, 1, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b(Shape{2, 1, 3}, std::vector<float>{10, 20, 30, 40, 50, 60});
    EXPECT_EQ(ii::add(a, b).shape, (Shape{2, 1, 3}));
    expect_data(ii::add(a, b), {11, 22, 33, 44, 55, 66});
    expect_data(ii::sub(b, a), {9, 18, 27, 36, 45, 54});
    expect_data(ii::mul(a, b), {10, 40, 90, 160, 250, 360});
    expect_data(ii::div(b, a), {10, 10, 10, 10, 10, 10});
}

// ---- активации -------------------------------------------------------------

TEST(Activations, ReluSigmoidSilu) {
    Tensor x(Shape{3}, std::vector<float>{-1, 0, 2});
    expect_data(ii::relu(x), {0, 0, 2});
    expect_data(ii::sigmoid(x),
                {1.0f / (1.0f + std::exp(1.0f)), 0.5f,
                 1.0f / (1.0f + std::exp(-2.0f))});
    // silu(0) == 0; silu(2) == 2*sigmoid(2).
    expect_data(ii::silu(x),
                {-1.0f / (1.0f + std::exp(1.0f)), 0.0f,
                 2.0f / (1.0f + std::exp(-2.0f))});
}

TEST(Activations, GeluKnownValues) {
    Tensor x(Shape{3}, std::vector<float>{0, 1, -1});
    // 0.5*x*(1+erf(x/sqrt2)).
    expect_data(ii::gelu(x), {0.0f, 0.8413447f, -0.1586553f});
}

TEST(Activations, LeakyClip) {
    Tensor x(Shape{4}, std::vector<float>{-2, -1, 1, 5});
    expect_data(ii::leaky_relu(x, 0.1f), {-0.2f, -0.1f, 1.0f, 5.0f});
    expect_data(ii::clip(x, -1.0f, 2.0f), {-1, -1, 1, 2});
}

// ---- softmax / layernorm ---------------------------------------------------

TEST(Softmax, TwoClassKnown) {
    // x=[0, ln3] -> exp=[1,3], сумма 4 -> [0.25, 0.75].
    Tensor x(Shape{2}, std::vector<float>{0.0f, std::log(3.0f)});
    expect_data(ii::softmax(x, -1), {0.25f, 0.75f});
}

TEST(Softmax, AlongAxisSumsToOne) {
    Tensor x(Shape{2, 3}, std::vector<float>{1, 2, 3, -1, 0, 1});
    Tensor y = ii::softmax(x, 1);
    for (int r = 0; r < 2; ++r) {
        float s = y.data[r * 3] + y.data[r * 3 + 1] + y.data[r * 3 + 2];
        EXPECT_NEAR(s, 1.0f, 1e-5f);
    }
}

TEST(Softmax, NumericallyStableLargeInputs) {
    // Без вычитания максимума здесь был бы overflow -> NaN.
    Tensor x(Shape{2}, std::vector<float>{1000.0f, 1000.0f});
    expect_data(ii::softmax(x, -1), {0.5f, 0.5f});
}

TEST(LayerNorm, LastAxisUnitScale) {
    Tensor x(Shape{1, 4}, std::vector<float>{1, 2, 3, 4});
    Tensor w(Shape{4}, std::vector<float>{1, 1, 1, 1});
    Tensor b;  // нет смещения
    Tensor y = ii::layer_norm(x, w, b, -1, 1e-5f);
    // (x-2.5)/sqrt(1.25) с поправкой на eps.
    float inv = 1.0f / std::sqrt(1.25f + 1e-5f);
    expect_data(y, {-1.5f * inv, -0.5f * inv, 0.5f * inv, 1.5f * inv}, 1e-3f);
}

// ---- matmul / gemm ---------------------------------------------------------

TEST(MatMul, TwoDimKnown) {
    Tensor a(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor b(Shape{2, 2}, std::vector<float>{5, 6, 7, 8});
    expect_data(ii::matmul(a, b), {19, 22, 43, 50});
    EXPECT_EQ(ii::matmul(a, b).shape, (Shape{2, 2}));
}

TEST(MatMul, NonSquare) {
    Tensor a(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    expect_data(ii::matmul(a, b), {22, 28, 49, 64});
}

TEST(MatMul, BatchedBroadcastRhs) {
    // A:(2,2,2) батч из двух матриц, B:(2,2) — broadcast по батчу.
    Tensor a(Shape{2, 2, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    Tensor b(Shape{2, 2}, std::vector<float>{1, 0, 0, 1});  // единичная
    Tensor y = ii::matmul(a, b);
    EXPECT_EQ(y.shape, (Shape{2, 2, 2}));
    expect_data(y, {1, 2, 3, 4, 5, 6, 7, 8});  // I не меняет
}

TEST(MatMul, VectorTimesMatrix) {
    // 1-D слева: (3) x (3,2) -> (2).
    Tensor a(Shape{3}, std::vector<float>{1, 2, 3});
    Tensor b(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor y = ii::matmul(a, b);
    EXPECT_EQ(y.shape, (Shape{2}));
    expect_data(y, {22, 28});
}

TEST(Gemm, NoTransWithBias) {
    Tensor a(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor c(Shape{2, 2}, std::vector<float>{1, 1, 1, 1});
    expect_data(ii::gemm(a, b, c, 1.0f, 1.0f, false, false),
                {23, 29, 50, 65});
}

TEST(Gemm, TransAEqualsTransposedInput) {
    // A^T форма (2,3) из (3,2) даёт ту же матрицу, что в NoTrans-тесте.
    Tensor a(Shape{3, 2}, std::vector<float>{1, 4, 2, 5, 3, 6});
    Tensor b(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor c;  // нет C, beta игнорируется
    expect_data(ii::gemm(a, b, c, 1.0f, 0.0f, true, false),
                {22, 28, 49, 64});
}

// ---- свёртка ---------------------------------------------------------------

TEST(Conv2D, IdentitySumWindow) {
    Tensor x(Shape{1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor w(Shape{1, 1, 2, 2}, std::vector<float>{1, 1, 1, 1});
    Tensor b;
    ii::Conv2DParams p;  // stride 1, no pad
    Tensor y = ii::conv2d(x, w, b, p);
    EXPECT_EQ(y.shape, (Shape{1, 1, 2, 2}));
    expect_data(y, {12, 16, 24, 28});
}

TEST(Conv2D, WithBias) {
    Tensor x(Shape{1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor w(Shape{1, 1, 2, 2}, std::vector<float>{1, 1, 1, 1});
    Tensor b(Shape{1}, std::vector<float>{1});
    expect_data(ii::conv2d(x, w, b, {}), {13, 17, 25, 29});
}

TEST(Conv2D, StrideAndPadSame) {
    // 3x3 вход, ядро 3x3 единиц, pad 1 со всех сторон, stride 1 -> 3x3 выход
    // (свёртка-«сумма соседей»). Центр = сумма всех 9 = 45.
    Tensor x(Shape{1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor w(Shape{1, 1, 3, 3}, std::vector<float>(9, 1.0f));
    Tensor b;
    ii::Conv2DParams p;
    p.pad = {1, 1, 1, 1};
    Tensor y = ii::conv2d(x, w, b, p);
    EXPECT_EQ(y.shape, (Shape{1, 1, 3, 3}));
    EXPECT_NEAR(y.data[4], 45.0f, 1e-4f);          // центр
    EXPECT_NEAR(y.data[0], 1 + 2 + 4 + 5, 1e-4f);  // угол: 4 валидных
}

TEST(Conv2D, DepthwiseGroups) {
    // 2 канала, groups=2 (depthwise): каждый канал свой фильтр 1x1.
    Tensor x(Shape{1, 2, 1, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor w(Shape{2, 1, 1, 1}, std::vector<float>{10, 100});
    Tensor b;
    ii::Conv2DParams p;
    p.groups = 2;
    Tensor y = ii::conv2d(x, w, b, p);
    EXPECT_EQ(y.shape, (Shape{1, 2, 1, 2}));
    expect_data(y, {10, 20, 300, 400});
}

// ---- пулинг ----------------------------------------------------------------

TEST(Pool, MaxAndAvg2x2) {
    Tensor x(Shape{1, 1, 4, 4},
             std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16});
    ii::PoolParams p;
    p.kernel = {2, 2};
    p.stride = {2, 2};
    expect_data(ii::maxpool2d(x, p), {6, 8, 14, 16});
    expect_data(ii::avgpool2d(x, p), {3.5f, 5.5f, 11.5f, 13.5f});
}

TEST(Pool, GlobalAverage) {
    Tensor x(Shape{1, 1, 4, 4},
             std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16});
    Tensor y = ii::global_average_pool(x);
    EXPECT_EQ(y.shape, (Shape{1, 1, 1, 1}));
    expect_data(y, {8.5f});
}

// ---- перестановки формы ----------------------------------------------------

TEST(Concat, Axis1) {
    Tensor a(Shape{1, 2}, std::vector<float>{1, 2});
    Tensor b(Shape{1, 3}, std::vector<float>{3, 4, 5});
    Tensor y = ii::concat({&a, &b}, 1);
    EXPECT_EQ(y.shape, (Shape{1, 5}));
    expect_data(y, {1, 2, 3, 4, 5});
}

TEST(Concat, Axis0Channels) {
    Tensor a(Shape{1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor b(Shape{1, 1, 2, 2}, std::vector<float>{5, 6, 7, 8});
    Tensor y = ii::concat({&a, &b}, 1);  // склейка по каналам
    EXPECT_EQ(y.shape, (Shape{1, 2, 2, 2}));
    expect_data(y, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(Reshape, InferAndZero) {
    Tensor x(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    EXPECT_EQ(ii::reshape(x, {-1}).shape, (Shape{6}));
    EXPECT_EQ(ii::reshape(x, {3, -1}).shape, (Shape{3, 2}));
    EXPECT_EQ(ii::reshape(x, {0, -1}).shape, (Shape{2, 3}));  // 0 = копия оси
    EXPECT_THROW(ii::reshape(x, {4, 2}), std::runtime_error);
}

TEST(Transpose, TwoDim) {
    Tensor x(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor y = ii::transpose(x, {1, 0});
    EXPECT_EQ(y.shape, (Shape{3, 2}));
    expect_data(y, {1, 4, 2, 5, 3, 6});
}

TEST(Transpose, DefaultReverses) {
    Tensor x(Shape{2, 1, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor y = ii::transpose(x, {});
    EXPECT_EQ(y.shape, (Shape{3, 1, 2}));
}

TEST(Upsample, NearestX2) {
    Tensor x(Shape{1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor y = ii::upsample_nearest(x, 2, 2);
    EXPECT_EQ(y.shape, (Shape{1, 1, 4, 4}));
    expect_data(y, {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
}

// ---- доп. поэлементные -----------------------------------------------------

TEST(MoreElementwise, UnaryAndBinary) {
    Tensor x(Shape{2}, std::vector<float>{4, 9});
    expect_data(ii::sqrt_(x), {2, 3});
    expect_data(ii::exp_(Tensor(Shape{2}, std::vector<float>{0, 1})), {1.0f, std::exp(1.0f)});
    expect_data(ii::abs_(Tensor(Shape{2}, std::vector<float>{-2, 3})), {2, 3});
    expect_data(ii::neg(Tensor(Shape{2}, std::vector<float>{1, -2})), {-1, 2});
    expect_data(ii::reciprocal(Tensor(Shape{2}, std::vector<float>{2, 4})), {0.5f, 0.25f});

    Tensor a(Shape{2}, std::vector<float>{2, 3}), b(Shape{2}, std::vector<float>{3, 2});
    expect_data(ii::pow_(a, b), {8, 9});
    expect_data(ii::min_(Tensor(Shape{2}, std::vector<float>{1, 5}),
                         Tensor(Shape{2}, std::vector<float>{3, 2})), {1, 2});
    expect_data(ii::max_(Tensor(Shape{2}, std::vector<float>{1, 5}),
                         Tensor(Shape{2}, std::vector<float>{3, 2})), {3, 5});
}

// ---- resize / split / slice / gather / shape / reduce ----------------------

TEST(Resize, NearestScales) {
    Tensor x(Shape{1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor y = ii::resize_nearest(x, {1, 1, 2, 2});
    EXPECT_EQ(y.shape, (Shape{1, 1, 4, 4}));
    expect_data(y, {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
}

TEST(Resize, NearestModeRoundPreferCeil) {
    // scale 1.5 по последней оси: 2 -> 3. coord asymmetric, round_prefer_ceil.
    // o=0 -> 0; o=1 -> round(0.667)=1; o=2 -> round(1.333)=1.
    Tensor x(Shape{3}, std::vector<float>{10, 20, 30});
    Tensor y = ii::resize(x, {1.5f}, ii::ResizeMode::Nearest,
                          ii::ResizeCoord::Asymmetric,
                          ii::ResizeNearest::RoundPreferCeil);
    EXPECT_EQ(y.shape, (Shape{4}));
    expect_data(y, {10, 20, 20, 30});  // floor(3*1.5)=4 точек
}

TEST(Resize, LinearHalfPixelUpsample) {
    // [1,3] -> 4 по ширине, half_pixel (как в OpenCV/torch bilinear).
    Tensor x(Shape{1, 1, 1, 2}, std::vector<float>{1, 3});
    Tensor y = ii::resize(x, {1, 1, 1, 2}, ii::ResizeMode::Linear,
                          ii::ResizeCoord::HalfPixel);
    EXPECT_EQ(y.shape, (Shape{1, 1, 1, 4}));
    expect_data(y, {1.0f, 1.5f, 2.5f, 3.0f});
}

TEST(Resize, LinearAlignCorners) {
    // align_corners сохраняет крайние значения: [1,3] -> [1, 1.667, 2.333, 3].
    Tensor x(Shape{1, 1, 1, 2}, std::vector<float>{1, 3});
    Tensor y = ii::resize(x, {1, 1, 1, 2}, ii::ResizeMode::Linear,
                          ii::ResizeCoord::AlignCorners);
    EXPECT_EQ(y.shape, (Shape{1, 1, 1, 4}));
    expect_data(y, {1.0f, 1.0f + 2.0f / 3.0f, 1.0f + 4.0f / 3.0f, 3.0f});
}

TEST(Resize, LinearBilinear2D) {
    // 2x2 -> 4x4 bilinear, align_corners: углы сохранены, центр интерполирован.
    Tensor x(Shape{1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor y = ii::resize(x, {1, 1, 2, 2}, ii::ResizeMode::Linear,
                          ii::ResizeCoord::AlignCorners);
    EXPECT_EQ(y.shape, (Shape{1, 1, 4, 4}));
    EXPECT_NEAR(y.data[0], 1.0f, 1e-4f);    // угол (0,0)
    EXPECT_NEAR(y.data[3], 2.0f, 1e-4f);    // угол (0,3)
    EXPECT_NEAR(y.data[12], 3.0f, 1e-4f);   // угол (3,0)
    EXPECT_NEAR(y.data[15], 4.0f, 1e-4f);   // угол (3,3)
    EXPECT_NEAR(y.data[5], 2.0f, 1e-4f);    // (1/3,1/3) -> 2.0
}

TEST(Split, AlongAxis) {
    Tensor x(Shape{2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto parts = ii::split(x, 1, {2, 2});
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0].shape, (Shape{2, 2}));
    expect_data(parts[0], {1, 2, 5, 6});
    expect_data(parts[1], {3, 4, 7, 8});
}

TEST(Slice, PositiveAndNegativeStep) {
    Tensor x(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor y = ii::slice(x, {1}, {3}, {1}, {});  // столбцы 1..2
    EXPECT_EQ(y.shape, (Shape{2, 2}));
    expect_data(y, {2, 3, 5, 6});

    Tensor v(Shape{5}, std::vector<float>{0, 1, 2, 3, 4});
    Tensor r = ii::slice(v, {4}, {-6}, {0}, {-1});  // реверс
    EXPECT_EQ(r.shape, (Shape{5}));
    expect_data(r, {4, 3, 2, 1, 0});
}

TEST(Gather, Axis0) {
    Tensor x(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor idx(Shape{2}, std::vector<float>{2, 0});
    Tensor y = ii::gather(x, idx, 0);
    EXPECT_EQ(y.shape, (Shape{2, 2}));
    expect_data(y, {5, 6, 1, 2});
}

TEST(ShapeUnsqueezeSqueeze, Basics) {
    Tensor x(Shape{2, 3, 4});
    expect_data(ii::shape_of(x), {2, 3, 4});

    Tensor v(Shape{3}, std::vector<float>{1, 2, 3});
    EXPECT_EQ(ii::unsqueeze(v, {0}).shape, (Shape{1, 3}));
    EXPECT_EQ(ii::unsqueeze(v, {1}).shape, (Shape{3, 1}));
    Tensor w(Shape{1, 3, 1}, std::vector<float>{1, 2, 3});
    EXPECT_EQ(ii::squeeze(w, {}).shape, (Shape{3}));
    EXPECT_EQ(ii::squeeze(w, {0}).shape, (Shape{3, 1}));
}

TEST(Reduce, MeanSumMaxAlongAxis) {
    Tensor x(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    expect_data(ii::reduce(x, {1}, false, 0), {6, 15});       // sum
    expect_data(ii::reduce(x, {1}, false, 1), {2, 5});        // mean
    expect_data(ii::reduce(x, {1}, false, 2), {3, 6});        // max
    EXPECT_EQ(ii::reduce(x, {1}, true, 0).shape, (Shape{2, 1}));
    // редукция по всем осям -> скаляр.
    Tensor all = ii::reduce(x, {}, false, 0);
    EXPECT_TRUE(all.shape.empty());
    expect_data(all, {21});
}

TEST(Reduce, MinKindAndKeepdims) {
    Tensor x(Shape{2, 3}, std::vector<float>{1, 5, 2, 4, 0, 6});
    expect_data(ii::reduce(x, {1}, false, 3), {1, 0});  // min по оси
    // keepdims сохраняет редуцируемую ось как 1.
    Tensor k = ii::reduce(x, {1}, true, 3);
    EXPECT_EQ(k.shape, (Shape{2, 1}));
    expect_data(k, {1, 0});
    // редукция по всем осям с keepdims -> форма из единиц.
    Tensor allk = ii::reduce(x, {}, true, 2);  // max всех
    EXPECT_EQ(allk.shape, (Shape{1, 1}));
    expect_data(allk, {6});
}

TEST(Reduce, MultiAxis3D) {
    // (2,2,2): сумма по осям 0 и 2 -> остаётся ось 1 размера 2.
    Tensor x(Shape{2, 2, 2},
             std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    // элементы с j=0: индексы 0,1,4,5 = 1+2+5+6 = 14; j=1: 3+4+7+8 = 22.
    Tensor y = ii::reduce(x, {0, 2}, false, 0);
    EXPECT_EQ(y.shape, (Shape{2}));
    expect_data(y, {14, 22});
}

// ---- активации: оставшиеся ветки ------------------------------------------

TEST(Activations, TanhAndScalars) {
    Tensor x(Shape{3}, std::vector<float>{-1, 0, 2});
    expect_data(ii::tanh_(x), {std::tanh(-1.0f), 0.0f, std::tanh(2.0f)});
    expect_data(ii::add_scalar(x, 10.0f), {9, 10, 12});
    expect_data(ii::mul_scalar(x, -2.0f), {2, 0, -4});
}

TEST(Activations, SigmoidStableLargeNegative) {
    // exp(+inf) не должен давать NaN: sigmoid(-100) ~ 0, sigmoid(100) ~ 1.
    Tensor x(Shape{2}, std::vector<float>{-100.0f, 100.0f});
    Tensor y = ii::sigmoid(x);
    EXPECT_NEAR(y.data[0], 0.0f, 1e-6f);
    EXPECT_NEAR(y.data[1], 1.0f, 1e-6f);
    EXPECT_FALSE(std::isnan(y.data[0]));
}

TEST(Activations, ClipOneSidedInfinity) {
    Tensor x(Shape{4}, std::vector<float>{-5, -1, 1, 5});
    // только нижняя граница (как Clip с min, без max).
    expect_data(ii::clip(x, 0.0f, INFINITY), {0, 0, 1, 5});
    // только верхняя граница.
    expect_data(ii::clip(x, -INFINITY, 0.0f), {-5, -1, 0, 0});
}

// ---- softmax: строчный (strided) путь и ошибка оси ------------------------

TEST(Softmax, MiddleAxisStrided) {
    // (2,2), softmax по оси 0 -> inner==2, ходим по входу с шагом.
    Tensor x(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor y = ii::softmax(x, 0);
    float a = std::exp(-2.0f), b = 1.0f, s = a + b;  // столбец [v, v+2]
    expect_data(y, {a / s, a / s, b / s, b / s});
}

TEST(Softmax, AxisOutOfRangeThrows) {
    Tensor x(Shape{2, 3});
    EXPECT_THROW(ii::softmax(x, 5), std::runtime_error);
}

// ---- layernorm: смещение и не-последняя ось -------------------------------

TEST(LayerNorm, WithWeightAndBias) {
    Tensor x(Shape{1, 4}, std::vector<float>{1, 2, 3, 4});
    Tensor w(Shape{4}, std::vector<float>{2, 2, 2, 2});
    Tensor b(Shape{4}, std::vector<float>{1, 1, 1, 1});
    Tensor y = ii::layer_norm(x, w, b, -1, 1e-5f);
    float inv = 1.0f / std::sqrt(1.25f + 1e-5f);
    expect_data(y, {-1.5f * 2 * inv + 1, -0.5f * 2 * inv + 1,
                    0.5f * 2 * inv + 1, 1.5f * 2 * inv + 1}, 1e-3f);
}

TEST(LayerNorm, NormalizesTailFromAxis) {
    // axis=1 на (2,2): каждая строка нормируется отдельно по 2 элементам.
    Tensor x(Shape{2, 2}, std::vector<float>{1, 3, 10, 14});
    Tensor w(Shape{2}, std::vector<float>{1, 1});
    Tensor b;
    Tensor y = ii::layer_norm(x, w, b, 1, 0.0f);
    // строка [1,3]: mean 2, var 1 -> [-1, 1]; строка [10,14]: mean 12, var 4 -> [-1, 1].
    expect_data(y, {-1, 1, -1, 1}, 1e-4f);
}

TEST(LayerNorm, SizeMismatchThrows) {
    Tensor x(Shape{1, 4}, std::vector<float>{1, 2, 3, 4});
    Tensor w(Shape{3}, std::vector<float>{1, 1, 1});  // не совпадает с хвостом 4
    Tensor b;
    EXPECT_THROW(ii::layer_norm(x, w, b, -1, 1e-5f), std::runtime_error);
}

// ---- matmul / gemm: оставшиеся ветки --------------------------------------

TEST(MatMul, MatrixTimesVector) {
    // (2,3) x (3) -> (2): правый операнд 1-D промоутится к (3,1).
    Tensor a(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b(Shape{3}, std::vector<float>{1, 1, 1});
    Tensor y = ii::matmul(a, b);
    EXPECT_EQ(y.shape, (Shape{2}));
    expect_data(y, {6, 15});
}

TEST(MatMul, VectorDotVector) {
    // (3) x (3) -> скаляр (оба 1-D схлопываются).
    Tensor a(Shape{3}, std::vector<float>{1, 2, 3});
    Tensor b(Shape{3}, std::vector<float>{4, 5, 6});
    Tensor y = ii::matmul(a, b);
    EXPECT_TRUE(y.shape.empty());
    expect_data(y, {32});
}

TEST(MatMul, BatchedBroadcastLhs) {
    // A:(2,2) единичная, B:(2,2,2) батч — broadcast A по батчу -> B без изменений.
    Tensor a(Shape{2, 2}, std::vector<float>{1, 0, 0, 1});
    Tensor b(Shape{2, 2, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    Tensor y = ii::matmul(a, b);
    EXPECT_EQ(y.shape, (Shape{2, 2, 2}));
    expect_data(y, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(MatMul, InnerMismatchThrows) {
    Tensor a(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    EXPECT_THROW(ii::matmul(a, b), std::runtime_error);
}

TEST(Gemm, AlphaBetaScaling) {
    Tensor a(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor b(Shape{2, 2}, std::vector<float>{1, 0, 0, 1});  // единичная
    Tensor c(Shape{2, 2}, std::vector<float>{1, 1, 1, 1});
    // 2*A*I + 3*C = 2*[1,2,3,4] + [3,3,3,3].
    expect_data(ii::gemm(a, b, c, 2.0f, 3.0f, false, false), {5, 7, 9, 11});
}

TEST(Gemm, TransBEqualsTransposedRhs) {
    // B^T: B:(2,3) транспонируется к (3,2). A:(2,3) x B^T:(3,2) -> (2,2).
    Tensor a(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor c;
    expect_data(ii::gemm(a, b, c, 1.0f, 0.0f, false, true), {14, 32, 32, 77});
}

TEST(Gemm, KMismatchThrows) {
    Tensor a(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor b(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor c;
    EXPECT_THROW(ii::gemm(a, b, c, 1.0f, 1.0f, false, false), std::runtime_error);
}

// ---- свёртка: оставшиеся пути (1x1, im2col Cg>1, дилатация, ошибки) --------

TEST(Conv2D, Pointwise1x1MultiChannel) {
    // 1x1-свёртка по 2 каналам, 2 фильтра-селектора (no_im2col-путь, Cg>1).
    Tensor x(Shape{1, 2, 2, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    Tensor w(Shape{2, 2, 1, 1}, std::vector<float>{1, 0, 0, 1});  // f0->ch0, f1->ch1
    Tensor b;
    Tensor y = ii::conv2d(x, w, b, {});
    EXPECT_EQ(y.shape, (Shape{1, 2, 2, 2}));
    expect_data(y, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(Conv2D, MultiChannelIm2col) {
    // 2 входных канала, ядро 2x2 единиц, без паддинга -> сумма всех 8 значений.
    Tensor x(Shape{1, 2, 2, 2}, std::vector<float>{1, 2, 3, 4, 10, 20, 30, 40});
    Tensor w(Shape{1, 2, 2, 2}, std::vector<float>(8, 1.0f));
    Tensor b;
    Tensor y = ii::conv2d(x, w, b, {});
    EXPECT_EQ(y.shape, (Shape{1, 1, 1, 1}));
    expect_data(y, {110});
}

TEST(Conv2D, Dilation) {
    // depthwise-путь (1 канал) с дилатацией 2: ядро 2x2 «видит» углы 3x3.
    Tensor x(Shape{1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor w(Shape{1, 1, 2, 2}, std::vector<float>{1, 1, 1, 1});
    Tensor b;
    ii::Conv2DParams p;
    p.dilation = {2, 2};
    Tensor y = ii::conv2d(x, w, b, p);  // OH=OW=1
    EXPECT_EQ(y.shape, (Shape{1, 1, 1, 1}));
    expect_data(y, {1 + 3 + 7 + 9});  // 20
}

TEST(Conv2D, MultiBatch) {
    // Два изображения в батче, depthwise 1x1 умножение на 10.
    Tensor x(Shape{2, 1, 1, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor w(Shape{1, 1, 1, 1}, std::vector<float>{10});
    Tensor b;
    Tensor y = ii::conv2d(x, w, b, {});
    EXPECT_EQ(y.shape, (Shape{2, 1, 1, 2}));
    expect_data(y, {10, 20, 30, 40});
}

TEST(Conv2D, BadShapeThrows) {
    Tensor x2d(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    Tensor w(Shape{1, 1, 2, 2}, std::vector<float>{1, 1, 1, 1});
    Tensor b;
    EXPECT_THROW(ii::conv2d(x2d, w, b, {}), std::runtime_error);  // не 4-D

    // каналы входа не согласованы с весами/группами.
    Tensor x(Shape{1, 3, 2, 2}, std::vector<float>(12, 1.0f));
    Tensor wbad(Shape{1, 2, 1, 1}, std::vector<float>{1, 1});  // ждёт C=2
    EXPECT_THROW(ii::conv2d(x, wbad, b, {}), std::runtime_error);
}

// ---- пулинг: паддинг и count_include_pad ----------------------------------

TEST(Pool, AvgCountIncludePad) {
    // Один пиксель, окно 2x2 с паддингом справа/снизу: 1 валидный + 3 «нулевых».
    Tensor x(Shape{1, 1, 1, 1}, std::vector<float>{4});
    ii::PoolParams p;
    p.kernel = {2, 2};
    p.stride = {1, 1};
    p.pad = {0, 0, 1, 1};  // bottom, right
    p.count_include_pad = false;
    expect_data(ii::avgpool2d(x, p), {4});       // среднее только по валидным
    p.count_include_pad = true;
    expect_data(ii::avgpool2d(x, p), {1});       // (4+0+0+0)/4
}

TEST(Pool, MaxWithPadding) {
    // Паддинг не влияет на max (нули-паддинги не учитываются в окне).
    Tensor x(Shape{1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4});
    ii::PoolParams p;
    p.kernel = {2, 2};
    p.stride = {1, 1};
    p.pad = {1, 1, 0, 0};  // top, left
    Tensor y = ii::maxpool2d(x, p);
    EXPECT_EQ(y.shape, (Shape{1, 1, 2, 2}));
    // верхний-левый выход «видит» только x[0,0]=1; центр -> max всех = 4.
    EXPECT_NEAR(y.data[0], 1.0f, 1e-4f);
    EXPECT_NEAR(y.data[3], 4.0f, 1e-4f);
}

// ---- concat: 3 входа и ошибки несовместимости -----------------------------

TEST(Concat, ThreeInputsAndMismatch) {
    Tensor a(Shape{2, 1}, std::vector<float>{1, 4});
    Tensor b(Shape{2, 2}, std::vector<float>{2, 3, 5, 6});
    Tensor c(Shape{2, 1}, std::vector<float>{7, 8});
    Tensor y = ii::concat({&a, &b, &c}, 1);
    EXPECT_EQ(y.shape, (Shape{2, 4}));
    expect_data(y, {1, 2, 3, 7, 4, 5, 6, 8});

    Tensor bad(Shape{3, 1}, std::vector<float>{0, 0, 0});  // расходится по оси 0
    EXPECT_THROW(ii::concat({&a, &bad}, 1), std::runtime_error);
    Tensor diffnd(Shape{2}, std::vector<float>{0, 0});
    EXPECT_THROW(ii::concat({&a, &diffnd}, 1), std::runtime_error);
}

// ---- transpose: проверка данных 3-D ---------------------------------------

TEST(Transpose, ThreeDimData) {
    Tensor x(Shape{2, 2, 2}, std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    Tensor y = ii::transpose(x, {1, 0, 2});  // меняем местами две внешние оси
    EXPECT_EQ(y.shape, (Shape{2, 2, 2}));
    expect_data(y, {0, 1, 4, 5, 2, 3, 6, 7});
    EXPECT_THROW(ii::transpose(x, {0, 1}), std::runtime_error);  // длина perm != ndim
}

// ---- resize: оставшиеся режимы nearest ------------------------------------

TEST(Resize, NearestFloorDownscale) {
    // scale 0.5, asymmetric, floor: [10,20,30,40] -> [10,30].
    Tensor x(Shape{4}, std::vector<float>{10, 20, 30, 40});
    Tensor y = ii::resize(x, {0.5f}, ii::ResizeMode::Nearest,
                          ii::ResizeCoord::Asymmetric, ii::ResizeNearest::Floor);
    EXPECT_EQ(y.shape, (Shape{2}));
    expect_data(y, {10, 30});
}

TEST(Resize, NearestCeilUpscale) {
    Tensor x(Shape{2}, std::vector<float>{10, 20});
    Tensor y = ii::resize(x, {1.5f}, ii::ResizeMode::Nearest,
                          ii::ResizeCoord::Asymmetric, ii::ResizeNearest::Ceil);
    EXPECT_EQ(y.shape, (Shape{3}));      // floor(2*1.5)
    expect_data(y, {10, 20, 20});        // o=0->0; o=1->ceil(.667)=1; o=2->ceil(1.333)=2->clamp1
}

// ---- slice / split / gather: оставшиеся ветки и ошибки ---------------------

TEST(Slice, ExplicitAxis) {
    Tensor x(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor y = ii::slice(x, {0}, {1}, {0}, {});  // первая строка по оси 0
    EXPECT_EQ(y.shape, (Shape{1, 3}));
    expect_data(y, {1, 2, 3});
}

TEST(Slice, ZeroStepThrows) {
    Tensor x(Shape{4}, std::vector<float>{1, 2, 3, 4});
    EXPECT_THROW(ii::slice(x, {0}, {4}, {0}, {0}), std::runtime_error);
}

TEST(Split, SumMismatchThrows) {
    Tensor x(Shape{2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_THROW(ii::split(x, 1, {2, 1}), std::runtime_error);  // 3 != 4
}

TEST(Gather, Axis1Negative) {
    Tensor x(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor idx(Shape{2}, std::vector<float>{-1, 0});  // последний и первый столбцы
    Tensor y = ii::gather(x, idx, 1);
    EXPECT_EQ(y.shape, (Shape{2, 2}));
    expect_data(y, {3, 1, 6, 4});
}

TEST(Gather, IndexOutOfRangeThrows) {
    Tensor x(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    Tensor idx(Shape{1}, std::vector<float>{5});  // вне [0,3)
    EXPECT_THROW(ii::gather(x, idx, 0), std::runtime_error);
}
