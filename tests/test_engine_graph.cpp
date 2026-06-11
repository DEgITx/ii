// Тесты графа/исполнителя движка `ii` (engine/graph.* + engine/kernels.*).
//
// Графы собираются вручную (без загрузчика ONNX — он тестируется отдельно),
// чтобы проверить именно механику: топологический прогон, диспетч по
// реестру ядер, разбор атрибутов нод, проброс ошибок.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "engine/graph.h"
#include "engine/ops.h"
#include "engine/tensor.h"

using ii::Attribute;
using ii::Executor;
using ii::Graph;
using ii::Node;
using ii::Shape;
using ii::Tensor;

namespace {

Node mk(const std::string& op, std::vector<std::string> ins,
        std::vector<std::string> outs, const std::string& name = "") {
    Node n;
    n.op_type = op;
    n.inputs = std::move(ins);
    n.outputs = std::move(outs);
    n.name = name.empty() ? op : name;
    return n;
}

}  // namespace

// Маленький MLP: y = softmax(relu(x*W1 + b1) * W2 + b2). Проверяем, что
// исполнитель проходит цепочку и выдаёт корректное распределение.
TEST(Executor, TinyMLP) {
    Graph g;
    g.input_names = {"x"};
    g.output_names = {"y"};

    // W1 = I(2), b1 = 0  -> relu(x). W2 = I(2), b2 = 0 -> softmax(relu(x)).
    g.initializers["W1"] = Tensor(Shape{2, 2}, std::vector<float>{1, 0, 0, 1});
    g.initializers["b1"] = Tensor(Shape{2}, std::vector<float>{0, 0});
    g.initializers["W2"] = Tensor(Shape{2, 2}, std::vector<float>{1, 0, 0, 1});
    g.initializers["b2"] = Tensor(Shape{2}, std::vector<float>{0, 0});

    g.nodes.push_back(mk("Gemm", {"x", "W1", "b1"}, {"h"}));
    g.nodes.push_back(mk("Relu", {"h"}, {"r"}));
    g.nodes.push_back(mk("Gemm", {"r", "W2", "b2"}, {"o"}));
    g.nodes.push_back(mk("Softmax", {"o"}, {"y"}));

    Executor ex(g);
    std::unordered_map<std::string, Tensor> inputs;
    // x = [[0, ln3]] -> relu без изменений -> softmax = [0.25, 0.75].
    inputs["x"] = Tensor(Shape{1, 2}, std::vector<float>{0.0f, std::log(3.0f)});

    ASSERT_TRUE(ex.run(inputs));
    const Tensor* y = ex.output(0);
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(y->shape, (Shape{1, 2}));
    EXPECT_NEAR(y->data[0], 0.25f, 1e-4f);
    EXPECT_NEAR(y->data[1], 0.75f, 1e-4f);
}

// CNN-путь: Conv -> Relu -> MaxPool с разбором атрибутов нод.
TEST(Executor, ConvReluPool) {
    Graph g;
    g.input_names = {"x"};
    g.output_names = {"y"};
    g.initializers["w"] =
        Tensor(Shape{1, 1, 2, 2}, std::vector<float>{1, 1, 1, 1});

    Node conv = mk("Conv", {"x", "w"}, {"c"});
    conv.attrs["strides"] = Attribute{{1, 1}, {}, ""};
    g.nodes.push_back(conv);
    g.nodes.push_back(mk("Relu", {"c"}, {"r"}));
    Node pool = mk("MaxPool", {"r"}, {"y"});
    pool.attrs["kernel_shape"] = Attribute{{2, 2}, {}, ""};
    pool.attrs["strides"] = Attribute{{1, 1}, {}, ""};
    g.nodes.push_back(pool);

    Executor ex(g);
    std::unordered_map<std::string, Tensor> in;
    // 4x4 -> conv2x2(sum) даёт 3x3 -> maxpool2x2 s1 -> 2x2.
    in["x"] = Tensor(Shape{1, 1, 4, 4},
                     std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8,
                                        9, 10, 11, 12, 13, 14, 15, 16});
    ASSERT_TRUE(ex.run(in));
    const Tensor* y = ex.output(0);
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(y->shape, (Shape{1, 1, 2, 2}));
    // conv: суммы окон 2x2; затем max по окну 2x2 шага 1.
    // conv-выход:
    //   14 18 22 / 30 34 38 / 46 50 54
    // maxpool 2x2 s1: 34 38 / 50 54.
    EXPECT_NEAR(y->data[0], 34.0f, 1e-4f);
    EXPECT_NEAR(y->data[1], 38.0f, 1e-4f);
    EXPECT_NEAR(y->data[2], 50.0f, 1e-4f);
    EXPECT_NEAR(y->data[3], 54.0f, 1e-4f);
}

// Композиция в стиле YOLOv8: Conv -> SiLU(Sigmoid+Mul) -> MaxPool ->
// Conv -> Resize(upsample) -> Concat со «skip» -> Conv-голова -> Reshape ->
// Sigmoid. Проверяем, что весь набор слоёв стыкуется и даёт конечный
// тензор правильной формы со значениями в [0,1].
TEST(Executor, YoloLikeGraph) {
    Graph g;
    g.input_names = {"x"};
    g.output_names = {"y"};
    // Лёгкие детерминированные веса (важна стыковка форм, не числа).
    g.initializers["w1"] = Tensor(Shape{8, 3, 3, 3}, 0.05f);
    g.initializers["w2"] = Tensor(Shape{8, 8, 3, 3}, 0.05f);
    g.initializers["wh"] = Tensor(Shape{4, 16, 1, 1}, 0.05f);
    g.initializers["scales"] = Tensor(Shape{4}, std::vector<float>{1, 1, 2, 2});
    g.initializers["newshape"] = Tensor(Shape{3}, std::vector<float>{1, 4, 256});

    auto pad1 = [](Node n) {
        n.attrs["pads"] = Attribute{{1, 1, 1, 1}, {}, ""};
        n.attrs["strides"] = Attribute{{1, 1}, {}, ""};
        return n;
    };
    // stem: Conv -> SiLU -> MaxPool/2
    g.nodes.push_back(pad1(mk("Conv", {"x", "w1"}, {"c1"})));
    g.nodes.push_back(mk("Sigmoid", {"c1"}, {"sg1"}));
    g.nodes.push_back(mk("Mul", {"c1", "sg1"}, {"a1"}));   // a1 = SiLU(c1) [1,8,16,16]
    Node pool = mk("MaxPool", {"a1"}, {"p1"});
    pool.attrs["kernel_shape"] = Attribute{{2, 2}, {}, ""};
    pool.attrs["strides"] = Attribute{{2, 2}, {}, ""};
    g.nodes.push_back(pool);                                // p1 [1,8,8,8]
    // deeper conv + upsample back to 16x16
    g.nodes.push_back(pad1(mk("Conv", {"p1", "w2"}, {"c2"})));
    g.nodes.push_back(mk("Resize", {"c2", "", "scales"}, {"up"}));  // [1,8,16,16]
    // concat skip(a1) + up -> [1,16,16,16] -> head conv -> [1,4,16,16]
    Node cat = mk("Concat", {"a1", "up"}, {"cc"});
    cat.attrs["axis"] = Attribute{{1}, {}, ""};
    g.nodes.push_back(cat);
    g.nodes.push_back(mk("Conv", {"cc", "wh"}, {"hd"}));    // 1x1, [1,4,16,16]
    g.nodes.push_back(mk("Reshape", {"hd", "newshape"}, {"flat"}));  // [1,4,256]
    g.nodes.push_back(mk("Sigmoid", {"flat"}, {"y"}));

    Executor ex(g);
    std::unordered_map<std::string, Tensor> in;
    in["x"] = Tensor(Shape{1, 3, 16, 16}, 0.5f);
    ASSERT_TRUE(ex.run(in));
    const Tensor* y = ex.output(0);
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(y->shape, (Shape{1, 4, 256}));
    for (float v : y->data) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
        EXPECT_FALSE(std::isnan(v));
    }
}

// Identity — проброс; Cast к int — усечение к нулю.
TEST(Executor, IdentityAndCast) {
    Graph g;
    g.input_names = {"x"};
    g.output_names = {"id", "ci"};
    g.nodes.push_back(mk("Identity", {"x"}, {"id"}));
    Node cast = mk("Cast", {"x"}, {"ci"});
    cast.attrs["to"] = Attribute{{7}, {}, ""};  // INT64 -> усечение
    g.nodes.push_back(cast);
    Executor ex(g);
    std::unordered_map<std::string, Tensor> in;
    in["x"] = Tensor(Shape{3}, std::vector<float>{1.9f, -1.9f, 2.0f});
    ASSERT_TRUE(ex.run(in));
    EXPECT_EQ(ex.value("id")->data, (std::vector<float>{1.9f, -1.9f, 2.0f}));
    EXPECT_EQ(ex.value("ci")->data, (std::vector<float>{1, -1, 2}));
}

TEST(Executor, UnknownOpFails) {
    Graph g;
    g.input_names = {"x"};
    g.output_names = {"y"};
    g.nodes.push_back(mk("NoSuchOp", {"x"}, {"y"}));
    Executor ex(g);
    std::unordered_map<std::string, Tensor> in;
    in["x"] = Tensor(Shape{1}, std::vector<float>{1});
    EXPECT_FALSE(ex.run(in));
}

TEST(Executor, MissingValueFails) {
    Graph g;
    g.input_names = {"x"};
    g.output_names = {"y"};
    g.nodes.push_back(mk("Relu", {"missing"}, {"y"}));
    Executor ex(g);
    std::unordered_map<std::string, Tensor> in;
    in["x"] = Tensor(Shape{1}, std::vector<float>{1});
    EXPECT_FALSE(ex.run(in));
}

// Реестр ядер расширяем извне: регистрируем свой слой и используем его.
TEST(Registry, CustomKernel) {
    ii::register_op("Doubler", [](auto& in, const Node&) {
        Tensor o = ii::mul_scalar(*in[0], 2.0f);
        return std::vector<Tensor>{std::move(o)};
    });
    Graph g;
    g.input_names = {"x"};
    g.output_names = {"y"};
    g.nodes.push_back(mk("Doubler", {"x"}, {"y"}));
    Executor ex(g);
    std::unordered_map<std::string, Tensor> in;
    in["x"] = Tensor(Shape{3}, std::vector<float>{1, 2, 3});
    ASSERT_TRUE(ex.run(in));
    EXPECT_EQ(ex.output(0)->data, (std::vector<float>{2, 4, 6}));
}
