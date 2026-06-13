// Тесты загрузчика ONNX движка `ii` (engine/onnx.cpp).
//
// Модели собираются прямо в тесте — кодируем минимальный protobuf
// ModelProto байтами, без внешних файлов и зависимостей. Так проверяется
// весь путь: разбор wire-формата -> Graph -> Executor -> число, а также
// мост inf::Engine (make_ii_engine) на временном .onnx-файле.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "engine/graph.h"
#include "engine/loader.h"
#include "engine/tensor.h"
#include "inference.h"

namespace inf { std::unique_ptr<Engine> make_ii_engine(); }  // из engine/backend.cpp

namespace {

// ---- крошечный protobuf-энкодер (ровно под нужды тестов) ------------------
void put_varint(std::string& s, std::uint64_t v) {
    while (v >= 0x80) { s.push_back((char)(v | 0x80)); v >>= 7; }
    s.push_back((char)v);
}
void put_tag(std::string& s, int field, int wt) {
    put_varint(s, ((std::uint64_t)field << 3) | wt);
}
void put_bytes(std::string& s, int field, const std::string& b) {
    put_tag(s, field, 2); put_varint(s, b.size()); s += b;
}
void put_str(std::string& s, int field, const std::string& v) { put_bytes(s, field, v); }
void put_int(std::string& s, int field, std::int64_t v) {
    put_tag(s, field, 0); put_varint(s, (std::uint64_t)v);
}
void put_float(std::string& s, int field, float f) {
    put_tag(s, field, 5);
    std::uint32_t u; std::memcpy(&u, &f, 4);
    for (int i = 0; i < 4; ++i) s.push_back((char)((u >> (8 * i)) & 0xFF));
}

// TensorProto через float_data.
std::string tensor_f(const std::string& name, std::vector<std::int64_t> dims,
                     std::vector<float> data) {
    std::string t;
    for (auto d : dims) put_int(t, 1, d);   // dims
    put_int(t, 2, 1);                        // data_type = FLOAT
    for (float f : data) put_float(t, 4, f); // float_data
    put_str(t, 8, name);                     // name
    return t;
}
// TensorProto через raw_data (float LE).
std::string tensor_raw(const std::string& name, std::vector<std::int64_t> dims,
                      std::vector<float> data) {
    std::string raw;
    for (float f : data) {
        std::uint32_t u; std::memcpy(&u, &f, 4);
        for (int i = 0; i < 4; ++i) raw.push_back((char)((u >> (8 * i)) & 0xFF));
    }
    std::string t;
    for (auto d : dims) put_int(t, 1, d);
    put_int(t, 2, 1);
    put_str(t, 8, name);
    put_bytes(t, 9, raw);                    // raw_data
    return t;
}
// ValueInfoProto со статической формой.
std::string value_info(const std::string& name, std::vector<std::int64_t> dims) {
    std::string shape;
    for (auto d : dims) { std::string dim; put_int(dim, 1, d); put_bytes(shape, 1, dim); }
    std::string tensor; put_int(tensor, 1, 1); put_bytes(tensor, 2, shape);
    std::string type; put_bytes(type, 1, tensor);
    std::string vi; put_str(vi, 1, name); put_bytes(vi, 2, type);
    return vi;
}
std::string node(const std::string& op, std::vector<std::string> ins,
                 std::vector<std::string> outs, const std::string& attrs = "") {
    std::string n;
    for (auto& i : ins) put_str(n, 1, i);
    for (auto& o : outs) put_str(n, 2, o);
    put_str(n, 4, op);
    n += attrs;  // уже закодированные AttributeProto (поле 5)
    return n;
}
std::string attr_value_tensor(const std::string& tensor_bytes) {
    std::string a; put_str(a, 1, "value"); put_bytes(a, 5, tensor_bytes);
    std::string wrap; put_bytes(wrap, 5, a);  // attribute (поле 5 ноды)
    return wrap;
}
// AttributeProto-обёртки для скалярных/списочных Constant-значений.
std::string attr_float(const std::string& name, float f) {
    std::string a; put_str(a, 1, name); put_float(a, 2, f);   // f (поле 2)
    std::string wrap; put_bytes(wrap, 5, a); return wrap;
}
std::string attr_int(const std::string& name, std::int64_t i) {
    std::string a; put_str(a, 1, name); put_int(a, 3, i);     // i (поле 3)
    std::string wrap; put_bytes(wrap, 5, a); return wrap;
}
std::string attr_floats(const std::string& name, std::vector<float> v) {
    std::string a; put_str(a, 1, name);
    for (float f : v) put_float(a, 7, f);                     // floats (поле 7)
    std::string wrap; put_bytes(wrap, 5, a); return wrap;
}
std::string attr_ints(const std::string& name, std::vector<std::int64_t> v) {
    std::string a; put_str(a, 1, name);
    for (auto x : v) put_int(a, 8, x);                        // ints (поле 8)
    std::string wrap; put_bytes(wrap, 5, a); return wrap;
}
std::string model(const std::string& graph_bytes) {
    std::string m; put_bytes(m, 7, graph_bytes); return m;
}

}  // namespace

// Add(X,B) -> Relu. Проверяем весь путь разбора и счёта.
TEST(Onnx, AddReluPipeline) {
    std::string g;
    put_bytes(g, 1, node("Add", {"X", "B"}, {"S"}));
    put_bytes(g, 1, node("Relu", {"S"}, {"Y"}));
    put_bytes(g, 5, tensor_f("B", {3}, {1, 1, 1}));   // initializer
    put_bytes(g, 11, value_info("X", {1, 3}));        // input
    put_bytes(g, 12, value_info("Y", {1, 3}));        // output
    std::string m = model(g);

    ii::Graph graph;
    std::string err;
    ASSERT_TRUE(ii::parse_onnx(m.data(), m.size(), graph, err)) << err;

    EXPECT_EQ(graph.input_names, (std::vector<std::string>{"X"}));
    EXPECT_EQ(graph.output_names, (std::vector<std::string>{"Y"}));
    EXPECT_EQ(graph.input_shapes["X"], (ii::Shape{1, 3}));
    ASSERT_EQ(graph.initializers.count("B"), 1u);
    EXPECT_EQ(graph.initializers["B"].data, (std::vector<float>{1, 1, 1}));
    EXPECT_EQ(graph.nodes.size(), 2u);

    ii::Executor ex(graph);
    std::unordered_map<std::string, ii::Tensor> in;
    in["X"] = ii::Tensor(ii::Shape{1, 3}, std::vector<float>{-2, 0, 5});
    ASSERT_TRUE(ex.run(in));
    const ii::Tensor* y = ex.output(0);
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(y->data, (std::vector<float>{0, 1, 6}));
}

// raw_data initializer декодируется корректно.
TEST(Onnx, RawDataInitializer) {
    std::string g;
    put_bytes(g, 1, node("Mul", {"X", "W"}, {"Y"}));
    put_bytes(g, 5, tensor_raw("W", {2}, {2, 3}));
    put_bytes(g, 11, value_info("X", {2}));
    put_bytes(g, 12, value_info("Y", {2}));
    std::string m = model(g);

    ii::Graph graph;
    std::string err;
    ASSERT_TRUE(ii::parse_onnx(m.data(), m.size(), graph, err)) << err;
    EXPECT_EQ(graph.initializers["W"].data, (std::vector<float>{2, 3}));

    ii::Executor ex(graph);
    std::unordered_map<std::string, ii::Tensor> in;
    in["X"] = ii::Tensor(ii::Shape{2}, std::vector<float>{5, 10});
    ASSERT_TRUE(ex.run(in));
    EXPECT_EQ(ex.output(0)->data, (std::vector<float>{10, 30}));
}

// Constant-нода сворачивается в initializer.
TEST(Onnx, ConstantFolded) {
    std::string g;
    put_bytes(g, 1, node("Constant", {}, {"C"}, attr_value_tensor(tensor_f("", {2}, {-1, 2}))));
    put_bytes(g, 1, node("Relu", {"C"}, {"Y"}));
    put_bytes(g, 12, value_info("Y", {2}));
    std::string m = model(g);

    ii::Graph graph;
    std::string err;
    ASSERT_TRUE(ii::parse_onnx(m.data(), m.size(), graph, err)) << err;
    // Constant исчез из нод, его выход стал константой.
    ASSERT_EQ(graph.initializers.count("C"), 1u);
    EXPECT_EQ(graph.nodes.size(), 1u);
    EXPECT_TRUE(graph.input_names.empty());

    ii::Executor ex(graph);
    ASSERT_TRUE(ex.run({}));
    EXPECT_EQ(ex.output(0)->data, (std::vector<float>{0, 2}));
}

// Constant через value_floats: сворачивается в 1-D initializer.
TEST(Onnx, ConstantValueFloats) {
    std::string g;
    put_bytes(g, 1, node("Constant", {}, {"C"},
                         attr_floats("value_floats", {-1, 2, 3})));
    put_bytes(g, 1, node("Relu", {"C"}, {"Y"}));
    put_bytes(g, 12, value_info("Y", {3}));
    std::string m = model(g);

    ii::Graph graph;
    std::string err;
    ASSERT_TRUE(ii::parse_onnx(m.data(), m.size(), graph, err)) << err;
    ASSERT_EQ(graph.initializers.count("C"), 1u);
    EXPECT_EQ(graph.initializers["C"].shape, (ii::Shape{3}));
    EXPECT_EQ(graph.initializers["C"].data, (std::vector<float>{-1, 2, 3}));

    ii::Executor ex(graph);
    ASSERT_TRUE(ex.run({}));
    EXPECT_EQ(ex.output(0)->data, (std::vector<float>{0, 2, 3}));
}

// Constant через value_ints: список целых -> 1-D float-тензор.
TEST(Onnx, ConstantValueInts) {
    std::string g;
    put_bytes(g, 1, node("Constant", {}, {"C"},
                         attr_ints("value_ints", {2, 5, 7})));
    put_bytes(g, 1, node("Identity", {"C"}, {"Y"}));
    put_bytes(g, 12, value_info("Y", {3}));
    std::string m = model(g);

    ii::Graph graph;
    std::string err;
    ASSERT_TRUE(ii::parse_onnx(m.data(), m.size(), graph, err)) << err;
    EXPECT_EQ(graph.initializers["C"].shape, (ii::Shape{3}));
    EXPECT_EQ(graph.initializers["C"].data, (std::vector<float>{2, 5, 7}));
}

// Constant через скаляр value_int: 0-мерный тензор (одно значение).
TEST(Onnx, ConstantScalarInt) {
    std::string g;
    put_bytes(g, 1, node("Constant", {}, {"C"}, attr_int("value_int", 7)));
    put_bytes(g, 1, node("Relu", {"C"}, {"Y"}));
    put_bytes(g, 12, value_info("Y", {}));
    std::string m = model(g);

    ii::Graph graph;
    std::string err;
    ASSERT_TRUE(ii::parse_onnx(m.data(), m.size(), graph, err)) << err;
    EXPECT_EQ(graph.initializers["C"].shape, (ii::Shape{}));  // скаляр
    EXPECT_EQ(graph.initializers["C"].data, (std::vector<float>{7}));

    ii::Executor ex(graph);
    ASSERT_TRUE(ex.run({}));
    EXPECT_EQ(ex.output(0)->data, (std::vector<float>{7}));
}

// Constant через скаляр value_float.
TEST(Onnx, ConstantScalarFloat) {
    std::string g;
    put_bytes(g, 1, node("Constant", {}, {"C"},
                         attr_float("value_float", 2.5f)));
    put_bytes(g, 1, node("Identity", {"C"}, {"Y"}));
    put_bytes(g, 12, value_info("Y", {}));
    std::string m = model(g);

    ii::Graph graph;
    std::string err;
    ASSERT_TRUE(ii::parse_onnx(m.data(), m.size(), graph, err)) << err;
    EXPECT_EQ(graph.initializers["C"].shape, (ii::Shape{}));
    EXPECT_EQ(graph.initializers["C"].data, (std::vector<float>{2.5f}));
}

TEST(Onnx, RejectsGarbage) {
    std::string junk = "not a protobuf at all";
    ii::Graph graph;
    std::string err;
    EXPECT_FALSE(ii::parse_onnx(junk.data(), junk.size(), graph, err));
    EXPECT_FALSE(err.empty());
}

// Полный путь через inf::Engine: загрузка .onnx-файла, заполнение входа,
// invoke, чтение выхода.
TEST(Onnx, EngineAdapterEndToEnd) {
    std::string g;
    put_bytes(g, 1, node("Add", {"X", "B"}, {"S"}));
    put_bytes(g, 1, node("Relu", {"S"}, {"Y"}));
    put_bytes(g, 5, tensor_f("B", {1, 3}, {10, 10, 10}));
    put_bytes(g, 11, value_info("X", {1, 3}));
    put_bytes(g, 12, value_info("Y", {1, 3}));
    std::string m = model(g);

    const char* path = "ii_adapter_test.onnx";
    { std::ofstream f(path, std::ios::binary); f.write(m.data(), (std::streamsize)m.size()); }

    auto eng = inf::make_ii_engine();
    inf::Engine::Options opts;
    ASSERT_TRUE(eng->load(path, opts));

    ASSERT_EQ(eng->inputs().size(), 1u);
    EXPECT_EQ(eng->inputs()[0].shape, (std::vector<int>{1, 3}));
    ASSERT_EQ(eng->outputs().size(), 1u);
    EXPECT_EQ(eng->outputs()[0].shape, (std::vector<int>{1, 3}));

    float* in = (float*)eng->input_data(0);
    in[0] = -20; in[1] = 0; in[2] = 5;   // +10 -> [-10,10,15] -> relu -> [0,10,15]
    ASSERT_TRUE(eng->invoke());
    const float* out = (const float*)eng->output_data(0);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 10.0f);
    EXPECT_FLOAT_EQ(out[2], 15.0f);

    std::remove(path);
}
