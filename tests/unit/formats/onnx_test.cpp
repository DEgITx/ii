// SPDX-License-Identifier: MIT
// The protobuf wire-format decoder and the ONNX importer, on models built in memory.
#include <ii/ii.h>

#include "formats/onnx/onnx_proto.h"
#include "formats/protobuf.h"
#include "support/test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <variant>

namespace ii {
namespace {

using ::testing::HasSubstr;
namespace pb = formats::pb;

// --------------------------------------------------------------------------------------------------------------------
// Protobuf

TEST(ProtobufTest, ReadsWhatTheWriterWrites) {
    pb::Writer inner;
    inner.string(1, "inner");
    pb::Writer w;
    w.varint(1, 300);
    w.string(2, "hello");
    w.float32(3, 1.5f);
    const std::int64_t values[] = {1, -1, 1LL << 40};
    w.packed_varints(4, values);
    w.varint(4, 7);  // the same repeated field, unpacked
    w.message(5, inner);
    pb::Reader r(w.data());
    pb::Field f;
    std::vector<std::int64_t> repeated;
    int seen = 0;
    while (r.next(f)) {
        ++seen;
        switch (f.number) {
        case 1:
            EXPECT_EQ(f.int64(), 300);
            break;
        case 2:
            EXPECT_EQ(f.string(), "hello");
            break;
        case 3:
            EXPECT_EQ(f.float32(), 1.5f);
            break;
        case 4:
            ASSERT_TRUE(pb::append_varints(f, repeated).ok());
            break;
        case 5: {
            pb::Reader nested(f.bytes);
            pb::Field g;
            ASSERT_TRUE(nested.next(g));
            EXPECT_EQ(g.string(), "inner");
            break;
        }
        default:
            ADD_FAILURE() << "unexpected field " << f.number;
        }
    }
    EXPECT_TRUE(r.status().ok());
    EXPECT_EQ(seen, 6);
    EXPECT_EQ(repeated, (std::vector<std::int64_t>{1, -1, 1LL << 40, 7}));
}

TEST(ProtobufTest, RejectsMalformedData) {
    const auto read_all = [](std::vector<std::uint8_t> bytes) {
        pb::Reader r(std::as_bytes(std::span<const std::uint8_t>(bytes)));
        pb::Field f;
        while (r.next(f)) {
        }
        return r.status();
    };
    EXPECT_FALSE(read_all({0x0A, 0x05, 'a'}).ok());  // length past the end
    EXPECT_FALSE(read_all({0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01}).ok());  // 11 bytes
    EXPECT_FALSE(read_all({0x08}).ok());              // missing varint
    EXPECT_FALSE(read_all({0x0B}).ok());              // group
    EXPECT_FALSE(read_all({0x0D, 0x01, 0x02}).ok());  // truncated fixed32
    EXPECT_FALSE(read_all({0x00}).ok());              // field number 0
    EXPECT_TRUE(read_all({}).ok());
}

// --------------------------------------------------------------------------------------------------------------------
// ONNX models built in memory

constexpr std::int32_t kFloat = 1;
constexpr std::int32_t kInt64 = 7;

pb::Writer attribute_int(std::string_view name, std::int64_t value) {
    pb::Writer a;
    a.string(1, name);
    a.varint(3, static_cast<std::uint64_t>(value));
    a.varint(20, 2);
    return a;
}

pb::Writer attribute_float(std::string_view name, float value) {
    pb::Writer a;
    a.string(1, name);
    a.float32(2, value);
    a.varint(20, 1);
    return a;
}

pb::Writer attribute_ints(std::string_view name, std::vector<std::int64_t> values) {
    pb::Writer a;
    a.string(1, name);
    a.packed_varints(8, values);
    a.varint(20, 7);
    return a;
}

/// TensorProto with raw data.
pb::Writer tensor_proto(std::string_view name, std::int32_t type, std::vector<std::int64_t> dims,
                        std::span<const std::byte> raw) {
    pb::Writer t;
    t.packed_varints(1, dims);
    t.varint(2, static_cast<std::uint64_t>(type));
    t.string(8, name);
    t.bytes(9, raw);
    return t;
}

pb::Writer float_tensor(std::string_view name, std::vector<std::int64_t> dims, const std::vector<float>& values) {
    return tensor_proto(name, kFloat, std::move(dims), std::as_bytes(std::span<const float>(values)));
}

pb::Writer int_tensor(std::string_view name, std::vector<std::int64_t> dims, const std::vector<std::int64_t>& values) {
    return tensor_proto(name, kInt64, std::move(dims), std::as_bytes(std::span<const std::int64_t>(values)));
}

using DimSpec = std::variant<std::int64_t, std::string>;

pb::Writer value_info(std::string_view name, std::int32_t type, const std::vector<DimSpec>& dims) {
    pb::Writer shape;
    for (const DimSpec& d : dims) {
        pb::Writer dim;
        if (const auto* v = std::get_if<std::int64_t>(&d)) {
            dim.varint(1, static_cast<std::uint64_t>(*v));
        } else {
            dim.string(2, std::get<std::string>(d));
        }
        shape.message(1, dim);
    }
    pb::Writer tensor;
    tensor.varint(1, static_cast<std::uint64_t>(type));
    tensor.message(2, shape);
    pb::Writer type_proto;
    type_proto.message(1, tensor);
    pb::Writer info;
    info.string(1, name);
    info.message(2, type_proto);
    return info;
}

pb::Writer node(std::string_view op, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs,
                const std::vector<pb::Writer>& attributes = {}, std::string_view domain = {}) {
    pb::Writer n;
    for (const std::string& i : inputs) {
        n.string(1, i);
    }
    for (const std::string& o : outputs) {
        n.string(2, o);
    }
    n.string(3, std::string(op) + "_node");
    n.string(4, op);
    for (const pb::Writer& a : attributes) {
        n.message(5, a);
    }
    if (!domain.empty()) {
        n.string(7, domain);
    }
    return n;
}

struct OnnxGraph {
    pb::Writer graph;
    OnnxGraph& add_node(const pb::Writer& n) {
        graph.message(1, n);
        return *this;
    }
    OnnxGraph& initializer(const pb::Writer& t) {
        graph.message(5, t);
        return *this;
    }
    OnnxGraph& input(const pb::Writer& info) {
        graph.message(11, info);
        return *this;
    }
    OnnxGraph& output(const pb::Writer& info) {
        graph.message(12, info);
        return *this;
    }
};

std::vector<std::byte> model_bytes(const OnnxGraph& g, std::int64_t opset = 17,
                                   const std::vector<pb::Writer>& functions = {}) {
    pb::Writer opset_id;
    opset_id.string(1, "");
    opset_id.varint(2, static_cast<std::uint64_t>(opset));
    pb::Writer m;
    m.varint(1, 8);
    m.string(2, "ii-test");
    m.message(7, g.graph);
    m.message(8, opset_id);
    for (const pb::Writer& f : functions) {
        m.message(25, f);
    }
    return m.buffer();
}

Result<std::vector<Tensor>> run(const std::vector<std::byte>& bytes, const std::vector<Tensor>& inputs,
                                const LoadOptions& options = {}) {
    II_ASSIGN_OR_RETURN(Model model, Model::from_onnx(ii_test::test_context(), bytes, options));
    return ii_test::run_model(model, inputs);
}

Tensor f32(std::initializer_list<float> values, Shape shape) {
    return Tensor::from<float>(values, shape).value();
}

TEST(OnnxImportTest, ElementwiseWithInitializer) {
    OnnxGraph g;
    g.initializer(float_tensor("b", {3}, {1, -10, 0.5f}))
        .input(value_info("x", kFloat, {2, 3}))
        .add_node(node("Add", {"x", "b"}, {"s"}))
        .add_node(node("Relu", {"s"}, {"y"}))
        .output(value_info("y", kFloat, {2, 3}));
    const Result<std::vector<Tensor>> out = run(model_bytes(g), {f32({1, 2, 3, -4, 5, -6}, {2, 3})});
    ASSERT_TRUE(out.ok()) << out.status().to_string();
    EXPECT_EQ(ii_test::compare((*out)[0], f32({2, 0, 3.5f, 0, 0, 0}, {2, 3}), {0, 0}), "");
}

TEST(OnnxImportTest, TracksShapeArithmeticSymbolically) {
    // y = Reshape(x, Concat(Unsqueeze(Shape(x)[0]), [2, 3])) for x [N, 6]
    OnnxGraph g;
    g.initializer(int_tensor("zero", {}, {0}))
        .initializer(int_tensor("axes", {1}, {0}))
        .initializer(int_tensor("tail", {2}, {2, 3}))
        .input(value_info("x", kFloat, {std::string("N"), 6}))
        .add_node(node("Shape", {"x"}, {"shape"}))
        .add_node(node("Gather", {"shape", "zero"}, {"n"}))
        .add_node(node("Unsqueeze", {"n", "axes"}, {"n1"}))
        .add_node(node("Concat", {"n1", "tail"}, {"target"}, {attribute_int("axis", 0)}))
        .add_node(node("Reshape", {"x", "target"}, {"y"}))
        .output(value_info("y", kFloat, {std::string("N"), 2, 3}));
    const Result<Model> model = Model::from_onnx(ii_test::test_context(), model_bytes(g));
    ASSERT_TRUE(model.ok()) << model.status().to_string();
    EXPECT_EQ(to_string(model->outputs()[0].shape), "[N, 2, 3]");
    for (const std::int64_t n : {1, 4}) {
        const Tensor x = ii_test::random_tensor(DType::kF32, Shape{n, 6}, 1);
        const Result<std::vector<Tensor>> out = ii_test::run_model(*model, {x});
        ASSERT_TRUE(out.ok()) << out.status().to_string();
        EXPECT_EQ((*out)[0].shape(), (Shape{n, 2, 3}));
    }
    // Fixing the symbol when loading makes the model static.
    LoadOptions options;
    options.dims = {{"N", 3}};
    const Result<Model> fixed = Model::from_onnx(ii_test::test_context(), model_bytes(g), options);
    ASSERT_TRUE(fixed.ok()) << fixed.status().to_string();
    EXPECT_EQ(to_string(fixed->inputs()[0].shape), "[3, 6]");
}

TEST(OnnxImportTest, ConvolutionWithBatchNormAndGemm) {
    const std::vector<float> w = {1, 0, -1, 2, 0.5f, 0, 0, 1, 1, -1, 0.25f, 0.5f, 1, 1, 1, 1, 0, 0};  // [2, 1, 3, 3]
    OnnxGraph g;
    g.initializer(float_tensor("w", {2, 1, 3, 3}, w))
        .initializer(float_tensor("scale", {2}, {2, 0.5f}))
        .initializer(float_tensor("bias", {2}, {0.1f, -0.2f}))
        .initializer(float_tensor("mean", {2}, {0.5f, -1}))
        .initializer(float_tensor("var", {2}, {4, 0.25f}))
        .initializer(float_tensor("fc", {3, 8}, std::vector<float>(24, 0.125f)))
        .initializer(float_tensor("fc_bias", {3}, {1, 2, 3}))
        .input(value_info("x", kFloat, {1, 1, 4, 4}))
        .add_node(node("Conv", {"x", "w"}, {"c"},
                       {attribute_ints("kernel_shape", {3, 3}), attribute_ints("pads", {1, 1, 1, 1}),
                        attribute_ints("strides", {2, 2})}))
        .add_node(
            node("BatchNormalization", {"c", "scale", "bias", "mean", "var"}, {"bn"}, {attribute_float("epsilon", 0)}))
        .add_node(node("Relu", {"bn"}, {"r"}))
        .add_node(node("Flatten", {"r"}, {"f"}))
        .add_node(node("Gemm", {"f", "fc", "fc_bias"}, {"y"},
                       {attribute_int("transB", 1), attribute_float("alpha", 2), attribute_float("beta", 0.5f)}))
        .output(value_info("y", kFloat, {1, 3}));
    const Tensor x = ii_test::random_tensor(DType::kF32, Shape{1, 1, 4, 4}, 5);
    const Result<std::vector<Tensor>> out = run(model_bytes(g), {x});
    ASSERT_TRUE(out.ok()) << out.status().to_string();

    // The same computation with the builder API.
    GraphBuilder b;
    const Var input = b.input("x", DType::kF32, {1, 1, 4, 4});
    const Var c =
        conv(input, b.constant(Tensor::from<float>(w, Shape{2, 1, 3, 3}).value()), {}, {{2, 2}, {}, {1, 1, 1, 1}});
    const Var bn = (c - b.constant(f32({0.5f, -1}, {2, 1, 1}))) / b.constant(f32({2, 0.5f}, {2, 1, 1})) *
                       b.constant(f32({2, 0.5f}, {2, 1, 1})) +
                   b.constant(f32({0.1f, -0.2f}, {2, 1, 1}));
    const Var flat = reshape(relu(bn), {1, 8});
    const Var y = matmul(flat, b.constant(Tensor::from<float>(std::vector<float>(24, 0.125f), Shape{3, 8}).value()),
                         false, true) *
                      2.0 +
                  b.constant(f32({0.5f, 1, 1.5f}, {3}));
    b.output("y", y);
    const Model expected_model = Model::from_builder(ii_test::test_context(), std::move(b)).value();
    const Result<std::vector<Tensor>> expected = ii_test::run_model(expected_model, {x});
    ASSERT_TRUE(expected.ok()) << expected.status().to_string();
    EXPECT_EQ(ii_test::compare((*out)[0], (*expected)[0], {1e-5, 1e-6}), "");
}

TEST(OnnxImportTest, InlinesFunctionsAndConstantIfBranches) {
    // Function custom.Shift(x) = x + offset (an attribute), and If(true) { y = Neg(f) } else { y = f }.
    pb::Writer function;
    function.string(1, "Shift");
    function.string(4, "fx");
    function.string(5, "fy");
    function.string(6, "offset");
    {
        pb::Writer constant = node("Constant", {}, {"k"});
        pb::Writer ref;
        ref.string(1, "value_float");
        ref.string(21, "offset");
        ref.varint(20, 1);
        constant.message(5, ref);
        function.message(7, constant);
        function.message(7, node("Add", {"fx", "k"}, {"fy"}));
    }
    {
        pb::Writer opset_id;
        opset_id.string(1, "");
        opset_id.varint(2, 17);
        function.message(9, opset_id);
    }
    function.string(10, "custom");

    pb::Writer then_branch;
    then_branch.message(1, node("Neg", {"f"}, {"t"}));
    then_branch.message(12, value_info("t", kFloat, {3}));
    pb::Writer else_branch;
    else_branch.message(1, node("Identity", {"f"}, {"e"}));
    else_branch.message(12, value_info("e", kFloat, {3}));
    pb::Writer then_attr;
    then_attr.string(1, "then_branch");
    then_attr.message(6, then_branch);
    then_attr.varint(20, 5);
    pb::Writer else_attr;
    else_attr.string(1, "else_branch");
    else_attr.message(6, else_branch);
    else_attr.varint(20, 5);

    pb::Writer cond;
    {
        const std::uint8_t one = 1;
        cond = tensor_proto("cond", 9, {}, std::as_bytes(std::span<const std::uint8_t>(&one, 1)));
    }
    OnnxGraph g;
    g.initializer(cond)
        .input(value_info("x", kFloat, {3}))
        .add_node(node("Shift", {"x"}, {"f"}, {attribute_float("offset", 2)}, "custom"))
        .add_node(node("If", {"cond"}, {"y"}, {then_attr, else_attr}))
        .output(value_info("y", kFloat, {3}));
    const Result<std::vector<Tensor>> out = run(model_bytes(g, 17, {function}), {f32({1, -2, 3}, {3})});
    ASSERT_TRUE(out.ok()) << out.status().to_string();
    EXPECT_EQ(ii_test::compare((*out)[0], f32({-3, 0, -5}, {3}), {0, 0}), "");
}

TEST(OnnxImportTest, ExternalDataStaysInsideTheModelDirectory) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ii_onnx_external";
    std::filesystem::create_directories(dir);
    const std::vector<float> weights = {1, 2, 3, 4, 5, 6};
    {
        std::ofstream data(dir / "weights.bin", std::ios::binary);
        const char pad[8] = {};
        data.write(pad, 8);
        data.write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size() * 4));
    }
    const auto make = [&](std::string_view location) {
        pb::Writer t;
        const std::int64_t dims[] = {2, 3};
        t.packed_varints(1, dims);
        t.varint(2, kFloat);
        t.string(8, "w");
        for (const auto& [key, value] : {std::pair<std::string, std::string>{"location", std::string(location)},
                                         {"offset", "8"},
                                         {"length", "24"}}) {
            pb::Writer entry;
            entry.string(1, key);
            entry.string(2, value);
            t.message(13, entry);
        }
        t.varint(14, 1);
        OnnxGraph g;
        g.initializer(t)
            .input(value_info("x", kFloat, {2, 3}))
            .add_node(node("Mul", {"x", "w"}, {"y"}))
            .output(value_info("y", kFloat, {2, 3}));
        return model_bytes(g);
    };
    const auto save = [&](const std::vector<std::byte>& bytes) {
        std::ofstream file(dir / "model.onnx", std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    };
    save(make("weights.bin"));
    const Result<Model> model = Model::load(ii_test::test_context(), dir / "model.onnx");
    ASSERT_TRUE(model.ok()) << model.status().to_string();
    const Result<std::vector<Tensor>> out = ii_test::run_model(*model, {f32({1, 1, 1, 2, 2, 2}, {2, 3})});
    ASSERT_TRUE(out.ok()) << out.status().to_string();
    EXPECT_EQ(ii_test::compare((*out)[0], f32({1, 2, 3, 8, 10, 12}, {2, 3}), {0, 0}), "");

    for (const char* bad : {"../weights.bin", "/etc/passwd", "C:/Windows/win.ini"}) {
        save(make(bad));
        const Result<Model> rejected = Model::load(ii_test::test_context(), dir / "model.onnx");
        EXPECT_FALSE(rejected.ok()) << bad;
    }
    // From memory there is no directory for external data.
    EXPECT_FALSE(Model::from_onnx(ii_test::test_context(), make("weights.bin")).ok());
    std::filesystem::remove_all(dir);
}

TEST(OnnxImportTest, ReportsUnsupportedOperatorsAndMalformedFiles) {
    OnnxGraph g;
    g.input(value_info("x", kFloat, {3}))
        .add_node(node("NonZero", {"x"}, {"y"}))
        .output(value_info("y", kInt64, {1, std::string("n")}));
    const Result<Model> model = Model::from_onnx(ii_test::test_context(), model_bytes(g));
    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), StatusCode::kUnsupported);
    EXPECT_THAT(model.status().message(), HasSubstr("NonZero"));

    // Every truncation of a valid model fails cleanly (or happens to be a smaller valid model).
    OnnxGraph valid;
    valid.initializer(float_tensor("b", {3}, {1, 2, 3}))
        .input(value_info("x", kFloat, {3}))
        .add_node(node("Add", {"x", "b"}, {"y"}))
        .output(value_info("y", kFloat, {3}));
    const std::vector<std::byte> bytes = model_bytes(valid);
    for (std::size_t size = 0; size < bytes.size(); ++size) {
        (void)Model::from_onnx(ii_test::test_context(), std::span<const std::byte>(bytes.data(), size));
    }
    const std::vector<std::byte> garbage(64, std::byte{0xFF});
    EXPECT_FALSE(Model::from_onnx(ii_test::test_context(), garbage).ok());
}

TEST(OnnxImportTest, RejectsNegativeConstantOfShapeDimensions) {
    // Found by fuzz_onnx: a known negative dimension was used as a vector size.
    pb::Writer value;
    value.string(1, "value");
    value.message(5, int_tensor("", {1}, {7}));
    value.varint(20, 4);  // TENSOR
    OnnxGraph g;
    g.initializer(int_tensor("shape", {1}, {-3}))
        .add_node(node("ConstantOfShape", {"shape"}, {"y"}, {value}))
        .output(value_info("y", kInt64, {3}));
    const Result<Model> model = Model::from_onnx(ii_test::test_context(), model_bytes(g));
    EXPECT_FALSE(model.ok());
}

TEST(OnnxImportTest, ConstantInputsReplaceParameterInputs) {
    // Reshape takes its target shape as an input: unknown when loading unless given as a constant.
    OnnxGraph g;
    g.input(value_info("x", kFloat, {2, 3}))
        .input(value_info("shape", kInt64, {2}))
        .add_node(node("Reshape", {"x", "shape"}, {"y"}))
        .output(value_info("y", kFloat, {3, 2}));
    const Result<Model> plain = Model::from_onnx(ii_test::test_context(), model_bytes(g));
    ASSERT_FALSE(plain.ok());
    EXPECT_THAT(plain.status().message(), HasSubstr("when the model is loaded"));
    LoadOptions options;
    options.constant_inputs = {{"shape", Tensor::from<std::int64_t>({3, 2}, Shape{2}).value()}};
    const Result<std::vector<Tensor>> out = run(model_bytes(g), {f32({1, 2, 3, 4, 5, 6}, {2, 3})}, options);
    ASSERT_TRUE(out.ok()) << out.status().to_string();
    EXPECT_EQ((*out)[0].shape(), (Shape{3, 2}));
}

TEST(OnnxImportTest, TensorProtoConversion) {
    // int32_data holds f16 bits; float_data holds floats.
    pb::Writer t;
    const std::int64_t dims[] = {2};
    t.packed_varints(1, dims);
    t.varint(2, 10);  // FLOAT16
    const std::int64_t bits[] = {0x3C00, 0xC000};
    t.packed_varints(5, bits);
    const Result<formats::onnx::OnnxTensor> proto = formats::onnx::parse_tensor(t.data());
    ASSERT_TRUE(proto.ok());
    const Result<Tensor> tensor = formats::onnx::to_tensor(*proto, nullptr, nullptr);
    ASSERT_TRUE(tensor.ok()) << tensor.status().to_string();
    EXPECT_EQ(tensor->dtype(), DType::kF16);
    EXPECT_EQ(tensor->values<f16>()[0].to_float(), 1.0f);
    EXPECT_EQ(tensor->values<f16>()[1].to_float(), -2.0f);
    // Raw data of the wrong size is rejected.
    const std::vector<float> three = {1, 2, 3};
    const Result<formats::onnx::OnnxTensor> wrong = formats::onnx::parse_tensor(
        tensor_proto("w", kFloat, {2, 2}, std::as_bytes(std::span<const float>(three))).data());
    ASSERT_TRUE(wrong.ok());
    EXPECT_FALSE(formats::onnx::to_tensor(*wrong, nullptr, nullptr).ok());
}

}  // namespace
}  // namespace ii
