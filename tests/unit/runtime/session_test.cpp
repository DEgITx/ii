// SPDX-License-Identifier: MIT
#include <ii/builder.h>
#include <ii/model.h>

#include "support/alloc_counter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace ii {
namespace {

using ::testing::HasSubstr;

Context make_context() {
    ContextOptions options;
    options.num_threads = 2;
    return Context::create(std::move(options)).value();
}

std::vector<float> values_of(const Tensor& tensor) {
    const auto span = tensor.contiguous().value().values<float>();
    return {span.begin(), span.end()};
}

Tensor floats(std::vector<float> values, const Shape& shape) {
    return Tensor::from(std::span<const float>(values), shape).value();
}

/// y = relu(x @ w + b) with x [N, 3], w [3, 2], b [2].
Model mlp_model(const Context& context) {
    GraphBuilder b("mlp");
    const Var x = b.input("x", DType::kF32, {"N", 3});
    const Var w = b.constant(floats({1, -1, 2, 0.5f, -3, 1}, Shape{3, 2}));
    const Var bias = b.constant(floats({0.5f, -0.25f}, Shape{2}));
    b.output("y", relu(matmul(x, w) + bias));
    return Model::from_builder(context, std::move(b)).value();
}

std::vector<float> mlp_reference(const std::vector<float>& x) {
    const float w[3][2] = {{1, -1}, {2, 0.5f}, {-3, 1}};
    const float b[2] = {0.5f, -0.25f};
    std::vector<float> y;
    for (std::size_t r = 0; r < x.size() / 3; ++r) {
        for (int j = 0; j < 2; ++j) {
            float acc = b[j];
            for (int k = 0; k < 3; ++k) {
                acc += x[r * 3 + static_cast<std::size_t>(k)] * w[k][j];
            }
            y.push_back(std::max(acc, 0.0f));
        }
    }
    return y;
}

TEST(SessionTest, RunsMlpWithDynamicBatch) {
    const Context context = make_context();
    const Model model = mlp_model(context);
    EXPECT_EQ(model.inputs().size(), 1u);
    EXPECT_EQ(to_string(model.inputs()[0].shape), "[N, 3]");
    const Program program = model.compile().value();
    Session session = program.new_session().value();
    for (const std::int64_t batch : {1, 4, 2, 4}) {
        std::vector<float> x(static_cast<std::size_t>(batch * 3));
        std::iota(x.begin(), x.end(), -3.0f);
        ASSERT_TRUE(session.set_input("x", floats(x, Shape{batch, 3})).ok());
        ASSERT_TRUE(session.run().ok());
        const Tensor y = session.output("y").value();
        EXPECT_EQ(y.shape(), (Shape{batch, 2}));
        EXPECT_EQ(values_of(y), mlp_reference(x));
    }
    EXPECT_THAT(program.describe(), HasSubstr("matmul"));
    EXPECT_THAT(program.describe(), HasSubstr("ii-ref"));
}

TEST(SessionTest, ConvenienceRunReturnsCopies) {
    const Context context = make_context();
    Session session = mlp_model(context).compile().value().new_session().value();
    const std::vector<Tensor> first = session.run({{"x", floats({1, 2, 3}, Shape{1, 3})}}).value();
    const std::vector<Tensor> second = session.run({{"x", floats({0, 0, 0}, Shape{1, 3})}}).value();
    EXPECT_EQ(values_of(first[0]), mlp_reference({1, 2, 3}));
    EXPECT_EQ(values_of(second[0]), mlp_reference({0, 0, 0}));
}

TEST(SessionTest, InputErrors) {
    const Context context = make_context();
    Session session = mlp_model(context).compile().value().new_session().value();
    EXPECT_THAT(session.run().to_string(), HasSubstr("input 'x' is not set"));
    EXPECT_THAT(session.set_input("z", floats({1, 2, 3}, Shape{1, 3})).to_string(), HasSubstr("no input 'z'"));
    EXPECT_THAT(session.set_input("x", Tensor::zeros(DType::kF64, Shape{1, 3}).value()).to_string(),
                HasSubstr("must be f32"));
    ASSERT_TRUE(session.set_input("x", floats({1, 2, 3, 4}, Shape{1, 4})).ok());
    EXPECT_THAT(session.run().to_string(), HasSubstr("expected f32[N, 3]"));
    EXPECT_THAT(session.output("y").status().to_string(), HasSubstr("before run()"));
}

TEST(SessionTest, SymbolsMustAgreeAcrossInputs) {
    const Context context = make_context();
    GraphBuilder b;
    const Var x = b.input("x", DType::kF32, {"N", 2});
    const Var y = b.input("y", DType::kF32, {"N", 2});
    const Var z = b.input("z", DType::kF32, {Dim("N") * 2 + 1});
    b.output("sum", x + y);
    b.output("z2", z * 2.0);
    Session session = Model::from_builder(context, std::move(b)).value().compile().value().new_session().value();
    ASSERT_TRUE(session.set_input("x", Tensor::zeros(DType::kF32, Shape{3, 2}).value()).ok());
    ASSERT_TRUE(session.set_input("y", Tensor::zeros(DType::kF32, Shape{3, 2}).value()).ok());
    ASSERT_TRUE(session.set_input("z", Tensor::zeros(DType::kF32, Shape{7}).value()).ok());
    EXPECT_TRUE(session.run().ok());
    ASSERT_TRUE(session.set_input("z", Tensor::zeros(DType::kF32, Shape{8}).value()).ok());
    EXPECT_THAT(session.run().to_string(), HasSubstr("expected 2*N + 1 = 7"));
    ASSERT_TRUE(session.set_input("y", Tensor::zeros(DType::kF32, Shape{4, 2}).value()).ok());
    EXPECT_THAT(session.run().to_string(), HasSubstr("N = 3 from another input"));
}

TEST(SessionTest, SolvesSymbolsFromAffineDimensions) {
    const Context context = make_context();
    GraphBuilder b;
    const Var z = b.input("z", DType::kF32, {Dim("N") * 2 + 1});
    b.output("r", reshape(slice(z, {0}, {1}, {2147483647}), {"N", 2}));
    // N is only known through 2*N + 1.
    Session session = Model::from_builder(context, std::move(b)).value().compile().value().new_session().value();
    std::vector<float> data(7);
    std::iota(data.begin(), data.end(), 0.0f);
    ASSERT_TRUE(session.set_input("z", floats(data, Shape{7})).ok());
    ASSERT_TRUE(session.run().ok()) << session.run().to_string();
    const Tensor r = session.output("r").value();
    EXPECT_EQ(r.shape(), (Shape{3, 2}));
    EXPECT_EQ(values_of(r), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(SessionTest, StatesPersistBetweenRuns) {
    const Context context = make_context();
    GraphBuilder b;
    const Var x = b.input("x", DType::kF32, {2});
    const Var total = b.state("total", DType::kF32, {2}, floats({10, 20}, Shape{2}));
    const Var next = total + x;
    b.update_state(total, next);
    b.output("total", next);
    Session session = Model::from_builder(context, std::move(b)).value().compile().value().new_session().value();
    EXPECT_EQ(values_of(session.state("total").value()), (std::vector<float>{10, 20}));
    for (int i = 1; i <= 3; ++i) {
        ASSERT_TRUE(session.set_input("x", floats({1, 2}, Shape{2})).ok());
        ASSERT_TRUE(session.run().ok());
        EXPECT_EQ(values_of(session.output("total").value()),
                  (std::vector<float>{10.0f + static_cast<float>(i), 20.0f + 2.0f * static_cast<float>(i)}));
    }
    ASSERT_TRUE(session.reset_states().ok());
    EXPECT_EQ(values_of(session.state("total").value()), (std::vector<float>{10, 20}));
}

TEST(SessionTest, StatesCanSwap) {
    const Context context = make_context();
    GraphBuilder b;
    const Var a = b.state("a", DType::kF32, {1}, floats({1}, Shape{1}));
    const Var c = b.state("c", DType::kF32, {1}, floats({2}, Shape{1}));
    b.update_state(a, c);
    b.update_state(c, a);
    b.output("sum", a + c);
    Session session = Model::from_builder(context, std::move(b)).value().compile().value().new_session().value();
    ASSERT_TRUE(session.run().ok());
    EXPECT_EQ(values_of(session.state("a").value()), std::vector<float>{2});
    EXPECT_EQ(values_of(session.state("c").value()), std::vector<float>{1});
}

TEST(SessionTest, BoundOutputsAreWrittenInPlace) {
    const Context context = make_context();
    Session session = mlp_model(context).compile().value().new_session().value();
    float out[4] = {};
    ASSERT_TRUE(session.set_output("y", Tensor::wrap(out, DType::kF32, Shape{2, 2})).ok());
    ASSERT_TRUE(session.set_input("x", floats({1, 2, 3, -1, 0, 1}, Shape{2, 3})).ok());
    ASSERT_TRUE(session.run().ok());
    EXPECT_EQ(std::vector<float>(out, out + 4), mlp_reference({1, 2, 3, -1, 0, 1}));
    EXPECT_EQ(session.output("y").value().data(), out);
    // A binding with the wrong shape is reported at run time.
    ASSERT_TRUE(session.set_input("x", floats({1, 2, 3}, Shape{1, 3})).ok());
    EXPECT_THAT(session.run().to_string(), HasSubstr("bound to a tensor of shape [2, 2]"));
    ASSERT_TRUE(session.clear_output("y").ok());
    EXPECT_TRUE(session.run().ok());
    EXPECT_THAT(session.set_output("y", Tensor::zeros(DType::kI32, Shape{1, 2}).value()).to_string(),
                HasSubstr("contiguous host f32"));
}

TEST(SessionTest, SteadyStateRunDoesNotAllocate) {
    ii_test::CountingAllocator allocator;
    ContextOptions options;
    options.allocator = &allocator;
    options.num_threads = 1;
    const Context context = Context::create(std::move(options)).value();
    // A model touching many kernel kinds, including workspaces (sdpa) and states.
    GraphBuilder b;
    const Var x = b.input("x", DType::kF32, {"N", 4});
    const Var w = b.constant(floats(std::vector<float>(16, 0.25f), Shape{4, 4}));
    const Var h = layer_norm(gelu(linear(x, w)), {}, {}, -1);
    const Var q = reshape(h, {1, 1, "N", 4});
    const Var a = sdpa(q, q, q, {}, true);
    const Var counter = b.state("counter", DType::kI64, {});
    b.update_state(counter, counter + b.scalar(1, DType::kI64));
    b.output("y", softmax(reshape(a, {"N", 4})) + reduce_sum(x, {1}, true));
    const Program program = Model::from_builder(context, std::move(b)).value().compile().value();
    Session session = program.new_session().value();
    const Tensor input = floats(std::vector<float>(12, 0.5f), Shape{3, 4});
    ASSERT_TRUE(session.set_input("x", input).ok());
    ASSERT_TRUE(session.run().ok());  // first run plans memory
    const std::int64_t arena_allocations = allocator.allocations.load();
    {
        ii_test::AllocationCounter allocations;
        for (int i = 0; i < 5; ++i) {
            ASSERT_TRUE(session.run().ok());
        }
        EXPECT_EQ(allocations.count(), 0) << "heap allocations in steady-state run()";
    }
    EXPECT_EQ(allocator.allocations.load(), arena_allocations);
    EXPECT_EQ(session.state("counter").value().values<std::int64_t>()[0], 6);
}

TEST(SessionTest, CompileErrors) {
    const Context context = make_context();
    EXPECT_THAT(mlp_model(context).compile({.backends = {"nope"}}).status().to_string(),
                HasSubstr("unknown backend 'nope'"));
    GraphBuilder b;
    b.output("y", relu(b.input("x", DType::kF32, {2})) + b.input("x", DType::kF32, {2}));
    EXPECT_THAT(Model::from_builder(context, std::move(b)).status().to_string(), HasSubstr("already used"));
}

}  // namespace
}  // namespace ii
