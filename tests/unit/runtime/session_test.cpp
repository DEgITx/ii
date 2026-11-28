// SPDX-License-Identifier: MIT
#include <ii/builder.h>
#include <ii/model.h>

#include "runtime/runtime.h"
#include "support/alloc_counter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ii {
namespace {

using ::testing::HasSubstr;

}  // namespace

/// Test access to the session's execution plan (Session keeps its implementation private).
struct SessionTestAccess {
    static std::size_t arena_size(const Session& session) { return session.impl_->plan.arena_sizes[0]; }
    static std::ptrdiff_t state_aliases(const Session& session) {
        const auto& alias = session.impl_->plan.state_alias;
        return std::count_if(alias.begin(), alias.end(), [](int state) { return state >= 0; });
    }
};

namespace {

std::size_t session_arena_size(const Session& session) {
    return SessionTestAccess::arena_size(session);
}

Context make_context() {
    ContextOptions options;
    options.num_threads = 2;
    return Context::create(std::move(options)).value();
}

/// A context with the CPU backends only, also when II_TEST_DEVICES adds simulated devices or II_PLUGIN_PATH a GPU
/// plugin: for tests of CPU plans.
Context make_cpu_context() {
    ContextOptions options;
    options.num_threads = 2;
    options.test_devices = "none";
    options.plugin_options = {{"cuda.devices", "none"}, {"tensorrt.devices", "none"}};
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
    const Context context = make_cpu_context();
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
    EXPECT_THAT(program.describe(), HasSubstr("linear"));  // matmul + bias fused
    EXPECT_THAT(program.describe(), HasSubstr("ii-cpu"));
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

/// The LLM pattern: a preallocated KV cache state, written in place by slice_update and read by sdpa with kv_len.
TEST(SessionTest, KvCacheStateIsUpdatedInPlace) {
    const Context context = make_cpu_context();
    GraphBuilder b;
    const Var x = b.input("x", DType::kF32, {1, 1, "S", 4});
    const Var pos = b.input("pos", DType::kI64, {});
    const Var cache = b.state("cache", DType::kF32, {1, 1, 8, 4});
    const Var written = slice_update(cache, x, pos, 2);
    b.update_state(cache, written);
    const Var used = pos + reshape(reduce_sum(fill(b, {"S"}, DType::kI64, 1)), {});
    b.output("y", sdpa(x, written, written, {}, true, 0.0, used));
    const Program program = Model::from_builder(context, std::move(b)).value().compile().value();
    Session session = program.new_session().value();
    const void* cache_memory = session.state("cache").value().data();
    std::vector<float> all;
    std::int64_t position = 0;
    for (const int length : {3, 1, 1, 2, 1}) {
        std::vector<float> chunk;
        chunk.reserve(static_cast<std::size_t>(length) * 4);
        for (int i = 0; i < length * 4; ++i) {
            chunk.push_back(static_cast<float>(position * 4 + i) * 0.1f);
        }
        all.insert(all.end(), chunk.begin(), chunk.end());
        ASSERT_TRUE(session.set_input("x", floats(chunk, Shape{1, 1, length, 4})).ok());
        ASSERT_TRUE(session.set_input("pos", Tensor::scalar<std::int64_t>(position).value()).ok());
        ASSERT_TRUE(session.run().ok());
        position += length;
        // slice_update wrote straight into the state (no arena buffer, no copy at the end of the run).
        EXPECT_EQ(SessionTestAccess::state_aliases(session), 1);
        // Every token so far is in the cache, which never moved.
        const Tensor state = session.state("cache").value();
        EXPECT_EQ(state.data(), cache_memory);
        const auto values = state.values<float>();
        for (std::size_t i = 0; i < all.size(); ++i) {
            ASSERT_EQ(values[i], all[i]) << "cache element " << i << " after " << position << " tokens";
        }
        // Causal attention of the last new token over all tokens: compare with a full recomputation.
        const auto y = session.output("y").value().values<float>();
        const auto last = static_cast<std::size_t>(position - 1);
        std::vector<double> weights;
        double max = -1e300;
        for (std::size_t j = 0; j <= last; ++j) {
            double s = 0;
            for (std::size_t d = 0; d < 4; ++d) {
                s += static_cast<double>(all[last * 4 + d]) * static_cast<double>(all[j * 4 + d]);
            }
            weights.push_back(s / 2.0);
            max = std::max(max, s / 2.0);
        }
        double sum = 0;
        for (double& w : weights) {
            w = std::exp(w - max);
            sum += w;
        }
        for (std::size_t d = 0; d < 4; ++d) {
            double expected = 0;
            for (std::size_t j = 0; j <= last; ++j) {
                expected += weights[j] / sum * static_cast<double>(all[j * 4 + d]);
            }
            EXPECT_NEAR(y[static_cast<std::size_t>(length - 1) * 4 + d], expected, 1e-5);
        }
    }
    // Writing past the end of the cache is a run-time error.
    ASSERT_TRUE(session.set_input("x", floats(std::vector<float>(4, 0.0f), Shape{1, 1, 1, 4})).ok());
    ASSERT_TRUE(session.set_input("pos", Tensor::scalar<std::int64_t>(8).value()).ok());
    EXPECT_THAT(session.run().to_string(), HasSubstr("slice_update"));
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
    ASSERT_TRUE(session.run().ok());  // the first run plans memory
    ASSERT_TRUE(session.run().ok());  // the second records the work of a device that can record it (CUDA graphs)
    const std::int64_t arena_allocations = allocator.allocations.load();
    {
        ii_test::AllocationCounter allocations;  // counts nothing under ThreadSanitizer (kCountsAllocations)
        for (int i = 0; i < 5; ++i) {
            ASSERT_TRUE(session.run().ok());
        }
        EXPECT_EQ(allocations.count(), 0) << "heap allocations in steady-state run()";
    }
    EXPECT_EQ(allocator.allocations.load(), arena_allocations);
    EXPECT_EQ(session.state("counter").value().values<std::int64_t>()[0], 7);
}

TEST(SessionTest, ViewsAndInPlaceOperationsShareMemory) {
    const Context context = make_context();
    GraphBuilder b;
    const Var x = b.input("x", DType::kF32, {64, 64});
    Var h = exp(x);  // a node output: views and in-place ops below can reuse its buffer
    for (int i = 0; i < 6; ++i) {
        h = reshape(i % 2 == 0 ? abs(h) : -h, i % 2 == 0 ? Dims{4096} : Dims{64, 64});
    }
    b.output("y", h);
    b.output("view_of_input", reshape(reshape(x, {4096}), {64, 64}) * 2.0);  // a view of a view of an input
    CompileOptions options;
    options.fuse = false;  // keep the chain as separate steps
    const Program program = Model::from_builder(context, std::move(b)).value().compile(options).value();
    Session session = program.new_session().value();
    std::vector<float> data(64 * 64);
    std::iota(data.begin(), data.end(), -2000.0f);
    for (float& v : data) {
        v /= 1000.0f;
    }
    ASSERT_TRUE(session.set_input("x", Tensor::wrap(data.data(), DType::kF32, Shape{64, 64})).ok());
    ASSERT_TRUE(session.run().ok());
    const Tensor y = session.output("y").value();
    const Tensor z = session.output("view_of_input").value();
    for (std::size_t i = 0; i < data.size(); ++i) {
        // abs, neg, abs, neg, abs, neg of exp(x) > 0: -exp(x)
        EXPECT_FLOAT_EQ(static_cast<const float*>(y.data())[i], -std::exp(data[i]));
        EXPECT_FLOAT_EQ(static_cast<const float*>(z.data())[i], 2.0f * data[i]);
    }
    // One 16 KB buffer for the whole chain (plus one for the second output), instead of one per step.
    const std::size_t buffer = 64 * 64 * sizeof(float);
    EXPECT_LE(session_arena_size(session), 2 * buffer + 1024);
}

TEST(SessionTest, ProfilesSteps) {
    const Context context = make_cpu_context();
    const Program program = mlp_model(context).compile().value();
    Session session = program.new_session().value();
    std::vector<float> x(6, 1.0f);
    ASSERT_TRUE(session.set_input("x", floats(x, Shape{2, 3})).ok());
    ASSERT_TRUE(session.run().ok());
    EXPECT_FALSE(session.profile().ok());  // not enabled

    session.set_profiling(true);
    ASSERT_TRUE(session.run().ok());
    const Profile profile = session.profile().value();
    ASSERT_EQ(profile.steps.size(), program.impl()->steps.size());
    double sum = 0;
    for (const StepProfile& step : profile.steps) {
        EXPECT_FALSE(step.op.empty());
        EXPECT_EQ(step.backend, "ii-cpu");
        EXPECT_GE(step.duration_us, 0.0);
        EXPECT_GE(step.start_us, sum - 1e-9);  // steps run one after another
        sum = step.start_us + step.duration_us;
    }
    EXPECT_LE(sum, profile.total_us + 1e-6);
    EXPECT_EQ(profile.steps[0].output, "f32[2, 2]");

    const Profile runs[] = {profile, profile};
    const std::string trace = to_chrome_trace(runs);
    EXPECT_THAT(trace, HasSubstr("\"traceEvents\""));
    EXPECT_THAT(trace, HasSubstr("\"name\": \"run 1\""));
    EXPECT_THAT(trace, HasSubstr("\"name\": \"linear\""));
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
