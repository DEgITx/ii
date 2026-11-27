// SPDX-License-Identifier: MIT
// Multi-device execution on the simulated devices of the ii-test-remote backend (no GPU needed): the planner's
// choices, transfers, states in device memory, memory limits, the device queue's semantics. Every result is
// compared with ii-ref.
#include <ii/builder.h>
#include <ii/model.h>

#include "runtime/runtime.h"
#include "support/alloc_counter.h"
#include "support/test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace ii {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

/// A device so fast and so close that the planner gives it every operation it supports.
constexpr std::string_view kFast = "launch=0,latency=0,link=100000,gflops=100000,bandwidth=100000";

Context device_context(std::string devices, Allocator* allocator = nullptr) {
    ContextOptions options;
    options.num_threads = 2;
    options.test_devices = std::move(devices);
    options.plugin_options = {{"cuda.devices", "none"}};  // only the simulated devices, also with a GPU plugin
    options.allocator = allocator;
    return Context::create(std::move(options)).value();
}

std::string fast(std::string_view extra = {}) {
    std::string spec(kFast);
    if (!extra.empty()) {
        spec += ",";
        spec += extra;
    }
    return spec;
}

/// Runs `model` as planned with `options` and on ii-ref alone, and compares the outputs.
void expect_matches_reference(const Model& model, const std::vector<Tensor>& inputs,
                              const CompileOptions& options = {}) {
    const Result<std::vector<Tensor>> actual = ii_test::run_model(model, inputs, options);
    ASSERT_TRUE(actual.ok()) << actual.status().to_string();
    CompileOptions reference;
    reference.backends = {"ii-ref"};
    const Result<std::vector<Tensor>> expected = ii_test::run_model(model, inputs, reference);
    ASSERT_TRUE(expected.ok()) << expected.status().to_string();
    ASSERT_EQ(actual->size(), expected->size());
    for (std::size_t i = 0; i < actual->size(); ++i) {
        const std::string mismatch = ii_test::compare((*actual)[i], (*expected)[i], {1e-5, 1e-5});
        EXPECT_TRUE(mismatch.empty()) << "output " << i << ": " << mismatch;
    }
}

int count_of(const std::string& text, std::string_view what) {
    int count = 0;
    for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + what.size())) {
        ++count;
    }
    return count;
}

/// `layers` layers relu(x @ w^T + b) of `width` features, x [N, width].
Model mlp(const Context& context, int layers, std::int64_t width) {
    GraphBuilder b("mlp");
    Var h = b.input("x", DType::kF32, {"N", width});
    for (int l = 0; l < layers; ++l) {
        const Var w = b.constant(ii_test::random_tensor(DType::kF32, Shape{width, width}, 100 + l, -0.1, 0.1));
        const Var bias = b.constant(ii_test::random_tensor(DType::kF32, Shape{width}, 200 + l, -0.1, 0.1));
        h = relu(linear(h, w, bias));
    }
    b.output("y", h);
    return Model::from_builder(context, std::move(b)).value();
}

Tensor input(std::int64_t rows, std::int64_t width, std::uint32_t seed = 1) {
    return ii_test::random_tensor(DType::kF32, Shape{rows, width}, seed);
}

TEST(MultiDevice, OperationsTheDeviceLacksRunOnTheHost) {
    const Context context = device_context(fast("ops=linear"));
    const Model model = mlp(context, 2, 32);
    const Program program = model.compile().value();
    const std::string plan = program.describe();
    // linear on the device, relu on the host: x and each result cross over.
    EXPECT_EQ(count_of(plan, "ii-test-remote @ test:0"), 2) << plan;
    EXPECT_EQ(count_of(plan, "ii-cpu @ cpu"), 2) << plan;
    EXPECT_EQ(count_of(plan, "transfer cpu -> test:0"), 2) << plan;
    EXPECT_EQ(count_of(plan, "transfer test:0 -> cpu"), 2) << plan;
    EXPECT_THAT(program.describe_plan(), HasSubstr("ii-test-remote: no kernel for 'relu'"));
    expect_matches_reference(model, {input(5, 32)});
}

TEST(MultiDevice, SlowLinksKeepSmallModelsOnTheHost) {
    const Context context = device_context("latency=1000,link=1");
    const Model model = mlp(context, 2, 32);
    const std::string plan = model.compile().value().describe();
    EXPECT_THAT(plan, Not(HasSubstr("test:0"))) << plan;
}

TEST(MultiDevice, LargeOperationsMoveToAFastDevice) {
    // A realistic device (launch and copy latencies): the large layers go to the device, the result comes back.
    const Context context = device_context("gflops=20000,bandwidth=800");
    const Model model = mlp(context, 4, 512);
    CompileOptions options;
    options.plan_dims = {{"N", 64}};
    const Program program = model.compile(options).value();
    const std::string plan = program.describe();
    EXPECT_EQ(count_of(plan, "transfer cpu -> test:0"), 1) << plan;  // in and out once
    EXPECT_EQ(count_of(plan, "transfer test:0 -> cpu"), 1) << plan;
    EXPECT_THAT(plan, Not(HasSubstr("ii-cpu"))) << plan;
    expect_matches_reference(model, {input(64, 512)}, options);
}

TEST(MultiDevice, TwoDevicesExchangeThroughTheHost) {
    const Context context = device_context(fast("ops=linear") + ";" + fast("ops=relu"));
    const Model model = mlp(context, 2, 16);
    const std::string plan = model.compile().value().describe();
    EXPECT_THAT(plan, HasSubstr("ii-test-remote.1 @ test:1")) << plan;
    EXPECT_THAT(plan, HasSubstr("transfer test:0 -> cpu")) << plan;
    EXPECT_THAT(plan, HasSubstr("transfer cpu -> test:1")) << plan;
    expect_matches_reference(model, {input(3, 16)});
}

TEST(MultiDevice, MemoryLimitSplitsTheGraphOnce) {
    // 8 layers of 256 KiB of weights each on a realistic device (copies cost latency) with room for about 4 of them
    // next to the activations: the first layers run on the host, the rest on the device, so the graph is cut once.
    const std::string device = "gflops=20000,bandwidth=800,memory=2M";
    CompileOptions options;
    options.plan_dims = {{"N", 256}};
    const auto check_plan = [](const std::string& plan, int& on_device) {
        EXPECT_EQ(count_of(plan, "transfer cpu -> test:0"), 1) << plan;
        EXPECT_EQ(count_of(plan, "transfer test:0 -> cpu"), 1) << plan;
        on_device = count_of(plan, "linear           f32[N, 256]              ii-test-remote");
        EXPECT_GE(on_device, 3) << plan;
        EXPECT_LE(on_device, 6) << plan;
    };
    const Context context = device_context(device);
    const Model model = mlp(context, 8, 256);
    int on_device = 0;
    {
        const Program program = model.compile(options).value();
        check_plan(program.describe(), on_device);
        EXPECT_THAT(program.describe_plan(), HasSubstr("moved to other devices to fit the memory limit"));
    }
    // The device refuses allocations beyond its memory, so running proves that the plan fits.
    expect_matches_reference(model, {input(256, 256)}, options);

    // The same limit given at compile time for a larger device.
    const Context large = device_context("gflops=20000,bandwidth=800,memory=1G");
    options.device_memory = {{"test:0", std::uint64_t{2} << 20}};
    int limited = 0;
    check_plan(mlp(large, 8, 256).compile(options).value().describe(), limited);
    EXPECT_EQ(limited, on_device);
}

TEST(MultiDevice, ModelThatFitsNowhereFails) {
    const Context context = device_context(fast("memory=1M"));
    CompileOptions options;
    options.backends = {"ii-test-remote"};  // nothing may run on the host
    const Result<Program> program = mlp(context, 8, 256).compile(options);
    ASSERT_FALSE(program.ok());
    EXPECT_EQ(program.status().code(), StatusCode::kOutOfMemory);
    EXPECT_THAT(program.status().to_string(), HasSubstr("does not fit into the memory of test:0"));
    options.device_memory = {{"cuda:7", 1}};
    EXPECT_EQ(mlp(context, 1, 8).compile(options).status().code(), StatusCode::kNotFound);
}

TEST(MultiDevice, DeviceOnlyPlan) {
    const Context context = device_context(fast());
    CompileOptions options;
    options.backends = {"ii-test-remote"};
    const Model model = mlp(context, 3, 8);
    const std::string plan = model.compile(options).value().describe();
    EXPECT_THAT(plan, Not(HasSubstr("ii-cpu"))) << plan;
    expect_matches_reference(model, {input(2, 8)}, options);
}

/// The LLM pattern: a KV cache state written in place by slice_update and read by sdpa.
Model kv_cache_model(const Context& context) {
    GraphBuilder b;
    const Var x = b.input("x", DType::kF32, {1, 1, "S", 4});
    const Var pos = b.input("pos", DType::kI64, {});
    const Var cache = b.state("cache", DType::kF32, {1, 1, 8, 4});
    const Var written = slice_update(cache, x, pos, 2);
    b.update_state(cache, written);
    const Var used = pos + reshape(reduce_sum(fill(b, {"S"}, DType::kI64, 1)), {});
    b.output("y", sdpa(x, written, written, {}, true, 0.0, used));
    return Model::from_builder(context, std::move(b)).value();
}

TEST(MultiDevice, StatesLiveInDeviceMemory) {
    const Context context = device_context(fast());
    const Context host = device_context("none");
    const Program program = kv_cache_model(context).compile().value();
    ASSERT_EQ(program.impl()->states.size(), 1u);
    EXPECT_EQ(program.impl()->states[0].device, 1);  // test:0
    Session session = program.new_session().value();
    Session expected = kv_cache_model(host).compile().value().new_session().value();
    std::int64_t position = 0;
    for (const int length : {3, 1, 2, 1}) {
        const Tensor x =
            ii_test::random_tensor(DType::kF32, Shape{1, 1, length, 4}, static_cast<std::uint32_t>(position));
        const Tensor pos = Tensor::scalar<std::int64_t>(position).value();
        for (Session* s : {&session, &expected}) {
            ASSERT_TRUE(s->set_input("x", x).ok());
            ASSERT_TRUE(s->set_input("pos", pos).ok());
            ASSERT_TRUE(s->run().ok());
        }
        position += length;
        EXPECT_EQ(ii_test::compare(session.output("y").value(), expected.output("y").value(), {1e-5, 1e-6}), "");
        // state() of a device state is a host copy.
        const Tensor state = session.state("cache").value();
        EXPECT_TRUE(state.device().is_host());
        EXPECT_EQ(ii_test::compare(state, expected.state("cache").value(), {0, 0}), "");
    }
    ASSERT_TRUE(session.reset_states().ok());
    const Tensor cleared = session.state("cache").value();
    for (const float value : cleared.values<float>()) {
        ASSERT_EQ(value, 0.0f);
    }
}

TEST(MultiDevice, RecordedWorkSeesNewData) {
    // From the second run with the same memory, the device's steps are recorded and replayed (like CUDA graphs):
    // decoding with a KV cache in device memory must still see every new position and input, also when a prompt of
    // another length (another plan) comes in between.
    const Context context = device_context(fast());
    const Context host = device_context("none");
    Session session = kv_cache_model(context).compile().value().new_session().value();
    Session expected = kv_cache_model(host).compile().value().new_session().value();
    std::int64_t position = 0;
    for (const int length : {1, 1, 1, 1, 2, 1, 1}) {
        SCOPED_TRACE(position);
        const Tensor x =
            ii_test::random_tensor(DType::kF32, Shape{1, 1, length, 4}, static_cast<std::uint32_t>(position + 50));
        const Tensor pos = Tensor::scalar<std::int64_t>(position).value();
        for (Session* s : {&session, &expected}) {
            ASSERT_TRUE(s->set_input("x", x).ok());
            ASSERT_TRUE(s->set_input("pos", pos).ok());
            ASSERT_TRUE(s->run().ok());
        }
        position += length;
        EXPECT_EQ(ii_test::compare(session.output("y").value(), expected.output("y").value(), {1e-5, 1e-6}), "");
    }
    EXPECT_EQ(ii_test::compare(session.state("cache").value(), expected.state("cache").value(), {0, 0}), "");
}

TEST(MultiDevice, ProfileShowsTransfers) {
    const Context context = device_context(fast("ops=linear"));
    Session session = mlp(context, 1, 8).compile().value().new_session().value();
    session.set_profiling(true);
    ASSERT_TRUE(session.set_input("x", input(2, 8)).ok());
    ASSERT_TRUE(session.run().ok());
    std::vector<std::string> backends;
    const Profile profile = session.profile().value();
    for (const StepProfile& step : profile.steps) {
        backends.push_back(step.backend);
    }
    EXPECT_EQ(backends, (std::vector<std::string>{"transfer", "ii-test-remote", "transfer", "ii-cpu"}));
}

TEST(MultiDevice, SteadyStateRunDoesNotAllocate) {
    ii_test::CountingAllocator allocator;
    const Context context = device_context(fast("ops=linear+sdpa+slice_update+reshape"), &allocator);
    Session session = kv_cache_model(context).compile().value().new_session().value();
    const Tensor x = ii_test::random_tensor(DType::kF32, Shape{1, 1, 1, 4}, 3);
    const Tensor pos = Tensor::scalar<std::int64_t>(2).value();
    ASSERT_TRUE(session.set_input("x", x).ok());
    ASSERT_TRUE(session.set_input("pos", pos).ok());
    ASSERT_TRUE(session.run().ok());  // the first run plans memory
    ASSERT_TRUE(session.run().ok());  // the second records the device's work
    const std::int64_t arena_allocations = allocator.allocations.load();
    {
        ii_test::AllocationCounter allocations;
        for (int i = 0; i < 5; ++i) {
            ASSERT_TRUE(session.run().ok());
        }
        EXPECT_EQ(allocations.count(), 0) << "heap allocations in steady-state run()";
    }
    EXPECT_EQ(allocator.allocations.load(), arena_allocations);
}

TEST(MultiDevice, KernelsRejectHostMemory) {
    // A kernel of the device given host memory reports the missing transfer instead of reading it.
    const Context context = device_context(fast());
    CompileOptions options;
    options.backends = {"ii-test-remote"};
    const Program program = mlp(context, 1, 4).compile(options).value();
    const detail::ProgramImpl& p = *program.impl();
    const detail::Step* linear_step = nullptr;
    for (const detail::Step& step : p.steps) {
        if (!step.is_transfer() && step.node->op_name() == "linear") {
            linear_step = &step;
        }
    }
    ASSERT_NE(linear_step, nullptr);
    std::vector<float> x(4, 1.0f);
    std::vector<float> w(16, 1.0f);
    std::vector<float> b(4, 0.0f);
    std::vector<float> y(4, 0.0f);
    const backend::TensorView inputs[] = {
        {x.data(), DType::kF32, Shape{1, 4}}, {w.data(), DType::kF32, Shape{4, 4}}, {b.data(), DType::kF32, Shape{4}}};
    const backend::TensorView outputs[] = {{y.data(), DType::kF32, Shape{1, 4}}};
    std::unique_ptr<backend::Queue> queue = p.devices[1].api->create_queue().value();
    backend::KernelContext kernel_context;
    kernel_context.inputs = inputs;
    kernel_context.outputs = outputs;
    kernel_context.executor = &context.executor();
    kernel_context.queue = queue.get();
    const Status status = linear_step->kernel->run(kernel_context);
    EXPECT_THAT(status.to_string(), HasSubstr("a transfer is missing"));
}

TEST(MultiDevice, QueueWorkHappensWhenSynchronized) {
    // Like a GPU stream: enqueued work (here a fill and a copy to the host) completes at synchronize(); reading
    // earlier sees old data. Fresh device memory is NaN.
    const Context context = device_context(fast());
    const Program program = mlp(context, 1, 4).compile().value();
    backend::DeviceApi& api = *program.impl()->devices[1].api;
    std::unique_ptr<backend::Queue> queue = api.create_queue().value();
    std::shared_ptr<Buffer> memory = api.allocate(16).value();
    float host[4] = {1, 2, 3, 4};
    ASSERT_TRUE(queue->copy_to_host(host, memory->data(), sizeof(host)).ok());
    EXPECT_EQ(host[0], 1.0f);  // not yet
    ASSERT_TRUE(queue->synchronize().ok());
    EXPECT_TRUE(std::isnan(host[0]));
    ASSERT_TRUE(queue->fill_zero(memory->data(), 16).ok());
    ASSERT_TRUE(queue->copy_to_host(host, memory->data(), sizeof(host)).ok());
    ASSERT_TRUE(queue->synchronize().ok());
    EXPECT_EQ(host[3], 0.0f);
    EXPECT_FALSE(queue->copy_to_host(host, host, sizeof(host)).ok());  // not device memory
    // Device memory is not readable through host tensor functions.
    const Tensor tensor = Tensor::view(memory, DType::kF32, Shape{4}, contiguous_strides(Shape{4})).value();
    EXPECT_FALSE(tensor.clone().ok());
    EXPECT_THAT(tensor.to_string(), HasSubstr("(test:0)"));
}

TEST(MultiDevice, ContextListsDevices) {
    const Context context = device_context(fast("memory=64M") + ";default");
    const std::vector<DeviceDescription> devices = context.devices();
    ASSERT_EQ(devices.size(), 3u);
    EXPECT_EQ(devices[0].id, "cpu");
    EXPECT_EQ(devices[0].backends, (std::vector<std::string>{"ii-cpu", "ii-ref"}));
    EXPECT_EQ(devices[1].id, "test:0");
    EXPECT_EQ(devices[1].memory_bytes, std::uint64_t{64} << 20);
    EXPECT_EQ(devices[2].backends, (std::vector<std::string>{"ii-test-remote.1"}));
    ContextOptions bad;
    bad.test_devices = "memory=lots";
    EXPECT_FALSE(Context::create(std::move(bad)).ok());
}

}  // namespace
}  // namespace ii
