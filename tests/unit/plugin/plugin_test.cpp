// SPDX-License-Identifier: MIT
// Backend plugins: loading (ii-backend-test-remote as a shared library), the checks of the function table, and the
// C interface itself — an in-process plugin made with the SDK runs ii-ref and the simulated device through it, so
// every node crosses the interface as text and every kernel, copy and queue operation as C calls.
#include <ii/builder.h>
#include <ii/model.h>
#include <ii/version.h>

#include "backends/ref/ref_backend.h"
#include "backends/test_remote/test_remote.h"
#include "core/platform/platform.h"
#include "plugin/host.h"
#include "plugin/sdk.h"
#include "runtime/runtime.h"
#include "support/alloc_counter.h"
#include "support/random_graph.h"
#include "support/test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

namespace ii {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr std::string_view kFast = "launch=0,latency=0,link=100000,gflops=100000,bandwidth=100000";

// --- An in-process plugin ---

/// ii-ref under another name (a host backend of the plugin).
class RenamedBackend final : public backend::Backend {
public:
    RenamedBackend(std::unique_ptr<backend::Backend> inner, std::string name)
        : inner_(std::move(inner)), name_(std::move(name)) {}

    std::string_view name() const noexcept override { return name_; }
    const backend::DeviceInfo& device_info() const noexcept override { return inner_->device_info(); }
    double estimate_us(const ir::CostContext& context) const override { return inner_->estimate_us(context); }
    Status supports(const ir::Node& node) const override { return inner_->supports(node); }
    Result<std::unique_ptr<backend::Kernel>> create_kernel(const ir::Node& node) const override {
        return inner_->create_kernel(node);
    }

private:
    std::unique_ptr<backend::Backend> inner_;
    std::string name_;
};

/// Options: "in_process.host" = "1" adds "ii-plugin-ref"; "test_devices" adds simulated devices; "in_process.log"
/// logs a message; "in_process.fail" fails.
Result<plugin::BackendList> create_in_process(const plugin::PluginHost& host) {
    if (host.option("in_process.fail") == "1") {
        return errors::invalid_argument("failing on request");
    }
    if (host.option("in_process.log") == "1") {
        host.logger().log(LogLevel::kWarning, "hello from the plugin");
    }
    plugin::BackendList backends;
    if (host.option("in_process.host") == "1") {
        backends.push_back(std::make_unique<RenamedBackend>(ref::create_ref_backend(), "ii-plugin-ref"));
    }
    if (const std::string spec = host.option("test_devices"); !spec.empty()) {
        II_ASSIGN_OR_RETURN(std::vector<test_remote::TestDeviceOptions> devices, test_remote::parse_test_devices(spec));
        for (const test_remote::TestDeviceOptions& device : devices) {
            backends.push_back(test_remote::create_test_remote_backend(device));
        }
    }
    return backends;
}

constexpr plugin::PluginDefinition kInProcess{"ii-in-process", "1.2.3", &create_in_process, true};

using Options = std::vector<std::pair<std::string, std::string>>;

/// A context without simulated devices or discovered plugins, plus the in-process plugin with `options`.
Context in_process_context(const Options& options, Allocator* allocator = nullptr) {
    ContextOptions context_options;
    context_options.num_threads = 2;
    context_options.test_devices = "none";
    context_options.search_plugins = false;
    context_options.allocator = allocator;
    const Context context = Context::create(std::move(context_options)).value();
    EXPECT_TRUE(
        plugin::add_plugin(*context.impl(), *plugin::make_plugin_table(kInProcess), "in-process", options).ok());
    return context;
}

/// A context loading the ii-backend-test-remote library.
Result<Context> library_context(std::string devices, Options plugin_options = {}) {
    ContextOptions options;
    options.num_threads = 2;
    options.test_devices = std::move(devices);
    options.search_plugins = false;
    options.plugin_paths = {ii_test::test_plugin_path()};
    options.plugin_options = std::move(plugin_options);
    return Context::create(std::move(options));
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
    b.output("y", softmax(h, -1));
    return Model::from_builder(context, std::move(b)).value();
}

/// Runs `model` restricted to `backends` and on ii-ref alone; expects equal outputs.
void expect_matches_reference(const Model& model, const std::vector<Tensor>& inputs,
                              std::vector<std::string> backends = {}) {
    CompileOptions options;
    options.backends = std::move(backends);
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

std::vector<std::string> profiled_backends(const Program& program, const std::vector<Tensor>& inputs) {
    Session session = program.new_session().value();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_TRUE(session.set_input(program.inputs()[i].name, inputs[i]).ok());
    }
    session.set_profiling(true);
    EXPECT_TRUE(session.run().ok());
    const Profile profile = session.profile().value();
    std::vector<std::string> backends;
    for (const StepProfile& step : profile.steps) {
        backends.push_back(step.backend);
    }
    return backends;
}

// --- Loading ---

TEST(Plugin, LoadsTheTestRemoteLibrary) {
    Result<Context> context = library_context(std::string(kFast));
    ASSERT_TRUE(context.ok()) << context.status().to_string();
    const std::vector<PluginDescription> plugins = context->plugins();
    ASSERT_EQ(plugins.size(), 1u);
    EXPECT_EQ(plugins[0].name, "ii-backend-test-remote");
    EXPECT_EQ(plugins[0].state, PluginState::kLoaded);
    EXPECT_EQ(plugins[0].version, II_VERSION_STRING);
    EXPECT_THAT(plugins[0].backends, ElementsAre("ii-test-remote"));
    EXPECT_EQ(to_string(plugins[0].state), "loaded");

    // The plugin's device appears like a built-in one, and models run on it.
    const std::vector<DeviceDescription> devices = context->devices();
    ASSERT_EQ(devices.size(), 2u);
    EXPECT_EQ(devices[1].id, "test:0");
    const Model model = mlp(*context, 3, 32);
    const std::vector<Tensor> inputs = {ii_test::random_tensor(DType::kF32, Shape{8, 32}, 1)};
    expect_matches_reference(model, inputs);
    const Program program = model.compile().value();
    EXPECT_THAT(profiled_backends(program, inputs), ::testing::Contains("ii-test-remote"));
}

TEST(Plugin, TwoContextsShareTheLibrary) {
    Result<Context> a = library_context(std::string(kFast));
    Result<Context> b = library_context(std::string(kFast) + ";" + std::string(kFast));
    ASSERT_TRUE(a.ok() && b.ok());
    EXPECT_THAT(a->plugins()[0].backends, ElementsAre("ii-test-remote"));
    EXPECT_THAT(b->plugins()[0].backends, ElementsAre("ii-test-remote", "ii-test-remote.1"));
    const std::vector<Tensor> inputs = {ii_test::random_tensor(DType::kF32, Shape{4, 16}, 2)};
    expect_matches_reference(mlp(*a, 2, 16), inputs);
    expect_matches_reference(mlp(*b, 2, 16), inputs);
}

TEST(Plugin, WithoutDevicesThePluginAddsNothing) {
    Result<Context> context = library_context("none");
    ASSERT_TRUE(context.ok()) << context.status().to_string();
    ASSERT_EQ(context->plugins().size(), 1u);
    EXPECT_EQ(context->plugins()[0].state, PluginState::kLoaded);
    EXPECT_TRUE(context->plugins()[0].backends.empty());
    EXPECT_EQ(context->devices().size(), 1u);
}

TEST(Plugin, UnavailablePluginIsListed) {
    Result<Context> context = library_context("none", {{"test_remote.unavailable", "1"}});
    ASSERT_TRUE(context.ok()) << context.status().to_string();
    ASSERT_EQ(context->plugins().size(), 1u);
    EXPECT_EQ(context->plugins()[0].state, PluginState::kUnavailable);
    EXPECT_THAT(context->plugins()[0].message, HasSubstr("unavailable on request"));
    EXPECT_EQ(context->devices().size(), 1u);
}

TEST(Plugin, ExplicitPathsMustLoad) {
    // A shared library that is not a plugin.
    ContextOptions options;
    options.search_plugins = false;
    options.plugin_paths = {ii_test::test_dynlib_path()};
    Result<Context> context = Context::create(options);
    ASSERT_FALSE(context.ok());
    EXPECT_THAT(context.status().message(), HasSubstr(II_PLUGIN_ENTRY_POINT));

    options.plugin_paths = {(std::filesystem::temp_directory_path() / "ii-backend-missing.dll").string()};
    EXPECT_FALSE(Context::create(options).ok());
}

TEST(Plugin, DirectoriesAreScanned) {
    const std::filesystem::path file = platform::path_from_utf8(ii_test::test_plugin_path());
    EXPECT_TRUE(plugin::is_plugin_file_name(file)) << file.string();
    ContextOptions options;
    options.search_plugins = false;
    options.test_devices = std::string(kFast);
    options.plugin_paths = {platform::path_to_utf8(file.parent_path())};
    options.plugin_options = {{"cuda.devices", "none"}};  // the directory may also hold the CUDA plugin
    auto test_plugins = [](const Context& context) {
        std::vector<PluginDescription> found;
        for (const PluginDescription& plugin : context.plugins()) {
            if (plugin.name == "ii-backend-test-remote") {
                found.push_back(plugin);
            }
        }
        return found;
    };
    Result<Context> context = Context::create(options);
    ASSERT_TRUE(context.ok()) << context.status().to_string();
    ASSERT_EQ(test_plugins(*context).size(), 1u);
    EXPECT_EQ(test_plugins(*context)[0].state, PluginState::kLoaded);

    // The same plugin named twice is loaded once.
    options.plugin_paths.push_back(ii_test::test_plugin_path());
    context = Context::create(options);
    ASSERT_TRUE(context.ok()) << context.status().to_string();
    EXPECT_EQ(test_plugins(*context).size(), 1u);
}

TEST(Plugin, FileNames) {
#if defined(_WIN32)
    EXPECT_TRUE(plugin::is_plugin_file_name("ii-backend-cuda.dll"));
    EXPECT_TRUE(plugin::is_plugin_file_name("libii-backend-cuda.dll"));
    EXPECT_FALSE(plugin::is_plugin_file_name("ii-backend-cuda.so"));
#elif !defined(__APPLE__)
    EXPECT_TRUE(plugin::is_plugin_file_name("libii-backend-cuda.so"));
#endif
    EXPECT_FALSE(plugin::is_plugin_file_name("ii.dll"));
    EXPECT_FALSE(plugin::is_plugin_file_name("libii.so"));
    EXPECT_FALSE(plugin::is_plugin_file_name("ii-backend-"));
}

TEST(Plugin, TableChecks) {
    EXPECT_THAT(plugin::validate_table(nullptr).message(), HasSubstr("does not support plugin ABI version 1"));
    ii_plugin_v1 table = *plugin::make_plugin_table(kInProcess);
    EXPECT_TRUE(plugin::validate_table(&table).ok());
    ii_plugin_v1 newer = table;
    newer.abi_version = 2;
    EXPECT_THAT(plugin::validate_table(&newer).message(), HasSubstr("plugin ABI version 2"));
    ii_plugin_v1 short_table = table;
    short_table.struct_size = 16;
    EXPECT_THAT(plugin::validate_table(&short_table).message(), HasSubstr("incomplete"));
    ii_plugin_v1 missing = table;
    missing.run = nullptr;
    EXPECT_THAT(plugin::validate_table(&missing).message(), HasSubstr("lacks required functions"));
}

// --- The interface, in process ---

TEST(Plugin, HostBackendThroughTheInterface) {
    // Every node goes to the plugin as text, and every kernel runs through the C interface.
    const Context context = in_process_context({{"in_process.host", "1"}});
    ASSERT_EQ(context.plugins().size(), 1u);
    EXPECT_EQ(context.plugins()[0].version, "1.2.3");
    EXPECT_THAT(context.plugins()[0].backends, ElementsAre("ii-plugin-ref"));
    for (std::uint32_t seed = 1; seed <= 12; ++seed) {
        SCOPED_TRACE(seed);
        const ii_test::RandomModel random = ii_test::random_model(context, seed);
        expect_matches_reference(random.model, random.inputs, {"ii-plugin-ref"});
    }
    const std::vector<Tensor> inputs = {ii_test::random_tensor(DType::kF32, Shape{5, 24}, 3)};
    const Model model = mlp(context, 2, 24);
    expect_matches_reference(model, inputs, {"ii-plugin-ref"});
    CompileOptions options;
    options.backends = {"ii-plugin-ref"};
    for (const std::string& backend : profiled_backends(model.compile(options).value(), inputs)) {
        EXPECT_TRUE(backend == "ii-plugin-ref" || backend == "view") << backend;
    }
}

TEST(Plugin, DeviceBackendThroughTheInterface) {
    const Context context = in_process_context({{"test_devices", std::string(kFast)}});
    EXPECT_THAT(context.plugins()[0].backends, ElementsAre("ii-test-remote"));
    const std::vector<Tensor> inputs = {ii_test::random_tensor(DType::kF32, Shape{8, 32}, 4)};
    const Model model = mlp(context, 3, 32);
    expect_matches_reference(model, inputs);
    expect_matches_reference(model, inputs, {"ii-test-remote"});
    for (std::uint32_t seed = 1; seed <= 6; ++seed) {
        SCOPED_TRACE(seed);
        const ii_test::RandomModel random = ii_test::random_model(context, seed);
        expect_matches_reference(random.model, random.inputs);
    }
}

TEST(Plugin, ErrorsCrossTheInterface) {
    // A kernel of the device given host memory: the plugin's error message reaches the caller.
    const Context context = in_process_context({{"test_devices", std::string(kFast)}});
    CompileOptions options;
    options.backends = {"ii-test-remote"};
    const Program program = mlp(context, 1, 4).compile(options).value();
    const detail::Step* linear_step = nullptr;
    for (const detail::Step& step : program.impl()->steps) {
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
    const backend::Backend* device = context.impl()->find_backend("ii-test-remote");
    ASSERT_NE(device, nullptr);
    std::unique_ptr<backend::Queue> queue = device->device_api()->create_queue().value();
    backend::KernelContext kernel_context;
    kernel_context.inputs = inputs;
    kernel_context.outputs = outputs;
    kernel_context.executor = &context.executor();
    kernel_context.queue = queue.get();
    const Status status = linear_step->kernel->run(kernel_context);
    EXPECT_EQ(status.code(), StatusCode::kInternal);
    EXPECT_THAT(status.message(), HasSubstr("a transfer is missing"));
}

TEST(Plugin, CreationErrorsAndLogging) {
    std::vector<std::string> messages;
    ContextOptions options;
    options.test_devices = "none";
    options.search_plugins = false;
    options.logger =
        Logger(LogLevel::kInfo, [&](LogLevel, std::string_view message) { messages.emplace_back(message); });
    const Context context = Context::create(std::move(options)).value();
    const ii_plugin_v1& table = *plugin::make_plugin_table(kInProcess);
    ASSERT_TRUE(plugin::add_plugin(*context.impl(), table, "a", Options{{"in_process.fail", "1"}}).ok());
    ASSERT_EQ(context.plugins().size(), 1u);
    EXPECT_EQ(context.plugins()[0].state, PluginState::kFailed);
    EXPECT_THAT(context.plugins()[0].message, HasSubstr("failing on request"));

    ASSERT_TRUE(plugin::add_plugin(*context.impl(), table, "b", Options{{"in_process.log", "1"}}).ok());
    EXPECT_THAT(messages, ::testing::Contains(HasSubstr("hello from the plugin")));

    // Another library with a loaded plugin's name is skipped.
    ASSERT_TRUE(plugin::add_plugin(*context.impl(), table, "c", {}).ok());
    ASSERT_EQ(context.plugins().size(), 3u);
    EXPECT_EQ(context.plugins()[2].state, PluginState::kFailed);
    EXPECT_THAT(context.plugins()[2].message, HasSubstr("already loaded"));

    // A backend whose name exists is skipped.
    const Context other = Context::create(ContextOptions{.test_devices = "none", .search_plugins = false}).value();
    ASSERT_TRUE(plugin::add_plugin(*other.impl(), table, "d", Options{{"in_process.host", "1"}}).ok());
    ASSERT_TRUE(plugin::add_plugin(*other.impl(), table, "d", Options{{"in_process.host", "1"}}).ok());
    EXPECT_EQ(other.backend_names().size(), 3u);
}

TEST(Plugin, SteadyStateRunDoesNotAllocate) {
    ii_test::CountingAllocator allocator;
    const Context context =
        in_process_context({{"in_process.host", "1"}, {"test_devices", std::string(kFast)}}, &allocator);
    const Model model = mlp(context, 3, 16);
    Session session = model.compile().value().new_session().value();
    ASSERT_TRUE(session.set_input("x", ii_test::random_tensor(DType::kF32, Shape{4, 16}, 5)).ok());
    ASSERT_TRUE(session.run().ok());
    CompileOptions host_only;
    host_only.backends = {"ii-plugin-ref"};
    Session host_session = model.compile(host_only).value().new_session().value();
    ASSERT_TRUE(host_session.set_input("x", ii_test::random_tensor(DType::kF32, Shape{4, 16}, 5)).ok());
    ASSERT_TRUE(host_session.run().ok());
    {
        ii_test::AllocationCounter allocations;
        for (int i = 0; i < 5; ++i) {
            ASSERT_TRUE(session.run().ok());
            ASSERT_TRUE(host_session.run().ok());
        }
        EXPECT_EQ(allocations.count(), 0) << "heap allocations in steady-state run()";
    }
}

TEST(Plugin, CheckBackends) {
    const Context context = in_process_context({{"in_process.host", "1"}, {"test_devices", std::string(kFast)}});
    const Result<std::vector<BackendCheck>> checks = check_backends(context);
    ASSERT_TRUE(checks.ok()) << checks.status().to_string();
    ASSERT_EQ(checks->size(), 3u);
    EXPECT_EQ((*checks)[0].backend, "ii-cpu");
    EXPECT_EQ((*checks)[1].backend, "ii-plugin-ref");
    EXPECT_EQ((*checks)[2].backend, "ii-test-remote");
    EXPECT_EQ((*checks)[2].device, "test:0");
    for (const BackendCheck& check : *checks) {
        SCOPED_TRACE(check.backend);
        EXPECT_EQ(check.failed, 0);
        EXPECT_GE(check.passed, 25);
        for (const BackendCheckCase& test : check.cases) {
            EXPECT_NE(test.state, BackendCheckState::kFailed) << test.name << ": " << test.message;
        }
    }
    EXPECT_EQ((*checks)[1].skipped, 0);  // everything ii-ref supports
    EXPECT_EQ((*checks)[2].skipped, 0);
    const std::vector<std::string> unknown = {"ii-nope"};
    EXPECT_EQ(check_backends(context, unknown).status().code(), StatusCode::kNotFound);
}

}  // namespace
}  // namespace ii
