// SPDX-License-Identifier: MIT
// The ii-backend-tensorrt plugin: parts of models compiled into TensorRT engines match ii-ref, alone and next to
// ii-cuda on the same GPU. Skipped when the plugin is not built or TensorRT or a GPU is missing.
#include <ii/ii.h>

#include "core/platform/platform.h"
#include "support/random_graph.h"
#include "support/test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ii;
using ::testing::HasSubstr;

std::string plugin_path(const char* variable, [[maybe_unused]] const char* built) {
    return platform::get_env(variable).value_or(built);
}

std::string tensorrt_plugin_path() {
#ifdef II_TEST_TENSORRT_PLUGIN_PATH
    return plugin_path("II_TEST_TENSORRT_PLUGIN_PATH", II_TEST_TENSORRT_PLUGIN_PATH);
#else
    return plugin_path("II_TEST_TENSORRT_PLUGIN_PATH", "");
#endif
}

std::string cuda_plugin_path() {
#ifdef II_TEST_CUDA_PLUGIN_PATH
    return plugin_path("II_TEST_CUDA_PLUGIN_PATH", II_TEST_CUDA_PLUGIN_PATH);
#else
    return plugin_path("II_TEST_CUDA_PLUGIN_PATH", "");
#endif
}

/// A directory for engines of one test (removed by the caller when it matters).
std::string engine_dir(std::string_view name) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / std::format("ii-tensorrt-{}", name);
    return platform::path_to_utf8(dir);
}

/// A context with the TensorRT plugin (and ii-cuda with `with_cuda`), or a reason to skip. Engines go to the
/// shared test cache directory unless the options name another.
Result<Context>
tensorrt_context(std::vector<std::pair<std::string, std::string>> plugin_options = {{"tensorrt.prefer", "1"}},
                 bool with_cuda = false) {
    plugin_options.emplace_back("tensorrt.devices", "all");  // overrides II_PLUGIN_OPTIONS
    plugin_options.emplace_back("tensorrt.cache", engine_dir("tests"));
    plugin_options.emplace_back("cuda.devices", "all");
    plugin_options.emplace_back("cuda.cache", "none");
    const std::string path = tensorrt_plugin_path();
    if (path.empty()) {
        return errors::unavailable("the TensorRT plugin is not built");
    }
    ContextOptions options;
    options.num_threads = 2;
    options.test_devices = "none";
    options.search_plugins = false;
    options.plugin_paths = {path};
    if (with_cuda) {
        if (cuda_plugin_path().empty()) {
            return errors::unavailable("the CUDA plugin is not built");
        }
        options.plugin_paths.push_back(cuda_plugin_path());
    }
    options.plugin_options = std::move(plugin_options);
    II_ASSIGN_OR_RETURN(Context context, Context::create(std::move(options)));
    for (const PluginDescription& plugin : context.plugins()) {
        if (plugin.state != PluginState::kLoaded) {
            return errors::unavailable("{}: {}", plugin.name, plugin.message);
        }
    }
    return context;
}

#define II_TENSORRT_CONTEXT(name, ...)                                                                                 \
    Result<Context> name##_result = tensorrt_context(__VA_ARGS__);                                                     \
    if (!name##_result.ok()) {                                                                                         \
        GTEST_SKIP() << name##_result.status().message();                                                              \
    }                                                                                                                  \
    const Context& name = *name##_result

std::vector<Tensor> random_inputs(const Model& model, std::uint32_t seed) {
    std::vector<Tensor> inputs;
    for (const TensorInfo& info : model.inputs()) {
        Shape shape;
        for (const Dim& dim : info.shape) {
            shape.push_back(dim.constant().value_or(0));
        }
        inputs.push_back(ii_test::random_tensor(info.dtype, shape, seed++));
    }
    return inputs;
}

/// Runs `model` with the given backends and with ii-ref and compares the outputs.
void expect_matches_reference(const Model& model, const std::vector<Tensor>& inputs,
                              std::vector<std::string> backends = {"ii-tensorrt"},
                              std::optional<ii_test::Tolerance> given = std::nullopt) {
    CompileOptions options;
    options.backends = std::move(backends);
    const Result<std::vector<Tensor>> actual = ii_test::run_model(model, inputs, options);
    ASSERT_TRUE(actual.ok()) << actual.status().to_string();
    CompileOptions reference;
    reference.backends = {"ii-ref"};
    const Result<std::vector<Tensor>> expected = ii_test::run_model(model, inputs, reference);
    ASSERT_TRUE(expected.ok()) << expected.status().to_string();
    ASSERT_EQ(actual->size(), expected->size());
    DType half = DType::kInvalid;
    for (const TensorInfo& info : model.inputs()) {
        if (info.dtype == DType::kBF16 || (info.dtype == DType::kF16 && half != DType::kBF16)) {
            half = info.dtype;
        }
    }
    for (std::size_t i = 0; i < actual->size(); ++i) {
        ii_test::Tolerance tolerance = ii_test::tolerance_for(half != DType::kInvalid ? half : (*expected)[i].dtype());
        if (half == DType::kInvalid && (*expected)[i].dtype() == DType::kF32) {
            tolerance = {1e-4, 1e-5};
        }
        if (given) {
            tolerance = *given;
        }
        const std::string mismatch = ii_test::compare((*actual)[i], (*expected)[i], tolerance);
        EXPECT_TRUE(mismatch.empty()) << "output " << i << ": " << mismatch;
    }
}

struct Case {
    const char* name;
    const char* text;
};

const Case kCases[] = {
    {"elementwise chain", R"(graph g {
  input %a : f32[4, 1, 8]
  input %b : f32[3, 1]
  %s : f32[4, 3, 8] = add(%a, %b)
  %t : f32[4, 3, 8] = silu(%s)
  %u : f32[4, 3, 8] = gelu(%t)
  %v : f32[4, 3, 8] = gelu(%s) {approximate = "tanh"}
  %w : f32[4, 3, 8] = sub(%u, %v)
  %e : f32[4, 3, 8] = exp(%w)
  %q : f32[4, 3, 8] = rsqrt(%e)
  %r : f32[4, 3, 8] = div(%q, %e)
  %m : f32[4, 3, 8] = max(%r, %b)
  %p : f32[4, 3, 8] = pow(%e, %m)
  output y = %p
  output z = %m
})"},
    {"unary", R"(graph g {
  input %a : f32[64]
  %n : f32[64] = neg(%a)
  %b : f32[64] = abs(%n)
  %s : f32[64] = sqrt(%b)
  %l : f32[64] = log(%s)
  %c : f32[64] = cos(%a)
  %i : f32[64] = sin(%c)
  %r : f32[64] = recip(%i)
  %e : f32[64] = erf(%a)
  %t : f32[64] = tanh(%e)
  %g : f32[64] = sigmoid(%t)
  %f : f32[64] = floor(%a)
  %h : f32[64] = ceil(%a)
  %u : f32[64] = relu(%a)
  output l = %l
  output r = %r
  output g = %g
  output f = %f
  output h = %h
  output u = %u
})"},
    {"f16 and bf16 chains", R"(graph g {
  input %a : f16[5, 7]
  input %b : f16[7]
  input %c : bf16[3, 4]
  %m : f16[5, 7] = mul(%a, %b)
  %t : f16[5, 7] = tanh(%m)
  %y : f32[5, 7] = cast(%t) {to = f32}
  %d : bf16[3, 4] = add(%c, %c)
  output y = %y
  output d = %d
})"},
    {"compare, logic and where", R"(graph g {
  input %a : f32[6, 4]
  input %b : f32[4]
  input %i : i32[6, 4]
  input %j : i64[4]
  %c : bool[6, 4] = gt(%a, %b)
  %y : f32[6, 4] = where(%c, %a, %b)
  %e : bool[6, 4] = le(%a, %b)
  %f : bool[6, 4] = ge(%a, %b)
  %n : bool[6, 4] = ne(%a, %b)
  %x : bool[6, 4] = xor(%c, %e)
  %o : bool[6, 4] = or(%x, %f)
  %d : bool[6, 4] = and(%o, %n)
  %k : bool[6, 4] = not(%d)
  %s : i32[6, 4] = add(%i, %i)
  %m : i64[4] = mul(%j, %j)
  %q : bool[4] = lt(%j, %m)
  output y = %y
  output k = %k
  output s = %s
  output m = %m
  output q = %q
})"},
    {"clamp, round, sign, casts", R"(graph g {
  input %a : f32[64]
  %c : f32[64] = clamp(%a) {min = -0.5, max = 1.25}
  %r : f32[64] = round(%a)
  %s : f32[64] = sign(%a)
  %i : i32[64] = cast(%r) {to = i32}
  %h : f16[64] = cast(%a) {to = f16}
  %b : bool[64] = cast(%s) {to = bool}
  output c = %c
  output r = %r
  output s = %s
  output i = %i
  output h = %h
  output b = %b
})"},
    {"reductions", R"(graph g {
  input %x : f32[3, 5, 7]
  %s : f32[3, 1, 7] = reduce_sum(%x) {axes = [1], keepdims = true}
  %m : f32[5] = reduce_max(%x) {axes = [0, 2]}
  %a : f32[] = reduce_mean(%x)
  %p : f32[3, 5] = reduce_prod(%x) {axes = [-1]}
  %n : f32[3, 7] = reduce_min(%x) {axes = [1]}
  output s = %s
  output m = %m
  output a = %a
  output p = %p
  output n = %n
})"},
    {"softmax then a sum of everything", R"(graph g {
  input %x : f32[4, 4]
  %s : f32[4, 4] = softmax(%x) {axis = -1}
  %t : f32[] = reduce_sum(%s)
  output s = %s
  output t = %t
})"},
    {"argmax and topk", R"(graph g {
  input %x : f32[4, 6, 3]
  input %r : f32[2, 1000]
  %a : i64[4, 3] = argmax(%x) {axis = 1}
  %b : i64[4, 1, 3] = argmin(%x) {axis = 1, keepdims = true}
  %v : f32[2, 5], %i : i64[2, 5] = topk(%r) {k = 5}
  %w : f32[4, 2, 3], %j : i64[4, 2, 3] = topk(%x) {k = 2, axis = 1, largest = false}
  output a = %a
  output b = %b
  output v = %v
  output i = %i
  output w = %w
  output j = %j
})"},
    {"softmax and norms", R"(graph g {
  input %x : f32[2, 5, 3]
  input %r : f32[3, 3000]
  input %n : f32[2, 4, 6]
  input %g : f32[4, 6]
  input %h : f32[4, 6]
  input %q : f32[6]
  %s : f32[2, 5, 3] = softmax(%x) {axis = 1}
  %l : f32[3, 3000] = log_softmax(%r)
  %y : f32[2, 4, 6] = layer_norm(%n, %g, %h) {axis = -2}
  %u : f32[2, 4, 6] = layer_norm(%n) {epsilon = 0.001}
  %z : f32[2, 4, 6] = rms_norm(%n, %q) {axis = -1}
  output s = %s
  output l = %l
  output y = %y
  output u = %u
  output z = %z
})"},
    {"matmul shapes", R"(graph g {
  input %a : f32[2, 1, 3, 4]
  input %b : f32[1, 3, 4, 5]
  input %v : f32[4]
  input %w : f32[2, 4, 5]
  input %p : f32[4, 3]
  input %q : f32[5, 4]
  input %x : f32[3, 4, 16]
  input %lw : f32[10, 16]
  input %lb : f32[10]
  %c : f32[2, 3, 3, 5] = matmul(%a, %b)
  %d : f32[2, 5] = matmul(%v, %w)
  %e : f32[3, 5] = matmul(%p, %q) {transpose_a = true, transpose_b = true}
  %y : f32[3, 4, 10] = linear(%x, %lw, %lb)
  output c = %c
  output d = %d
  output e = %e
  output y = %y
})"},
    {"f16 linear", R"(graph g {
  input %x : f16[6, 32]
  input %w : f16[32, 8]
  input %b : f16[8]
  %y : f16[6, 8] = linear(%x, %w, %b) {transpose_w = false}
  output y = %y
})"},
    {"convolutions", R"(graph g {
  input %x : f32[1, 4, 9, 9]
  input %dw : f32[4, 1, 3, 3]
  input %s : f32[2, 3, 10]
  input %sw : f32[4, 3, 3]
  input %p : f32[2, 8, 5, 5]
  input %pw : f32[6, 8, 1, 1]
  input %pb : f32[6]
  input %v : f32[1, 2, 4, 5, 6]
  input %vw : f32[3, 2, 3, 3, 3]
  %d : f32[1, 4, 5, 5] = conv(%x, %dw, _) {groups = 4, strides = [2, 2], pads = [1, 1, 1, 1]}
  %e : f32[2, 4, 6] = conv(%s, %sw, _) {dilations = [2]}
  %f : f32[2, 6, 5, 5] = conv(%p, %pw, %pb)
  %g : f32[2, 6, 5, 5] = conv(%p, %pw, %pb) {auto_pad = "same_upper"}
  %h : f32[1, 3, 4, 5, 6] = conv(%v, %vw, _) {pads = [1, 1, 1, 1, 1, 1]}
  output d = %d
  output e = %e
  output f = %f
  output g = %g
  output h = %h
})"},
    {"conv with epilogue", R"(graph g {
  input %x : f32[2, 4, 6, 6]
  input %r : f32[2, 4, 6, 6]
  input %w : f32[4, 4, 3, 3]
  const %b : f32[4] = f32[4] {0.5, -0.5, 0.25, 0}
  %0 : f32[2, 4, 6, 6] = conv(%x, %w, %b) {pads = [1, 1, 1, 1]}
  %2 : f32[2, 4, 6, 6] = add(%0, %r)
  %3 : f32[2, 4, 6, 6] = relu(%2)
  output y = %3
})"},
    {"transposed convolution", R"(graph g {
  input %x : f32[1, 2, 3, 3]
  const %w : f32[2, 1, 2, 2] = f32[2, 1, 2, 2] {0.5, -1, 0.25, 2, 1, 0.75, -0.5, 1.5}
  %y : f32[1, 1, 6, 6] = conv_transpose(%x, %w, _) {strides = [2, 2]}
  output y = %y
})"},
    {"pooling", R"(graph g {
  input %x : f32[1, 2, 7, 7]
  input %z : f32[1, 2, 5, 5]
  %a : f32[1, 2, 4, 4] = avg_pool(%x) {kernel = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1], count_include_pad = true}
  %b : f32[1, 2, 4, 4] = avg_pool(%x) {kernel = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1]}
  %m : f32[1, 2, 3, 3] = max_pool(%z) {kernel = [2, 2], strides = [2, 2], ceil_mode = true}
  output a = %a
  output b = %b
  output m = %m
})"},
    {"resize", R"(graph g {
  input %x : f32[1, 2, 4, 4]
  %n : f32[1, 2, 8, 8] = resize(%x) {shape = [1, 2, 8, 8]}
  %l : f32[1, 2, 6, 10] = resize(%x) {shape = [1, 2, 6, 10], mode = "linear"}
  %c : f32[1, 2, 7, 7] = resize(%x) {shape = [1, 2, 7, 7], mode = "cubic", coordinate_mode = "align_corners"}
  %a : f32[1, 2, 8, 8] = resize(%x) {shape = [1, 2, 8, 8], coordinate_mode = "asymmetric", nearest_mode = "floor"}
  output n = %n
  output l = %l
  output c = %c
  output a = %a
})"},
    {"movement", R"(graph g {
  input %x : f32[2, 5]
  input %t : f32[3, 4]
  input %u : f32[2, 3]
  input %v : f32[2, 5]
  input %w : f32[2, 6]
  const %i : i32[3] = i32[3] {-1, 0, 2}
  %r : f32[4, 9] = pad(%x) {pads = [1, 2, 1, 2], mode = "reflect"}
  %e : f32[4, 9] = pad(%x) {pads = [1, 2, 1, 2], mode = "edge"}
  %c : f32[3, 7] = pad(%x) {pads = [0, 1, 1, 1], value = 1.5}
  %g : f32[3, 3] = gather(%t, %i) {axis = 1}
  %l : f32[4, 6] = tile(%u) {repeats = [2, 2]}
  %f : f32[3, 4] = flip(%t) {axes = [0, 1]}
  %s : f32[2, 2] = slice(%w) {axes = [1], starts = [4], ends = [0], steps = [-2]}
  %k : f32[2, 10] = concat(%x, %v) {axis = 1}
  %p : f32[4, 3] = permute(%t) {perm = [1, 0]}
  %x2 : f32[2, 1, 5] = reshape(%x) {shape = [2, 1, 5]}
  %b : f32[2, 3, 5] = expand(%x2) {shape = [2, 3, 5]}
  %m : f32[2, 6] = cumsum(%w) {axis = 1}
  %n : f32[2, 6] = cumsum(%w) {axis = 1, exclusive = true, reverse = true}
  output r = %r
  output e = %e
  output c = %c
  output g = %g
  output l = %l
  output f = %f
  output s = %s
  output k = %k
  output p = %p
  output b = %b
  output m = %m
  output n = %n
})"},
    {"split, scatter and creation", R"(graph g {
  input %x : f32[3, 4]
  input %d : f32[3, 4]
  input %u : f32[1, 4]
  const %i : i64[1, 4] = i64[1, 4] {2, 0, -1, 1}
  %a : f32[1, 4], %b : f32[2, 4] = split(%x) {axis = 0, sizes = [1, 2]}
  %s : f32[3, 4] = scatter(%d, %i, %u) {axis = 0}
  %o : i64[2, 3] = iota() {shape = [2, 3], axis = 1, start = 1, step = 2}
  %f : f32[2, 4] = fill() {shape = [2, 4], value = 0.5}
  %y : f32[2, 4] = add(%b, %f)
  output a = %a
  output y = %y
  output s = %s
  output o = %o
})"},
    {"attention", R"(graph g {
  input %q : f32[1, 4, 6, 16]
  input %k : f32[1, 2, 9, 16]
  input %v : f32[1, 2, 9, 16]
  input %m : bool[6, 9]
  %a : f32[1, 4, 6, 16] = sdpa(%q, %k, %v, %m, _)
  %c : f32[1, 4, 6, 16] = sdpa(%q, %k, %v, _, _) {causal = true}
  output a = %a
  output c = %c
})"},
};

const char* case_text(std::string_view name) {
    for (const Case& test : kCases) {
        if (name == test.name) {
            return test.text;
        }
    }
    return "";
}

TEST(TensorRT, DeviceIsListed) {
    II_TENSORRT_CONTEXT(context, {});
    EXPECT_THAT(context.plugins()[0].backends, ::testing::Contains("ii-tensorrt"));
    EXPECT_THAT(context.backend_names(), ::testing::Contains("ii-tensorrt"));
}

TEST(TensorRT, NoDevicesOnRequest) {
    II_TENSORRT_CONTEXT(probe, {});
    (void)probe;
    ContextOptions options;
    options.test_devices = "none";
    options.search_plugins = false;
    options.plugin_paths = {tensorrt_plugin_path()};
    options.plugin_options = {{"tensorrt.devices", "none"}};
    const Context context = Context::create(std::move(options)).value();
    ASSERT_EQ(context.plugins().size(), 1u);
    EXPECT_EQ(context.plugins()[0].state, PluginState::kLoaded);
    EXPECT_TRUE(context.plugins()[0].backends.empty());
}

TEST(TensorRT, OperationsMatchTheReference) {
    II_TENSORRT_CONTEXT(context);
    for (const Case& test : kCases) {
        SCOPED_TRACE(test.name);
        const Result<Model> model = Model::from_text(context, test.text);
        ASSERT_TRUE(model.ok()) << model.status().to_string();
        CompileOptions options;
        options.backends = {"ii-tensorrt"};
        const Result<Program> program = model->compile(options);
        ASSERT_TRUE(program.ok()) << program.status().to_string();
        EXPECT_THAT(program->describe(), HasSubstr("subgraph"));
        expect_matches_reference(*model, random_inputs(*model, 7));
    }
}

TEST(TensorRT, SharesTheGpuWithIiCuda) {
    // TensorRT compiles the parts it can; ii-cuda runs the operation it cannot (integer modulo) on the same GPU, in
    // the same memory and on the same stream: no copies through the host between them.
    II_TENSORRT_CONTEXT(context, {{"tensorrt.prefer", "1"}, {"cuda.prefer", "1"}}, true);
    const Model model = Model::from_text(context, R"(graph g {
  input %x : f32[8, 16]
  input %i : i32[8, 16]
  input %w : f32[16, 16]
  %a : f32[8, 16] = matmul(%x, %w)
  %b : f32[8, 16] = tanh(%a)
  %c : i32[8, 16] = cast(%b) {to = i32}
  %m : i32[8, 16] = mod(%i, %c)
  %f : f32[8, 16] = cast(%m) {to = f32}
  %g : f32[8, 16] = add(%f, %b)
  %h : f32[8, 16] = softmax(%g)
  output y = %h
})")
                            .value();
    CompileOptions options;
    options.backends = {"ii-tensorrt", "ii-cuda"};
    const Program program = model.compile(options).value();
    const std::string plan = program.describe();
    EXPECT_THAT(plan, HasSubstr("ii-tensorrt @ cuda:0"));
    EXPECT_THAT(plan, HasSubstr("mod              i32[8, 16]"));
    EXPECT_THAT(plan, HasSubstr("ii-cuda @ cuda:0"));
    // Copies only for the three inputs and the output.
    std::size_t copies = 0;
    for (const std::string_view copy : {"transfer cpu -> cuda:0", "transfer cuda:0 -> cpu"}) {
        for (std::size_t at = plan.find(copy); at != std::string::npos; at = plan.find(copy, at + 1)) {
            ++copies;
        }
    }
    EXPECT_EQ(copies, 4u) << plan;
    std::vector<Tensor> inputs = random_inputs(model, 3);
    inputs[1] = ii_test::random_tensor(DType::kI32, Shape{8, 16}, 4, -50, 50);
    expect_matches_reference(model, inputs, {"ii-tensorrt", "ii-cuda"});
}

TEST(TensorRT, PlannerChoosesTensorRTForCnns) {
    // Without preferences, the estimate of a whole CNN as one engine beats ii-cuda's kernels one by one.
    II_TENSORRT_CONTEXT(context, {{"cuda.prefer", "0"}, {"tensorrt.prefer", "0"}}, true);  // over II_PLUGIN_OPTIONS
    GraphBuilder b("cnn");
    Var x = b.input("x", DType::kF32, {Dim("N"), 32, 64, 64});
    for (int layer = 0; layer < 6; ++layer) {
        const Var w = b.constant(ii_test::random_tensor(DType::kF32, Shape{32, 32, 3, 3}, 10 + layer, -0.2, 0.2));
        WindowOptions window;
        window.pads = {1, 1, 1, 1};
        x = relu(conv(x, w, {}, window));
    }
    b.output("y", x);
    const Model model = Model::from_builder(context, std::move(b)).value();
    CompileOptions options;
    options.plan_dims = {{"N", 16}};
    const Program program = model.compile(options).value();
    EXPECT_THAT(program.describe(), HasSubstr("subgraph"));
    EXPECT_THAT(program.describe_plan(), HasSubstr("ii-tensorrt compiles 1 part"));
}

TEST(TensorRT, DynamicShapesUseProfiles) {
    // One engine covers 1 .. 2 * N of the first N (at least 8); a larger N builds a second engine.
    const std::string cache = engine_dir("profiles");
    std::filesystem::remove_all(platform::path_from_utf8(cache));
    II_TENSORRT_CONTEXT(context, {{"tensorrt.prefer", "1"}, {"tensorrt.cache", cache}});
    const Model model = Model::from_text(context, R"(graph g {
  input %x : f32[N, S, 16]
  const %w : f32[16] = f32[16] {0.5, -1, 0.25, 2, 1, 0.75, -0.5, 1.5, 0.5, -1, 0.25, 2, 1, 0.75, -0.5, 1.5}
  %a : f32[N, S, 16] = mul(%x, %w)
  %r : f32[N * S, 16] = reshape(%a) {shape = [N * S, 16]}
  %s : f32[N * S, 16] = softmax(%r)
  %t : f32[N, S, 16] = reshape(%s) {shape = [N, S, 16]}
  %y : f32[N, 16] = reduce_mean(%t) {axes = [1]}
  output y = %y
})")
                            .value();
    CompileOptions options;
    options.backends = {"ii-tensorrt"};
    const Program program = model.compile(options).value();
    CompileOptions reference_options;
    reference_options.backends = {"ii-ref"};
    const Program reference = model.compile(reference_options).value();
    Session session = program.new_session().value();
    Session expected = reference.new_session().value();
    auto engines = [&] {
        std::size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(platform::path_from_utf8(cache))) {
            count += entry.path().extension() == ".engine" ? 1 : 0;
        }
        return count;
    };
    std::uint32_t seed = 1;
    for (const auto& [n, s, built] :
         std::vector<std::tuple<int, int, std::size_t>>{{2, 3, 1}, {5, 7, 1}, {1, 1, 1}, {20, 3, 2}, {3, 3, 2}}) {
        SCOPED_TRACE(std::format("N = {}, S = {}", n, s));
        const Tensor x = ii_test::random_tensor(DType::kF32, Shape{n, s, 16}, seed++);
        for (Session* run : {&session, &expected}) {
            ASSERT_TRUE(run->set_input("x", x).ok());
            const Status status = run->run();
            ASSERT_TRUE(status.ok()) << status.to_string();
        }
        EXPECT_EQ(ii_test::compare(session.output("y").value().contiguous().value(),
                                   expected.output("y").value().contiguous().value(), {1e-4, 1e-5}),
                  "");
        EXPECT_EQ(engines(), built);
    }
}

TEST(TensorRT, EnginesAreCached) {
    // A second context finds the engine on disk instead of building it; another TensorRT option builds another.
    const std::string cache = engine_dir("cache");
    std::filesystem::remove_all(platform::path_from_utf8(cache));
    const char* text = case_text("convolutions");  // five independent parts: five engines
    std::size_t first = 0;
    for (int pass = 0; pass < 3; ++pass) {
        SCOPED_TRACE(pass);
        std::vector<std::pair<std::string, std::string>> plugin_options = {{"tensorrt.prefer", "1"},
                                                                           {"tensorrt.cache", cache}};
        if (pass == 2) {
            plugin_options.emplace_back("tensorrt.level", "1");
        }
        II_TENSORRT_CONTEXT(context, plugin_options);
        const Model model = Model::from_text(context, text).value();
        expect_matches_reference(model, random_inputs(model, 5));
        std::size_t engines = 0;
        for (const auto& entry : std::filesystem::directory_iterator(platform::path_from_utf8(cache))) {
            engines += entry.path().extension() == ".engine" ? 1 : 0;
        }
        if (pass == 0) {
            first = engines;
            EXPECT_GT(first, 0u);
        }
        EXPECT_EQ(engines, pass == 2 ? 2 * first : first);
    }
    std::filesystem::remove_all(platform::path_from_utf8(cache));
}

TEST(TensorRT, RecordedRunsUseNewInputs) {
    // TensorRT engines and ii-cuda kernels in one recorded segment (a CUDA graph): every run computes with its own
    // inputs and states, also after the shapes change and back.
    II_TENSORRT_CONTEXT(context, {{"tensorrt.prefer", "1"}, {"cuda.prefer", "1"}}, true);
    const Model model = Model::from_text(context, R"(graph g {
  input %x : f32[N, 16]
  input %w : f32[16, 16]
  input %i : i32[N, 16]
  state %acc : f32[1, 16]
  %a : f32[N, 16] = matmul(%x, %w)
  %b : f32[N, 16] = tanh(%a)
  %k : i32[N, 16] = mod(%i, %i)
  %f : f32[N, 16] = cast(%k) {to = f32}
  %g : f32[N, 16] = add(%b, %f)
  %c : f32[1, 16] = reduce_sum(%g) {axes = [0], keepdims = true}
  %d : f32[1, 16] = add(%c, %acc)
  %e : f32[N, 16] = mul(%g, %d)
  %h : f32[N, 16] = sigmoid(%e)
  update %acc = %d
  output y = %h
})")
                            .value();
    std::vector<Program> programs;
    for (const std::vector<std::string>& backends :
         {std::vector<std::string>{"ii-tensorrt", "ii-cuda"}, std::vector<std::string>{"ii-ref"}}) {
        CompileOptions options;
        options.backends = backends;
        programs.push_back(model.compile(options).value());
    }
    EXPECT_THAT(programs[0].describe(), HasSubstr("ii-tensorrt"));
    Session gpu = programs[0].new_session().value();
    Session ref = programs[1].new_session().value();
    const Tensor w = ii_test::random_tensor(DType::kF32, Shape{16, 16}, 2, -0.5, 0.5);
    std::uint32_t seed = 1;
    for (const std::int64_t n : {3, 3, 3, 3, 5, 5, 5, 3, 3}) {
        SCOPED_TRACE(std::format("run {} (N = {})", seed, n));
        const Tensor x = ii_test::random_tensor(DType::kF32, Shape{n, 16}, seed++);
        const Tensor i = ii_test::random_tensor(DType::kI32, Shape{n, 16}, seed++, 1, 9);
        for (Session* session : {&gpu, &ref}) {
            ASSERT_TRUE(session->set_input("x", x).ok());
            ASSERT_TRUE(session->set_input("w", w).ok());
            ASSERT_TRUE(session->set_input("i", i).ok());
            const Status status = session->run();
            ASSERT_TRUE(status.ok()) << status.to_string();
        }
        const Tensor actual = gpu.output("y").value().contiguous().value();
        const Tensor expected = ref.output("y").value().contiguous().value();
        EXPECT_EQ(ii_test::compare(actual, expected, {1e-4, 1e-4}), "");
    }
}

TEST(TensorRT, SessionsRunConcurrently) {
    II_TENSORRT_CONTEXT(context);
    const Model model = Model::from_text(context, case_text("matmul shapes")).value();
    CompileOptions options;
    options.backends = {"ii-tensorrt"};
    const Program program = model.compile(options).value();
    const std::vector<Tensor> inputs = random_inputs(model, 3);
    CompileOptions reference_options;
    reference_options.backends = {"ii-ref"};
    const std::vector<Tensor> expected = ii_test::run_model(model, inputs, reference_options).value();
    auto run = [&] {
        Session session = program.new_session().value();
        for (int repeat = 0; repeat < 20; ++repeat) {
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                EXPECT_TRUE(session.set_input(program.inputs()[i].name, inputs[i]).ok());
            }
            EXPECT_TRUE(session.run().ok());
            for (std::size_t o = 0; o < expected.size(); ++o) {
                EXPECT_EQ(ii_test::compare(session.output(program.outputs()[o].name).value().contiguous().value(),
                                           expected[o], {1e-4, 1e-5}),
                          "");
            }
        }
    };
    std::thread a(run);
    std::thread b(run);
    a.join();
    b.join();
}

TEST(TensorRT, RandomModelsMatchTheReference) {
    II_TENSORRT_CONTEXT(context);
    for (std::uint32_t seed = 1; seed <= 16; ++seed) {
        SCOPED_TRACE(seed);
        const ii_test::RandomModel random = ii_test::random_model(context, seed);
        // TensorRT preferred; operations it lacks run on the CPU.
        expect_matches_reference(random.model, random.inputs, {});
    }
}

}  // namespace
