// SPDX-License-Identifier: MIT
// The ii-backend-cuda plugin: models run with ii-cuda alone (inputs and outputs copied to and from the GPU) match
// ii-ref. Skipped when the plugin is not built or the machine has no usable NVIDIA GPU.
#include <ii/ii.h>

#include "core/platform/platform.h"
#include "support/random_graph.h"
#include "support/test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ii;
using ::testing::HasSubstr;

/// The plugin file: II_TEST_CUDA_PLUGIN_PATH, else the one built with the tests; empty when there is none.
std::string cuda_plugin_path() {
#ifdef II_TEST_CUDA_PLUGIN_PATH
    return platform::get_env("II_TEST_CUDA_PLUGIN_PATH").value_or(II_TEST_CUDA_PLUGIN_PATH);
#else
    return platform::get_env("II_TEST_CUDA_PLUGIN_PATH").value_or("");
#endif
}

/// A context with the CUDA plugin, or a reason to skip.
Result<Context> cuda_context(std::vector<std::pair<std::string, std::string>> plugin_options = {{"cuda.prefer", "1"}}) {
    plugin_options.emplace_back("cuda.devices", "all");  // overrides II_PLUGIN_OPTIONS
    const std::string path = cuda_plugin_path();
    if (path.empty()) {
        return errors::unavailable("the CUDA plugin is not built");
    }
    ContextOptions options;
    options.num_threads = 2;
    options.test_devices = "none";
    options.search_plugins = false;
    options.plugin_paths = {path};
    options.plugin_options = std::move(plugin_options);
    II_ASSIGN_OR_RETURN(Context context, Context::create(std::move(options)));
    if (context.plugins().empty() || context.plugins()[0].state != PluginState::kLoaded) {
        return errors::unavailable("ii-backend-cuda: {}",
                                   context.plugins().empty() ? "not loaded" : context.plugins()[0].message);
    }
    return context;
}

#define II_CUDA_CONTEXT(name, ...)                                                                                     \
    Result<Context> name##_result = cuda_context(__VA_ARGS__);                                                         \
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

/// Runs `model` with ii-cuda alone and with ii-ref and compares the outputs.
void expect_cuda_matches_reference(const Model& model, const std::vector<Tensor>& inputs,
                                   std::vector<std::string> backends = {"ii-cuda"}) {
    CompileOptions options;
    options.backends = std::move(backends);
    const Result<std::vector<Tensor>> actual = ii_test::run_model(model, inputs, options);
    ASSERT_TRUE(actual.ok()) << actual.status().to_string();
    CompileOptions reference;
    reference.backends = {"ii-ref"};
    const Result<std::vector<Tensor>> expected = ii_test::run_model(model, inputs, reference);
    ASSERT_TRUE(expected.ok()) << expected.status().to_string();
    ASSERT_EQ(actual->size(), expected->size());
    // Fused chains keep intermediate values in f32 where the reference rounds every result to its dtype: models
    // with 16-bit inputs are compared with the tolerance of the coarsest 16-bit type among them.
    DType half = DType::kInvalid;
    for (const TensorInfo& info : model.inputs()) {
        if (info.dtype == DType::kBF16 || (info.dtype == DType::kF16 && half != DType::kBF16)) {
            half = info.dtype;
        }
    }
    for (std::size_t i = 0; i < actual->size(); ++i) {
        const bool rounded = half != DType::kInvalid;
        ii_test::Tolerance tolerance = ii_test::tolerance_for(rounded ? half : (*expected)[i].dtype());
        if (!rounded && (*expected)[i].dtype() == DType::kF32) {
            tolerance = {1e-4, 1e-5};  // f32 accumulation (the reference accumulates in f64)
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
    {"broadcast bf16 chain", R"(graph g {
  input %a : bf16[4, 1, 8]
  input %b : bf16[3, 1]
  %s : bf16[4, 3, 8] = add(%a, %b)
  %y : bf16[4, 3, 8] = silu(%s)
  output y = %y
})"},
    {"f16 chain with cast", R"(graph g {
  input %a : f16[5, 7]
  input %b : f16[7]
  %m : f16[5, 7] = mul(%a, %b)
  %t : f16[5, 7] = tanh(%m)
  %y : f32[5, 7] = cast(%t) {to = f32}
  output y = %y
})"},
    {"integer arithmetic", R"(graph g {
  input %a : i64[32]
  input %b : i64[32]
  %d : i64[32] = div(%a, %b)
  %m : i64[32] = mod(%a, %b)
  %f : i64[32] = mod(%a, %b) {fmod = true}
  %p : i64[32] = pow(%a, %b)
  %n : i64[32] = neg(%a)
  output d = %d
  output m = %m
  output f = %f
  output p = %p
  output n = %n
})"},
    {"i32 compare and where", R"(graph g {
  input %a : i32[6, 4]
  input %b : i32[4]
  %c : bool[6, 4] = gt(%a, %b)
  %y : i32[6, 4] = where(%c, %a, %b)
  %e : bool[6, 4] = eq(%a, %b)
  %x : bool[6, 4] = xor(%c, %e)
  output y = %y
  output x = %x
})"},
    {"clamp, round, sign", R"(graph g {
  input %a : f32[64]
  %c : f32[64] = clamp(%a) {min = -0.5, max = 1.25}
  %r : f32[64] = round(%a)
  %s : f32[64] = sign(%a)
  %m : f32[64] = mod(%a, %c)
  output c = %c
  output r = %r
  output s = %s
  output m = %m
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
    {"long reductions", R"(graph g {
  input %x : f32[2, 6000]
  input %i : i64[3, 5000]
  %s : f32[2] = reduce_sum(%x) {axes = [1]}
  %t : f32[] = reduce_max(%x)
  %k : i64[3] = reduce_sum(%i) {axes = [1]}
  output s = %s
  output t = %t
  output k = %k
})"},
    {"argmax and argmin", R"(graph g {
  input %x : f32[4, 6, 3]
  input %r : f32[2, 1000]
  %a : i64[4, 3] = argmax(%x) {axis = 1}
  %b : i64[4, 1, 3] = argmin(%x) {axis = 1, keepdims = true, select_last_index = true}
  %c : i64[2] = argmax(%r) {axis = 1}
  output a = %a
  output b = %b
  output c = %c
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
  %z : f32[2, 4, 6] = rms_norm(%n, %q) {axis = -1}
  output s = %s
  output l = %l
  output y = %y
  output z = %z
})"},
    {"matmul shapes", R"(graph g {
  input %a : f32[2, 1, 3, 4]
  input %b : f32[1, 3, 4, 5]
  input %v : f32[4]
  input %w : f32[2, 4, 5]
  input %p : f32[4, 3]
  input %q : f32[5, 4]
  %c : f32[2, 3, 3, 5] = matmul(%a, %b)
  %d : f32[2, 5] = matmul(%v, %w)
  %e : f32[3, 5] = matmul(%p, %q) {transpose_a = true, transpose_b = true}
  output c = %c
  output d = %d
  output e = %e
})"},
    {"linear with bf16 weights", R"(graph g {
  input %x : f32[3, 4, 16]
  input %w : bf16[10, 16]
  input %b : f32[10]
  %y : f32[3, 4, 10] = linear(%x, %w, %b)
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
  %d : f32[1, 4, 5, 5] = conv(%x, %dw, _) {groups = 4, strides = [2, 2], pads = [1, 1, 1, 1]}
  %e : f32[2, 4, 6] = conv(%s, %sw, _) {dilations = [2]}
  %f : f32[2, 6, 5, 5] = conv(%p, %pw, %pb)
  output d = %d
  output e = %e
  output f = %f
})"},
    {"f16 convolution", R"(graph g {
  input %x : f16[2, 3, 8, 8]
  input %w : f16[4, 3, 3, 3]
  input %b : f16[4]
  %y : f16[2, 4, 8, 8] = conv(%x, %w, %b) {pads = [1, 1, 1, 1]}
  output y = %y
})"},
    {"conv with epilogue", R"(graph g {
  input %x : f32[2, 4, 6, 6]
  input %w : f32[4, 4, 3, 3]
  input %b : f32[4]
  input %r : f32[2, 4, 6, 6]
  input %k : f32[4, 1, 1]
  %0 : f32[2, 4, 6, 6] = conv(%x, %w, %b) {pads = [1, 1, 1, 1]}
  %1 : f32[2, 4, 6, 6] = mul(%0, %k)
  %2 : f32[2, 4, 6, 6] = add(%1, %r)
  %3 : f32[2, 4, 6, 6] = relu(%2)
  output y = %3
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
  output n = %n
  output l = %l
  output c = %c
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
})"},
    {"split", R"(graph g {
  input %x : f32[3, 4]
  %a : f32[1, 4], %b : f32[2, 4] = split(%x) {axis = 0, sizes = [1, 2]}
  output a = %a
  output b = %b
})"},
};

TEST(Cuda, DeviceIsListed) {
    II_CUDA_CONTEXT(context, {});
    EXPECT_THAT(context.plugins()[0].backends, ::testing::Contains("ii-cuda"));
    bool found = false;
    for (const DeviceDescription& device : context.devices()) {
        if (device.id.starts_with("cuda:")) {
            found = true;
            EXPECT_FALSE(device.name.empty());
            EXPECT_GT(device.memory_bytes, 0u);
        }
    }
    EXPECT_TRUE(found);
}

TEST(Cuda, NoDevicesOnRequest) {
    II_CUDA_CONTEXT(probe, {});
    (void)probe;
    ContextOptions options;
    options.test_devices = "none";
    options.search_plugins = false;
    options.plugin_paths = {cuda_plugin_path()};
    options.plugin_options = {{"cuda.devices", "none"}};
    const Context context = Context::create(std::move(options)).value();
    ASSERT_EQ(context.plugins().size(), 1u);
    EXPECT_EQ(context.plugins()[0].state, PluginState::kLoaded);
    EXPECT_TRUE(context.plugins()[0].backends.empty());
}

TEST(Cuda, OperationsMatchTheReference) {
    II_CUDA_CONTEXT(context);
    for (const Case& test : kCases) {
        SCOPED_TRACE(test.name);
        const Result<Model> model = Model::from_text(context, test.text);
        ASSERT_TRUE(model.ok()) << model.status().to_string();
        expect_cuda_matches_reference(*model, random_inputs(*model, 7));
    }
}

TEST(Cuda, ConvolutionEpilogueIsFused) {
    II_CUDA_CONTEXT(context);
    const Model model = Model::from_text(context, kCases[14].text).value();
    CompileOptions options;
    options.backends = {"ii-cuda"};
    EXPECT_THAT(model.compile(options).value().describe(), HasSubstr("fused_conv"));
}

TEST(Cuda, CastsSaturate) {
    II_CUDA_CONTEXT(context);
    const Model model = Model::from_text(context, R"(graph g {
  input %x : f32[10]
  %a : i32[10] = cast(%x) {to = i32}
  %b : u8[10] = cast(%x) {to = u8}
  %c : bool[10] = cast(%x) {to = bool}
  %d : f16[10] = cast(%x) {to = f16}
  %e : i64[10] = cast(%x) {to = i64}
  output a = %a
  output b = %b
  output c = %c
  output d = %d
  output e = %e
})")
                            .value();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> values = {std::nanf(""), inf, -inf, 3e10f, -3e10f, 2.7f, -2.7f, 0.5f, 300.f, 1e20f};
    Tensor x = Tensor::empty(DType::kF32, Shape{10}).value();
    std::copy(values.begin(), values.end(), x.values<float>().begin());
    expect_cuda_matches_reference(model, {x});
}

TEST(Cuda, KernelErrorsReachTheCaller) {
    // A gather index out of range is found on the GPU and reported when the session synchronizes, like on the CPU.
    II_CUDA_CONTEXT(context);
    const Model model = Model::from_text(context, R"(graph g {
  input %t : f32[10, 4]
  input %i : i64[3]
  %y : f32[3, 4] = gather(%t, %i) {axis = 0}
  output y = %y
})")
                            .value();
    CompileOptions options;
    options.backends = {"ii-cuda"};
    const Program program = model.compile(options).value();
    Session session = program.new_session().value();
    Tensor indices = Tensor::empty(DType::kI64, Shape{3}).value();
    auto values = indices.values<std::int64_t>();
    values[0] = 1;
    values[1] = 12;
    values[2] = -1;
    ASSERT_TRUE(session.set_input("t", ii_test::random_tensor(DType::kF32, Shape{10, 4}, 1)).ok());
    ASSERT_TRUE(session.set_input("i", indices).ok());
    const Status status = session.run();
    EXPECT_EQ(status.code(), StatusCode::kOutOfRange) << status.to_string();
    EXPECT_THAT(std::string(status.message()), HasSubstr("gather index 12 is out of range for axis size 10"));
    values[1] = 9;  // the error is cleared: the next run succeeds
    ASSERT_TRUE(session.set_input("i", indices).ok());
    EXPECT_TRUE(session.run().ok());
}

TEST(Cuda, RandomModelsMatchTheReference) {
    II_CUDA_CONTEXT(context);
    for (std::uint32_t seed = 1; seed <= 16; ++seed) {
        SCOPED_TRACE(seed);
        const ii_test::RandomModel random = ii_test::random_model(context, seed);
        // ii-cuda preferred; operations it lacks run on the CPU.
        expect_cuda_matches_reference(random.model, random.inputs, {});
    }
}

TEST(Cuda, SessionsRunConcurrently) {
    II_CUDA_CONTEXT(context);
    const Model model = Model::from_text(context, kCases[12].text).value();
    CompileOptions options;
    options.backends = {"ii-cuda"};
    const Program program = model.compile(options).value();
    const std::vector<Tensor> inputs = random_inputs(model, 3);
    auto run = [&] {
        Session session = program.new_session().value();
        std::vector<Tensor> outputs;
        for (int repeat = 0; repeat < 20; ++repeat) {
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                EXPECT_TRUE(session.set_input(program.inputs()[i].name, inputs[i]).ok());
            }
            EXPECT_TRUE(session.run().ok());
            outputs.clear();
            for (const TensorInfo& info : program.outputs()) {
                outputs.push_back(session.output(info.name).value().contiguous().value());
            }
        }
        return outputs;
    };
    std::vector<Tensor> first;
    std::vector<Tensor> second;
    std::thread other([&] { second = run(); });
    first = run();
    other.join();
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(ii_test::compare(first[i], second[i], {0, 0}), "");
    }
}

}  // namespace
