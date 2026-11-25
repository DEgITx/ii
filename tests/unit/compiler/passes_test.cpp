// SPDX-License-Identifier: MIT
#include "backends/ref/ref_backend.h"
#include "compiler/passes.h"
#include "ir/text.h"
#include "support/test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace ii {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

std::unique_ptr<ir::Graph> parse(std::string_view text) {
    Result<std::unique_ptr<ir::Graph>> graph = ir::parse_graph(text);
    EXPECT_TRUE(graph.ok()) << graph.status().to_string();
    return graph.ok() ? std::move(*graph) : std::make_unique<ir::Graph>();
}

TEST(PassesTest, DeadCodeElimination) {
    auto graph = parse(R"(graph {
      input %x : f32[2]
      const %k : f32[1] = f32[1] {1}
      %a : f32[2] = neg(%x)
      %b : f32[2] = exp(%x)
      %c : f32[2] = add(%b, %k)
      state %s : f32[2]
      %u : f32[2] = abs(%x)
      update %s = %u
      output y = %a
    })");
    EXPECT_EQ(compiler::eliminate_dead_code(*graph), 2u);
    ASSERT_TRUE(graph->verify().ok());
    EXPECT_EQ(graph->to_text(), R"(graph {
  input %x : f32[2]
  state %s : f32[2]
  %a : f32[2] = neg(%x)
  %u : f32[2] = abs(%x)
  update %s = %u
  output y = %a
}
)");
}

TEST(PassesTest, ConstantFolding) {
    auto graph = parse(R"(graph {
      input %x : f32[2]
      const %a : f32[2] = f32[2] {1, 2}
      const %b : f32[2] = f32[2] {3, 4}
      %s : f32[2] = add(%a, %b)
      %m : f32[2] = mul(%s, %s)
      %y : f32[2] = add(%x, %m)
      %r : f32[2] = random_uniform() {shape = [2]}
      %z : f32[2] = add(%r, %m)
      output y = %y
      output z = %z
    })");
    const auto ref = ref::create_ref_backend();
    EXPECT_EQ(compiler::fold_constants(*graph, *ref).value(), 2u);
    ASSERT_TRUE(graph->verify().ok());
    const std::string text = graph->to_text();
    EXPECT_THAT(text, HasSubstr("const %m : f32[2] = f32[2] {16, 36}"));
    EXPECT_THAT(text, Not(HasSubstr("= mul(")));
    EXPECT_THAT(text, HasSubstr("random_uniform"));
    EXPECT_THAT(text, Not(HasSubstr("const %a")));  // unused constants are removed
}

TEST(PassesTest, ConstantFoldingDoesNotInflate) {
    auto graph = parse(R"(graph {
      const %one : f32[] = f32[] {1}
      %big : f32[1024, 1024] = expand(%one) {shape = [1024, 1024]}
      %small : f32[2] = expand(%one) {shape = [2]}
      output big = %big
      output small = %small
    })");
    const auto ref = ref::create_ref_backend();
    EXPECT_EQ(compiler::fold_constants(*graph, *ref).value(), 1u);
    EXPECT_THAT(graph->to_text(), HasSubstr("= expand(%one) {shape = [1024, 1024]}"));
}

TEST(PassesTest, CommonSubexpressionElimination) {
    auto graph = parse(R"(graph {
      input %x : f32[2]
      %a : f32[2] = exp(%x)
      %b : f32[2] = exp(%x)
      %c : f32[2] = clamp(%x) {max = 1, min = 0}
      %d : f32[2] = clamp(%x) {max = 2, min = 0}
      %s : f32[2] = add(%a, %b)
      %t : f32[2] = add(%a, %b)
      %r1 : f32[2] = random_uniform() {shape = [2]}
      %r2 : f32[2] = random_uniform() {shape = [2]}
      %u : f32[2] = mul(%s, %t)
      %v : f32[2] = add(%c, %d)
      %w : f32[2] = add(%r1, %r2)
      output u = %u
      output v = %v
      output w = %w
    })");
    EXPECT_EQ(compiler::eliminate_common_subexpressions(*graph), 2u);  // %b, %t; different attributes and random
    ASSERT_TRUE(graph->verify().ok());
    EXPECT_THAT(graph->to_text(), HasSubstr("%u : f32[2] = mul(%s, %s)"));
}

TEST(PassesTest, FusesMatmulAndBiasIntoLinear) {
    auto graph = parse(R"(graph {
      input %x : f32[N, 3]
      const %w : f32[3, 2] = f32[3, 2] {1, 2, 3, 4, 5, 6}
      const %b : f32[2] = f32[2] {1, 2}
      const %wt : f32[2, 3] = f32[2, 3] {1, 2, 3, 4, 5, 6}
      %m : f32[N, 2] = matmul(%x, %w)
      %y : f32[N, 2] = add(%b, %m)
      %m2 : f32[N, 2] = matmul(%x, %wt) {transpose_b = true}
      %y2 : f32[N, 2] = add(%m2, %b)
      %m3 : f32[N, 2] = matmul(%x, %w)
      %y3 : f32[N, 2] = add(%m3, %m3)
      output y = %y
      output y2 = %y2
      output y3 = %y3
    })");
    EXPECT_EQ(compiler::fuse_patterns(*graph).value(), 2u);
    ASSERT_TRUE(graph->verify().ok());
    const std::string text = graph->to_text();
    EXPECT_THAT(text, HasSubstr("%y : f32[N, 2] = linear(%x, %w, %b) {transpose_w = false}"));
    EXPECT_THAT(text, HasSubstr("%y2 : f32[N, 2] = linear(%x, %wt, %b) {transpose_w = true}"));
    EXPECT_THAT(text, HasSubstr("matmul(%x, %w)"));  // %m3 has two uses: not a bias pattern
}

TEST(PassesTest, FusesElementwiseChains) {
    auto graph = parse(R"(graph {
      input %x : f32[N, 8]
      input %b : f32[8]
      input %m : bool[N, 8]
      %a : f32[N, 8] = add(%x, %b)
      %s : f32[N, 8] = silu(%a)
      %c : f32[N, 8] = where(%m, %s, %x)
      %r : f32[N] = reduce_sum(%c) {axes = [1]}
      %e : f32[N] = exp(%r)
      %h : f16[N] = cast(%e) {to = f16}
      output y = %h
      output c = %c
    })");
    EXPECT_EQ(compiler::fuse_elementwise(*graph).value(), 2u);
    ASSERT_TRUE(graph->verify().ok());
    const std::string text = graph->to_text();
    EXPECT_THAT(text, HasSubstr("%c : f32[N, 8] = fused_elementwise(%x, %b, %m)"));
    EXPECT_THAT(text, HasSubstr("%h : f16[N] = fused_elementwise(%r)"));
    EXPECT_THAT(text, HasSubstr("reduce_sum"));
    EXPECT_THAT(text, Not(HasSubstr("%s : f32[N, 8] = silu")));
    // The fused program decomposes back into the original operations.
    EXPECT_EQ(
        compiler::decompose(*graph, [](const ir::Node& node) { return node.op_name() == "fused_elementwise"; }).value(),
        2u);
    ASSERT_TRUE(graph->verify().ok());
    EXPECT_THAT(graph->to_text(), HasSubstr("silu("));
}

TEST(PassesTest, ElementwiseFusionKeepsSharedAndBroadcastValues) {
    auto graph = parse(R"(graph {
      input %x : f32[4, 8]
      input %v : f32[8]
      %e : f32[8] = exp(%v)
      %a : f32[4, 8] = mul(%x, %e)
      %t : f32[4, 8] = tanh(%a)
      %u : f32[4, 8] = neg(%t)
      output u = %u
      output t = %t
    })");
    // %e has a different shape (it would be recomputed per row) and %t is a graph output: neither is absorbed.
    EXPECT_EQ(compiler::fuse_elementwise(*graph).value(), 1u);
    ASSERT_TRUE(graph->verify().ok());
    const std::string text = graph->to_text();
    EXPECT_THAT(text, HasSubstr("%e : f32[8] = exp(%v)"));
    EXPECT_THAT(text, HasSubstr("%t : f32[4, 8] = fused_elementwise(%x, %e)"));
    EXPECT_THAT(text, HasSubstr("%u : f32[4, 8] = neg(%t)"));
}

TEST(PassesTest, DecompositionKeepsTypes) {
    auto graph = parse(R"(graph {
      input %q : f32[B, 4, S, 8]
      input %k : f32[B, 2, T, 8]
      input %v : f32[B, 2, T, 8]
      %y : f32[B, 4, S, 8] = sdpa(%q, %k, %v, _) {causal = true}
      output y = %y
    })");
    EXPECT_GT(compiler::decompose(*graph, [](const ir::Node&) { return true; }).value(), 1u);
    ASSERT_TRUE(graph->verify().ok()) << graph->verify().to_string();
    const std::string text = graph->to_text();
    EXPECT_THAT(text, Not(HasSubstr("sdpa(")));
    EXPECT_THAT(text, Not(HasSubstr("softmax(")));
    EXPECT_THAT(text, HasSubstr("matmul("));
}

struct DecompositionCase {
    const char* name;
    const char* text;
    std::vector<Shape> input_shapes;
};

class DecompositionTest : public ::testing::TestWithParam<DecompositionCase> {};

TEST_P(DecompositionTest, MatchesNativeKernel) {
    const DecompositionCase& c = GetParam();
    const Context context = ii_test::test_context();
    const Model model = Model::from_text(context, c.text).value();
    std::vector<Tensor> inputs;
    const std::vector<TensorInfo> infos = model.inputs();
    for (std::size_t i = 0; i < c.input_shapes.size(); ++i) {
        const bool mask = infos[i].dtype == DType::kBool;
        inputs.push_back(ii_test::random_tensor(infos[i].dtype, c.input_shapes[i], static_cast<std::uint32_t>(17 + i),
                                                mask ? 0.0 : -2.0, mask ? 1.6 : 2.0));
    }
    const std::vector<Tensor> native = ii_test::run_model(model, inputs).value();
    CompileOptions options;
    options.decompose_composites = true;
    const Result<std::vector<Tensor>> decomposed = ii_test::run_model(model, inputs, options);
    ASSERT_TRUE(decomposed.ok()) << decomposed.status().to_string();
    // The decomposed graph contains no composite operations.
    const std::string text = model.compile(options).value().to_text();
    for (const char* op : {"gelu(", "silu(", "softmax(", "layer_norm(", "rms_norm(", "linear(", "sdpa("}) {
        EXPECT_THAT(text, Not(HasSubstr(std::string(" = ") + op)));
    }
    ASSERT_EQ(native.size(), decomposed->size());
    for (std::size_t i = 0; i < native.size(); ++i) {
        const std::string mismatch = ii_test::compare((*decomposed)[i], native[i], {2e-5, 2e-6});
        EXPECT_TRUE(mismatch.empty()) << mismatch;
    }
}

const DecompositionCase kDecompositionCases[] = {
    {"gelu", "graph { input %x : f32[3, 5]\n %y : f32[3, 5] = gelu(%x)\n output y = %y }", {{3, 5}}},
    {"gelu_tanh",
     "graph { input %x : f32[7]\n %y : f32[7] = gelu(%x) {approximate = \"tanh\"}\n output y = %y }",
     {{7}}},
    {"silu", "graph { input %x : f32[N]\n %y : f32[N] = silu(%x)\n output y = %y }", {{9}}},
    {"softmax", "graph { input %x : f32[2, N]\n %y : f32[2, N] = softmax(%x)\n output y = %y }", {{2, 6}}},
    {"softmax_axis0",
     "graph { input %x : f32[4, 3]\n %y : f32[4, 3] = softmax(%x) {axis = 0}\n output y = %y }",
     {{4, 3}}},
    {"log_softmax", "graph { input %x : f32[3, 4]\n %y : f32[3, 4] = log_softmax(%x)\n output y = %y }", {{3, 4}}},
    {"layer_norm",
     "graph { input %x : f32[2, 3, 4]\n input %g : f32[3, 4]\n input %b : f32[3, 4]\n %y : f32[2, 3, 4] = "
     "layer_norm(%x, %g, %b) {axis = 1}\n output y = %y }",
     {{2, 3, 4}, {3, 4}, {3, 4}}},
    {"layer_norm_plain", "graph { input %x : f32[5, 8]\n %y : f32[5, 8] = layer_norm(%x)\n output y = %y }", {{5, 8}}},
    {"rms_norm",
     "graph { input %x : f32[N, 6]\n input %g : f32[6]\n %y : f32[N, 6] = rms_norm(%x, %g)\n output y = %y }",
     {{3, 6}, {6}}},
    {"linear",
     "graph { input %x : f32[2, N, 4]\n input %w : f32[5, 4]\n input %b : f32[5]\n %y : f32[2, N, 5] = linear(%x, "
     "%w, %b)\n output y = %y }",
     {{2, 3, 4}, {5, 4}, {5}}},
    {"linear_no_transpose",
     "graph { input %x : f32[3, 4]\n input %w : f32[4, 2]\n %y : f32[3, 2] = linear(%x, %w) {transpose_w = false}\n "
     "output y = %y }",
     {{3, 4}, {4, 2}}},
    {"sdpa",
     "graph { input %q : f32[1, 2, 3, 4]\n input %k : f32[1, 2, 5, 4]\n input %v : f32[1, 2, 5, 3]\n %y : "
     "f32[1, 2, 3, 3] = sdpa(%q, %k, %v)\n output y = %y }",
     {{1, 2, 3, 4}, {1, 2, 5, 4}, {1, 2, 5, 3}}},
    {"sdpa_gqa_causal_symbolic",
     "graph { input %q : f32[B, 4, S, 8]\n input %k : f32[B, 2, T, 8]\n input %v : f32[B, 2, T, 8]\n %y : f32[B, 4, "
     "S, 8] = sdpa(%q, %k, %v, _) {causal = true, scale = 0.3}\n output y = %y }",
     {{2, 4, 3, 8}, {2, 2, 5, 8}, {2, 2, 5, 8}}},
    {"sdpa_masks",
     "graph { input %q : f32[1, 2, 3, 4]\n input %k : f32[1, 2, 4, 4]\n input %v : f32[1, 2, 4, 4]\n input %m : "
     "f32[3, 4]\n %y : f32[1, 2, 3, 4] = sdpa(%q, %k, %v, %m)\n output y = %y }",
     {{1, 2, 3, 4}, {1, 2, 4, 4}, {1, 2, 4, 4}, {3, 4}}},
};

INSTANTIATE_TEST_SUITE_P(Composites, DecompositionTest, ::testing::ValuesIn(kDecompositionCases),
                         [](const ::testing::TestParamInfo<DecompositionCase>& param_info) {
                             return std::string(param_info.param.name);
                         });

}  // namespace
}  // namespace ii
