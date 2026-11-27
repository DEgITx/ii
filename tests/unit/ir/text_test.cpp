// SPDX-License-Identifier: MIT
#include "ir/text.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace ii::ir {
namespace {

using ::testing::AnyOf;
using ::testing::HasSubstr;

/// Parses, verifies, prints and re-parses; the two printed forms must be identical.
std::string round_trip(std::string_view text) {
    Result<std::unique_ptr<Graph>> graph = parse_graph(text);
    EXPECT_TRUE(graph.ok()) << graph.status().to_string() << "\n" << text;
    if (!graph.ok()) {
        return {};
    }
    const std::string printed = (*graph)->to_text();
    Result<std::unique_ptr<Graph>> again = parse_graph(printed);
    EXPECT_TRUE(again.ok()) << again.status().to_string() << "\n" << printed;
    if (again.ok()) {
        EXPECT_EQ((*again)->to_text(), printed);
    }
    return printed;
}

std::string parse_error(std::string_view text) {
    Result<std::unique_ptr<Graph>> graph = parse_graph(text);
    EXPECT_FALSE(graph.ok()) << text;
    return graph.ok() ? std::string() : graph.status().to_string();
}

TEST(IrTextTest, RoundTripsAllConstructs) {
    const std::string printed = round_trip(R"(
        # A small graph exercising every statement.
        graph "my graph" {
          input %x : f32[N, 4]
          input %"in put" : f32[N, 4]
          state %steps : i64[] = i64[] {0}
          state %h : f32[1, 4]
          const %w : f32[4, 2] = f32[4, 2] {1, 2.5, -3, 4e-08, inf, -inf, 7, 8}
          const %flags : bool[3] = bool[3] {true, false, true}
          const %q : q4_0[1, 32] = q4_0[1, 32] 0x000102030405060708090a0b0c0d0e0f1011
          %sum : f32[N, 4] = add(%x, %"in put")
          %0 : f32[N, 2] = matmul(%sum, %w) {transpose_b = false}
          %a : f32[N, 1], %b : f32[N, 1] = split(%0) {axis = 1, sizes = [1, 1]}
          %n : f32[N, 4] = layer_norm(%sum, _, _) {epsilon = 0.001}
          %r : f32[N, 2, 2] = reshape(%n) {shape = [N, 2, -1]}
          %one : i64[] = add(%steps, %steps)
          %g : f32[1, 4] = gelu(%h) {approximate = "tanh"}
          update %h = %g
          update %steps = %one
          output y = %a
          output "z z" = %r
        }
    )");
    EXPECT_THAT(printed, HasSubstr("graph \"my graph\" {"));
    EXPECT_THAT(printed, HasSubstr("input %\"in put\" : f32[N, 4]"));
    EXPECT_THAT(printed, HasSubstr("{1, 2.5, -3, 4e-08, inf, -inf, 7, 8}"));
    EXPECT_THAT(printed, HasSubstr("%n : f32[N, 4] = layer_norm(%sum, _, _) {axis = -1, epsilon = 0.001}"));
    EXPECT_THAT(printed, HasSubstr("{shape = [N, 2, -1]}"));
    EXPECT_THAT(printed, HasSubstr("{approximate = \"tanh\"}"));
    EXPECT_THAT(printed, HasSubstr("output \"z z\" = %r"));
}

TEST(IrTextTest, AttributeValues) {
    const std::string printed = round_trip(R"(
        graph {
          input %x : f32[2, 3]
          %c : i32[2, 3] = cast(%x) {to = i32}
          %k : f32[2, 3] = clamp(%x) {max = 6, min = -inf}
          %p : f32[3, 2] = permute(%x) {perm = [1, 0]}
          %f : f32[2, 3] = fill() {dtype = f32, shape = [2, 3], value = 0.5}
          output c = %c
          output k = %k
          output p = %p
          output f = %f
        }
    )");
    EXPECT_THAT(printed, HasSubstr("{to = i32}"));
    EXPECT_THAT(printed, HasSubstr("{max = 6.0, min = -inf}"));
}

TEST(IrTextTest, ElidedConstants) {
    PrintOptions options;
    options.max_constant_elements = 2;
    Graph graph;
    Value* c = graph.add_constant(Tensor::from({1.0f, 2.0f, 3.0f}, Shape{3}).value(), "big");
    graph.add_output("c", c);
    const std::string text = graph.to_text(options);
    EXPECT_THAT(text, HasSubstr("const %big : f32[3] = f32[3] {...}"));
    EXPECT_THAT(parse_error(text), HasSubstr("elided data"));
    ParseOptions parse_options;
    parse_options.resolve_constant = [](std::string_view name, const TensorType& type) -> Result<Tensor> {
        EXPECT_EQ(name, "big");
        return Tensor::zeros(type.dtype, *type.static_shape());
    };
    EXPECT_TRUE(parse_graph(text, parse_options).ok());
}

TEST(IrTextTest, Errors) {
    EXPECT_THAT(parse_error("grph {}"), HasSubstr("1:1: expected 'graph'"));
    EXPECT_THAT(parse_error("graph {\n  %y : f32[2] = relu(%x)\n}"), HasSubstr("2:22: undefined value %x"));
    EXPECT_THAT(parse_error("graph {\n input %x : f32[2]\n %y : f32[2] = rel(%x)\n}"), HasSubstr("unknown operation"));
    EXPECT_THAT(parse_error("graph { input %x : f33[2] }"), HasSubstr("expected a dtype"));
    EXPECT_THAT(parse_error("graph { input %x : f32[2] input %x : f32[2] }"), HasSubstr("defined twice"));
    EXPECT_THAT(parse_error("graph { input %x : f32[2]\n %y : f32[3] = relu(%x) }"),
                HasSubstr("shape inference gives f32[2]"));
    EXPECT_THAT(parse_error("graph { const %c : f32[2] = f32[2] {1, 2, 3} }"), HasSubstr("too many values"));
    EXPECT_THAT(parse_error("graph { const %c : f32[2] = f32[2] {1} }"), HasSubstr("1 values given"));
    EXPECT_THAT(parse_error("graph { const %c : i8[1] = i8[1] {300} }"), HasSubstr("invalid i8 value"));
    EXPECT_THAT(parse_error("graph { input %x : f32[2]\n %y : f32[2] = relu(%x) {foo = 1} }"),
                HasSubstr("no attribute 'foo'"));
    EXPECT_THAT(parse_error("graph { input %x : f32[2] "), HasSubstr("unexpected end"));
    EXPECT_THAT(parse_error("graph { input %x : f32[2+] }"), HasSubstr("invalid dimension"));
    EXPECT_THAT(parse_error("graph {} extra"), HasSubstr("unexpected text"));
    // Literal sizes are checked against the text before allocating (found by fuzz_ir_text).
    EXPECT_THAT(parse_error("graph { const %c : i64[3333333333333333333] = i64[3333333333333333333] {0} }"),
                HasSubstr("literal shorter than"));
    EXPECT_THAT(parse_error("graph { const %c : f32[4000000000, 4000000000, 4000000000] = "
                            "f32[4000000000, 4000000000, 4000000000] {0} }"),
                AnyOf(HasSubstr("invalid literal shape"), HasSubstr("invalid dimension")));
    EXPECT_THAT(parse_error("graph { const %q : q4_0[1, 3200] = q4_0[1, 3200] 0x0001 }"),
                HasSubstr("hex literal shorter than"));
}

}  // namespace
}  // namespace ii::ir
