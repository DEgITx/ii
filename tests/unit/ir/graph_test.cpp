// SPDX-License-Identifier: MIT
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/text.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace ii::ir {
namespace {

using ::testing::HasSubstr;

TensorType f32(Dims shape) {
    return TensorType{DType::kF32, std::move(shape)};
}

TEST(GraphTest, BuildVerifyAndPrint) {
    Graph graph;
    graph.set_name("mlp");
    Builder b(graph);
    Value* x = graph.add_input("x", f32({"N", 4}));
    Value* w = b.constant(Tensor::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{4, 2}).value());
    Value* h = b.op("matmul", {x, w});
    Value* y = b.op("relu", {h});
    ASSERT_TRUE(b.ok()) << b.status().to_string();
    graph.add_output("y", y);
    ASSERT_TRUE(graph.verify().ok()) << graph.verify().to_string();

    EXPECT_EQ(h->type().to_string(), "f32[N, 2]");
    EXPECT_EQ(y->producer()->op_name(), "relu");
    EXPECT_EQ(h->uses().size(), 1u);
    EXPECT_EQ(graph.num_nodes(), 2u);
    EXPECT_EQ(graph.to_text(), R"(graph mlp {
  input %x : f32[N, 4]
  const %0 : f32[4, 2] = f32[4, 2] {1, 2, 3, 4, 5, 6, 7, 8}
  %1 : f32[N, 2] = matmul(%x, %0) {transpose_a = false, transpose_b = false}
  %2 : f32[N, 2] = relu(%1)
  output y = %2
}
)");
}

TEST(GraphTest, BuilderReportsFirstError) {
    Graph graph;
    Builder b(graph);
    Value* x = graph.add_input("x", f32({2, 3}));
    Value* y = graph.add_input("y", f32({4, 5}));
    EXPECT_EQ(b.op("add", {x, y}), nullptr);
    EXPECT_FALSE(b.ok());
    EXPECT_THAT(b.status().to_string(), HasSubstr("not broadcastable"));
    EXPECT_EQ(b.op("relu", {x}), nullptr);  // nothing happens after an error
    EXPECT_EQ(graph.num_nodes(), 0u);
}

TEST(GraphTest, BuilderValidatesAttributes) {
    Graph graph;
    Builder b(graph);
    Value* x = graph.add_input("x", f32({2, 3}));
    EXPECT_THAT(b.create_node("softmax", {x}, {{"axs", 1}}).status().to_string(), HasSubstr("no attribute 'axs'"));
    EXPECT_THAT(b.create_node("softmax", {x}, {{"axis", "one"}}).status().to_string(), HasSubstr("must be int"));
    EXPECT_THAT(b.create_node("softmax", {x}, {{"axis", 2}}).status().to_string(), HasSubstr("out of range"));
    EXPECT_THAT(b.create_node("cast", {x}).status().to_string(), HasSubstr("requires attribute 'to'"));
    EXPECT_THAT(b.create_node("nope", {x}).status().to_string(), HasSubstr("unknown operation"));
    EXPECT_THAT(b.create_node("relu", {x, x}).status().to_string(), HasSubstr("expects 1 inputs"));
    EXPECT_THAT(b.create_node("layer_norm", {nullptr}).status().to_string(), HasSubstr("is required"));
    // Conversions: int -> float, ints -> dims.
    Node* clamp = b.create_node("clamp", {x}, {{"min", 0}}).value();
    EXPECT_EQ(kind_of(*clamp->attrs().find("min")), AttrKind::kFloat);
    Node* reshape = b.create_node("reshape", {x}, {{"shape", std::vector<std::int64_t>{3, 2}}}).value();
    EXPECT_EQ(kind_of(*reshape->attrs().find("shape")), AttrKind::kDims);
}

TEST(GraphTest, ReplaceAndErase) {
    Graph graph;
    Builder b(graph);
    Value* x = graph.add_input("x", f32({3}));
    Value* a = b.op("neg", {x});
    Value* c = b.op("exp", {a});
    graph.add_output("c", c);
    Value* d = b.op("abs", {x});
    graph.replace_all_uses(a, d);
    EXPECT_TRUE(a->uses().empty());
    EXPECT_EQ(c->producer()->input(0), d);
    // The abs node was appended after exp, which now uses it: restore topological order.
    EXPECT_FALSE(graph.verify().ok());
    ASSERT_TRUE(graph.topological_sort().ok());
    ASSERT_TRUE(graph.verify().ok()) << graph.verify().to_string();
    graph.erase_nodes_if([](const Node& node) { return node.op_name() == "neg"; });
    EXPECT_EQ(graph.num_nodes(), 2u);
    EXPECT_TRUE(graph.verify().ok());
    graph.replace_all_uses(c, d);
    EXPECT_EQ(graph.outputs()[0].value, d);
}

TEST(GraphTest, CloneIsDeep) {
    Graph graph;
    Builder b(graph);
    Value* x = graph.add_input("x", f32({"N"}));
    Value* s = graph.add_state("s", f32({"N"}));
    Value* y = b.op("add", {x, s});
    graph.set_state_update(s, y);
    graph.add_output("y", y);
    std::unique_ptr<Graph> copy = graph.clone();
    ASSERT_TRUE(copy->verify().ok());
    EXPECT_EQ(copy->to_text(), graph.to_text());
    EXPECT_NE(copy->outputs()[0].value, y);
    EXPECT_EQ(copy->states()[0].update, copy->outputs()[0].value);
}

TEST(GraphTest, VerifyCatchesErrors) {
    Graph graph;
    Builder b(graph);
    Value* x = graph.add_input("x", f32({2}));
    Value* y = b.op("neg", {x});
    graph.add_output("y", y);
    y->set_type(f32({3}));
    EXPECT_THAT(graph.verify().to_string(), HasSubstr("shape inference gives f32[2]"));
    y->set_type(f32({2}));
    graph.add_output("y", y);
    EXPECT_THAT(graph.verify().to_string(), HasSubstr("not unique"));
}

TEST(GraphTest, CycleDetection) {
    Graph graph;
    Builder b(graph);
    Value* x = graph.add_input("x", f32({2}));
    Value* a = b.op("neg", {x});
    Value* c = b.op("abs", {a});
    a->producer()->set_input(0, c);
    EXPECT_THAT(graph.topological_sort().to_string(), HasSubstr("cycle"));
    EXPECT_EQ(graph.num_nodes(), 2u);
}

}  // namespace
}  // namespace ii::ir
