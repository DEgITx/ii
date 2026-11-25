// SPDX-License-Identifier: MIT
// Property tests over random models: constant folding, decomposition and the IR text round trip must not change
// results, and every backend must agree with ii-ref (compare_backends).
#include "ir/text.h"
#include "runtime/runtime.h"
#include "support/random_graph.h"
#include "support/test_util.h"

#include <gtest/gtest.h>

namespace ii {
namespace {

class PassInvarianceTest : public ::testing::TestWithParam<std::uint32_t> {};

void expect_same(const std::vector<Tensor>& actual, const std::vector<Tensor>& expected, std::string_view what) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const std::string mismatch = ii_test::compare(actual[i], expected[i], {1e-4, 1e-5});
        EXPECT_TRUE(mismatch.empty()) << what << ", output " << i << ":\n" << mismatch;
    }
}

TEST_P(PassInvarianceTest, PassesPreserveResults) {
    const Context context = ii_test::test_context();
    const ii_test::RandomModel random = ii_test::random_model(context, GetParam());
    CompileOptions plain;
    plain.fold_constants = false;
    const std::vector<Tensor> expected = ii_test::run_model(random.model, random.inputs, plain).value();

    expect_same(ii_test::run_model(random.model, random.inputs).value(), expected, "constant folding");
    CompileOptions decomposed;
    decomposed.decompose_composites = true;
    expect_same(ii_test::run_model(random.model, random.inputs, decomposed).value(), expected, "decomposition");

    // Text round trip of the optimized graph.
    const std::string text = random.model.to_text(-1);
    const Model reparsed = Model::from_text(context, text).value();
    EXPECT_EQ(reparsed.to_text(-1), text);
    expect_same(ii_test::run_model(reparsed, random.inputs, plain).value(), expected, "text round trip");
}

TEST_P(PassInvarianceTest, BackendsAgreeWithReference) {
    const Context context = ii_test::test_context();
    const ii_test::RandomModel random = ii_test::random_model(context, GetParam());
    for (const std::string& backend : context.backend_names()) {
        const std::string mismatch =
            ii_test::compare_backends(random.model, random.inputs, backend, "ii-ref", {1e-4, 1e-5}).value();
        EXPECT_TRUE(mismatch.empty()) << backend << ": " << mismatch;
    }
}

TEST_P(PassInvarianceTest, FoldingShrinksTheGraph) {
    const Context context = ii_test::test_context();
    const ii_test::RandomModel random = ii_test::random_model(context, GetParam());
    CompileOptions plain;
    plain.fold_constants = false;
    const std::size_t before = random.model.compile(plain).value().impl()->steps.size();
    const std::size_t after = random.model.compile().value().impl()->steps.size();
    EXPECT_LT(after, before);  // the generator always adds a constant subgraph
}

INSTANTIATE_TEST_SUITE_P(Seeds, PassInvarianceTest, ::testing::Range(std::uint32_t{1}, std::uint32_t{41}));

}  // namespace
}  // namespace ii
