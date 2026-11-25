// SPDX-License-Identifier: MIT
#include "core/platform/dynlib.h"
#include "core/platform/platform.h"
#include "support/test_util.h"

#include <gtest/gtest.h>

namespace ii::platform {
namespace {

TEST(DynLib, LoadsLibraryAndResolvesFunction) {
    Result<DynLib> library = DynLib::open(path_from_utf8(ii_test::test_dynlib_path()));
    ASSERT_TRUE(library.ok()) << library.status().to_string();
    EXPECT_TRUE(library->is_open());

    using AnswerFn = int (*)();
    const Result<AnswerFn> answer = library->function<AnswerFn>("ii_test_answer");
    ASSERT_TRUE(answer.ok()) << answer.status().to_string();
    EXPECT_EQ((*answer)(), 42);
}

TEST(DynLib, MissingSymbol) {
    Result<DynLib> library = DynLib::open(path_from_utf8(ii_test::test_dynlib_path()));
    ASSERT_TRUE(library.ok());
    const Result<void*> symbol = library->symbol("ii_no_such_symbol");
    ASSERT_FALSE(symbol.ok());
    EXPECT_EQ(symbol.status().code(), StatusCode::kNotFound);
}

TEST(DynLib, MissingLibrary) {
    const Result<DynLib> library = DynLib::open("ii-no-such-library-6f1c2.dll");
    ASSERT_FALSE(library.ok());
    EXPECT_EQ(library.status().code(), StatusCode::kNotFound);
}

TEST(DynLib, MoveTransfersOwnership) {
    Result<DynLib> library = DynLib::open(path_from_utf8(ii_test::test_dynlib_path()));
    ASSERT_TRUE(library.ok());
    DynLib moved = std::move(library).value();
    EXPECT_TRUE(moved.is_open());
    DynLib other;
    EXPECT_FALSE(other.is_open());
    EXPECT_FALSE(other.symbol("ii_test_answer").ok());
    other = std::move(moved);
    EXPECT_TRUE(other.is_open());
    EXPECT_FALSE(moved.is_open());  // NOLINT(bugprone-use-after-move)
}

}  // namespace
}  // namespace ii::platform
