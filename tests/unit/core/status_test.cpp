// SPDX-License-Identifier: MIT
#include <ii/status.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

namespace ii {
namespace {

TEST(Status, DefaultIsOk) {
    const Status status;
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kOk);
    EXPECT_TRUE(status.message().empty());
    EXPECT_EQ(status.to_string(), "OK");
}

TEST(Status, ErrorCarriesCodeAndMessage) {
    const Status status(StatusCode::kNotFound, "no such tensor");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
    EXPECT_EQ(status.message(), "no such tensor");
    EXPECT_EQ(status.to_string(), "NotFound: no such tensor");
}

TEST(Status, OkCodeWithMessageIsOk) {
    const Status status(StatusCode::kOk, "ignored");
    EXPECT_TRUE(status.ok());
    EXPECT_TRUE(status.message().empty());
}

TEST(Status, CopyAndMove) {
    const Status original(StatusCode::kIoError, "disk");
    Status copy = original;
    EXPECT_EQ(copy.code(), StatusCode::kIoError);
    EXPECT_EQ(copy.message(), "disk");
    EXPECT_EQ(original.message(), "disk");

    Status moved = std::move(copy);
    EXPECT_EQ(moved.code(), StatusCode::kIoError);

    copy = moved;
    EXPECT_EQ(copy.message(), "disk");
    copy = Status();
    EXPECT_TRUE(copy.ok());
}

TEST(Status, AddContext) {
    Status status = errors::data_corrupted("bad header");
    status.add_context("model.gguf");
    EXPECT_EQ(status.message(), "model.gguf: bad header");

    Status ok;
    ok.add_context("ignored");
    EXPECT_TRUE(ok.ok());
}

TEST(Status, ErrorFactoriesFormat) {
    const Status status = errors::invalid_argument("expected {} dims, got {}", 4, 3);
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "expected 4 dims, got 3");

    EXPECT_EQ(errors::unsupported("x").code(), StatusCode::kUnsupported);
    EXPECT_EQ(errors::internal("x").code(), StatusCode::kInternal);
    EXPECT_EQ(make_status(StatusCode::kCancelled, "stop {}", 1).message(), "stop 1");
}

TEST(Status, CodeNames) {
    EXPECT_EQ(to_string(StatusCode::kOk), "OK");
    EXPECT_EQ(to_string(StatusCode::kOutOfMemory), "OutOfMemory");
    EXPECT_EQ(to_string(StatusCode::kDataCorrupted), "DataCorrupted");
}

TEST(Result, HoldsValue) {
    Result<int> result = 42;
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result.status().ok());
    EXPECT_EQ(result.value(), 42);
    EXPECT_EQ(*result, 42);
}

TEST(Result, HoldsError) {
    const Result<int> result = errors::not_found("missing");
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kNotFound);
    EXPECT_EQ(result.value_or(7), 7);
}

TEST(Result, FromOkStatusBecomesInternalError) {
    const Result<int> result = Status();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInternal);
}

TEST(Result, MoveOnlyValue) {
    Result<std::unique_ptr<int>> result = std::make_unique<int>(5);
    ASSERT_TRUE(result.ok());
    std::unique_ptr<int> value = std::move(result).value();
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, 5);
}

TEST(Result, InPlaceAndArrow) {
    const Result<std::string> result(std::in_place, 3, 'x');
    EXPECT_EQ(result->size(), 3u);
    EXPECT_EQ(*result, "xxx");
}

TEST(Result, ConvertingConstruction) {
    const Result<std::string> result = "literal";
    EXPECT_EQ(*result, "literal");
}

TEST(ResultDeathTest, ValueOnErrorTerminates) {
    const Result<int> result = errors::not_found("the answer");
    EXPECT_DEATH({ (void)result.value(); }, "the answer");
}

// --- Propagation macros ---

Status fail_if(bool fail) {
    if (fail) {
        return errors::io_error("failed");
    }
    return {};
}

Result<int> parse(int value) {
    if (value < 0) {
        return errors::out_of_range("negative: {}", value);
    }
    return value * 2;
}

Status use_try(bool fail) {
    II_TRY(fail_if(fail));
    II_TRY(parse(1));
    return {};
}

Result<int> use_assign(int value) {
    II_ASSIGN_OR_RETURN(const int doubled, parse(value));
    II_ASSIGN_OR_RETURN(int again, parse(doubled));
    return again + 1;
}

TEST(StatusMacros, TryPropagatesErrors) {
    EXPECT_TRUE(use_try(false).ok());
    const Status status = use_try(true);
    EXPECT_EQ(status.code(), StatusCode::kIoError);
}

TEST(StatusMacros, AssignOrReturn) {
    const Result<int> ok = use_assign(3);
    ASSERT_TRUE(ok.ok());
    EXPECT_EQ(*ok, 13);

    const Result<int> error = use_assign(-1);
    ASSERT_FALSE(error.ok());
    EXPECT_EQ(error.status().code(), StatusCode::kOutOfRange);
    EXPECT_EQ(error.status().message(), "negative: -1");
}

}  // namespace
}  // namespace ii
