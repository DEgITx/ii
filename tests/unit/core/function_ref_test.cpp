// SPDX-License-Identifier: MIT
#include <ii/function_ref.h>

#include <gtest/gtest.h>

#include <string>

namespace ii {
namespace {

int call(FunctionRef<int(int)> fn, int value) {
    return fn(value);
}

int add_one(int value) {
    return value + 1;
}

TEST(FunctionRef, Lambda) {
    int captured = 10;
    EXPECT_EQ(call([&](int v) { return v + captured; }, 5), 15);
}

TEST(FunctionRef, FreeFunction) {
    EXPECT_EQ(call(add_one, 1), 2);
}

TEST(FunctionRef, MutableState) {
    int counter = 0;
    auto increment = [&counter](int v) { return counter += v; };
    const FunctionRef<int(int)> ref = increment;
    ref(2);
    ref(3);
    EXPECT_EQ(counter, 5);
}

TEST(FunctionRef, ConstCallable) {
    const auto fn = [](int v) { return v * 3; };
    EXPECT_EQ(call(fn, 4), 12);
}

TEST(FunctionRef, ForwardsReferences) {
    std::string text = "a";
    const FunctionRef<void(std::string&)> append = [](std::string& s) { s += "b"; };
    append(text);
    EXPECT_EQ(text, "ab");
}

}  // namespace
}  // namespace ii
