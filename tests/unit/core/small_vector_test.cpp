// SPDX-License-Identifier: MIT
#include "core/small_vector.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ii {
namespace {

/// Counts live instances to detect leaks and double destruction.
struct Tracked {
    static inline int live = 0;
    int value = 0;

    Tracked() { ++live; }
    explicit Tracked(int v) : value(v) { ++live; }
    Tracked(const Tracked& other) : value(other.value) { ++live; }
    Tracked(Tracked&& other) noexcept : value(std::exchange(other.value, -1)) { ++live; }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&& other) noexcept {
        value = std::exchange(other.value, -1);
        return *this;
    }
    ~Tracked() { --live; }
    friend bool operator==(const Tracked& a, const Tracked& b) { return a.value == b.value; }
};

class SmallVectorTest : public ::testing::Test {
protected:
    void SetUp() override { Tracked::live = 0; }
    void TearDown() override { EXPECT_EQ(Tracked::live, 0) << "leaked or double-destroyed elements"; }
};

TEST_F(SmallVectorTest, StartsEmptyAndInline) {
    const SmallVector<int, 4> v;
    EXPECT_TRUE(v.empty());
    EXPECT_EQ(v.size(), 0u);
    EXPECT_EQ(v.capacity(), 4u);
    EXPECT_TRUE(v.is_inline());
}

TEST_F(SmallVectorTest, StaysInlineUpToCapacityThenGrows) {
    SmallVector<int, 4> v;
    for (int i = 0; i < 4; ++i) {
        v.push_back(i);
    }
    EXPECT_TRUE(v.is_inline());
    v.push_back(4);
    EXPECT_FALSE(v.is_inline());
    ASSERT_EQ(v.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(v[static_cast<std::size_t>(i)], i);
    }
}

TEST_F(SmallVectorTest, NonTrivialElementsAcrossGrowth) {
    SmallVector<Tracked, 2> v;
    for (int i = 0; i < 100; ++i) {
        v.emplace_back(i);
    }
    EXPECT_EQ(Tracked::live, 100);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(v[static_cast<std::size_t>(i)].value, i);
    }
    v.pop_back();
    EXPECT_EQ(Tracked::live, 99);
    v.clear();
    EXPECT_EQ(Tracked::live, 0);
}

TEST_F(SmallVectorTest, PushBackOwnElementWhileGrowing) {
    SmallVector<std::string, 2> v{"first", "second"};
    v.push_back(v[0]);  // the argument refers into the buffer that is reallocated
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[2], "first");
}

TEST_F(SmallVectorTest, ResizeWithOwnElementWhileGrowing) {
    SmallVector<std::string, 2> v{"x"};
    v.resize(10, v[0]);
    ASSERT_EQ(v.size(), 10u);
    for (const auto& s : v) {
        EXPECT_EQ(s, "x");
    }
}

TEST_F(SmallVectorTest, CopyInlineAndHeap) {
    SmallVector<Tracked, 3> small{Tracked(1), Tracked(2)};
    SmallVector<Tracked, 3> large;
    for (int i = 0; i < 10; ++i) {
        large.emplace_back(i);
    }
    const SmallVector<Tracked, 3> small_copy = small;
    const SmallVector<Tracked, 3> large_copy = large;
    EXPECT_EQ(small_copy, small);
    EXPECT_EQ(large_copy, large);
    EXPECT_TRUE(small_copy.is_inline());
}

TEST_F(SmallVectorTest, MoveInlineAndHeap) {
    SmallVector<Tracked, 3> small{Tracked(1), Tracked(2)};
    SmallVector<Tracked, 3> moved_small = std::move(small);
    ASSERT_EQ(moved_small.size(), 2u);
    EXPECT_EQ(moved_small[1].value, 2);
    EXPECT_TRUE(small.empty());  // NOLINT(bugprone-use-after-move): moved-from state is specified

    SmallVector<Tracked, 3> large;
    for (int i = 0; i < 10; ++i) {
        large.emplace_back(i);
    }
    const Tracked* heap = large.data();
    SmallVector<Tracked, 3> moved_large = std::move(large);
    EXPECT_EQ(moved_large.data(), heap);  // the heap buffer is stolen, not copied
    EXPECT_TRUE(large.empty());           // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(large.is_inline());       // NOLINT(bugprone-use-after-move)
}

TEST_F(SmallVectorTest, AssignmentReplacesContents) {
    SmallVector<Tracked, 2> a;
    for (int i = 0; i < 5; ++i) {
        a.emplace_back(i);
    }
    SmallVector<Tracked, 2> b{Tracked(9)};
    b = a;
    EXPECT_EQ(b, a);
    b = std::move(a);
    EXPECT_EQ(b.size(), 5u);
    b = SmallVector<Tracked, 2>{Tracked(7)};
    ASSERT_EQ(b.size(), 1u);
    EXPECT_EQ(b[0].value, 7);
    auto& self = b;
    b = self;  // self-assignment
    EXPECT_EQ(b.size(), 1u);
}

TEST_F(SmallVectorTest, ResizeValueInitializes) {
    SmallVector<int, 4> v;
    v.resize(6);
    ASSERT_EQ(v.size(), 6u);
    for (const int x : v) {
        EXPECT_EQ(x, 0);
    }
    v.resize(2);
    EXPECT_EQ(v.size(), 2u);
}

TEST_F(SmallVectorTest, IteratorConstruction) {
    const std::vector<int> source{1, 2, 3, 4, 5, 6};
    const SmallVector<int, 4> v(source.begin(), source.end());
    EXPECT_EQ(std::vector<int>(v.begin(), v.end()), source);
}

TEST_F(SmallVectorTest, MoveOnlyElements) {
    SmallVector<std::unique_ptr<int>, 2> v;
    for (int i = 0; i < 5; ++i) {
        v.push_back(std::make_unique<int>(i));
    }
    SmallVector<std::unique_ptr<int>, 2> moved = std::move(v);
    EXPECT_EQ(*moved[4], 4);
}

}  // namespace
}  // namespace ii
