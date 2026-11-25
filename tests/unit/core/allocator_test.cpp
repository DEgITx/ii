// SPDX-License-Identifier: MIT
#include <ii/allocator.h>

#include "core/aligned_buffer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

namespace ii {
namespace {

bool is_aligned(const void* ptr, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

TEST(DefaultAllocator, HonorsAlignment) {
    Allocator& allocator = default_allocator();
    for (const std::size_t alignment : {1u, 8u, 16u, 64u, 256u, 4096u}) {
        for (const std::size_t size : {1u, 7u, 64u, 1000u, 1u << 20}) {
            void* ptr = allocator.allocate(size, alignment);
            ASSERT_NE(ptr, nullptr);
            EXPECT_TRUE(is_aligned(ptr, alignment)) << "alignment " << alignment;
            std::memset(ptr, 0xAB, size);  // the memory must be writable
            allocator.deallocate(ptr, size, alignment);
        }
    }
}

TEST(DefaultAllocator, ZeroSizeReturnsUniquePointer) {
    Allocator& allocator = default_allocator();
    void* a = allocator.allocate(0, kDefaultAlignment);
    void* b = allocator.allocate(0, kDefaultAlignment);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a, b);
    allocator.deallocate(a, 0, kDefaultAlignment);
    allocator.deallocate(b, 0, kDefaultAlignment);
}

TEST(DefaultAllocator, RejectsNonPowerOfTwoAlignment) {
    EXPECT_EQ(default_allocator().allocate(64, 48), nullptr);
}

TEST(DefaultAllocator, DeallocateNullIsNoOp) {
    default_allocator().deallocate(nullptr, 0, kDefaultAlignment);
}

class CountingAllocator final : public Allocator {
public:
    void* allocate(std::size_t size, std::size_t alignment) noexcept override {
        ++allocations;
        return default_allocator().allocate(size, alignment);
    }
    void deallocate(void* ptr, std::size_t size, std::size_t alignment) noexcept override {
        if (ptr != nullptr) {
            ++deallocations;
        }
        default_allocator().deallocate(ptr, size, alignment);
    }
    int allocations = 0;
    int deallocations = 0;
};

TEST(AlignedBuffer, UsesGivenAllocatorAndReleases) {
    CountingAllocator allocator;
    {
        auto buffer = AlignedBuffer::allocate(1000, 128, allocator);
        ASSERT_TRUE(buffer.ok());
        EXPECT_EQ(buffer->size(), 1000u);
        EXPECT_TRUE(is_aligned(buffer->data(), 128));
        AlignedBuffer moved = std::move(buffer).value();
        EXPECT_EQ(moved.size(), 1000u);
        EXPECT_EQ(allocator.allocations, 1);
        EXPECT_EQ(allocator.deallocations, 0);
    }
    EXPECT_EQ(allocator.deallocations, 1);
}

TEST(AlignedBuffer, ReportsOutOfMemory) {
    class FailingAllocator final : public Allocator {
    public:
        void* allocate(std::size_t, std::size_t) noexcept override { return nullptr; }
        void deallocate(void*, std::size_t, std::size_t) noexcept override {}
    } failing;
    const auto buffer = AlignedBuffer::allocate(16, 64, failing);
    ASSERT_FALSE(buffer.ok());
    EXPECT_EQ(buffer.status().code(), StatusCode::kOutOfMemory);
}

}  // namespace
}  // namespace ii
