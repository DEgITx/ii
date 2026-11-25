// SPDX-License-Identifier: MIT
/// @file
/// Counts heap allocations made through the global `operator new` on the current thread (the test binary replaces
/// the global allocation functions), and an `ii::Allocator` that counts its calls.
#pragma once

#include <ii/allocator.h>

#include <atomic>
#include <cstdint>

namespace ii_test {

/// Counts `operator new` calls on this thread while alive.
class AllocationCounter {
public:
    AllocationCounter();
    ~AllocationCounter();
    AllocationCounter(const AllocationCounter&) = delete;
    AllocationCounter& operator=(const AllocationCounter&) = delete;

    std::int64_t count() const noexcept;

private:
    std::int64_t start_;
    bool previous_;
};

/// An allocator forwarding to the default allocator and counting calls.
class CountingAllocator final : public ii::Allocator {
public:
    void* allocate(std::size_t size, std::size_t alignment) noexcept override {
        allocations.fetch_add(1);
        return ii::default_allocator().allocate(size, alignment);
    }
    void deallocate(void* ptr, std::size_t size, std::size_t alignment) noexcept override {
        if (ptr != nullptr) {
            deallocations.fetch_add(1);
        }
        ii::default_allocator().deallocate(ptr, size, alignment);
    }

    std::atomic<std::int64_t> allocations{0};
    std::atomic<std::int64_t> deallocations{0};
};

}  // namespace ii_test
