// SPDX-License-Identifier: MIT
// Replacement global allocation functions that count allocations per thread while an AllocationCounter is active.
#include "support/alloc_counter.h"

#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace {

thread_local bool g_counting = false;
thread_local std::int64_t g_count = 0;

void* allocate_aligned(std::size_t size, std::size_t alignment) {
    if (size == 0) {
        size = 1;
    }
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment < sizeof(void*) ? sizeof(void*) : alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void free_aligned(void* ptr) noexcept {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void* counted_new(std::size_t size) {
    if (g_counting) {
        ++g_count;
    }
    void* ptr = std::malloc(size == 0 ? 1 : size);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void* counted_new_aligned(std::size_t size, std::align_val_t alignment) {
    if (g_counting) {
        ++g_count;
    }
    void* ptr = allocate_aligned(size, static_cast<std::size_t>(alignment));
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

}  // namespace

namespace ii_test {

AllocationCounter::AllocationCounter() : start_(g_count), previous_(g_counting) {
    g_counting = true;
}

AllocationCounter::~AllocationCounter() {
    g_counting = previous_;
}

std::int64_t AllocationCounter::count() const noexcept {
    return g_count - start_;
}

}  // namespace ii_test

// NOLINTBEGIN(misc-new-delete-overloads, cert-dcl54-cpp)
void* operator new(std::size_t size) {
    return counted_new(size);
}
void* operator new[](std::size_t size) {
    return counted_new(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return counted_new(size);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return counted_new(size);
    } catch (...) {
        return nullptr;
    }
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    return counted_new_aligned(size, alignment);
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return counted_new_aligned(size, alignment);
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    try {
        return counted_new_aligned(size, alignment);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    try {
        return counted_new_aligned(size, alignment);
    } catch (...) {
        return nullptr;
    }
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}
void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}
void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}
void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}
void operator delete(void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
}
void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
}
void operator delete(void* ptr, std::align_val_t) noexcept {
    free_aligned(ptr);
}
void operator delete[](void* ptr, std::align_val_t) noexcept {
    free_aligned(ptr);
}
void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
    free_aligned(ptr);
}
void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
    free_aligned(ptr);
}
void operator delete(void* ptr, std::align_val_t, const std::nothrow_t&) noexcept {
    free_aligned(ptr);
}
void operator delete[](void* ptr, std::align_val_t, const std::nothrow_t&) noexcept {
    free_aligned(ptr);
}
// NOLINTEND(misc-new-delete-overloads, cert-dcl54-cpp)
