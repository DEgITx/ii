// SPDX-License-Identifier: MIT
#include <ii/allocator.h>

#include "core/platform/platform.h"

#include <algorithm>
#include <bit>
#include <cstddef>

namespace ii {
namespace {

class DefaultAllocator final : public Allocator {
public:
    void* allocate(std::size_t size, std::size_t alignment) noexcept override {
        if (!std::has_single_bit(alignment)) {
            return nullptr;
        }
        alignment = std::max(alignment, alignof(std::max_align_t));
        size = std::max<std::size_t>(size, 1);
        return platform::aligned_allocate(size, alignment);
    }

    void deallocate(void* ptr, std::size_t /*size*/, std::size_t /*alignment*/) noexcept override {
        platform::aligned_free(ptr);
    }
};

}  // namespace

Allocator& default_allocator() noexcept {
    static DefaultAllocator allocator;
    return allocator;
}

}  // namespace ii
