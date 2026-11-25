// SPDX-License-Identifier: MIT
/// @file
/// Memory allocation interface. Applications can route all large ii allocations (weights, activation arenas,
/// scratch buffers) through their own allocator.
#pragma once

#include <ii/export.h>

#include <cstddef>

namespace ii {

/// Default alignment of ii buffers: one cache line, which also satisfies every SIMD instruction set.
inline constexpr std::size_t kDefaultAlignment = 64;

/// An aligned memory allocator.
///
/// Implementations must be thread-safe. `alignment` is always a power of two.
class Allocator {
public:
    virtual ~Allocator() = default;

    /// Allocates at least `size` bytes aligned to `alignment`. Returns nullptr on failure.
    /// A zero `size` returns a valid, unique pointer that must be passed to `deallocate`.
    virtual void* allocate(std::size_t size, std::size_t alignment) noexcept = 0;

    /// Releases memory returned by `allocate` with the same `size` and `alignment`. `ptr` may be nullptr.
    virtual void deallocate(void* ptr, std::size_t size, std::size_t alignment) noexcept = 0;

protected:
    Allocator() = default;
    Allocator(const Allocator&) = default;
    Allocator& operator=(const Allocator&) = default;
};

/// The process-wide default allocator (the platform's aligned heap allocation). Stateless and thread-safe.
II_API Allocator& default_allocator() noexcept;

}  // namespace ii
