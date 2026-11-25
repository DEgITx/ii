// SPDX-License-Identifier: MIT
/// @file
/// Static memory planning: places buffers with known lifetimes into one arena so that buffers whose lifetimes do
/// not overlap share memory ("greedy by size", as in TensorFlow Lite).
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ii::runtime {

/// A buffer needed from step `first` to step `last` (inclusive).
struct MemoryRequest {
    std::size_t size = 0;
    int first = 0;
    int last = 0;
};

struct MemoryPlan {
    std::vector<std::size_t> offsets;  ///< per request
    std::size_t total = 0;             ///< arena size
};

/// Assigns offsets (multiples of `alignment`) so that requests with overlapping lifetimes never overlap in memory.
/// Larger buffers are placed first, each at the lowest offset that fits between already placed, simultaneously
/// live buffers.
MemoryPlan plan_memory(std::span<const MemoryRequest> requests, std::size_t alignment = 64);

}  // namespace ii::runtime
