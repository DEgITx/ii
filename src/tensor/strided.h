// SPDX-License-Identifier: MIT
/// @file
/// Helpers for walking strided N-dimensional index spaces.
#pragma once

#include <ii/shape.h>

#include <cstdint>

namespace ii {

/// Iterates over all index tuples of `shape` in row-major order, maintaining the element offsets of up to `kArrays`
/// strided arrays. The innermost dimension is left to the caller: `fn(offsets)` is called once per *row*, i.e. for
/// every index of the outer `rank - 1` dimensions, with the offsets of the first element of that row.
///
/// For a rank-0 shape `fn` is called once with zero offsets.
template <int kArrays, class Fn>
void for_each_row(const Shape& shape, const Strides* const (&strides)[kArrays], Fn&& fn) {
    const int rank = shape.rank();
    std::int64_t offsets[kArrays] = {};
    if (rank <= 1) {
        fn(static_cast<const std::int64_t*>(offsets));
        return;
    }
    for (int d = 0; d < rank; ++d) {
        if (shape[d] == 0) {
            return;
        }
    }
    std::int64_t index[kMaxRank] = {};
    const int outer = rank - 1;
    for (;;) {
        fn(static_cast<const std::int64_t*>(offsets));
        int d = outer - 1;
        for (; d >= 0; --d) {
            ++index[d];
            for (int a = 0; a < kArrays; ++a) {
                offsets[a] += (*strides[a])[d];
            }
            if (index[d] < shape[d]) {
                break;
            }
            for (int a = 0; a < kArrays; ++a) {
                offsets[a] -= (*strides[a])[d] * shape[d];
            }
            index[d] = 0;
        }
        if (d < 0) {
            return;
        }
    }
}

/// Converts a linear row-major index into per-dimension indices of `shape`.
inline void unravel_index(std::int64_t linear, const Shape& shape, std::int64_t* index) noexcept {
    for (int d = shape.rank() - 1; d >= 0; --d) {
        const std::int64_t size = shape[d];
        index[d] = size == 0 ? 0 : linear % size;
        linear = size == 0 ? 0 : linear / size;
    }
}

/// Dot product of an index tuple with strides.
inline std::int64_t offset_of(const std::int64_t* index, const Strides& strides) noexcept {
    std::int64_t offset = 0;
    for (int d = 0; d < strides.rank(); ++d) {
        offset += index[d] * strides[d];
    }
    return offset;
}

}  // namespace ii
