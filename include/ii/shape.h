// SPDX-License-Identifier: MIT
/// @file
/// Concrete tensor shapes.
#pragma once

#include <ii/export.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>

namespace ii {

/// Maximum tensor rank supported by ii.
inline constexpr int kMaxRank = 8;

namespace detail {
/// Terminates the program: a shape with more than `kMaxRank` dimensions was constructed.
[[noreturn]] II_API void shape_rank_overflow(std::size_t rank) noexcept;
}  // namespace detail

/// A list of at most `kMaxRank` integers stored inline (no heap allocation). Used for the concrete shape of a tensor
/// (sizes per dimension, outermost first) and for strides (`Strides`).
///
/// A rank-0 shape `{}` describes a scalar with one element.
class Shape {
public:
    constexpr Shape() noexcept = default;

    constexpr Shape(std::initializer_list<std::int64_t> dims) noexcept { assign(dims.begin(), dims.size()); }

    constexpr explicit Shape(std::span<const std::int64_t> dims) noexcept { assign(dims.data(), dims.size()); }

    /// A shape of `rank` dimensions, all equal to `value`.
    static constexpr Shape filled(int rank, std::int64_t value) noexcept {
        Shape shape;
        shape.resize(rank, value);
        return shape;
    }

    constexpr int rank() const noexcept { return rank_; }
    constexpr bool empty() const noexcept { return rank_ == 0; }
    constexpr int size() const noexcept { return rank_; }

    constexpr std::int64_t operator[](int index) const noexcept { return dims_[index]; }
    constexpr std::int64_t& operator[](int index) noexcept { return dims_[index]; }

    /// Element `index`, counted from the back when negative (`at_signed(-1)` is the last dimension).
    constexpr std::int64_t at_signed(int index) const noexcept { return dims_[index < 0 ? rank_ + index : index]; }

    constexpr const std::int64_t* data() const noexcept { return dims_; }
    constexpr std::int64_t* data() noexcept { return dims_; }
    constexpr const std::int64_t* begin() const noexcept { return dims_; }
    constexpr const std::int64_t* end() const noexcept { return dims_ + rank_; }
    constexpr std::int64_t* begin() noexcept { return dims_; }
    constexpr std::int64_t* end() noexcept { return dims_ + rank_; }

    constexpr std::span<const std::int64_t> dims() const noexcept { return {dims_, static_cast<std::size_t>(rank_)}; }

    /// Product of all dimensions (1 for a scalar).
    constexpr std::int64_t numel() const noexcept {
        std::int64_t result = 1;
        for (int i = 0; i < rank_; ++i) {
            result *= dims_[i];
        }
        return result;
    }

    constexpr void push_back(std::int64_t value) noexcept {
        if (rank_ >= kMaxRank) {
            detail::shape_rank_overflow(static_cast<std::size_t>(rank_) + 1);
        }
        dims_[rank_++] = value;
    }

    constexpr void resize(int rank, std::int64_t value = 0) noexcept {
        if (rank < 0 || rank > kMaxRank) {
            detail::shape_rank_overflow(static_cast<std::size_t>(rank));
        }
        for (int i = rank_; i < rank; ++i) {
            dims_[i] = value;
        }
        rank_ = rank;
    }

    constexpr void clear() noexcept { rank_ = 0; }

    friend constexpr bool operator==(const Shape& a, const Shape& b) noexcept {
        return a.rank_ == b.rank_ && std::equal(a.begin(), a.end(), b.begin());
    }

    /// "[2, 3, 4]".
    II_API std::string to_string() const;

private:
    constexpr void assign(const std::int64_t* values, std::size_t count) noexcept {
        if (count > static_cast<std::size_t>(kMaxRank)) {
            detail::shape_rank_overflow(count);
        }
        rank_ = static_cast<int>(count);
        for (int i = 0; i < rank_; ++i) {
            dims_[i] = values[i];
        }
    }

    int rank_ = 0;
    std::int64_t dims_[kMaxRank] = {};
};

/// Element strides of a tensor (distance in elements between neighbours along each dimension).
using Strides = Shape;

/// Row-major (C-order) strides of a contiguous tensor with `shape`.
constexpr Strides contiguous_strides(const Shape& shape) noexcept {
    Strides strides = Shape::filled(shape.rank(), 1);
    std::int64_t stride = 1;
    for (int i = shape.rank() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

}  // namespace ii
