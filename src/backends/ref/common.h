// SPDX-License-Identifier: MIT
/// @file
/// Shared machinery of the reference backend.
///
/// ii-ref favours clarity and accuracy over speed: floating-point values are computed in double precision and
/// rounded once when stored; integers are computed in 64 bits and wrap around (modulo 2^bits) when stored.
#pragma once

#include <ii/half.h>

#include "backend/backend.h"
#include "core/assert.h"
#include "tensor/strided.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

namespace ii::ref {

using backend::Kernel;
using backend::KernelContext;
using backend::TensorView;

// ------------------------------------------------------------------------------------------------------------------
// Storage and compute types

/// Storage type of each plain dtype. Booleans are stored as one byte holding 0 or 1 (read as "non-zero").
template <DType kDType>
struct StorageOf;
template <>
struct StorageOf<DType::kF32> {
    using type = float;
};
template <>
struct StorageOf<DType::kF16> {
    using type = f16;
};
template <>
struct StorageOf<DType::kBF16> {
    using type = bf16;
};
template <>
struct StorageOf<DType::kF64> {
    using type = double;
};
template <>
struct StorageOf<DType::kI8> {
    using type = std::int8_t;
};
template <>
struct StorageOf<DType::kU8> {
    using type = std::uint8_t;
};
template <>
struct StorageOf<DType::kI16> {
    using type = std::int16_t;
};
template <>
struct StorageOf<DType::kU16> {
    using type = std::uint16_t;
};
template <>
struct StorageOf<DType::kI32> {
    using type = std::int32_t;
};
template <>
struct StorageOf<DType::kU32> {
    using type = std::uint32_t;
};
template <>
struct StorageOf<DType::kI64> {
    using type = std::int64_t;
};
template <>
struct StorageOf<DType::kU64> {
    using type = std::uint64_t;
};

/// Marker storage type for kBool.
struct Bool8 {
    std::uint8_t value;
};
template <>
struct StorageOf<DType::kBool> {
    using type = Bool8;
};

template <class S>
inline constexpr bool kIsFloatStorage =
    std::is_same_v<S, float> || std::is_same_v<S, double> || std::is_same_v<S, f16> || std::is_same_v<S, bf16>;
template <class S>
inline constexpr bool kIsBoolStorage = std::is_same_v<S, Bool8>;
template <class S>
inline constexpr bool kIsSignedStorage = std::is_integral_v<S> && std::is_signed_v<S>;
template <class S>
inline constexpr bool kIsUnsignedStorage = std::is_integral_v<S> && std::is_unsigned_v<S>;

/// Compute type: double for floats, int64 for signed, uint64 for unsigned integers, bool for booleans.
template <class S>
using ComputeOf =
    std::conditional_t<kIsFloatStorage<S>, double,
                       std::conditional_t<kIsSignedStorage<S>, std::int64_t,
                                          std::conditional_t<kIsUnsignedStorage<S>, std::uint64_t, bool>>>;

/// Type tag passed to dispatch callbacks.
template <class S>
struct Tag {
    using type = S;
};

// ------------------------------------------------------------------------------------------------------------------
// Conversions

/// Rounds a double to float with "round to odd", so that a second rounding to a narrower format (f16, bf16) gives
/// the correctly rounded result of the original double.
inline float double_to_float_round_odd(double value) noexcept {
    if (std::isnan(value)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float f = static_cast<float>(value);
    if (static_cast<double>(f) == value) {
        return f;
    }
    if (std::fabs(static_cast<double>(f)) > std::fabs(value)) {
        f = std::nextafter(f, 0.0f);  // truncate toward zero
    }
    return std::bit_cast<float>(std::bit_cast<std::uint32_t>(f) | 1u);  // inexact: force the lowest bit
}

template <class S>
ComputeOf<S> load(const S& value) noexcept {
    if constexpr (std::is_same_v<S, f16> || std::is_same_v<S, bf16>) {
        return static_cast<double>(value.to_float());
    } else if constexpr (kIsBoolStorage<S>) {
        return value.value != 0;
    } else {
        return static_cast<ComputeOf<S>>(value);
    }
}

/// Converts any compute value to storage type `S` with `cast` semantics: floats round to nearest even; floats to
/// integers truncate toward zero and saturate (NaN -> 0); integers wrap around; anything to bool is "!= 0".
template <class S, class C>
S store(C value) noexcept {
    if constexpr (kIsBoolStorage<S>) {
        if constexpr (std::is_floating_point_v<C>) {
            return Bool8{static_cast<std::uint8_t>(value != 0.0 ? 1 : 0)};  // NaN is true
        } else {
            return Bool8{static_cast<std::uint8_t>(value != 0 ? 1 : 0)};
        }
    } else if constexpr (std::is_same_v<S, f16> || std::is_same_v<S, bf16>) {
        return S(double_to_float_round_odd(static_cast<double>(value)));
    } else if constexpr (std::is_floating_point_v<S>) {
        return static_cast<S>(value);
    } else if constexpr (std::is_floating_point_v<C>) {
        // Float to integer: truncate, saturate, NaN -> 0.
        if (std::isnan(value)) {
            return S{0};
        }
        const double truncated = std::trunc(static_cast<double>(value));
        if (truncated <= static_cast<double>(std::numeric_limits<S>::lowest())) {
            return std::numeric_limits<S>::lowest();
        }
        if (truncated >= static_cast<double>(std::numeric_limits<S>::max())) {
            return std::numeric_limits<S>::max();
        }
        return static_cast<S>(truncated);
    } else {
        // Integer (or bool) to integer: modular conversion (well defined since C++20).
        return static_cast<S>(value);
    }
}

/// Reads element `index` of a view as `S`.
template <class S>
const S& at(const TensorView& view, std::int64_t index) noexcept {
    return static_cast<const S*>(view.data)[index];
}

/// Calls `fn(Tag<S>{})` with the storage type of `dtype`; returns `fn`'s Status. Unsupported dtypes fail.
template <class Fn>
Status dispatch(DType dtype, Fn&& fn) {
    switch (dtype) {
    case DType::kF32:
        return fn(Tag<float>{});
    case DType::kF16:
        return fn(Tag<f16>{});
    case DType::kBF16:
        return fn(Tag<bf16>{});
    case DType::kF64:
        return fn(Tag<double>{});
    case DType::kI8:
        return fn(Tag<std::int8_t>{});
    case DType::kU8:
        return fn(Tag<std::uint8_t>{});
    case DType::kI16:
        return fn(Tag<std::int16_t>{});
    case DType::kU16:
        return fn(Tag<std::uint16_t>{});
    case DType::kI32:
        return fn(Tag<std::int32_t>{});
    case DType::kU32:
        return fn(Tag<std::uint32_t>{});
    case DType::kI64:
        return fn(Tag<std::int64_t>{});
    case DType::kU64:
        return fn(Tag<std::uint64_t>{});
    case DType::kBool:
        return fn(Tag<Bool8>{});
    default:
        return errors::unsupported("ii-ref: dtype {} is not supported here", to_string(dtype));
    }
}

/// Copies `count` elements of `size` bytes each.
inline void copy_elements(void* dst, const void* src, std::int64_t count, std::size_t size) noexcept {
    if (count > 0) {
        std::memmove(dst, src, static_cast<std::size_t>(count) * size);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Wrapping integer arithmetic (avoids signed overflow UB)

inline std::int64_t wrap_add(std::int64_t a, std::int64_t b) noexcept {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
}
inline std::int64_t wrap_sub(std::int64_t a, std::int64_t b) noexcept {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b));
}
inline std::int64_t wrap_mul(std::int64_t a, std::int64_t b) noexcept {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b));
}
inline std::uint64_t wrap_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a + b;
}
inline std::uint64_t wrap_sub(std::uint64_t a, std::uint64_t b) noexcept {
    return a - b;
}
inline std::uint64_t wrap_mul(std::uint64_t a, std::uint64_t b) noexcept {
    return a * b;
}

// ------------------------------------------------------------------------------------------------------------------
// Shapes

/// Strides (in elements) to read a dense tensor of shape `in` broadcast to shape `out` (0 for broadcast dimensions).
inline Strides broadcast_strides(const Shape& in, const Shape& out) noexcept {
    Strides strides = Shape::filled(out.rank(), 0);
    const Strides dense = contiguous_strides(in);
    const int offset = out.rank() - in.rank();
    for (int d = 0; d < in.rank(); ++d) {
        strides[offset + d] = (in[d] == 1 && out[offset + d] != 1) ? 0 : dense[d];
    }
    return strides;
}

/// Normalizes an axis attribute (already validated by shape inference).
inline int axis_of(std::int64_t axis, int rank) noexcept {
    return static_cast<int>(axis < 0 ? axis + rank : axis);
}

/// Product of `shape[begin, end)`.
inline std::int64_t product(const Shape& shape, int begin, int end) noexcept {
    std::int64_t result = 1;
    for (int d = begin; d < end; ++d) {
        result *= shape[d];
    }
    return result;
}

// ------------------------------------------------------------------------------------------------------------------
// Kernel helpers

/// A kernel implemented by a callable `Status(const KernelContext&)` capturing the node's parsed attributes.
template <class Fn>
class FunctionKernel final : public Kernel {
public:
    explicit FunctionKernel(Fn fn) : fn_(std::move(fn)) {}
    Status run(const KernelContext& context) const override { return fn_(context); }

private:
    Fn fn_;
};

template <class Fn>
std::unique_ptr<Kernel> make_kernel(Fn fn) {
    return std::make_unique<FunctionKernel<Fn>>(std::move(fn));
}

/// Registration functions for the kernel groups.
void register_elementwise_kernels(backend::KernelBackend& backend);
void register_reduction_kernels(backend::KernelBackend& backend);
void register_linalg_kernels(backend::KernelBackend& backend);
void register_movement_kernels(backend::KernelBackend& backend);
void register_creation_kernels(backend::KernelBackend& backend);
void register_composite_kernels(backend::KernelBackend& backend);

}  // namespace ii::ref
