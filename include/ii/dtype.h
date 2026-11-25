// SPDX-License-Identifier: MIT
/// @file
/// Element data types of tensors.
///
/// Besides the usual floating-point, integer and boolean types, ii has *block-quantized* types whose memory layout is
/// byte-compatible with GGUF (llama.cpp), so quantized weights can be used directly from a memory-mapped file. A
/// block type packs `block_size` consecutive elements of the innermost dimension into `block_bytes` bytes.
#pragma once

#include <ii/export.h>
#include <ii/half.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>

namespace ii {

/// The element type of a tensor.
enum class DType : std::uint8_t {
    kInvalid = 0,  ///< No type; the dtype of an empty `Tensor`.

    // Floating point
    kF32,   ///< IEEE 754 binary32.
    kF16,   ///< IEEE 754 binary16 (`ii::f16`).
    kBF16,  ///< bfloat16 (`ii::bf16`).
    kF64,   ///< IEEE 754 binary64. Supported by the reference backend and tests; rarely by fast backends.

    // Integer
    kI8,
    kU8,
    kI16,
    kU16,
    kI32,
    kU32,
    kI64,
    kU64,
    kBool,  ///< One byte per element, 0 or 1.

    // Block-quantized, GGUF-compatible layouts. 32 elements per block unless noted.
    kQ4_0,  ///< 4-bit, f16 scale.
    kQ4_1,  ///< 4-bit, f16 scale and minimum.
    kQ5_0,  ///< 5-bit, f16 scale.
    kQ5_1,  ///< 5-bit, f16 scale and minimum.
    kQ8_0,  ///< 8-bit, f16 scale.
    kQ8_1,  ///< 8-bit, f16 scale and f16 sum.
    kQ2_K,  ///< "K-quant" super-blocks of 256 elements.
    kQ3_K,
    kQ4_K,
    kQ5_K,
    kQ6_K,
    kQ8_K,

    kCount  ///< Number of dtypes (not a dtype).
};

/// Broad classification of a dtype.
enum class DTypeKind : std::uint8_t {
    kInvalid,
    kFloat,
    kSignedInt,
    kUnsignedInt,
    kBool,
    kBlockQuantized,
};

/// Static properties of a dtype.
struct DTypeInfo {
    std::string_view name;                 ///< Canonical lower-case name, e.g. "f32", "q4_k".
    DTypeKind kind = DTypeKind::kInvalid;  ///< Classification.
    std::uint16_t block_size = 1;          ///< Elements per block; 1 for non-block types.
    std::uint16_t block_bytes = 0;         ///< Bytes per block; the element size for non-block types.
};

namespace detail {
inline constexpr std::array<DTypeInfo, static_cast<std::size_t>(DType::kCount)> kDTypeInfos{{
    {"invalid", DTypeKind::kInvalid, 1, 0},
    {"f32", DTypeKind::kFloat, 1, 4},
    {"f16", DTypeKind::kFloat, 1, 2},
    {"bf16", DTypeKind::kFloat, 1, 2},
    {"f64", DTypeKind::kFloat, 1, 8},
    {"i8", DTypeKind::kSignedInt, 1, 1},
    {"u8", DTypeKind::kUnsignedInt, 1, 1},
    {"i16", DTypeKind::kSignedInt, 1, 2},
    {"u16", DTypeKind::kUnsignedInt, 1, 2},
    {"i32", DTypeKind::kSignedInt, 1, 4},
    {"u32", DTypeKind::kUnsignedInt, 1, 4},
    {"i64", DTypeKind::kSignedInt, 1, 8},
    {"u64", DTypeKind::kUnsignedInt, 1, 8},
    {"bool", DTypeKind::kBool, 1, 1},
    {"q4_0", DTypeKind::kBlockQuantized, 32, 18},
    {"q4_1", DTypeKind::kBlockQuantized, 32, 20},
    {"q5_0", DTypeKind::kBlockQuantized, 32, 22},
    {"q5_1", DTypeKind::kBlockQuantized, 32, 24},
    {"q8_0", DTypeKind::kBlockQuantized, 32, 34},
    {"q8_1", DTypeKind::kBlockQuantized, 32, 36},
    {"q2_k", DTypeKind::kBlockQuantized, 256, 84},
    {"q3_k", DTypeKind::kBlockQuantized, 256, 110},
    {"q4_k", DTypeKind::kBlockQuantized, 256, 144},
    {"q5_k", DTypeKind::kBlockQuantized, 256, 176},
    {"q6_k", DTypeKind::kBlockQuantized, 256, 210},
    {"q8_k", DTypeKind::kBlockQuantized, 256, 292},
}};
}  // namespace detail

/// Properties of `dtype`. `dtype` must be a valid enumerator other than `kCount`.
constexpr const DTypeInfo& dtype_info(DType dtype) noexcept {
    return detail::kDTypeInfos[static_cast<std::size_t>(dtype)];
}

/// Canonical name of `dtype`, e.g. "f32".
constexpr std::string_view to_string(DType dtype) noexcept {
    return dtype_info(dtype).name;
}

/// Parses a canonical dtype name. Returns nullopt for unknown names and for "invalid".
constexpr std::optional<DType> parse_dtype(std::string_view name) noexcept {
    for (std::size_t i = 1; i < detail::kDTypeInfos.size(); ++i) {
        if (detail::kDTypeInfos[i].name == name) {
            return static_cast<DType>(i);
        }
    }
    return std::nullopt;
}

constexpr bool is_float(DType dtype) noexcept {
    return dtype_info(dtype).kind == DTypeKind::kFloat;
}
constexpr bool is_integer(DType dtype) noexcept {
    const DTypeKind kind = dtype_info(dtype).kind;
    return kind == DTypeKind::kSignedInt || kind == DTypeKind::kUnsignedInt;
}
constexpr bool is_signed_integer(DType dtype) noexcept {
    return dtype_info(dtype).kind == DTypeKind::kSignedInt;
}
constexpr bool is_block_quantized(DType dtype) noexcept {
    return dtype_info(dtype).kind == DTypeKind::kBlockQuantized;
}

/// Size in bytes of one element of a non-block dtype; 0 for block-quantized and invalid dtypes.
constexpr std::size_t element_size(DType dtype) noexcept {
    const DTypeInfo& info = dtype_info(dtype);
    return info.block_size == 1 ? info.block_bytes : 0;
}

/// Bytes occupied by `count` consecutive elements. For block types `count` is rounded up to whole blocks.
constexpr std::size_t storage_size(DType dtype, std::int64_t count) noexcept {
    const DTypeInfo& info = dtype_info(dtype);
    const auto blocks = (static_cast<std::uint64_t>(count) + info.block_size - 1) / info.block_size;
    return static_cast<std::size_t>(blocks * info.block_bytes);
}

/// Maps a C++ element type to its dtype: `dtype_of<float>() == DType::kF32`.
template <class T>
constexpr DType dtype_of() noexcept {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, float>) {
        return DType::kF32;
    } else if constexpr (std::is_same_v<U, f16>) {
        return DType::kF16;
    } else if constexpr (std::is_same_v<U, bf16>) {
        return DType::kBF16;
    } else if constexpr (std::is_same_v<U, double>) {
        return DType::kF64;
    } else if constexpr (std::is_same_v<U, bool>) {
        return DType::kBool;
    } else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == 1) {
        return DType::kI8;
    } else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == 1) {
        return DType::kU8;
    } else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == 2) {
        return DType::kI16;
    } else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == 2) {
        return DType::kU16;
    } else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == 4) {
        return DType::kI32;
    } else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == 4) {
        return DType::kU32;
    } else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && sizeof(U) == 8) {
        return DType::kI64;
    } else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && sizeof(U) == 8) {
        return DType::kU64;
    } else {
        return DType::kInvalid;
    }
}

/// True when `T` has a corresponding dtype.
template <class T>
inline constexpr bool kHasDType = dtype_of<T>() != DType::kInvalid;

}  // namespace ii
