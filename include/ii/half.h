// SPDX-License-Identifier: MIT
/// @file
/// 16-bit floating-point storage types: IEEE 754 binary16 (`f16`) and bfloat16 (`bf16`).
///
/// These are storage types: convert to `float` for arithmetic. Conversions from `float` round to nearest, ties to
/// even; NaNs stay NaNs (and become quiet NaNs); overflow produces infinity. The conversions are portable scalar code
/// — vectorized kernels use hardware conversion instructions where available.
#pragma once

#include <bit>
#include <cstdint>

namespace ii {

namespace detail {

constexpr std::uint16_t f32_to_f16_bits(float value) noexcept {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
    const std::uint32_t abs = x & 0x7FFFFFFFu;

    if (abs >= 0x7F800000u) {  // Inf or NaN
        if (abs > 0x7F800000u) {
            // Quiet NaN; keep the high payload bits.
            return static_cast<std::uint16_t>(sign | 0x7E00u | ((abs >> 13) & 0x03FFu));
        }
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (abs >= 0x477FF000u) {  // >= 65520 (halfway above the largest finite f16) rounds to infinity
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (abs < 0x38800000u) {       // below 2^-14: subnormal f16 or zero
        if (abs <= 0x33000000u) {  // <= 2^-25: rounds to zero (2^-25 is a tie, zero is even)
            return sign;
        }
        const std::uint32_t exponent = abs >> 23;
        const std::uint32_t mantissa = (abs & 0x007FFFFFu) | 0x00800000u;
        const std::uint32_t shift = 126u - exponent;  // 14..24
        std::uint32_t quotient = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t half = 1u << (shift - 1u);
        if (remainder > half || (remainder == half && (quotient & 1u) != 0)) {
            ++quotient;  // may carry into the smallest normal number, which is the correct encoding
        }
        return static_cast<std::uint16_t>(sign | quotient);
    }
    // Normal: rebias the exponent from 127 to 15 and round the 13 dropped mantissa bits.
    const std::uint32_t rebiased = abs - 0x38000000u;
    std::uint32_t result = rebiased >> 13;
    const std::uint32_t remainder = rebiased & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0)) {
        ++result;  // a mantissa carry correctly increments the exponent
    }
    return static_cast<std::uint16_t>(sign | result);
}

constexpr float f16_bits_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1Fu;
    std::uint32_t mantissa = bits & 0x03FFu;

    if (exponent == 0x1Fu) {  // Inf or NaN
        return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
    }
    if (exponent == 0) {
        if (mantissa == 0) {
            return std::bit_cast<float>(sign);
        }
        // Subnormal: normalize.
        std::uint32_t shift = 0;
        while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        mantissa &= 0x03FFu;
        return std::bit_cast<float>(sign | ((113u - shift) << 23) | (mantissa << 13));
    }
    return std::bit_cast<float>(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

constexpr std::uint16_t f32_to_bf16_bits(float value) noexcept {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(value);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) {  // NaN: truncate and force a quiet NaN
        return static_cast<std::uint16_t>((x >> 16) | 0x0040u);
    }
    const std::uint32_t rounding_bias = 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<std::uint16_t>((x + rounding_bias) >> 16);
}

constexpr float bf16_bits_to_f32(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

}  // namespace detail

/// IEEE 754 binary16: 1 sign, 5 exponent, 10 mantissa bits.
struct f16 {
    std::uint16_t bits = 0;

    constexpr f16() noexcept = default;
    constexpr explicit f16(float value) noexcept : bits(detail::f32_to_f16_bits(value)) {}

    static constexpr f16 from_bits(std::uint16_t raw) noexcept {
        f16 result;
        result.bits = raw;
        return result;
    }

    constexpr float to_float() const noexcept { return detail::f16_bits_to_f32(bits); }
    constexpr explicit operator float() const noexcept { return to_float(); }
};

/// bfloat16: 1 sign, 8 exponent, 7 mantissa bits (the upper half of an IEEE binary32).
struct bf16 {
    std::uint16_t bits = 0;

    constexpr bf16() noexcept = default;
    constexpr explicit bf16(float value) noexcept : bits(detail::f32_to_bf16_bits(value)) {}

    static constexpr bf16 from_bits(std::uint16_t raw) noexcept {
        bf16 result;
        result.bits = raw;
        return result;
    }

    constexpr float to_float() const noexcept { return detail::bf16_bits_to_f32(bits); }
    constexpr explicit operator float() const noexcept { return to_float(); }
};

static_assert(sizeof(f16) == 2 && sizeof(bf16) == 2);

}  // namespace ii
