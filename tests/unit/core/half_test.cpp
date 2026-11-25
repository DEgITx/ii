// SPDX-License-Identifier: MIT
// f16/bf16 conversions are verified against independent reference implementations that work in double precision:
// exhaustively for every 16-bit pattern and for all rounding midpoints, plus millions of random f32 inputs.
#include <ii/half.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace ii {
namespace {

// --- Compile-time checks -------------------------------------------------------------------------------------------

static_assert(f16(1.0f).bits == 0x3C00);
static_assert(f16(-2.0f).bits == 0xC000);
static_assert(f16(65504.0f).bits == 0x7BFF);
static_assert(f16::from_bits(0x3C00).to_float() == 1.0f);
static_assert(bf16(1.0f).bits == 0x3F80);
static_assert(bf16::from_bits(0x3F80).to_float() == 1.0f);

// --- Reference implementations ---------------------------------------------------------------------------------------

double f16_reference_value(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 0x1F;
    const int mantissa = bits & 0x3FF;
    const double sign = (bits & 0x8000) ? -1.0 : 1.0;
    if (exponent == 0x1F) {
        return mantissa ? std::numeric_limits<double>::quiet_NaN() : sign * std::numeric_limits<double>::infinity();
    }
    if (exponent == 0) {
        return sign * std::ldexp(mantissa, -24);
    }
    return sign * std::ldexp(1024 + mantissa, exponent - 25);
}

/// Round-to-nearest-even f32 -> f16 by searching the table of all finite f16 values.
class F16Reference {
public:
    F16Reference() {
        for (std::uint32_t bits = 0; bits <= 0x7BFF; ++bits) {
            positive_.push_back(f16_reference_value(static_cast<std::uint16_t>(bits)));
        }
    }

    std::uint16_t convert(float value) const {
        const auto sign = static_cast<std::uint16_t>((std::bit_cast<std::uint32_t>(value) >> 16) & 0x8000);
        const double magnitude = std::fabs(static_cast<double>(value));
        if (magnitude >= 65520.0) {  // includes infinity; 65520 is the tie between 65504 and 2^16
            return sign | 0x7C00;
        }
        // Largest finite f16 <= magnitude.
        const auto it = std::upper_bound(positive_.begin(), positive_.end(), magnitude);
        const auto lower = static_cast<std::uint16_t>((it - positive_.begin()) - 1);
        const double lower_value = positive_[lower];
        const double upper_value = lower == 0x7BFF ? 65536.0 : positive_[lower + 1u];
        const double down = magnitude - lower_value;
        const double up = upper_value - magnitude;
        std::uint16_t result = lower;
        if (up < down || (up == down && (lower & 1u) != 0)) {
            result = static_cast<std::uint16_t>(lower + 1u);
        }
        return sign | result;
    }

private:
    std::vector<double> positive_;
};

/// Round-to-nearest-even f32 -> bf16 by comparing the two neighbouring bf16 values in double precision.
std::uint16_t bf16_reference(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = bits & 0x80000000u;
    const std::uint32_t magnitude_bits = bits & 0x7FFFFFFFu;
    const std::uint32_t lower = magnitude_bits & 0xFFFF0000u;
    const std::uint32_t upper = lower + 0x10000u;
    const auto as_double = [](std::uint32_t b) {
        return b >= 0x7F800000u ? std::ldexp(1.0, 128) : static_cast<double>(std::bit_cast<float>(b));
    };
    const double magnitude = as_double(magnitude_bits);
    const double down = magnitude - as_double(lower);
    const double up = as_double(upper) - magnitude;
    std::uint32_t result = lower;
    if (up < down || (up == down && ((lower >> 16) & 1u) != 0)) {
        result = upper;
    }
    return static_cast<std::uint16_t>((sign | result) >> 16);
}

bool f16_is_nan(std::uint16_t bits) {
    return (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0;
}

bool bf16_is_nan(std::uint16_t bits) {
    return (bits & 0x7F80) == 0x7F80 && (bits & 0x007F) != 0;
}

std::vector<float> random_floats(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        values.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(rng())));
    }
    return values;
}

// --- f16 -------------------------------------------------------------------------------------------------------------

TEST(F16, ToFloatMatchesReferenceExhaustively) {
    for (std::uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        const auto h = static_cast<std::uint16_t>(bits);
        const float value = f16::from_bits(h).to_float();
        const double expected = f16_reference_value(h);
        if (std::isnan(expected)) {
            ASSERT_TRUE(std::isnan(value)) << std::hex << bits;
        } else {
            ASSERT_EQ(static_cast<double>(value), expected) << std::hex << bits;
            ASSERT_EQ(std::signbit(value), (bits & 0x8000) != 0) << std::hex << bits;
        }
    }
}

TEST(F16, RoundTripIsExactExhaustively) {
    for (std::uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        const auto h = static_cast<std::uint16_t>(bits);
        const std::uint16_t back = f16(f16::from_bits(h).to_float()).bits;
        if (f16_is_nan(h)) {
            // NaNs stay NaNs and become quiet; quiet NaNs keep their payload.
            ASSERT_TRUE(f16_is_nan(back)) << std::hex << bits;
            ASSERT_EQ(back, h | 0x0200) << std::hex << bits;
        } else {
            ASSERT_EQ(back, h) << std::hex << bits;
        }
    }
}

TEST(F16, RoundsMidpointsToNearestEven) {
    const F16Reference reference;
    for (std::uint32_t bits = 0; bits < 0x7BFF; ++bits) {
        const double low = f16_reference_value(static_cast<std::uint16_t>(bits));
        const double high = f16_reference_value(static_cast<std::uint16_t>(bits + 1));
        const auto midpoint = static_cast<float>((low + high) / 2);  // exact: f16 midpoints fit in f32
        for (const float value : {midpoint, std::nextafter(midpoint, 0.0f), std::nextafter(midpoint, 1e9f)}) {
            ASSERT_EQ(f16(value).bits, reference.convert(value)) << value;
            ASSERT_EQ(f16(-value).bits, reference.convert(-value)) << -value;
        }
    }
}

TEST(F16, MatchesReferenceOnRandomFloats) {
    const F16Reference reference;
    for (const float value : random_floats(2'000'000, 1234)) {
        const std::uint16_t actual = f16(value).bits;
        if (std::isnan(value)) {
            ASSERT_TRUE(f16_is_nan(actual));
            ASSERT_EQ(actual & 0x8000, (std::bit_cast<std::uint32_t>(value) >> 16) & 0x8000);
        } else {
            ASSERT_EQ(actual, reference.convert(value)) << std::hexfloat << value;
        }
    }
}

TEST(F16, EdgeCases) {
    const F16Reference reference;
    const float cases[] = {0.0f,
                           -0.0f,
                           65504.0f,
                           65519.99f,
                           65520.0f,
                           1e10f,
                           std::numeric_limits<float>::infinity(),
                           std::ldexp(1.0f, -24),
                           std::ldexp(1.0f, -25),
                           std::nextafter(std::ldexp(1.0f, -25), 1.0f),
                           std::ldexp(1.5f, -25),
                           std::ldexp(1.0f, -14),
                           std::numeric_limits<float>::denorm_min(),
                           std::numeric_limits<float>::min(),
                           std::numeric_limits<float>::max()};
    for (const float value : cases) {
        EXPECT_EQ(f16(value).bits, reference.convert(value)) << value;
        EXPECT_EQ(f16(-value).bits, reference.convert(-value)) << -value;
    }
    EXPECT_EQ(f16(std::ldexp(1.0f, -25)).bits, 0x0000);  // tie rounds to even (zero)
    EXPECT_EQ(f16(std::ldexp(1.5f, -25)).bits, 0x0001);
    EXPECT_EQ(f16(65520.0f).bits, 0x7C00);
    EXPECT_EQ(f16(-0.0f).bits, 0x8000);
}

// --- bf16 ------------------------------------------------------------------------------------------------------------

TEST(Bf16, RoundTripIsExactExhaustively) {
    for (std::uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        const auto h = static_cast<std::uint16_t>(bits);
        const float value = bf16::from_bits(h).to_float();
        ASSERT_EQ(std::bit_cast<std::uint32_t>(value), bits << 16);
        const std::uint16_t back = bf16(value).bits;
        if (bf16_is_nan(h)) {
            ASSERT_TRUE(bf16_is_nan(back)) << std::hex << bits;
            ASSERT_EQ(back, h | 0x0040) << std::hex << bits;
        } else {
            ASSERT_EQ(back, h) << std::hex << bits;
        }
    }
}

TEST(Bf16, RoundsMidpointsToNearestEven) {
    for (std::uint32_t bits = 0; bits < 0x7F80; ++bits) {
        const std::uint32_t midpoint_bits = (bits << 16) | 0x8000u;
        for (const std::uint32_t b : {midpoint_bits - 1, midpoint_bits, midpoint_bits + 1}) {
            const float value = std::bit_cast<float>(b);
            ASSERT_EQ(bf16(value).bits, bf16_reference(value)) << std::hex << b;
            ASSERT_EQ(bf16(-value).bits, bf16_reference(-value)) << std::hex << b;
        }
    }
}

TEST(Bf16, MatchesReferenceOnRandomFloats) {
    for (const float value : random_floats(2'000'000, 5678)) {
        const std::uint16_t actual = bf16(value).bits;
        if (std::isnan(value)) {
            ASSERT_TRUE(bf16_is_nan(actual));
        } else {
            ASSERT_EQ(actual, bf16_reference(value)) << std::hexfloat << value;
        }
    }
}

TEST(Bf16, OverflowRoundsToInfinity) {
    EXPECT_EQ(bf16(std::numeric_limits<float>::max()).bits, 0x7F80);
    EXPECT_EQ(bf16(-std::numeric_limits<float>::infinity()).bits, 0xFF80);
}

}  // namespace
}  // namespace ii
