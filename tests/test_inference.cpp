// Юнит-тесты бэкенд-нейтральных хелперов inference.h :
// имена/размеры dtype и конвертация IEEE 754 binary16 -> binary32.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

#include "inference.h"

using inf::DType;

// ---- dtype_name / dtype_size ---------------------------------------------

TEST(DType, Names) {
    EXPECT_STREQ(inf::dtype_name(DType::Float32), "float32");
    EXPECT_STREQ(inf::dtype_name(DType::Float16), "float16");
    EXPECT_STREQ(inf::dtype_name(DType::Int8), "int8");
    EXPECT_STREQ(inf::dtype_name(DType::UInt8), "uint8");
    EXPECT_STREQ(inf::dtype_name(DType::Int32), "int32");
}

TEST(DType, Sizes) {
    EXPECT_EQ(inf::dtype_size(DType::Float32), 4u);
    EXPECT_EQ(inf::dtype_size(DType::Int32), 4u);
    EXPECT_EQ(inf::dtype_size(DType::UInt32), 4u);
    EXPECT_EQ(inf::dtype_size(DType::Int64), 8u);
    EXPECT_EQ(inf::dtype_size(DType::Float16), 2u);
    EXPECT_EQ(inf::dtype_size(DType::Int16), 2u);
    EXPECT_EQ(inf::dtype_size(DType::Int8), 1u);
    EXPECT_EQ(inf::dtype_size(DType::UInt8), 1u);
    EXPECT_EQ(inf::dtype_size(DType::Bool), 1u);
}

// ---- half_to_float (заголовочный inline) ----------------------------------

TEST(HalfToFloat, CommonValues) {
    EXPECT_FLOAT_EQ(inf::half_to_float(0x0000), 0.0f);   // +0
    EXPECT_FLOAT_EQ(inf::half_to_float(0x8000), -0.0f);  // -0
    EXPECT_FLOAT_EQ(inf::half_to_float(0x3C00), 1.0f);   // 1.0
    EXPECT_FLOAT_EQ(inf::half_to_float(0x4000), 2.0f);   // 2.0
    EXPECT_FLOAT_EQ(inf::half_to_float(0xC000), -2.0f);  // -2.0
    EXPECT_FLOAT_EQ(inf::half_to_float(0x3800), 0.5f);   // 0.5
}

TEST(HalfToFloat, SpecialValues) {
    EXPECT_TRUE(std::isinf(inf::half_to_float(0x7C00)));  // +inf
    EXPECT_GT(inf::half_to_float(0x7C00), 0.0f);
    EXPECT_TRUE(std::isinf(inf::half_to_float(0xFC00)));  // -inf
    EXPECT_LT(inf::half_to_float(0xFC00), 0.0f);
    EXPECT_TRUE(std::isnan(inf::half_to_float(0x7E00)));  // NaN
}

TEST(HalfToFloat, Subnormal) {
    // Наименьший положительный субнормал half = 2^-24 ≈ 5.96e-8.
    EXPECT_NEAR(inf::half_to_float(0x0001), 5.9604645e-8f, 1e-12f);
}
