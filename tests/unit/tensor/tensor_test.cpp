// SPDX-License-Identifier: MIT
#include <ii/tensor.h>

#include <gtest/gtest.h>

#include <cstring>
#include <numeric>
#include <vector>

namespace ii {
namespace {

std::vector<float> to_vector(const Tensor& tensor) {
    const Tensor dense = tensor.contiguous().value();
    const auto values = dense.values<float>();
    return {values.begin(), values.end()};
}

Tensor iota(const Shape& shape) {
    std::vector<float> values(static_cast<std::size_t>(shape.numel()));
    std::iota(values.begin(), values.end(), 0.0f);
    return Tensor::from(std::span<const float>(values), shape).value();
}

TEST(DTypeTest, Table) {
    EXPECT_EQ(to_string(DType::kF32), "f32");
    EXPECT_EQ(element_size(DType::kF16), 2u);
    EXPECT_EQ(element_size(DType::kQ4_0), 0u);
    EXPECT_EQ(storage_size(DType::kQ4_0, 64), 36u);
    EXPECT_EQ(storage_size(DType::kQ4_K, 512), 288u);
    EXPECT_EQ(storage_size(DType::kI64, 3), 24u);
    EXPECT_EQ(parse_dtype("bf16"), DType::kBF16);
    EXPECT_EQ(parse_dtype("q6_k"), DType::kQ6_K);
    EXPECT_EQ(parse_dtype("invalid"), std::nullopt);
    EXPECT_EQ(parse_dtype("f31"), std::nullopt);
    static_assert(dtype_of<float>() == DType::kF32);
    static_assert(dtype_of<std::int64_t>() == DType::kI64);
    static_assert(dtype_of<f16>() == DType::kF16);
    static_assert(dtype_of<bool>() == DType::kBool);
    static_assert(!kHasDType<void*>);
    for (int i = 1; i < static_cast<int>(DType::kCount); ++i) {
        const auto dtype = static_cast<DType>(i);
        EXPECT_EQ(parse_dtype(to_string(dtype)), dtype);
        EXPECT_GT(dtype_info(dtype).block_bytes, 0);
    }
}

TEST(ShapeTest, Basics) {
    Shape shape{2, 3, 4};
    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape.numel(), 24);
    EXPECT_EQ(shape.to_string(), "[2, 3, 4]");
    EXPECT_EQ(contiguous_strides(shape), (Shape{12, 4, 1}));
    EXPECT_EQ(Shape{}.numel(), 1);
    shape.push_back(5);
    EXPECT_EQ(shape.numel(), 120);
    EXPECT_EQ(shape.at_signed(-1), 5);
    EXPECT_FALSE(shape == (Shape{2, 3, 4}));
}

TEST(ShapeDeathTest, RankOverflow) {
    EXPECT_DEATH((Shape{1, 2, 3, 4, 5, 6, 7, 8, 9}), "exceeds the maximum");
}

TEST(TensorTest, CreateAndAccess) {
    const Tensor empty;
    EXPECT_FALSE(empty.defined());
    EXPECT_EQ(empty.to_string(), "<empty tensor>");

    const Tensor t = Tensor::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}).value();
    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.dtype(), DType::kF32);
    EXPECT_EQ(t.nbytes(), 24u);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.values<float>()[4], 5.0f);
    EXPECT_EQ(t.to_string(), "f32[2, 3] {1, 2, 3, 4, 5, 6}");
    EXPECT_EQ(t.to_string(2), "f32[2, 3] {1, 2, ...}");
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(t.data()) % kDefaultAlignment, 0u);

    const Tensor z = Tensor::zeros(DType::kI32, Shape{5}).value();
    for (const std::int32_t v : z.values<std::int32_t>()) {
        EXPECT_EQ(v, 0);
    }
    const Tensor s = Tensor::scalar(7.5).value();
    EXPECT_EQ(s.rank(), 0);
    EXPECT_EQ(s.dtype(), DType::kF64);
    EXPECT_EQ(s.values<double>()[0], 7.5);
}

TEST(TensorTest, CreationErrors) {
    EXPECT_EQ(Tensor::from({1.0f, 2.0f}, Shape{3}).status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(Tensor::empty(DType::kF32, Shape{-1}).status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(Tensor::empty(DType::kInvalid, Shape{1}).status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(Tensor::empty(DType::kQ4_0, Shape{4, 33}).status().code(), StatusCode::kInvalidArgument);
    EXPECT_TRUE(Tensor::empty(DType::kQ4_0, Shape{4, 64}).ok());
    EXPECT_EQ(Tensor::empty(DType::kF32, Shape{1LL << 40, 1LL << 40}).status().code(), StatusCode::kOutOfRange);
}

TEST(TensorDeathTest, WrongTypeAccess) {
    const Tensor t = Tensor::zeros(DType::kF32, Shape{2}).value();
    EXPECT_DEATH((void)t.values<std::int32_t>(), "element type is f32");
    const Tensor p = iota(Shape{2, 3}).permute({1, 0}).value();
    EXPECT_DEATH((void)p.values<float>(), "not contiguous");
}

TEST(TensorTest, WrapIsZeroCopy) {
    float data[6] = {0, 1, 2, 3, 4, 5};
    const Tensor t = Tensor::wrap(data, DType::kF32, Shape{3, 2});
    EXPECT_EQ(t.data(), data);
    EXPECT_FALSE(t.buffer()->owns_memory());
    data[5] = 42;
    EXPECT_EQ(t.values<float>()[5], 42);
}

TEST(TensorTest, Reshape) {
    const Tensor t = iota(Shape{2, 3, 4});
    const Tensor r = t.reshape({4, -1}).value();
    EXPECT_EQ(r.shape(), (Shape{4, 6}));
    EXPECT_EQ(r.data(), t.data());
    EXPECT_FALSE(t.reshape({5, -1}).ok());
    EXPECT_FALSE(t.reshape({-1, -1}).ok());
    EXPECT_FALSE(t.reshape({7}).ok());
    EXPECT_FALSE(t.permute({2, 1, 0}).value().reshape({24}).ok());
}

TEST(TensorTest, PermuteAndContiguous) {
    const Tensor t = iota(Shape{2, 3});
    const Tensor p = t.permute({1, 0}).value();
    EXPECT_EQ(p.shape(), (Shape{3, 2}));
    EXPECT_FALSE(p.is_contiguous());
    EXPECT_EQ(to_vector(p), (std::vector<float>{0, 3, 1, 4, 2, 5}));
    EXPECT_FALSE(t.permute({0, 0}).ok());
    EXPECT_FALSE(t.permute({0}).ok());

    const Tensor t3 = iota(Shape{2, 3, 4});
    const Tensor p3 = t3.permute({2, 0, 1}).value();
    const std::vector<float> values = to_vector(p3);
    // p3[k][i][j] == t3[i][j][k] == i*12 + j*4 + k
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                EXPECT_EQ(values[static_cast<std::size_t>(k * 6 + i * 3 + j)], static_cast<float>(i * 12 + j * 4 + k));
            }
        }
    }
}

TEST(TensorTest, Slice) {
    const Tensor t = iota(Shape{4, 5});
    EXPECT_EQ(to_vector(t.slice(0, 1, 3).value()), (std::vector<float>{5, 6, 7, 8, 9, 10, 11, 12, 13, 14}));
    EXPECT_EQ(to_vector(t.slice(1, 0, 5, 2).value()), (std::vector<float>{0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19}));
    EXPECT_EQ(to_vector(t.slice(-1, -2, 100).value()), (std::vector<float>{3, 4, 8, 9, 13, 14, 18, 19}));
    EXPECT_EQ(t.slice(0, 3, 1).value().shape(), (Shape{0, 5}));
    EXPECT_EQ(to_vector(t.slice(0, 3, 1).value()), std::vector<float>{});
    EXPECT_FALSE(t.slice(2, 0, 1).ok());
    EXPECT_FALSE(t.slice(0, 0, 1, 0).ok());
    const Tensor q = Tensor::zeros(DType::kQ8_0, Shape{2, 64}).value();
    EXPECT_TRUE(q.slice(1, 32, 64).ok());
    EXPECT_FALSE(q.slice(1, 16, 48).ok());
    EXPECT_TRUE(q.slice(0, 1, 2).ok());
}

TEST(TensorTest, Expand) {
    const Tensor t = iota(Shape{3, 1});
    const Tensor e = t.expand(Shape{2, 3, 4}).value();
    EXPECT_EQ(e.strides(), (Shape{0, 1, 0}));
    const std::vector<float> values = to_vector(e);
    EXPECT_EQ(values.size(), 24u);
    EXPECT_EQ(values[5], 1.0f);
    EXPECT_EQ(values[12 + 8], 2.0f);
    EXPECT_FALSE(t.expand(Shape{3, 2}).value().is_contiguous());
    EXPECT_FALSE(t.expand(Shape{2, 4}).ok());
    EXPECT_FALSE(t.expand(Shape{4}).ok());
}

TEST(TensorTest, CopyFrom) {
    const Tensor src = iota(Shape{2, 3}).permute({1, 0}).value();
    const Tensor dst = Tensor::zeros(DType::kF32, Shape{3, 2}).value();
    ASSERT_TRUE(dst.copy_from(src).ok());
    EXPECT_EQ(to_vector(dst), (std::vector<float>{0, 3, 1, 4, 2, 5}));
    EXPECT_FALSE(dst.copy_from(iota(Shape{2, 3})).ok());
    EXPECT_FALSE(dst.copy_from(Tensor::zeros(DType::kI32, Shape{3, 2}).value()).ok());

    // Strided destination.
    const Tensor big = Tensor::zeros(DType::kF32, Shape{3, 4}).value();
    const Tensor window = big.slice(1, 1, 3).value();
    ASSERT_TRUE(window.copy_from(iota(Shape{3, 2})).ok());
    EXPECT_EQ(to_vector(big), (std::vector<float>{0, 0, 1, 0, 0, 2, 3, 0, 0, 4, 5, 0}));
}

TEST(TensorTest, View) {
    auto buffer = Buffer::allocate(64).value();
    EXPECT_TRUE(Tensor::view(buffer, DType::kF32, Shape{4, 4}, Shape{4, 1}).ok());
    EXPECT_FALSE(Tensor::view(buffer, DType::kF32, Shape{4, 5}, Shape{5, 1}).ok());
    EXPECT_FALSE(Tensor::view(buffer, DType::kF32, Shape{4, 4}, Shape{4, 1}, 4).ok());
    EXPECT_TRUE(Tensor::view(buffer, DType::kF32, Shape{0, 100}, Shape{100, 1}).ok());
    EXPECT_FALSE(Tensor::view(buffer, DType::kF32, Shape{4}, Shape{1, 1}).ok());
    EXPECT_FALSE(Tensor::view(nullptr, DType::kF32, Shape{4}, Shape{1}).ok());
}

TEST(TensorTest, BlockQuantizedClone) {
    Tensor q = Tensor::empty(DType::kQ4_0, Shape{3, 64}).value();
    auto* bytes = static_cast<std::uint8_t*>(q.data());
    for (std::size_t i = 0; i < q.nbytes(); ++i) {
        bytes[i] = static_cast<std::uint8_t>(i * 7);
    }
    const Tensor row = q.slice(0, 1, 3).value();
    const Tensor copy = row.clone().value();
    EXPECT_EQ(copy.nbytes(), 2u * 36u);
    EXPECT_EQ(std::memcmp(copy.data(), bytes + 36, copy.nbytes()), 0);
}

}  // namespace
}  // namespace ii
