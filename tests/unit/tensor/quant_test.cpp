// SPDX-License-Identifier: MIT
// Block-quantization codecs against gguf-py (tests/data/quant/references.npz, scripts/gen_quant_references.py).
#include "tensor/quant.h"

#include <ii/io.h>

#include "support/test_util.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace ii {
namespace {

const std::map<std::string, Tensor>& references() {
    static const std::map<std::string, Tensor> arrays = [] {
        std::map<std::string, Tensor> result;
        auto loaded = load_npz(std::filesystem::path(ii_test::test_data_dir()) / "quant" / "references.npz");
        if (!loaded.ok()) {
            ADD_FAILURE() << loaded.status().to_string();
            return result;
        }
        for (auto& [name, tensor] : *loaded) {
            result.emplace(name, std::move(tensor));
        }
        return result;
    }();
    return arrays;
}

const std::vector<DType> kAllBlockTypes = {DType::kQ4_0, DType::kQ4_1, DType::kQ5_0, DType::kQ5_1,
                                           DType::kQ8_0, DType::kQ8_1, DType::kQ2_K, DType::kQ3_K,
                                           DType::kQ4_K, DType::kQ5_K, DType::kQ6_K, DType::kQ8_K};

TEST(QuantTest, DequantizesLikeGgufPy) {
    int checked = 0;
    for (const DType dtype : kAllBlockTypes) {
        const std::string name(to_string(dtype));
        const auto blocks = references().find(name + "_blocks");
        if (blocks == references().end()) {
            continue;  // gguf-py does not implement this type
        }
        const Tensor& expected = references().at(name + "_dequant");
        std::vector<float> out(static_cast<std::size_t>(expected.numel()));
        ASSERT_TRUE(quant::can_dequantize(dtype));
        quant::dequantize(dtype, blocks->second.data(), out.data(), expected.numel());
        const auto want = expected.values<float>();
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_NEAR(out[i], want[i], 1e-6f * std::max(1.0f, std::fabs(want[i]))) << name << " element " << i;
        }
        ++checked;
    }
    EXPECT_EQ(checked, 10);
}

TEST(QuantTest, QuantizesLikeGgufPy) {
    int checked = 0;
    for (const DType dtype : kAllBlockTypes) {
        const std::string name(to_string(dtype));
        const auto input = references().find(name + "_input");
        if (input == references().end()) {
            continue;
        }
        const Tensor& expected = references().at(name + "_quant");
        std::vector<std::uint8_t> out(static_cast<std::size_t>(expected.numel()));
        ASSERT_TRUE(quant::can_quantize(dtype));
        quant::quantize(dtype, input->second.data_as<float>(), out.data(), input->second.numel());
        const auto want = expected.values<std::uint8_t>();
        int mismatches = 0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            mismatches += out[i] != want[i] ? 1 : 0;
        }
        EXPECT_EQ(mismatches, 0) << name;
        ++checked;
    }
    EXPECT_EQ(checked, 5);
}

/// Every quantizer round-trips smooth random data with an error typical for its bit width.
TEST(QuantTest, RoundTripErrorMatchesBitWidth) {
    const std::map<DType, double> max_relative_rmse = {
        {DType::kQ4_0, 0.12},  {DType::kQ4_1, 0.10},  {DType::kQ5_0, 0.06},  {DType::kQ5_1, 0.05},
        {DType::kQ8_0, 0.006}, {DType::kQ8_1, 0.006}, {DType::kQ2_K, 0.40},  {DType::kQ3_K, 0.20},
        {DType::kQ4_K, 0.10},  {DType::kQ5_K, 0.05},  {DType::kQ6_K, 0.025}, {DType::kQ8_K, 0.008},
    };
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    constexpr int kCount = 256 * 16;
    std::vector<float> input(kCount);
    for (float& v : input) {
        v = normal(rng);
    }
    for (const auto& [dtype, limit] : max_relative_rmse) {
        std::vector<std::uint8_t> blocks(storage_size(dtype, kCount));
        std::vector<float> output(kCount);
        quant::quantize(dtype, input.data(), blocks.data(), kCount);
        quant::dequantize(dtype, blocks.data(), output.data(), kCount);
        double error = 0;
        double energy = 0;
        for (int i = 0; i < kCount; ++i) {
            error += (output[i] - input[i]) * (output[i] - input[i]);
            energy += input[i] * input[i];
        }
        EXPECT_LT(std::sqrt(error / energy), limit) << to_string(dtype);
    }
}

TEST(QuantTest, ZeroBlocksStayZero) {
    for (const DType dtype : kAllBlockTypes) {
        const int count = dtype_info(dtype).block_size;
        std::vector<float> zeros(static_cast<std::size_t>(count), 0.0f);
        std::vector<std::uint8_t> blocks(storage_size(dtype, count), 0xFF);
        std::vector<float> output(static_cast<std::size_t>(count), 1.0f);
        quant::quantize(dtype, zeros.data(), blocks.data(), count);
        quant::dequantize(dtype, blocks.data(), output.data(), count);
        for (const float v : output) {
            ASSERT_EQ(v, 0.0f) << to_string(dtype);
        }
    }
    EXPECT_FALSE(quant::check_row(DType::kQ4_K, 300).ok());
    EXPECT_TRUE(quant::check_row(DType::kQ4_K, 512).ok());
}

}  // namespace
}  // namespace ii
