// SPDX-License-Identifier: MIT
// Quantized dot-product kernels of every ii-cpu ISA variant against an exact computation: the activations are
// quantized by the kernel table, then the dequantized weights and activations are multiplied in double precision.
#include "backends/cpu/cpu_backend.h"
#include "core/isa.h"
#include "support/test_util.h"
#include "tensor/quant.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>

namespace ii::cpu {
namespace {

struct TypeCase {
    DType dtype;
    QType qtype;
};

const TypeCase kTypes[] = {
    {DType::kQ4_0, QType::kQ4_0}, {DType::kQ4_1, QType::kQ4_1}, {DType::kQ5_0, QType::kQ5_0},
    {DType::kQ5_1, QType::kQ5_1}, {DType::kQ8_0, QType::kQ8_0}, {DType::kQ2_K, QType::kQ2_K},
    {DType::kQ3_K, QType::kQ3_K}, {DType::kQ4_K, QType::kQ4_K}, {DType::kQ5_K, QType::kQ5_K},
    {DType::kQ6_K, QType::kQ6_K},
};

bool k_type(DType dtype) {
    return dtype_info(dtype).block_size == 256;
}

/// Dequantized activation row (from either activation layout).
std::vector<double> activation_values(const std::uint8_t* a, std::int64_t k, bool k_layout) {
    std::vector<double> values(static_cast<std::size_t>(k));
    const std::int64_t block = k_layout ? 256 : 32;
    const std::int64_t bytes = k_layout ? 292 : 40;
    for (std::int64_t b = 0; b < k / block; ++b) {
        const std::uint8_t* p = a + b * bytes;
        float d = 0;
        std::memcpy(&d, p, 4);
        const auto* q = reinterpret_cast<const std::int8_t*>(p + (k_layout ? 4 : 8));
        for (std::int64_t i = 0; i < block; ++i) {
            values[static_cast<std::size_t>(b * block + i)] = static_cast<double>(d) * q[i];
        }
    }
    return values;
}

class QuantKernelTest : public ::testing::TestWithParam<Isa> {};

TEST_P(QuantKernelTest, DotProductsAreExact) {
    const KernelTable& table = kernel_table(GetParam());
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (const TypeCase& t : kTypes) {
        const std::int64_t k = k_type(t.dtype) ? 768 : 160;
        const int max_rows = 6;
        // Weights: random values quantized; activations: random rows quantized by the kernel table.
        std::vector<float> wf(static_cast<std::size_t>(k));
        for (float& v : wf) {
            v = normal(rng);
        }
        std::vector<std::uint8_t> w(storage_size(t.dtype, k));
        quant::quantize(t.dtype, wf.data(), w.data(), k);
        std::vector<float> wd(static_cast<std::size_t>(k));
        quant::dequantize(t.dtype, w.data(), wd.data(), k);

        const std::int64_t a_bytes = k_type(t.dtype) ? k / 256 * 292 : k / 32 * 40;
        std::vector<std::uint8_t> act(static_cast<std::size_t>(a_bytes * max_rows));
        for (int r = 0; r < max_rows; ++r) {
            std::vector<float> x(static_cast<std::size_t>(k));
            for (float& v : x) {
                v = normal(rng) * (r + 1.0f);
            }
            (k_type(t.dtype) ? table.quantize_q8k : table.quantize_q8x32)(x.data(), act.data() + r * a_bytes, k);
        }
        for (const int rows : {1, 3, 4}) {
            for (int start = 0; start + rows <= max_rows; start += 2) {
                float out[8] = {};
                table.qdot[static_cast<int>(t.qtype)](k, w.data(), act.data() + start * a_bytes, a_bytes, rows, out, 2);
                for (int r = 0; r < rows; ++r) {
                    const std::vector<double> a =
                        activation_values(act.data() + (start + r) * a_bytes, k, k_type(t.dtype));
                    double expected = 0;
                    double magnitude = 0;
                    for (std::int64_t i = 0; i < k; ++i) {
                        expected +=
                            static_cast<double>(wd[static_cast<std::size_t>(i)]) * a[static_cast<std::size_t>(i)];
                        magnitude += std::fabs(static_cast<double>(wd[static_cast<std::size_t>(i)]) *
                                               a[static_cast<std::size_t>(i)]);
                    }
                    EXPECT_NEAR(out[2 * r], expected, 1e-5 * magnitude + 1e-6)
                        << isa_name(GetParam()) << " " << to_string(t.dtype) << " rows " << rows << " row " << r;
                }
            }
        }
    }
}

TEST_P(QuantKernelTest, ActivationQuantizationMatchesTheReference) {
    const KernelTable& table = kernel_table(GetParam());
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0f, 2.0f);
    std::vector<float> x(512);
    for (float& v : x) {
        v = normal(rng);
    }
    x[7] = 0.0f;
    // q8_K is ggml's layout: identical bytes to the reference quantizer.
    std::vector<std::uint8_t> ours(2 * 292);
    std::vector<std::uint8_t> reference(2 * 292);
    table.quantize_q8k(x.data(), ours.data(), 512);
    quant::quantize(DType::kQ8_K, x.data(), reference.data(), 512);
    EXPECT_EQ(ours, reference) << isa_name(GetParam());
    // q8x32: scale amax / 127 and rounding half to even.
    std::vector<std::uint8_t> blocks(16 * 40);
    table.quantize_q8x32(x.data(), blocks.data(), 512);
    const std::vector<double> values = activation_values(blocks.data(), 512, false);
    for (std::size_t i = 0; i < values.size(); ++i) {
        float amax = 0;
        for (std::size_t j = i / 32 * 32; j < i / 32 * 32 + 32; ++j) {
            amax = std::max(amax, std::fabs(x[j]));
        }
        EXPECT_NEAR(values[i], x[i], amax / 127.0 * 0.5 + 1e-6);
    }
}

TEST_P(QuantKernelTest, PanelKernelsMatchRowKernels) {
    const KernelTable& table = kernel_table(GetParam());
    std::mt19937 rng(21);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    int checked = 0;
    for (const TypeCase& t : kTypes) {
        const QGemm8Fn gemm8 = table.qgemm8[static_cast<int>(t.qtype)];
        if (gemm8 == nullptr) {
            continue;
        }
        const std::int64_t k = k_type(t.dtype) ? 512 : 96;
        const std::int64_t n = 16;
        std::vector<float> wf(static_cast<std::size_t>(n * k));
        for (float& v : wf) {
            v = normal(rng);
        }
        std::vector<std::uint8_t> w(storage_size(t.dtype, n * k));
        quant::quantize(t.dtype, wf.data(), w.data(), n * k);
        auto panels = pack_quantized_panels(t.qtype, t.dtype, w.data(), n, k);
        ASSERT_TRUE(panels.ok()) << panels.status().to_string();
        const std::int64_t panel_stride = quantized_panel_stride(t.qtype, k);
        const std::int64_t a_bytes = k_type(t.dtype) ? k / 256 * 292 : k / 32 * 40;
        const int rows = 4;
        std::vector<std::uint8_t> act(static_cast<std::size_t>(a_bytes * rows));
        for (int r = 0; r < rows; ++r) {
            std::vector<float> x(static_cast<std::size_t>(k));
            for (float& v : x) {
                v = normal(rng);
            }
            (k_type(t.dtype) ? table.quantize_q8k : table.quantize_q8x32)(x.data(), act.data() + r * a_bytes, k);
        }
        const std::size_t w_row = storage_size(t.dtype, k);
        for (const int count : {1, 2, 3, 4}) {
            for (std::int64_t panel = 0; panel < n / 8; ++panel) {
                float ours[4 * 8] = {};
                gemm8(k, static_cast<const std::uint8_t*>(panels->data()) + panel * panel_stride, act.data(), a_bytes,
                      count, ours, 8);
                for (int c = 0; c < 8; ++c) {
                    float expected[4] = {};
                    table.qdot[static_cast<int>(t.qtype)](k, w.data() + (panel * 8 + c) * w_row, act.data(), a_bytes,
                                                          count, expected, 1);
                    for (int r = 0; r < count; ++r) {
                        EXPECT_NEAR(ours[r * 8 + c], expected[r], 2e-5 * (std::fabs(expected[r]) + 1.0))
                            << isa_name(GetParam()) << " " << to_string(t.dtype) << " rows " << count << " column "
                            << panel * 8 + c;
                    }
                }
            }
        }
        ++checked;
    }
    if (std::string_view(table.isa) != "generic" && std::string_view(table.isa) != "neon") {
        EXPECT_EQ(checked, 5) << table.isa;
    }
}

std::vector<Isa> isas() {
    return usable_isas(cpu_info());
}

INSTANTIATE_TEST_SUITE_P(Variants, QuantKernelTest, ::testing::ValuesIn(isas()),
                         [](const ::testing::TestParamInfo<Isa>& param_info) {
                             return std::string(isa_name(param_info.param));
                         });

}  // namespace
}  // namespace ii::cpu
