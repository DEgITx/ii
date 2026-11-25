// SPDX-License-Identifier: MIT
// ii-cpu: accuracy of the vectorized math functions and differential tests against ii-ref (GEMM shapes and layouts,
// broadcasting, reductions, normalizations, attention, data movement), for every ISA variant this machine can run.
// Models are compiled for ii-cpu alone, so every node is guaranteed to run on the backend under test.
#include <ii/builder.h>
#include <ii/model.h>

#include "backends/cpu/cpu_backend.h"
#include "core/isa.h"
#include "support/test_util.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace ii::cpu {
namespace {

std::vector<std::string> isa_names() {
    std::vector<std::string> names;
    for (const Isa isa : usable_isas(cpu_info())) {
        names.emplace_back(isa_name(isa));
    }
    return names;
}

const Context& context_for(const std::string& isa) {
    static std::mutex mutex;
    static std::map<std::string, Context> contexts;
    const std::lock_guard lock(mutex);
    auto it = contexts.find(isa);
    if (it == contexts.end()) {
        ContextOptions options;
        options.cpu_isa = isa;
        options.num_threads = 4;
        it = contexts.emplace(isa, Context::create(std::move(options)).value()).first;
    }
    return it->second;
}

class CpuBackendTest : public ::testing::TestWithParam<std::string> {
protected:
    const Context& context() const { return context_for(GetParam()); }

    /// Builds a model with `build`, runs it on ii-cpu only and on ii-ref, and compares all outputs.
    void expect_matches_reference(const std::function<void(GraphBuilder&)>& build, const std::vector<Tensor>& inputs,
                                  ii_test::Tolerance tolerance) const {
        GraphBuilder b;
        build(b);
        const Result<Model> model = Model::from_builder(context(), std::move(b));
        ASSERT_TRUE(model.ok()) << model.status().to_string();
        CompileOptions cpu;
        cpu.backends = {"ii-cpu"};
        CompileOptions ref;
        ref.backends = {"ii-ref"};
        const Result<std::vector<Tensor>> actual = ii_test::run_model(*model, inputs, cpu);
        ASSERT_TRUE(actual.ok()) << actual.status().to_string();
        const Result<std::vector<Tensor>> expected = ii_test::run_model(*model, inputs, ref);
        ASSERT_TRUE(expected.ok()) << expected.status().to_string();
        ASSERT_EQ(actual->size(), expected->size());
        for (std::size_t i = 0; i < actual->size(); ++i) {
            const std::string mismatch = ii_test::compare((*actual)[i], (*expected)[i], tolerance);
            EXPECT_TRUE(mismatch.empty()) << "output " << i << ": " << mismatch;
        }
    }
};

Dims dims_of(const Shape& shape) {
    return Dims(shape.begin(), shape.end());
}

// --------------------------------------------------------------------------------------------------------------------
// Vectorized math: maximum error over dense sweeps, measured in units in the last place of the correctly rounded
// result (plus a small absolute floor where the result passes through zero).

struct MathCase {
    const char* name;
    EwOp op;
    double (*reference)(double);
    double low;
    double high;
    bool log_spaced;
    double max_ulps;
    double abs_floor;
};

double ref_sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
double ref_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}
double ref_gelu_tanh(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) * (x + 0.044715 * x * x * x)));
}
double ref_silu(double x) {
    return x / (1.0 + std::exp(-x));
}
double ref_exp(double x) {
    return std::exp(x);
}
double ref_log(double x) {
    return std::log(x);
}
double ref_tanh(double x) {
    return std::tanh(x);
}
double ref_erf(double x) {
    return std::erf(x);
}

const MathCase kMathCases[] = {
    {"exp", EwOp::kExp, ref_exp, -87.0, 88.5, false, 2, 0},
    {"log", EwOp::kLog, ref_log, 1e-44, 3e38, true, 2, 0},
    {"tanh", EwOp::kTanh, ref_tanh, -12.0, 12.0, false, 3, 0},
    {"sigmoid", EwOp::kSigmoid, ref_sigmoid, -80.0, 80.0, false, 4, 1e-30},
    {"erf", EwOp::kErf, ref_erf, -6.0, 6.0, false, 8, 1e-9},
    {"gelu", EwOp::kGelu, ref_gelu, -12.0, 12.0, false, 8, 1.5e-6},
    {"gelu_tanh", EwOp::kGeluTanh, ref_gelu_tanh, -12.0, 12.0, false, 8, 1.5e-6},
    {"silu", EwOp::kSilu, ref_silu, -80.0, 80.0, false, 4, 1e-30},
};

std::vector<float> run_unary(const KernelTable& table, EwOp op, const std::vector<float>& x) {
    const EwInstr code[] = {{EwOp::kLoad, 0, 0, 0, 0, 0, 0}, {op, 1, 0, 0, 0, 0, 0}};
    const EwProgram program{code, 2, 1, Elem::kF32};
    const EwInput input{x.data(), Elem::kF32, false};
    std::vector<float> y(x.size());
    table.ew_run(program, &input, y.data(), static_cast<std::int64_t>(x.size()));
    return y;
}

class CpuMathTest : public ::testing::TestWithParam<std::string> {};

TEST_P(CpuMathTest, ElementaryFunctionsAreAccurate) {
    const KernelTable& table = kernel_table(*parse_isa(GetParam()));
    for (const MathCase& c : kMathCases) {
        constexpr int kCount = 200001;
        std::vector<float> x(kCount);
        for (int i = 0; i < kCount; ++i) {
            const double t = static_cast<double>(i) / (kCount - 1);
            x[static_cast<std::size_t>(i)] =
                static_cast<float>(c.log_spaced ? std::exp(std::log(c.low) + t * (std::log(c.high) - std::log(c.low)))
                                                : c.low + t * (c.high - c.low));
        }
        const std::vector<float> y = run_unary(table, c.op, x);
        double worst = 0;
        float worst_x = 0;
        for (int i = 0; i < kCount; ++i) {
            const float xi = x[static_cast<std::size_t>(i)];
            const double expected = c.reference(static_cast<double>(xi));
            const auto rounded = static_cast<float>(expected);
            const float ulp =
                std::nextafter(std::fabs(rounded), std::numeric_limits<float>::infinity()) - std::fabs(rounded);
            const double error = std::fabs(static_cast<double>(y[static_cast<std::size_t>(i)]) - expected);
            const double ulps = std::max(0.0, error - c.abs_floor) / static_cast<double>(ulp);
            if (ulps > worst) {
                worst = ulps;
                worst_x = xi;
            }
        }
        EXPECT_LE(worst, c.max_ulps) << c.name << ": worst error " << worst << " ulp at x = " << worst_x;
    }
}

TEST_P(CpuMathTest, SpecialValues) {
    const KernelTable& table = kernel_table(*parse_isa(GetParam()));
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> x = {inf, -inf, nan, 0.0f, -0.0f, 1e-40f};
    auto at = [&](EwOp op, std::size_t i) { return run_unary(table, op, x)[i]; };
    EXPECT_EQ(at(EwOp::kExp, 0), inf);
    EXPECT_EQ(at(EwOp::kExp, 1), 0.0f);
    EXPECT_TRUE(std::isnan(at(EwOp::kExp, 2)));
    EXPECT_EQ(at(EwOp::kExp, 3), 1.0f);
    EXPECT_EQ(at(EwOp::kLog, 0), inf);
    EXPECT_TRUE(std::isnan(at(EwOp::kLog, 1)));
    EXPECT_EQ(at(EwOp::kLog, 3), -inf);
    EXPECT_NEAR(at(EwOp::kLog, 5), std::log(1e-40), 1e-4);
    EXPECT_EQ(at(EwOp::kTanh, 0), 1.0f);
    EXPECT_EQ(at(EwOp::kTanh, 1), -1.0f);
    EXPECT_TRUE(std::isnan(at(EwOp::kTanh, 2)));
    EXPECT_EQ(at(EwOp::kSigmoid, 0), 1.0f);
    EXPECT_EQ(at(EwOp::kSigmoid, 1), 0.0f);
    EXPECT_EQ(at(EwOp::kErf, 0), 1.0f);
    EXPECT_EQ(at(EwOp::kErf, 1), -1.0f);
    EXPECT_TRUE(std::isnan(at(EwOp::kErf, 2)));
    EXPECT_TRUE(std::isnan(at(EwOp::kRelu, 2)));
    EXPECT_TRUE(std::isnan(at(EwOp::kSigmoid, 2)));
}

TEST_P(CpuMathTest, ConversionsRoundToNearestEven) {
    const KernelTable& table = kernel_table(*parse_isa(GetParam()));
    std::mt19937 rng(7);
    std::vector<float> x(10007);
    for (float& v : x) {
        const std::uint32_t bits = rng();
        std::memcpy(&v, &bits, sizeof v);
        if (std::isnan(v)) {
            v = 1.0f;
        }
    }
    x[0] = 65520.0f;  // rounds to f16 infinity
    x[1] = 5.96e-8f;  // f16 subnormal range
    x[2] = std::numeric_limits<float>::quiet_NaN();
    std::vector<std::uint16_t> half(x.size());
    std::vector<std::uint16_t> brain(x.size());
    table.convert(x.data(), Elem::kF32, half.data(), Elem::kF16, static_cast<std::int64_t>(x.size()));
    table.convert(x.data(), Elem::kF32, brain.data(), Elem::kBF16, static_cast<std::int64_t>(x.size()));
    for (std::size_t i = 0; i < x.size(); ++i) {
        const f16 expected_half(x[i]);
        const bf16 expected_brain(x[i]);
        if (std::isnan(x[i])) {
            EXPECT_TRUE(std::isnan(f16::from_bits(half[i]).to_float()));
            EXPECT_TRUE(std::isnan(bf16::from_bits(brain[i]).to_float()));
            continue;
        }
        EXPECT_EQ(half[i], expected_half.bits) << "f16 of " << x[i];
        EXPECT_EQ(brain[i], expected_brain.bits) << "bf16 of " << x[i];
    }
}

INSTANTIATE_TEST_SUITE_P(Isa, CpuMathTest, ::testing::ValuesIn(isa_names()));

// --------------------------------------------------------------------------------------------------------------------
// GEMM

struct MatmulCase {
    Shape a;
    Shape b;
    bool transpose_a = false;
    bool transpose_b = false;
    bool constant_b = false;
    DType dtype = DType::kF32;
};

ii_test::Tolerance gemm_tolerance(DType dtype, std::int64_t k) {
    const double scale = std::sqrt(static_cast<double>(std::max<std::int64_t>(k, 1)));
    switch (dtype) {
    case DType::kF16:
        return {4e-3, 2e-3 * scale};
    case DType::kBF16:
        return {3e-2, 2e-2 * scale};
    default:
        return {1e-4, 2e-6 * scale};
    }
}

TEST_P(CpuBackendTest, MatmulMatchesReference) {
    const std::vector<MatmulCase> cases = {
        {{1, 1}, {1, 1}},
        {{5, 7}, {7, 3}},
        {{6, 16}, {16, 16}},
        {{14, 32}, {32, 32}},
        {{13, 300}, {300, 37}},
        {{29, 1100}, {1100, 70}},   // K above the KC block
        {{1, 1100}, {1100, 50}},    // GEMV
        {{3}, {3, 4}},              // 1-D a
        {{4, 3}, {3}},              // 1-D b
        {{2, 1, 5, 6}, {3, 6, 4}},  // batch broadcasting
        {{7, 5}, {9, 7}, true, true},
        {{4, 9, 5}, {4, 9, 6}, true, false},
        {{8, 33}, {40, 33}, false, true, true},  // constant B, transposed (linear layout)
        {{3, 8, 33}, {33, 40}, false, false, true},
        {{2, 4, 33}, {2, 33, 17}, false, false, true},  // batched constant B
        {{0, 5}, {5, 3}},
        {{4, 0}, {0, 3}},  // K = 0: zeros
        {{200, 300}, {300, 257}},
        {{17, 64}, {64, 40}, false, false, false, DType::kF16},
        {{17, 64}, {40, 64}, false, true, true, DType::kF16},
        {{9, 70}, {70, 33}, false, false, true, DType::kBF16},
        {{1, 300}, {300, 70}, false, false, true, DType::kBF16},
    };
    std::uint32_t seed = 1;
    for (const MatmulCase& c : cases) {
        SCOPED_TRACE(c.a.to_string() + " x " + c.b.to_string() + (c.transpose_a ? " ta" : "") +
                     (c.transpose_b ? " tb" : "") + (c.constant_b ? " const" : "") + " " +
                     std::string(to_string(c.dtype)));
        const Tensor a = ii_test::random_tensor(c.dtype, c.a, seed++);
        const Tensor bt = ii_test::random_tensor(c.dtype, c.b, seed++);
        const std::int64_t k = c.a.rank() == 1 ? c.a[0] : (c.transpose_a ? c.a[c.a.rank() - 2] : c.a[c.a.rank() - 1]);
        std::vector<Tensor> inputs = {a};
        if (!c.constant_b) {
            inputs.push_back(bt);
        }
        expect_matches_reference(
            [&](GraphBuilder& b) {
                const Var x = b.input("a", c.dtype, dims_of(c.a));
                const Var y = c.constant_b ? b.constant(bt) : b.input("b", c.dtype, dims_of(c.b));
                b.output("c", matmul(x, y, c.transpose_a, c.transpose_b));
            },
            inputs, gemm_tolerance(c.dtype, k));
    }
}

TEST_P(CpuBackendTest, LinearMatchesReference) {
    struct LinearCase {
        Shape x;
        std::int64_t out;
        bool transpose_w;
        bool bias;
        DType dtype;
    };
    const std::vector<LinearCase> cases = {
        {{2, 3, 64}, 48, true, true, DType::kF32},     {{5, 1100}, 33, true, false, DType::kF32},
        {{1, 64}, 4096 / 16, true, true, DType::kF32}, {{7, 20}, 9, false, true, DType::kF32},
        {{6, 40}, 24, true, true, DType::kF16},        {{6, 40}, 24, false, true, DType::kBF16},
        {{31}, 17, true, true, DType::kF32},
    };
    std::uint32_t seed = 100;
    for (const LinearCase& c : cases) {
        SCOPED_TRACE(c.x.to_string() + " -> " + std::to_string(c.out));
        const std::int64_t in = c.x[c.x.rank() - 1];
        const Tensor x = ii_test::random_tensor(c.dtype, c.x, seed++);
        const Tensor w = ii_test::random_tensor(c.dtype, c.transpose_w ? Shape{c.out, in} : Shape{in, c.out}, seed++);
        const Tensor bias = ii_test::random_tensor(c.dtype, Shape{c.out}, seed++);
        expect_matches_reference(
            [&](GraphBuilder& b) {
                const Var input = b.input("x", c.dtype, dims_of(c.x));
                b.output("y", linear(input, b.constant(w), c.bias ? b.constant(bias) : Var(), c.transpose_w));
            },
            {x}, gemm_tolerance(c.dtype, in));
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Elementwise

TEST_P(CpuBackendTest, BroadcastingMatchesReference) {
    const std::vector<std::pair<Shape, Shape>> shapes = {
        {{4, 1, 5}, {3, 1}},    {{2, 3, 4}, {4}},   {{1}, {7, 9}},       {{}, {5, 6}},
        {{6, 1}, {1, 6}},       {{70000}, {70000}}, {{300, 257}, {257}}, {{2, 1, 3, 1}, {1, 4, 1, 5}},
        {{513, 129}, {513, 1}},
    };
    std::uint32_t seed = 200;
    for (const auto& [sa, sb] : shapes) {
        SCOPED_TRACE(sa.to_string() + " and " + sb.to_string());
        const Tensor a = ii_test::random_tensor(DType::kF32, sa, seed++);
        const Tensor c = ii_test::random_tensor(DType::kF32, sb, seed++);
        expect_matches_reference(
            [&](GraphBuilder& b) {
                const Var x = b.input("a", DType::kF32, dims_of(sa));
                const Var y = b.input("b", DType::kF32, dims_of(sb));
                b.output("sum", x + y);
                b.output("where", where(less(x, y), x * y, y - x));
                b.output("max", maximum(x, y));
            },
            {a, c}, {1e-6, 1e-7});
    }
}

TEST_P(CpuBackendTest, UnaryAndBinaryOpsMatchReference) {
    const Shape shape{1000};
    const Tensor a = ii_test::random_tensor(DType::kF32, shape, 300, -3.0, 3.0);
    const Tensor c = ii_test::random_tensor(DType::kF32, shape, 301, 0.1, 3.0);
    expect_matches_reference(
        [&](GraphBuilder& b) {
            const Var x = b.input("a", DType::kF32, dims_of(shape));
            const Var y = b.input("b", DType::kF32, dims_of(shape));
            b.output("neg", -x);
            b.output("abs", abs(x));
            b.output("sign", sign(x));
            b.output("exp", exp(x));
            b.output("log", log(y));
            b.output("sqrt", sqrt(y));
            b.output("rsqrt", rsqrt(y));
            b.output("recip", recip(y));
            b.output("sin", sin(x));
            b.output("cos", cos(x));
            b.output("tanh", tanh(x));
            b.output("sigmoid", sigmoid(x));
            b.output("erf", erf(x));
            b.output("relu", relu(x));
            b.output("floor", floor(x * 2.0));
            b.output("ceil", ceil(x * 2.0));
            b.output("round", round(x * 2.0));
            b.output("div", x / y);
            b.output("pow", pow(y, x));
            b.output("mod", mod(x, y));
            b.output("fmod", mod(x, y, true));
            b.output("clamp", clamp(x, -1.0, 0.5));
            b.output("gelu", gelu(x));
            b.output("gelu_tanh", gelu(x, true));
            b.output("silu", silu(x));
            b.output("ge", cast(greater_equal(x, y), DType::kF32));
            b.output("logic", logical_xor(logical_and(less(x, y), not_equal(x, y)), logical_not(equal(x, y))));
        },
        {a, c}, {1e-5, 1e-6});
}

TEST_P(CpuBackendTest, HalfPrecisionElementwiseMatchesReference) {
    for (const DType dtype : {DType::kF16, DType::kBF16}) {
        const Shape shape{3, 333};
        const Tensor a = ii_test::random_tensor(dtype, shape, 400);
        const Tensor c = ii_test::random_tensor(dtype, Shape{333}, 401);
        expect_matches_reference(
            [&](GraphBuilder& b) {
                const Var x = b.input("a", dtype, dims_of(shape));
                const Var y = b.input("b", dtype, {333});
                b.output("y", silu(x * y + x));
                b.output("cast", cast(x, DType::kF32));
                b.output("down", cast(cast(x, DType::kF32) * 3.0, dtype));
            },
            {a, c}, ii_test::tolerance_for(dtype));
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Reductions, normalizations, attention

TEST_P(CpuBackendTest, ReductionsMatchReference) {
    const Shape shape{6, 37, 5};
    const Tensor x = ii_test::random_tensor(DType::kF32, shape, 500, 0.5, 1.5);
    const std::vector<std::vector<std::int64_t>> axes_list = {{}, {0}, {1}, {2}, {0, 1}, {1, 2}, {-1}};
    for (const auto& axes : axes_list) {
        expect_matches_reference(
            [&](GraphBuilder& b) {
                const Var input = b.input("x", DType::kF32, dims_of(shape));
                b.output("sum", reduce_sum(input, axes));
                b.output("mean", reduce_mean(input, axes, true));
                b.output("max", reduce_max(input, axes));
                b.output("min", reduce_min(input, axes));
                b.output("prod", reduce_prod(input, axes));
            },
            {x}, {1e-5, 1e-6});
    }
}

TEST_P(CpuBackendTest, SoftmaxAndNormsMatchReference) {
    const Shape shape{3, 70, 9};
    const Tensor x = ii_test::random_tensor(DType::kF32, shape, 600, -4, 4);
    const Tensor scale = ii_test::random_tensor(DType::kF32, Shape{70, 9}, 601);
    const Tensor bias = ii_test::random_tensor(DType::kF32, Shape{70, 9}, 602);
    const Tensor row_scale = ii_test::random_tensor(DType::kF32, Shape{9}, 603);
    expect_matches_reference(
        [&](GraphBuilder& b) {
            const Var input = b.input("x", DType::kF32, dims_of(shape));
            b.output("softmax_last", softmax(input));
            b.output("softmax_mid", softmax(input, 1));
            b.output("softmax_first", softmax(input, 0));
            b.output("log_softmax", log_softmax(input, 1));
            b.output("layer_norm", layer_norm(input, b.constant(scale), b.constant(bias), 1));
            b.output("layer_norm_plain", layer_norm(input));
            b.output("rms_norm", rms_norm(input, b.constant(row_scale)));
            b.output("rms_norm_two", rms_norm(input, {}, 1));
        },
        {x}, {2e-5, 2e-6});
}

TEST_P(CpuBackendTest, AttentionMatchesReference) {
    struct AttentionCase {
        std::int64_t batch, heads, kv_heads, sq, skv, d, dv;
        bool causal;
        int mask;  // 0 none, 1 bool [sq, skv], 2 additive [b, 1, 1, skv]
    };
    const std::vector<AttentionCase> cases = {
        {1, 2, 2, 3, 5, 4, 3, false, 0}, {2, 4, 2, 7, 7, 16, 16, true, 0}, {1, 8, 1, 1, 33, 64, 64, true, 0},
        {2, 2, 1, 5, 9, 8, 8, false, 1}, {2, 3, 3, 4, 6, 5, 7, false, 2},  {1, 2, 2, 40, 40, 32, 32, true, 2},
    };
    std::uint32_t seed = 700;
    for (const AttentionCase& c : cases) {
        SCOPED_TRACE(::testing::Message() << "heads " << c.heads << "/" << c.kv_heads << " sq " << c.sq << " skv "
                                          << c.skv << " causal " << c.causal << " mask " << c.mask);
        const Shape qs{c.batch, c.heads, c.sq, c.d};
        const Shape ks{c.batch, c.kv_heads, c.skv, c.d};
        const Shape vs{c.batch, c.kv_heads, c.skv, c.dv};
        std::vector<Tensor> inputs = {ii_test::random_tensor(DType::kF32, qs, seed++),
                                      ii_test::random_tensor(DType::kF32, ks, seed++),
                                      ii_test::random_tensor(DType::kF32, vs, seed++)};
        Shape mask_shape;
        if (c.mask == 1) {
            mask_shape = Shape{c.sq, c.skv};
            Tensor mask = ii_test::random_tensor(DType::kBool, mask_shape, seed++);
            for (std::int64_t i = 0; i < c.sq; ++i) {
                static_cast<std::uint8_t*>(mask.data())[i * c.skv] = 1;  // no fully masked rows
            }
            inputs.push_back(mask);
        } else if (c.mask == 2) {
            mask_shape = Shape{c.batch, 1, 1, c.skv};
            inputs.push_back(ii_test::random_tensor(DType::kF32, mask_shape, seed++, -3, 0));
        }
        expect_matches_reference(
            [&](GraphBuilder& b) {
                const Var q = b.input("q", DType::kF32, dims_of(qs));
                const Var k = b.input("k", DType::kF32, dims_of(ks));
                const Var v = b.input("v", DType::kF32, dims_of(vs));
                const Var mask = c.mask == 0
                                     ? Var()
                                     : b.input("mask", c.mask == 1 ? DType::kBool : DType::kF32, dims_of(mask_shape));
                b.output("y", sdpa(q, k, v, mask, c.causal));
            },
            inputs, {1e-4, 1e-5});
    }
}

// --------------------------------------------------------------------------------------------------------------------
// Data movement

TEST_P(CpuBackendTest, MovementMatchesReference) {
    for (const DType dtype : {DType::kF32, DType::kF16, DType::kI64, DType::kU8, DType::kBool}) {
        SCOPED_TRACE(std::string(to_string(dtype)));
        const Shape shape{3, 40, 5, 33};
        const Tensor x = ii_test::random_tensor(dtype, shape, 800, -100, 100);
        const Tensor y = ii_test::random_tensor(dtype, Shape{3, 7, 5, 33}, 801, -100, 100);
        expect_matches_reference(
            [&](GraphBuilder& b) {
                const Var input = b.input("x", dtype, dims_of(shape));
                const Var other = b.input("y", dtype, {3, 7, 5, 33});
                b.output("transpose", permute(input, {0, 1, 3, 2}));
                b.output("permute", permute(input, {2, 0, 3, 1}));
                b.output("swap", permute(input, {1, 0, 2, 3}));
                b.output("reshape", reshape(input, {120, 165}));
                b.output("concat", concat({input, other, input}, 1));
                b.output("concat_last", concat({input, input}, -1));
            },
            {x, y}, {0, 0});
    }
}

INSTANTIATE_TEST_SUITE_P(Isa, CpuBackendTest, ::testing::ValuesIn(isa_names()));

// --------------------------------------------------------------------------------------------------------------------

TEST(CpuBackend, ReportsSelectedIsaAndRejectsUnknownOnes) {
    ContextOptions options;
    options.cpu_isa = "no-such-isa";
    EXPECT_FALSE(Context::create(std::move(options)).ok());
    const Context context = ii_test::test_context();
    const std::vector<std::string> names = context.backend_names();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "ii-cpu");
    EXPECT_EQ(names[1], "ii-ref");
}

TEST(CpuBackend, UnsupportedNodesFallBackToReference) {
    const Context context = ii_test::test_context();
    GraphBuilder b;
    const Var x = b.input("x", DType::kI32, {4});
    b.output("y", cumsum(x * x, 0));  // integer arithmetic and cumsum are not accelerated
    const Program program = Model::from_builder(context, std::move(b)).value().compile().value();
    EXPECT_NE(program.describe().find("ii-ref"), std::string::npos);
}

}  // namespace
}  // namespace ii::cpu
