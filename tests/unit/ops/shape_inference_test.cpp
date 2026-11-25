// SPDX-License-Identifier: MIT
// Shape and type inference of every OpSet v1 operation, written in the IR text format: parsing verifies that the
// declared result types equal the inferred ones.
#include "ir/builder.h"
#include "ir/text.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <set>
#include <string>

namespace ii::ir {
namespace {

using ::testing::HasSubstr;

struct Case {
    const char* name;
    const char* body;   ///< statements inside "graph { ... }"
    const char* error;  ///< nullptr: must succeed; otherwise a substring of the expected error
};

const Case kCases[] = {
    // Elementwise
    {"unary", "input %x : f32[N, 3]\n %y : f32[N, 3] = exp(%x)", nullptr},
    {"unary_int_rejected", "input %x : i32[3]\n %y : i32[3] = exp(%x)", "must be a float type"},
    {"relu_int", "input %x : i32[3]\n %y : i32[3] = relu(%x)", nullptr},
    {"abs_unsigned", "input %x : u8[3]\n %y : u8[3] = abs(%x)", nullptr},
    {"neg_unsigned_rejected", "input %x : u8[3]\n %y : u8[3] = neg(%x)", "signed"},
    {"not", "input %x : bool[3]\n %y : bool[3] = not(%x)", nullptr},
    {"binary_broadcast", "input %a : f32[N, 1, 4]\n input %b : f32[3, 1]\n %y : f32[N, 3, 4] = add(%a, %b)", nullptr},
    {"binary_symbolic", "input %a : f32[N, S]\n input %b : f32[S]\n %y : f32[N, S] = mul(%a, %b)", nullptr},
    {"binary_symbolic_mismatch", "input %a : f32[N]\n input %b : f32[M]\n %y : f32[N] = add(%a, %b)",
     "not broadcastable"},
    {"binary_dtype_mismatch", "input %a : f32[2]\n input %b : f16[2]\n %y : f32[2] = add(%a, %b)", "same dtype"},
    {"compare", "input %a : i64[2]\n input %b : i64[]\n %y : bool[2] = lt(%a, %b)", nullptr},
    {"eq_bool", "input %a : bool[2]\n %y : bool[2] = eq(%a, %a)", nullptr},
    {"logical", "input %a : bool[2]\n %y : bool[2] = xor(%a, %a)", nullptr},
    {"mod", "input %a : i32[2]\n %y : i32[2] = mod(%a, %a) {fmod = true}", nullptr},
    {"where", "input %c : bool[N, 1]\n input %a : f32[3]\n input %b : f32[]\n %y : f32[N, 3] = where(%c, %a, %b)",
     nullptr},
    {"cast", "input %x : f32[2]\n %y : bf16[2] = cast(%x) {to = bf16}", nullptr},
    {"cast_quantized_rejected", "input %x : f32[32]\n %y : q4_0[32] = cast(%x) {to = q4_0}", "non-quantized"},
    {"clamp", "input %x : i32[2]\n %y : i32[2] = clamp(%x) {min = 0, max = 6}", nullptr},
    {"clamp_bad", "input %x : f32[2]\n %y : f32[2] = clamp(%x) {min = 1, max = 0}", "min <= max"},
    // Reductions
    {"reduce_sum", "input %x : f32[N, 3, 4]\n %y : f32[N, 4] = reduce_sum(%x) {axes = [1]}", nullptr},
    {"reduce_keepdims",
     "input %x : f32[N, 3, 4]\n %y : f32[N, 1, 1] = reduce_max(%x) {axes = [-1, 1], keepdims = true}", nullptr},
    {"reduce_all", "input %x : f32[N, 3]\n %y : f32[] = reduce_mean(%x)", nullptr},
    {"reduce_dup_axes", "input %x : f32[2, 3]\n %y : f32[2] = reduce_sum(%x) {axes = [1, -1]}", "repeated"},
    {"reduce_min_prod",
     "input %x : i32[2, 3]\n %a : i32[2] = reduce_min(%x) {axes = [1]}\n %b : i32[3] = reduce_prod(%x) {axes = [0]}",
     nullptr},
    {"mean_int_rejected", "input %x : i32[2]\n %y : i32[] = reduce_mean(%x)", "float"},
    {"argmax", "input %x : f32[N, 5]\n %y : i64[N, 1] = argmax(%x) {axis = 1, keepdims = true}", nullptr},
    {"argmin", "input %x : f16[N, 5]\n %y : i64[5] = argmin(%x) {select_last_index = true}", nullptr},
    {"topk", "input %x : f32[N, 10]\n %v : f32[N, 3], %i : i64[N, 3] = topk(%x) {k = 3}", nullptr},
    {"topk_axis0", "input %x : i32[5, 4]\n %v : i32[2, 4], %i : i64[2, 4] = topk(%x) {k = 2, axis = 0}", nullptr},
    {"topk_too_large", "input %x : f32[2, 3]\n %v : f32[2, 4], %i : i64[2, 4] = topk(%x) {k = 4}", "exceeds"},
    // Image
    {"resize", "input %x : f32[N, 3, H, W]\n %y : f32[N, 3, 2*H, 2*W] = resize(%x) {shape = [N, 3, 2*H, 2*W]}",
     nullptr},
    {"resize_linear_int_rejected",
     "input %x : i32[1, 4]\n %y : i32[1, 8] = resize(%x) {shape = [1, 8], mode = \"linear\"}", "float"},
    {"resize_rank_mismatch", "input %x : f32[1, 4]\n %y : f32[8] = resize(%x) {shape = [8]}", "rank"},
    {"resize_bad_mode", "input %x : f32[4]\n %y : f32[8] = resize(%x) {shape = [8], mode = \"area\"}", "mode"},
    // matmul
    {"matmul", "input %a : f32[B, M, K]\n input %b : f32[K, 7]\n %y : f32[B, M, 7] = matmul(%a, %b)", nullptr},
    {"matmul_transposes",
     "input %a : f32[4, 3]\n input %b : f32[2, 5, 4]\n %y : f32[2, 3, 5] = matmul(%a, %b) {transpose_a = true, "
     "transpose_b = true}",
     nullptr},
    {"matmul_vector", "input %a : f32[4]\n input %b : f32[2, 4, 3]\n %y : f32[2, 3] = matmul(%a, %b)", nullptr},
    {"matmul_vector_vector", "input %a : f32[4]\n input %b : f32[4]\n %y : f32[] = matmul(%a, %b)", nullptr},
    {"matmul_mismatch", "input %a : f32[2, 3]\n input %b : f32[4, 5]\n %y : f32[2, 5] = matmul(%a, %b)",
     "inner dimensions differ"},
    {"matmul_quantized",
     "input %a : f32[N, 64]\n input %b : q4_0[8, 64]\n %y : f32[N, 8] = matmul(%a, %b) "
     "{transpose_b = true}",
     nullptr},
    // conv / pool
    {"conv", "input %x : f32[N, 3, H, W]\n input %w : f32[8, 3, 3, 3]\n %y : f32[N, 8, H - 2, W - 2] = conv(%x, %w)",
     nullptr},
    {"conv_strided_padded",
     "input %x : f32[1, 4, 224, 224]\n input %w : f32[8, 2, 3, 3]\n input %b : f32[8]\n %y : f32[1, 8, 112, 112] = "
     "conv(%x, %w, %b) {groups = 2, pads = [1, 1, 1, 1], strides = [2, 2]}",
     nullptr},
    {"conv_symbolic_strided",
     "input %x : f32[1, 3, H]\n input %w : f32[2, 3, 3]\n %y : f32[1, 2, (H - 1) / 2 + 1] = conv(%x, %w) {pads = "
     "[1, 1], strides = [2]}",
     nullptr},
    {"conv_same",
     "input %x : f32[1, 1, H]\n input %w : f32[1, 1, 4]\n %y : f32[1, 1, (H + 2) / 3] = conv(%x, %w) "
     "{auto_pad = \"same_upper\", strides = [3]}",
     nullptr},
    {"conv_bad_channels", "input %x : f32[1, 3, 8]\n input %w : f32[2, 2, 3]\n %y : f32[1, 2, 6] = conv(%x, %w)",
     "input channels"},
    {"conv_transpose",
     "input %x : f32[1, 4, 5, 5]\n input %w : f32[4, 3, 3, 3]\n %y : f32[1, 3, 11, 11] = conv_transpose(%x, %w) "
     "{output_padding = [0, 0], pads = [0, 0, 0, 0], strides = [2, 2]}",
     nullptr},
    {"max_pool",
     "input %x : f32[1, 2, 5, 5]\n %y : f32[1, 2, 3, 3] = max_pool(%x) {ceil_mode = true, kernel = [2, 2], "
     "strides = [2, 2]}",
     nullptr},
    {"avg_pool", "input %x : f32[1, 2, 6]\n %y : f32[1, 2, 6] = avg_pool(%x) {kernel = [3], pads = [1, 1]}", nullptr},
    {"pool_bad_pad", "input %x : f32[1, 2, 6]\n %y : f32[1, 2, 6] = max_pool(%x) {kernel = [2], pads = [2, 0]}",
     "padding must be smaller"},
    // Data movement
    {"reshape_infer", "input %x : f32[N, 6]\n %y : f32[N, 2, 3] = reshape(%x) {shape = [N, -1, 3]}", nullptr},
    {"reshape_symbolic", "input %x : f32[N, S, 8]\n %y : f32[N*S, 8] = reshape(%x) {shape = [N*S, 8]}", nullptr},
    {"reshape_infer_symbolic", "input %x : f32[N, S, 8]\n %y : f32[N*S*2, 4] = reshape(%x) {shape = [-1, 4]}", nullptr},
    {"reshape_bad", "input %x : f32[2, 3]\n %y : f32[4] = reshape(%x) {shape = [4]}", "cannot reshape"},
    {"permute", "input %x : f32[A, B, C]\n %y : f32[C, A, B] = permute(%x) {perm = [2, 0, 1]}", nullptr},
    {"slice", "input %x : f32[10, 8]\n %y : f32[3, 8] = slice(%x) {axes = [0], ends = [-1], starts = [3], steps = [2]}",
     nullptr},
    {"slice_negative_step",
     "input %x : f32[10]\n %y : f32[5] = slice(%x) {axes = [0], ends = [-100], starts = [-1], "
     "steps = [-2]}",
     nullptr},
    {"slice_symbolic",
     "input %x : f32[S, 4]\n %y : f32[S - 1, 2] = slice(%x) {axes = [0, 1], ends = [2147483647, 4], "
     "starts = [1, 2]}",
     nullptr},
    {"expand", "input %x : f32[3, 1]\n %y : f32[N, 3, 4] = expand(%x) {shape = [N, 1, 4]}", nullptr},
    {"concat", "input %a : f32[N, 2]\n input %b : f32[N, M]\n %y : f32[N, M + 2] = concat(%a, %b) {axis = -1}",
     nullptr},
    {"concat_mismatch", "input %a : f32[N, 2]\n input %b : f32[M, 2]\n %y : f32[N, 4] = concat(%a, %b) {axis = 1}",
     "differ outside"},
    {"split", "input %x : f32[N, 5]\n %a : f32[N, 2], %b : f32[N, 3] = split(%x) {axis = 1, sizes = [2, 3]}", nullptr},
    {"split_bad", "input %x : f32[N, 5]\n %a : f32[N, 2] = split(%x) {axis = 1, sizes = [2]}", "sum to 2"},
    {"pad", "input %x : f32[N, 3]\n %y : f32[N + 1, 1] = pad(%x) {mode = \"edge\", pads = [1, -1, 0, -1]}", nullptr},
    {"pad_reflect_bad", "input %x : f32[2]\n %y : f32[6] = pad(%x) {mode = \"reflect\", pads = [2, 2]}",
     "smaller than the dimension"},
    {"gather", "input %d : f32[A, B, C]\n input %i : i32[2, 3]\n %y : f32[A, 2, 3, C] = gather(%d, %i) {axis = 1}",
     nullptr},
    {"gather_bad_index", "input %d : f32[4]\n input %i : f32[2]\n %y : f32[2] = gather(%d, %i)", "i32 or i64"},
    {"scatter",
     "input %d : f32[4, 3]\n input %i : i64[2, 3]\n input %u : f32[2, 3]\n %y : f32[4, 3] = scatter(%d, "
     "%i, %u) {reduction = \"add\"}",
     nullptr},
    {"tile", "input %x : f32[N, 2]\n %y : f32[N*3, 2] = tile(%x) {repeats = [3, 1]}", nullptr},
    {"flip", "input %x : f32[N, 2]\n %y : f32[N, 2] = flip(%x) {axes = [0, 1]}", nullptr},
    {"cumsum", "input %x : i32[N, 2]\n %y : i32[N, 2] = cumsum(%x) {axis = 0, reverse = true}", nullptr},
    // Creation
    {"iota", "%y : f32[S, 3] = iota() {axis = 1, dtype = f32, shape = [S, 3], start = 1, step = 0.5}", nullptr},
    {"fill", "%y : bool[2] = fill() {dtype = bool, shape = [2], value = 1}", nullptr},
    {"random", "%y : f16[4] = random_normal() {dtype = f16, seed = 7, shape = [4]}", nullptr},
    {"random_int_rejected", "%y : i32[4] = random_uniform() {dtype = i32, shape = [4]}", "float type"},
    // Quantization
    {"quantize",
     "input %x : f32[N, 3]\n input %s : f32[3]\n input %z : u8[3]\n %y : u8[N, 3] = quantize(%x, %s, %z) "
     "{dtype = u8}",
     nullptr},
    {"quantize_bad_axis", "input %x : f32[N, 3]\n input %s : f32[4]\n %y : i8[N, 3] = quantize(%x, %s)",
     "per-axis scale"},
    {"dequantize", "input %x : i8[2, 3]\n input %s : f16[]\n %y : f16[2, 3] = dequantize(%x, %s)", nullptr},
    {"dequantize_block", "input %x : q8_0[4, 64]\n %y : f32[4, 64] = dequantize(%x)", nullptr},
    // Composites
    {"gelu", "input %x : bf16[3]\n %y : bf16[3] = gelu(%x)", nullptr},
    {"gelu_bad", "input %x : f32[3]\n %y : f32[3] = gelu(%x) {approximate = \"fast\"}", "none or tanh"},
    {"silu", "input %x : f32[3]\n %y : f32[3] = silu(%x)", nullptr},
    {"log_softmax", "input %x : f32[N, 10]\n %y : f32[N, 10] = log_softmax(%x) {axis = 0}", nullptr},
    {"softmax", "input %x : f32[N, 10]\n %y : f32[N, 10] = softmax(%x)", nullptr},
    {"layer_norm",
     "input %x : f32[N, S, 8]\n input %g : f32[S, 8]\n %y : f32[N, S, 8] = layer_norm(%x, %g, _) "
     "{axis = 1}",
     nullptr},
    {"layer_norm_bad_scale", "input %x : f32[N, 8]\n input %g : f32[4]\n %y : f32[N, 8] = layer_norm(%x, %g)",
     "must have shape [8]"},
    {"rms_norm", "input %x : f16[N, 8]\n input %g : f16[8]\n %y : f16[N, 8] = rms_norm(%x, %g)", nullptr},
    {"linear",
     "input %x : f32[N, S, 4]\n input %w : f32[6, 4]\n input %b : f32[6]\n %y : f32[N, S, 6] = linear(%x, "
     "%w, %b)",
     nullptr},
    {"linear_mismatch", "input %x : f32[N, 5]\n input %w : f32[6, 4]\n %y : f32[N, 6] = linear(%x, %w)",
     "input features"},
    {"sdpa_gqa",
     "input %q : f32[B, 8, S, 16]\n input %k : f32[B, 2, T, 16]\n input %v : f32[B, 2, T, 32]\n input %m : bool[S, "
     "T]\n %y : f32[B, 8, S, 32] = sdpa(%q, %k, %v, %m) {causal = true}",
     nullptr},
    {"sdpa_bad_heads",
     "input %q : f32[1, 3, 4, 8]\n input %k : f32[1, 2, 4, 8]\n input %v : f32[1, 2, 4, 8]\n %y : "
     "f32[1, 3, 4, 8] = sdpa(%q, %k, %v)",
     "multiple of key/value heads"},
};

class ShapeInferenceTest : public ::testing::TestWithParam<Case> {};

TEST_P(ShapeInferenceTest, Infers) {
    const Case& c = GetParam();
    const std::string text = std::string("graph {\n ") + c.body + "\n}\n";
    const Result<std::unique_ptr<Graph>> graph = parse_graph(text);
    if (c.error == nullptr) {
        EXPECT_TRUE(graph.ok()) << graph.status().to_string() << "\n" << text;
    } else {
        ASSERT_FALSE(graph.ok()) << text;
        EXPECT_THAT(graph.status().to_string(), HasSubstr(c.error));
    }
}

INSTANTIATE_TEST_SUITE_P(OpSet, ShapeInferenceTest, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<Case>& param_info) {
                             return std::string(param_info.param.name);
                         });

TEST(OpSetTest, EveryOperationHasAnInferenceCase) {
    std::set<std::string, std::less<>> covered;
    for (const Case& c : kCases) {
        const std::string body = c.body;
        for (const auto& schema : default_op_registry().all()) {
            if (body.find(" = " + std::string(schema->name) + "(") != std::string::npos) {
                covered.insert(std::string(schema->name));
            }
        }
    }
    for (const auto& schema : default_op_registry().all()) {
        if (schema->has_flag(kOpInternal)) {
            continue;  // covered by the fusion tests
        }
        const bool unary_or_binary = schema->category == "unary" || schema->category == "binary";
        if (!unary_or_binary) {
            EXPECT_TRUE(covered.contains(schema->name)) << "no shape inference case for '" << schema->name << "'";
        }
        EXPECT_FALSE(schema->doc.empty()) << schema->name;
    }
}

}  // namespace
}  // namespace ii::ir
