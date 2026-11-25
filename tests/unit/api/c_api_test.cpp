// SPDX-License-Identifier: MIT
#include <ii/c_api.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

using ::testing::HasSubstr;

constexpr const char* kModel = R"(graph {
  input %x : f32[N, 2]
  const %k : f32[] = f32[] {3}
  %y : f32[N, 2] = mul(%x, %k)
  output y = %y
})";

ii_tensor make_tensor(float* data, std::int64_t rows) {
    ii_tensor t;
    std::memset(&t, 0, sizeof t);
    t.dtype = II_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = 2;
    t.data = data;
    return t;
}

TEST(CApiTest, RunsAModel) {
    ii_context_options options{sizeof(ii_context_options), 1};
    ii_context* context = nullptr;
    ASSERT_EQ(ii_context_create(&options, &context), II_OK);
    ii_model* model = nullptr;
    ASSERT_EQ(ii_model_from_text(context, kModel, std::strlen(kModel), &model), II_OK) << ii_last_error();
    ii_compile_options compile{sizeof(ii_compile_options), "ii-ref", 1};
    ii_program* program = nullptr;
    ASSERT_EQ(ii_model_compile(model, &compile, &program), II_OK) << ii_last_error();
    EXPECT_EQ(ii_program_num_inputs(program), 1);
    EXPECT_STREQ(ii_program_input_name(program, 0), "x");
    EXPECT_STREQ(ii_program_output_name(program, 0), "y");
    EXPECT_EQ(ii_program_input_name(program, 5), nullptr);
    ii_session* session = nullptr;
    ASSERT_EQ(ii_session_create(program, &session), II_OK);
    // Handles keep their parents alive.
    ii_program_destroy(program);
    ii_model_destroy(model);
    ii_context_destroy(context);

    float input[6] = {1, 2, 3, 4, 5, 6};
    const ii_tensor x = make_tensor(input, 3);
    ASSERT_EQ(ii_session_set_input(session, "x", &x), II_OK);
    ASSERT_EQ(ii_session_run(session), II_OK) << ii_last_error();
    ii_tensor y;
    ASSERT_EQ(ii_session_get_output(session, "y", &y), II_OK);
    EXPECT_EQ(y.dtype, II_DTYPE_F32);
    EXPECT_EQ(y.rank, 2);
    EXPECT_EQ(y.shape[0], 3);
    EXPECT_EQ(static_cast<const float*>(y.data)[5], 18.0f);

    float out[6] = {};
    const ii_tensor bound = make_tensor(out, 3);
    ASSERT_EQ(ii_session_set_output(session, "y", &bound), II_OK);
    ASSERT_EQ(ii_session_run(session), II_OK);
    EXPECT_EQ(out[0], 3.0f);
    ii_session_destroy(session);
}

TEST(CApiTest, ReportsErrors) {
    ii_context* context = nullptr;
    ASSERT_EQ(ii_context_create(nullptr, &context), II_OK);
    ii_model* model = nullptr;
    const char* bad = "graph { input %x : f32[2]\n %y : f32[2] = nope(%x) }";
    EXPECT_EQ(ii_model_from_text(context, bad, std::strlen(bad), &model), II_INVALID_ARGUMENT);
    EXPECT_EQ(model, nullptr);
    EXPECT_THAT(std::string(ii_last_error()), HasSubstr("unknown operation 'nope'"));
    EXPECT_STREQ(ii_status_name(II_INVALID_ARGUMENT), "InvalidArgument");
    EXPECT_EQ(ii_model_from_text(nullptr, bad, 1, &model), II_INVALID_ARGUMENT);

    ASSERT_EQ(ii_model_from_text(context, kModel, std::strlen(kModel), &model), II_OK);
    ii_compile_options compile{sizeof(ii_compile_options), "ii-ref,missing", 1};
    ii_program* program = nullptr;
    EXPECT_EQ(ii_model_compile(model, &compile, &program), II_NOT_FOUND);
    ASSERT_EQ(ii_model_compile(model, nullptr, &program), II_OK);
    ii_session* session = nullptr;
    ASSERT_EQ(ii_session_create(program, &session), II_OK);
    float data[4] = {};
    ii_tensor wrong = make_tensor(data, 2);
    wrong.dtype = II_DTYPE_I32;
    EXPECT_EQ(ii_session_set_input(session, "x", &wrong), II_INVALID_ARGUMENT);
    wrong.dtype = static_cast<ii_dtype>(99);
    EXPECT_EQ(ii_session_set_input(session, "x", &wrong), II_INVALID_ARGUMENT);
    EXPECT_EQ(ii_session_set_input(session, "x", nullptr), II_INVALID_ARGUMENT);
    EXPECT_EQ(ii_session_run(session), II_INVALID_ARGUMENT);
    EXPECT_THAT(std::string(ii_last_error()), HasSubstr("not set"));
    ii_session_destroy(session);
    ii_program_destroy(program);
    ii_model_destroy(model);
    ii_context_destroy(context);
    ii_context_destroy(nullptr);
    EXPECT_STRNE(ii_version(), "");
}

TEST(CApiTest, LoadsModelFiles) {
    ii_context* context = nullptr;
    ASSERT_EQ(ii_context_create(nullptr, &context), II_OK);
    // The ii text format through ii_model_load, from a file.
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "ii_c_api_model.ii.txt";
    {
        std::ofstream file(path, std::ios::binary);
        file << kModel;
    }
    ii_model* model = nullptr;
    const std::u8string utf8 = path.u8string();
    const std::string path_text(utf8.begin(), utf8.end());
    ASSERT_EQ(ii_model_load(context, path_text.c_str(), &model), II_OK) << ii_last_error();
    ii_model_destroy(model);
    std::filesystem::remove(path);
    model = nullptr;
    EXPECT_EQ(ii_model_load(context, path_text.c_str(), &model), II_NOT_FOUND);
    EXPECT_EQ(model, nullptr);
    // Garbage is not an ONNX model.
    const unsigned char garbage[16] = {0xFF, 0xFF, 0xFF, 0xFF};
    EXPECT_NE(ii_model_from_onnx(context, garbage, sizeof garbage, &model), II_OK);
    EXPECT_EQ(model, nullptr);
    EXPECT_EQ(ii_model_from_onnx(context, nullptr, 4, &model), II_INVALID_ARGUMENT);
    ii_context_destroy(context);
}

}  // namespace
