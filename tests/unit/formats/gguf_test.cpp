// SPDX-License-Identifier: MIT
// GGUF and safetensors readers: files written by gguf-py and safetensors (scripts/gen_format_samples.py), the GGUF
// writer round trip and malformed input.
#include <ii/gguf.h>

#include "formats/safetensors.h"
#include "support/test_util.h"
#include "tensor/quant.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace ii {
namespace {

using ::testing::HasSubstr;

std::filesystem::path sample(const char* name) {
    return std::filesystem::path(ii_test::test_data_dir()) / "formats" / name;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> chars((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(chars.size());
    std::memcpy(bytes.data(), chars.data(), chars.size());
    return bytes;
}

TEST(GgufTest, ReadsFilesWrittenByGgufPy) {
    const auto file = gguf::File::open(sample("sample.gguf"));
    ASSERT_TRUE(file.ok()) << file.status().to_string();
    EXPECT_EQ(file->version(), 3u);
    EXPECT_EQ(file->get_string("general.architecture"), "llama");
    EXPECT_EQ(file->get_string("general.name"), "sample");
    EXPECT_EQ(file->get_int("llama.block_count"), 2);
    EXPECT_EQ(file->get_float("llama.rope.freq_base"), 10000.0);
    EXPECT_EQ(file->find("sample.flag")->as_bool(), true);
    EXPECT_EQ(file->get_int("sample.negative"), -5);
    EXPECT_EQ(*file->find("sample.strings")->as_strings(), (std::vector<std::string>{"a", "bc", ""}));
    EXPECT_EQ(*file->find("sample.ints")->as_ints(), (std::vector<std::int64_t>{1, -2, 3}));
    EXPECT_EQ(*file->find("sample.floats")->as_floats(), (std::vector<double>{0.5, -1.25}));
    EXPECT_EQ(file->get_string("sample.unicode"), "привет ✓");
    EXPECT_EQ(file->find("missing"), nullptr);

    ASSERT_EQ(file->tensors().size(), 4u);
    const Tensor f32 = file->tensor("f32").value();
    EXPECT_EQ(f32.dtype(), DType::kF32);
    EXPECT_EQ(f32.shape(), (Shape{3, 4}));
    EXPECT_EQ(f32.values<float>()[11], 11.0f);
    const Tensor f16 = file->tensor("f16").value();
    EXPECT_EQ(f16.shape(), (Shape{2, 4}));
    EXPECT_EQ(f16.values<ii::f16>()[3].to_float(), 0.75f);
    const Tensor q8 = file->tensor("q8_0").value();
    EXPECT_EQ(q8.dtype(), DType::kQ8_0);
    EXPECT_EQ(q8.shape(), (Shape{2, 64}));
    EXPECT_EQ(q8.nbytes(), 4u * 34u);
    EXPECT_EQ(file->tensor("i32").value().values<std::int32_t>()[1], -8);
    EXPECT_FALSE(file->tensor("nope").ok());
    // Tensors are views of the mapping and keep it alive.
    const Tensor kept = [] { return gguf::File::open(sample("sample.gguf")).value().tensor("f32").value(); }();
    EXPECT_EQ(kept.values<float>()[5], 5.0f);
}

TEST(GgufTest, WriterRoundTrips) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "ii_gguf_roundtrip.gguf";
    std::vector<float> values(512);
    std::mt19937 rng(3);
    std::normal_distribution<float> normal;
    for (float& v : values) {
        v = normal(rng);
    }
    {
        gguf::Writer writer;
        writer.set("general.architecture", gguf::Value::string("llama"));
        writer.set("llama.context_length", gguf::Value::u32(4096));
        writer.set("tokenizer.ggml.tokens", gguf::Value::strings({"<s>", "</s>", "hello"}));
        writer.set("tokenizer.ggml.scores", gguf::Value::floats({0.0, -1.5, -2.0}));
        writer.set("llama.context_length", gguf::Value::u32(2048));  // replaces, keeps the position
        writer.add_tensor("w", Tensor::from(std::span<const float>(values), Shape{2, 256}).value());
        writer.add_tensor("q", DType::kQ4_K, Shape{2, 256}, [&]() -> Result<Tensor> {
            II_ASSIGN_OR_RETURN(Tensor t, Tensor::empty(DType::kQ4_K, Shape{2, 256}));
            quant::quantize(DType::kQ4_K, values.data(), t.data(), 512);
            return t;
        });
        writer.set_alignment(64);
        ASSERT_TRUE(writer.write(path).ok());
    }
    const auto file = gguf::File::open(path);
    ASSERT_TRUE(file.ok()) << file.status().to_string();
    EXPECT_EQ(file->alignment(), 64u);
    EXPECT_EQ(file->metadata()[1].first, "llama.context_length");
    EXPECT_EQ(file->get_int("llama.context_length"), 2048);
    EXPECT_EQ(file->find("tokenizer.ggml.tokens")->array_size(), 3u);
    EXPECT_EQ((*file->find("tokenizer.ggml.scores")->as_floats())[1], -1.5);
    const Tensor w = file->tensor("w").value();
    EXPECT_EQ(w.values<float>()[300], values[300]);
    const Tensor q = file->tensor("q").value();
    EXPECT_EQ(q.dtype(), DType::kQ4_K);
    std::vector<float> back(512);
    quant::dequantize(DType::kQ4_K, q.data(), back.data(), 512);
    EXPECT_NEAR(back[100], values[100], 0.3f);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(q.data()) % 64, 0u);
    std::filesystem::remove(path);
}

TEST(GgufTest, RejectsMalformedFiles) {
    const std::vector<std::byte> good = read_file(sample("sample.gguf"));
    EXPECT_THAT(gguf::File::parse({}).status().to_string(), HasSubstr("end of file"));
    std::vector<std::byte> bad = good;
    bad[0] = std::byte{'X'};
    EXPECT_THAT(gguf::File::parse(bad).status().to_string(), HasSubstr("magic"));
    bad = good;
    bad[4] = std::byte{9};  // version 9
    EXPECT_THAT(gguf::File::parse(bad).status().to_string(), HasSubstr("version"));
    // Every truncation before the end of the tensor data fails cleanly (gguf-py pads the file after it).
    std::size_t data_end = 0;
    const gguf::File parsed = gguf::File::parse(good).value();
    for (const gguf::TensorInfo& info : parsed.tensors()) {
        data_end = std::max<std::size_t>(data_end, info.offset + info.nbytes);
    }
    for (std::size_t size = 0; size < data_end; size += 7) {
        EXPECT_FALSE(gguf::File::parse(std::span<const std::byte>(good.data(), size)).ok()) << size;
    }
    // A huge tensor count.
    bad = good;
    bad[8] = std::byte{0xFF};
    bad[13] = std::byte{0x7F};
    EXPECT_FALSE(gguf::File::parse(bad).ok());
}

TEST(SafetensorsTest, ReadsFilesWrittenBySafetensors) {
    const auto file = formats::SafetensorsFile::open(sample("sample.safetensors"));
    ASSERT_TRUE(file.ok()) << file.status().to_string();
    ASSERT_EQ(file->tensors().size(), 4u);
    ASSERT_EQ(file->metadata().size(), 1u);
    EXPECT_EQ(file->metadata()[0].second, "pt");
    const Tensor a = file->tensor(*file->find("a.weight")).value();
    EXPECT_EQ(a.shape(), (Shape{2, 3}));
    EXPECT_EQ(a.values<float>()[4], 4.0f);
    EXPECT_EQ(file->tensor(*file->find("b.bias")).value().values<f16>()[2].to_float(), 3.0f);
    EXPECT_EQ(file->tensor(*file->find("c.ids")).value().values<std::int64_t>()[3], 4);
    const Tensor scalar = file->tensor(*file->find("scalar")).value();
    EXPECT_EQ(scalar.rank(), 0);
    EXPECT_EQ(scalar.values<double>()[0], 2.5);

    const auto model = formats::SafetensorsModel::open(sample("sample.safetensors"));
    ASSERT_TRUE(model.ok());
    EXPECT_EQ(model->names().front(), "a.weight");
    EXPECT_TRUE(model->contains("c.ids"));
    EXPECT_FALSE(model->tensor("missing").ok());
}

TEST(SafetensorsTest, RejectsMalformedFiles) {
    const std::vector<std::byte> good = read_file(sample("sample.safetensors"));
    for (std::size_t size = 0; size < good.size(); size += 5) {
        EXPECT_FALSE(formats::SafetensorsFile::parse(std::span<const std::byte>(good.data(), size), nullptr).ok())
            << size;
    }
    std::vector<std::byte> bad = good;
    bad[7] = std::byte{1};  // header size beyond the file
    EXPECT_FALSE(formats::SafetensorsFile::parse(bad, nullptr).ok());
}

}  // namespace
}  // namespace ii
