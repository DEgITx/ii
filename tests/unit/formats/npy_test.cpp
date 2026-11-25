// SPDX-License-Identifier: MIT
// NumPy .npy / .npz reading and writing, DEFLATE decompression and CRC-32.
#include <ii/io.h>

#include "core/compress.h"
#include "support/test_util.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>

namespace ii {
namespace {

std::filesystem::path data_path(const std::string& name) {
    return std::filesystem::path(ii_test::test_data_dir()) / "npy" / name;
}

std::filesystem::path temp_path(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("ii_npy_test_" + name);
}

std::vector<std::byte> from_hex(const std::string& hex) {
    std::vector<std::byte> bytes;
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        bytes.push_back(static_cast<std::byte>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return bytes;
}

TEST(NpyTest, RoundTripsEveryDtype) {
    const std::vector<std::pair<DType, Shape>> cases = {
        {DType::kF32, {2, 3}}, {DType::kF64, {4}},    {DType::kF16, {3, 1, 2}}, {DType::kI8, {5}},
        {DType::kU8, {2, 2}},  {DType::kI16, {3}},    {DType::kU16, {3}},       {DType::kI32, {1, 7}},
        {DType::kU32, {2}},    {DType::kI64, {2, 2}}, {DType::kU64, {3}},       {DType::kBool, {6}},
        {DType::kF32, {}},     {DType::kI64, {0, 3}},
    };
    std::uint32_t seed = 1;
    for (const auto& [dtype, shape] : cases) {
        SCOPED_TRACE(std::string(to_string(dtype)) + shape.to_string());
        const Tensor tensor = ii_test::random_tensor(dtype, shape, seed++, -100, 100);
        const Result<std::vector<std::byte>> bytes = serialize_npy(tensor);
        ASSERT_TRUE(bytes.ok()) << bytes.status().to_string();
        EXPECT_EQ((bytes->size() - tensor.nbytes()) % 64, 0u) << "data must start 64-byte aligned";
        const Result<Tensor> parsed = parse_npy(*bytes);
        ASSERT_TRUE(parsed.ok()) << parsed.status().to_string();
        EXPECT_EQ(parsed->dtype(), dtype);
        EXPECT_EQ(parsed->shape(), shape);
        EXPECT_EQ(ii_test::compare(*parsed, tensor, {0, 0}), "");
    }
    // bfloat16 has no NumPy type.
    EXPECT_FALSE(serialize_npy(ii_test::random_tensor(DType::kBF16, Shape{2}, 1)).ok());
}

TEST(NpyTest, ReadsFilesWrittenByNumPy) {
    const Result<Tensor> fortran = load_npy(data_path("fortran.npy"));
    ASSERT_TRUE(fortran.ok()) << fortran.status().to_string();
    ASSERT_EQ(fortran->shape(), (Shape{2, 3}));
    for (std::int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(fortran->values<std::int32_t>()[static_cast<std::size_t>(i)], i);  // C order after conversion
    }
    const Result<Tensor> half = load_npy(data_path("f16.npy"));
    ASSERT_TRUE(half.ok()) << half.status().to_string();
    EXPECT_EQ(half->dtype(), DType::kF16);
    EXPECT_EQ(half->values<f16>()[1].to_float(), -1.25f);
    EXPECT_EQ(half->values<f16>()[2].to_float(), 65504.0f);
}

TEST(NpyTest, ReadsStoredAndCompressedArchives) {
    const Result<std::vector<NamedTensor>> stored = load_npz(data_path("stored.npz"));
    ASSERT_TRUE(stored.ok()) << stored.status().to_string();
    ASSERT_EQ(stored->size(), 2u);
    EXPECT_EQ((*stored)[0].first, "x");
    EXPECT_EQ((*stored)[0].second.values<double>()[4], 1.0);
    EXPECT_EQ((*stored)[1].first, "mask");
    EXPECT_EQ((*stored)[1].second.dtype(), DType::kBool);

    const Result<std::vector<NamedTensor>> compressed = load_npz(data_path("compressed.npz"));
    ASSERT_TRUE(compressed.ok()) << compressed.status().to_string();
    std::map<std::string, Tensor> arrays(compressed->begin(), compressed->end());
    ASSERT_EQ(arrays.size(), 3u);
    EXPECT_EQ(arrays["a"].shape(), (Shape{3, 4}));
    EXPECT_EQ(arrays["a"].values<float>()[11], 11.0f);
    EXPECT_EQ(arrays["b"].values<std::int64_t>()[1], -2);
    EXPECT_EQ(arrays["c"].rank(), 0);
}

TEST(NpyTest, WritesArchivesThatReadBack) {
    const std::vector<NamedTensor> arrays = {
        {"logits", ii_test::random_tensor(DType::kF32, Shape{2, 5}, 7)},
        {"ids", ii_test::random_tensor(DType::kI64, Shape{3}, 8, 0, 50)},
        {"empty", ii_test::random_tensor(DType::kF32, Shape{0}, 9)},
    };
    const std::filesystem::path path = temp_path("roundtrip.npz");
    ASSERT_TRUE(save_npz(path, arrays).ok());
    const Result<std::vector<NamedTensor>> loaded = load_npz(path);
    ASSERT_TRUE(loaded.ok()) << loaded.status().to_string();
    ASSERT_EQ(loaded->size(), arrays.size());
    for (std::size_t i = 0; i < arrays.size(); ++i) {
        EXPECT_EQ((*loaded)[i].first, arrays[i].first);
        EXPECT_EQ(ii_test::compare((*loaded)[i].second, arrays[i].second, {0, 0}), "");
    }
    std::filesystem::remove(path);
}

TEST(NpyTest, RejectsMalformedFiles) {
    const std::vector<std::byte> good = serialize_npy(ii_test::random_tensor(DType::kF32, Shape{4}, 3)).value();
    for (std::size_t size = 0; size < good.size(); size += 7) {
        EXPECT_FALSE(parse_npy(std::span<const std::byte>(good.data(), size)).ok()) << size;
    }
    using namespace std::string_literals;
    const std::string big_endian =
        "\x93NUMPY\x01\x00\x40\x00{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }         \n\x01\x02\x03\x04"s;
    EXPECT_FALSE(parse_npy(std::as_bytes(std::span<const char>(big_endian.data(), big_endian.size()))).ok());
    // Every truncation of an archive fails cleanly.
    std::ifstream file(data_path("compressed.npz"), std::ios::binary);
    const std::string archive((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    for (std::size_t size = 0; size < archive.size(); size += 13) {
        const std::filesystem::path path = temp_path("truncated.npz");
        std::ofstream(path, std::ios::binary).write(archive.data(), static_cast<std::streamsize>(size));
        EXPECT_FALSE(load_npz(path).ok()) << size;
    }
    std::filesystem::remove(temp_path("truncated.npz"));
}

TEST(CompressTest, InflatesStoredFixedAndDynamicBlocks) {
    std::ifstream in(data_path("deflate_cases.txt"));
    std::map<std::string, std::vector<std::byte>> streams;
    std::string name;
    std::string hex;
    while (in >> name) {
        if (name[0] == '#') {
            std::getline(in, hex);
            continue;
        }
        in >> hex;
        streams[name] = from_hex(hex);
    }
    ASSERT_EQ(streams.size(), 4u);
    const std::vector<std::byte>& plain = streams["plain"];
    for (const char* kind : {"stored", "fixed", "dynamic"}) {
        SCOPED_TRACE(kind);
        const Result<std::vector<std::byte>> out = inflate(streams[kind], plain.size());
        ASSERT_TRUE(out.ok()) << out.status().to_string();
        EXPECT_EQ(*out, plain);
        // The output limit is enforced; truncated and corrupted streams fail without crashing.
        EXPECT_FALSE(inflate(streams[kind], plain.size() - 1).ok());
        const std::vector<std::byte>& stream = streams[kind];
        for (std::size_t size = 0; size + 1 < stream.size(); size += 17) {
            EXPECT_FALSE(inflate(std::span<const std::byte>(stream.data(), size), plain.size()).ok());
        }
        std::vector<std::byte> corrupted = stream;
        for (std::size_t i = 0; i < corrupted.size(); i += 11) {
            corrupted[i] ^= std::byte{0x5A};
            (void)inflate(corrupted, plain.size());  // must not crash
        }
    }
}

TEST(CompressTest, Crc32) {
    const std::string text = "123456789";
    EXPECT_EQ(crc32(std::as_bytes(std::span<const char>(text.data(), text.size()))), 0xCBF43926u);
    EXPECT_EQ(crc32({}), 0u);
}

}  // namespace
}  // namespace ii
