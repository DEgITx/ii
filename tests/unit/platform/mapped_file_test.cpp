// SPDX-License-Identifier: MIT
#include "core/platform/mapped_file.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace ii::platform {
namespace {

namespace fs = std::filesystem;

class MappedFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device random;
        dir_ = fs::temp_directory_path() / ("ii_mapped_file_test_" + std::to_string(random()));
        fs::create_directories(dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    fs::path write_file(const fs::path& name, const std::vector<char>& content) const {
        const fs::path path = dir_ / name;
        std::ofstream file(path, std::ios::binary);
        file.write(content.data(), static_cast<std::streamsize>(content.size()));
        return path;
    }

    fs::path dir_;
};

TEST_F(MappedFileTest, MapsFileContents) {
    std::vector<char> content(1 << 20);
    for (std::size_t i = 0; i < content.size(); ++i) {
        content[i] = static_cast<char>(i * 7 + 3);
    }
    const fs::path path = write_file("data.bin", content);

    const Result<MappedFile> file = MappedFile::open(path);
    ASSERT_TRUE(file.ok()) << file.status().to_string();
    ASSERT_EQ(file->size(), content.size());
    EXPECT_EQ(std::memcmp(file->data(), content.data(), content.size()), 0);
    EXPECT_EQ(file->bytes().size(), content.size());
}

TEST_F(MappedFileTest, UnicodePath) {
    const fs::path path = write_file(fs::path(u8"модель_模型.bin"), {'i', 'i'});
    const Result<MappedFile> file = MappedFile::open(path);
    ASSERT_TRUE(file.ok()) << file.status().to_string();
    ASSERT_EQ(file->size(), 2u);
    EXPECT_EQ(static_cast<const char*>(file->data())[1], 'i');
}

TEST_F(MappedFileTest, EmptyFile) {
    const fs::path path = write_file("empty.bin", {});
    const Result<MappedFile> file = MappedFile::open(path);
    ASSERT_TRUE(file.ok()) << file.status().to_string();
    EXPECT_EQ(file->size(), 0u);
    EXPECT_TRUE(file->bytes().empty());
}

TEST_F(MappedFileTest, MissingFile) {
    const Result<MappedFile> file = MappedFile::open(dir_ / "missing.bin");
    ASSERT_FALSE(file.ok());
    EXPECT_EQ(file.status().code(), StatusCode::kNotFound);
    EXPECT_NE(file.status().message().find("missing.bin"), std::string_view::npos);
}

TEST_F(MappedFileTest, MoveKeepsMappingAlive) {
    const fs::path path = write_file("move.bin", {'a', 'b', 'c'});
    Result<MappedFile> file = MappedFile::open(path);
    ASSERT_TRUE(file.ok());
    MappedFile moved = std::move(file).value();
    ASSERT_EQ(moved.size(), 3u);
    EXPECT_EQ(static_cast<const char*>(moved.data())[2], 'c');
    MappedFile target;
    target = std::move(moved);
    EXPECT_EQ(target.size(), 3u);
    EXPECT_EQ(moved.size(), 0u);  // NOLINT(bugprone-use-after-move)
}

}  // namespace
}  // namespace ii::platform
