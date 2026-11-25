// SPDX-License-Identifier: MIT
/// @file
/// Reading safetensors files (Hugging Face): an 8-byte little-endian header size, a JSON header
/// `{"name": {"dtype": "F32", "shape": [..], "data_offsets": [begin, end]}, "__metadata__": {..}}` and the raw data.
/// Tensors are returned as zero-copy views of the memory-mapped file when their data is aligned for the dtype.
#pragma once

#include <ii/status.h>
#include <ii/tensor.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ii::formats {

struct SafetensorsTensor {
    std::string name;
    std::string dtype_name;         ///< as written ("F32", "BF16", "F8_E4M3", ...)
    DType dtype = DType::kInvalid;  ///< kInvalid for types ii does not support
    Shape shape;
    std::size_t offset = 0;  ///< from the start of the file
    std::size_t nbytes = 0;
};

class SafetensorsFile {
public:
    static Result<SafetensorsFile> open(const std::filesystem::path& path);
    /// Parses a file held in memory (`keep_alive` owns `data`; null: tensors are copied).
    static Result<SafetensorsFile> parse(std::span<const std::byte> data, std::shared_ptr<const void> keep_alive);

    const std::vector<SafetensorsTensor>& tensors() const noexcept { return tensors_; }
    const std::vector<std::pair<std::string, std::string>>& metadata() const noexcept { return metadata_; }
    const SafetensorsTensor* find(std::string_view name) const noexcept;
    Result<Tensor> tensor(const SafetensorsTensor& info) const;

private:
    std::shared_ptr<const void> keep_alive_;
    std::span<const std::byte> data_;
    std::vector<SafetensorsTensor> tensors_;
    std::vector<std::pair<std::string, std::string>> metadata_;
};

/// The tensors of a Hugging Face model: one `.safetensors` file, or a directory with `model.safetensors` or the
/// shards listed in `model.safetensors.index.json`.
class SafetensorsModel {
public:
    static Result<SafetensorsModel> open(const std::filesystem::path& path);

    /// Every tensor name (sorted).
    std::vector<std::string> names() const;
    bool contains(std::string_view name) const noexcept;
    const SafetensorsTensor* find(std::string_view name) const noexcept;
    Result<Tensor> tensor(std::string_view name) const;

private:
    std::vector<SafetensorsFile> files_;
};

}  // namespace ii::formats
