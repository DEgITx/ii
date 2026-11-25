// SPDX-License-Identifier: MIT
#include "formats/safetensors.h"

#include "core/json.h"
#include "core/platform/mapped_file.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <set>
#include <sstream>

namespace ii::formats {

namespace {

constexpr std::uint64_t kMaxHeader = std::uint64_t{100} << 20;

DType dtype_from_name(std::string_view name) noexcept {
    struct Entry {
        std::string_view name;
        DType dtype;
    };
    constexpr Entry kTypes[] = {
        {"F64", DType::kF64},   {"F32", DType::kF32}, {"F16", DType::kF16}, {"BF16", DType::kBF16},
        {"I64", DType::kI64},   {"I32", DType::kI32}, {"I16", DType::kI16}, {"I8", DType::kI8},
        {"U64", DType::kU64},   {"U32", DType::kU32}, {"U16", DType::kU16}, {"U8", DType::kU8},
        {"BOOL", DType::kBool},
    };
    for (const Entry& entry : kTypes) {
        if (entry.name == name) {
            return entry.dtype;
        }
    }
    return DType::kInvalid;
}

}  // namespace

Result<SafetensorsFile> SafetensorsFile::open(const std::filesystem::path& path) {
    II_ASSIGN_OR_RETURN(platform::MappedFile mapped, platform::MappedFile::open(path));
    auto owner = std::make_shared<platform::MappedFile>(std::move(mapped));
    const std::span<const std::byte> bytes = owner->bytes();
    Result<SafetensorsFile> file = parse(bytes, std::move(owner));
    if (!file.ok()) {
        return Status(file.status()).add_context(path.string());
    }
    return file;
}

Result<SafetensorsFile> SafetensorsFile::parse(std::span<const std::byte> data,
                                               std::shared_ptr<const void> keep_alive) {
    if (data.size() < 8) {
        return errors::data_corrupted("not a safetensors file (too small)");
    }
    std::uint64_t header_size = 0;
    std::memcpy(&header_size, data.data(), 8);
    if (header_size > kMaxHeader || header_size > data.size() - 8) {
        return errors::data_corrupted("safetensors header of {} bytes exceeds the file", header_size);
    }
    const std::string_view text(reinterpret_cast<const char*>(data.data() + 8), static_cast<std::size_t>(header_size));
    II_ASSIGN_OR_RETURN(const Json header, Json::parse(text, 16));
    if (!header.is_object()) {
        return errors::data_corrupted("safetensors header is not a JSON object");
    }
    SafetensorsFile file;
    file.keep_alive_ = std::move(keep_alive);
    file.data_ = data;
    const std::size_t data_start = 8 + static_cast<std::size_t>(header_size);
    const std::size_t data_size = data.size() - data_start;
    for (const auto& [name, entry] : header.as_object()) {
        if (name == "__metadata__") {
            for (const auto& [key, value] : entry.as_object()) {
                file.metadata_.emplace_back(key, value.is_string() ? value.as_string() : std::string());
            }
            continue;
        }
        const Json* dtype = entry.find("dtype");
        const Json* shape = entry.find("shape");
        const Json* offsets = entry.find("data_offsets");
        if (dtype == nullptr || !dtype->is_string() || shape == nullptr || !shape->is_array() || offsets == nullptr ||
            !offsets->is_array() || offsets->size() != 2) {
            return errors::data_corrupted("safetensors tensor '{}' has a malformed description", name);
        }
        SafetensorsTensor info;
        info.name = name;
        info.dtype_name = dtype->as_string();
        info.dtype = dtype_from_name(info.dtype_name);
        if (shape->size() > static_cast<std::size_t>(kMaxRank)) {
            return errors::unsupported("safetensors tensor '{}' has rank {}", name, shape->size());
        }
        std::uint64_t count = 1;
        for (const Json& dim : shape->as_array()) {
            const std::optional<std::int64_t> d = dim.as_int();
            if (!d || *d < 0 || *d > (std::int64_t{1} << 40) ||
                (*d != 0 && count > (std::uint64_t{1} << 46) / static_cast<std::uint64_t>(*d))) {
                return errors::data_corrupted("safetensors tensor '{}' has an invalid shape", name);
            }
            count *= static_cast<std::uint64_t>(*d);
            info.shape.push_back(*d);
        }
        const std::optional<std::int64_t> begin = offsets->as_array()[0].as_int();
        const std::optional<std::int64_t> end = offsets->as_array()[1].as_int();
        if (!begin || !end || *begin < 0 || *end < *begin || static_cast<std::uint64_t>(*end) > data_size) {
            return errors::data_corrupted("safetensors tensor '{}' has data offsets outside the file", name);
        }
        info.offset = data_start + static_cast<std::size_t>(*begin);
        info.nbytes = static_cast<std::size_t>(*end - *begin);
        if (info.dtype != DType::kInvalid &&
            info.nbytes != storage_size(info.dtype, static_cast<std::int64_t>(count))) {
            return errors::data_corrupted("safetensors tensor '{}' ({}{}) has {} bytes of data", name, info.dtype_name,
                                          info.shape.to_string(), info.nbytes);
        }
        file.tensors_.push_back(std::move(info));
    }
    return file;
}

const SafetensorsTensor* SafetensorsFile::find(std::string_view name) const noexcept {
    for (const SafetensorsTensor& info : tensors_) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

Result<Tensor> SafetensorsFile::tensor(const SafetensorsTensor& info) const {
    if (info.dtype == DType::kInvalid) {
        return errors::unsupported("safetensors tensor '{}' has dtype {}, which ii does not support", info.name,
                                   info.dtype_name);
    }
    const std::byte* data = data_.data() + info.offset;
    const std::size_t alignment = std::max<std::size_t>(element_size(info.dtype), 1);
    if (keep_alive_ != nullptr && reinterpret_cast<std::uintptr_t>(data) % alignment == 0 && info.nbytes > 0) {
        auto buffer = Buffer::wrap(const_cast<std::byte*>(data), info.nbytes, keep_alive_);
        return Tensor::view(std::move(buffer), info.dtype, info.shape, contiguous_strides(info.shape));
    }
    return Tensor::from_bytes(info.dtype, info.shape, data, info.nbytes);
}

Result<SafetensorsModel> SafetensorsModel::open(const std::filesystem::path& path) {
    SafetensorsModel model;
    std::error_code error;
    if (!std::filesystem::is_directory(path, error)) {
        II_ASSIGN_OR_RETURN(SafetensorsFile file, SafetensorsFile::open(path));
        model.files_.push_back(std::move(file));
        return model;
    }
    const std::filesystem::path index_path = path / "model.safetensors.index.json";
    if (std::filesystem::exists(index_path, error)) {
        std::ifstream stream(index_path, std::ios::binary);
        std::stringstream buffer;
        buffer << stream.rdbuf();
        II_ASSIGN_OR_RETURN(const Json index, Json::parse(buffer.str()));
        const Json* map = index.find("weight_map");
        if (map == nullptr || !map->is_object()) {
            return errors::data_corrupted("{}: no weight_map", index_path.string());
        }
        std::set<std::string> shards;
        for (const auto& [name, file] : map->as_object()) {
            const std::string& shard = file.as_string();
            // Shards must be plain file names inside the model directory.
            if (shard.empty() || shard.find_first_of("/\\:") != std::string::npos || shard == "." || shard == "..") {
                return errors::data_corrupted("{}: invalid shard name '{}'", index_path.string(), shard);
            }
            shards.insert(shard);
        }
        for (const std::string& shard : shards) {
            II_ASSIGN_OR_RETURN(SafetensorsFile file, SafetensorsFile::open(path / shard));
            model.files_.push_back(std::move(file));
        }
        return model;
    }
    II_ASSIGN_OR_RETURN(SafetensorsFile file, SafetensorsFile::open(path / "model.safetensors"));
    model.files_.push_back(std::move(file));
    return model;
}

std::vector<std::string> SafetensorsModel::names() const {
    std::vector<std::string> result;
    for (const SafetensorsFile& file : files_) {
        for (const SafetensorsTensor& info : file.tensors()) {
            result.push_back(info.name);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool SafetensorsModel::contains(std::string_view name) const noexcept {
    return find(name) != nullptr;
}

const SafetensorsTensor* SafetensorsModel::find(std::string_view name) const noexcept {
    for (const SafetensorsFile& file : files_) {
        if (const SafetensorsTensor* info = file.find(name)) {
            return info;
        }
    }
    return nullptr;
}

Result<Tensor> SafetensorsModel::tensor(std::string_view name) const {
    for (const SafetensorsFile& file : files_) {
        if (const SafetensorsTensor* info = file.find(name)) {
            return file.tensor(*info);
        }
    }
    return errors::not_found("the model has no tensor '{}'", name);
}

}  // namespace ii::formats
