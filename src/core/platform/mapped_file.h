// SPDX-License-Identifier: MIT
#pragma once

#include <ii/status.h>

#include <cstddef>
#include <filesystem>
#include <span>
#include <utility>

namespace ii::platform {

/// A read-only memory mapping of a whole file.
///
/// Model weights are used directly from the mapping (zero-copy); pages are loaded lazily by the OS and shared between
/// processes mapping the same file.
class MappedFile {
public:
    /// Maps `path` read-only. An empty file produces a valid mapping with no data.
    static Result<MappedFile> open(const std::filesystem::path& path);

    MappedFile() noexcept = default;
    MappedFile(MappedFile&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            unmap();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile() { unmap(); }

    std::span<const std::byte> bytes() const noexcept { return {static_cast<const std::byte*>(data_), size_}; }
    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    MappedFile(const void* data, std::size_t size) noexcept : data_(data), size_(size) {}
    void unmap() noexcept;

    const void* data_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace ii::platform
