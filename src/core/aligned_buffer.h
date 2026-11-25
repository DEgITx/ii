// SPDX-License-Identifier: MIT
#pragma once

#include <ii/allocator.h>
#include <ii/status.h>

#include <cstddef>
#include <span>
#include <utility>

namespace ii {

/// An owning, aligned, uninitialized byte buffer obtained from an `Allocator`.
class AlignedBuffer {
public:
    AlignedBuffer() noexcept = default;

    /// Allocates `size` bytes. Fails with kOutOfMemory.
    static Result<AlignedBuffer> allocate(std::size_t size, std::size_t alignment = kDefaultAlignment,
                                          Allocator& allocator = default_allocator()) {
        void* data = allocator.allocate(size, alignment);
        if (data == nullptr) {
            return errors::out_of_memory("cannot allocate {} bytes", size);
        }
        return AlignedBuffer(allocator, data, size, alignment);
    }

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : allocator_(std::exchange(other.allocator_, nullptr)),
          data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)),
          alignment_(other.alignment_) {}

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            release();
            allocator_ = std::exchange(other.allocator_, nullptr);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            alignment_ = other.alignment_;
        }
        return *this;
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    ~AlignedBuffer() { release(); }

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    template <class T>
    std::span<T> as_span() noexcept {
        return {static_cast<T*>(data_), size_ / sizeof(T)};
    }

private:
    AlignedBuffer(Allocator& allocator, void* data, std::size_t size, std::size_t alignment) noexcept
        : allocator_(&allocator), data_(data), size_(size), alignment_(alignment) {}

    void release() noexcept {
        if (data_ != nullptr) {
            allocator_->deallocate(data_, size_, alignment_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    Allocator* allocator_ = nullptr;
    void* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t alignment_ = kDefaultAlignment;
};

}  // namespace ii
