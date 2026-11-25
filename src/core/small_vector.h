// SPDX-License-Identifier: MIT
#pragma once

#include "core/assert.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace ii {

/// A vector that stores up to `N` elements inline and moves to the heap beyond that.
///
/// Used for small, frequently created sequences (tensor shapes and strides, node inputs) to avoid heap allocations.
/// Iterators and references are invalidated by any operation that may grow the vector, as with std::vector.
template <class T, std::size_t N>
class SmallVector {
    static_assert(N > 0, "use std::vector for N == 0");

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    SmallVector() noexcept : data_(inline_data()) {}

    explicit SmallVector(size_type count) : SmallVector() { resize(count); }

    SmallVector(size_type count, const T& value) : SmallVector() { resize(count, value); }

    SmallVector(std::initializer_list<T> values) : SmallVector() { append(values.begin(), values.end()); }

    template <std::input_iterator It>
    SmallVector(It first, It last) : SmallVector() {
        append(first, last);
    }

    SmallVector(const SmallVector& other) : SmallVector() { append(other.begin(), other.end()); }

    SmallVector(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) : SmallVector() {
        take(std::move(other));
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            clear();
            append(other.begin(), other.end());
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this != &other) {
            clear();
            release_heap();
            take(std::move(other));
        }
        return *this;
    }

    SmallVector& operator=(std::initializer_list<T> values) {
        clear();
        append(values.begin(), values.end());
        return *this;
    }

    ~SmallVector() {
        clear();
        release_heap();
    }

    // --- Capacity ---
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    bool is_inline() const noexcept { return data_ == inline_data(); }
    static constexpr size_type inline_capacity() noexcept { return N; }

    void reserve(size_type new_capacity) {
        if (new_capacity > capacity_) {
            reallocate(new_capacity);
        }
    }

    // --- Element access ---
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T& operator[](size_type index) noexcept {
        II_ASSERT(index < size_);
        return data_[index];
    }
    const T& operator[](size_type index) const noexcept {
        II_ASSERT(index < size_);
        return data_[index];
    }

    T& front() noexcept { return (*this)[0]; }
    const T& front() const noexcept { return (*this)[0]; }
    T& back() noexcept { return (*this)[size_ - 1]; }
    const T& back() const noexcept { return (*this)[size_ - 1]; }

    iterator begin() noexcept { return data_; }
    iterator end() noexcept { return data_ + size_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator end() const noexcept { return data_ + size_; }
    const_iterator cbegin() const noexcept { return data_; }
    const_iterator cend() const noexcept { return data_ + size_; }

    // --- Modifiers ---
    template <class... Args>
    T& emplace_back(Args&&... args) {
        if (size_ == capacity_) {
            return emplace_back_with_growth(std::forward<Args>(args)...);
        }
        T* element = std::construct_at(data_ + size_, std::forward<Args>(args)...);
        ++size_;
        return *element;
    }

    void push_back(const T& value) { emplace_back(value); }
    void push_back(T&& value) { emplace_back(std::move(value)); }

    void pop_back() noexcept {
        II_ASSERT(size_ > 0);
        --size_;
        std::destroy_at(data_ + size_);
    }

    void clear() noexcept {
        std::destroy(data_, data_ + size_);
        size_ = 0;
    }

    void resize(size_type count) {
        if (count < size_) {
            std::destroy(data_ + count, data_ + size_);
        } else if (count > size_) {
            reserve(count);
            std::uninitialized_value_construct(data_ + size_, data_ + count);
        }
        size_ = count;
    }

    void resize(size_type count, const T& value) {
        if (count < size_) {
            std::destroy(data_ + count, data_ + size_);
        } else if (count > size_) {
            if (count > capacity_) {
                // `value` may refer to an element of this vector: copy it before reallocating.
                T copy(value);
                reserve(count);
                std::uninitialized_fill(data_ + size_, data_ + count, copy);
            } else {
                std::uninitialized_fill(data_ + size_, data_ + count, value);
            }
        }
        size_ = count;
    }

    template <std::input_iterator It>
    void append(It first, It last) {
        if constexpr (std::forward_iterator<It>) {
            reserve(size_ + static_cast<size_type>(std::distance(first, last)));
        }
        for (; first != last; ++first) {
            emplace_back(*first);
        }
    }

    friend bool operator==(const SmallVector& a, const SmallVector& b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end());
    }

private:
    T* inline_data() noexcept { return std::launder(reinterpret_cast<T*>(storage_)); }
    const T* inline_data() const noexcept { return std::launder(reinterpret_cast<const T*>(storage_)); }

    size_type grown_capacity(size_type minimum) const noexcept { return std::max(minimum, capacity_ * 2); }

    void reallocate(size_type new_capacity) {
        T* new_data = std::allocator<T>().allocate(new_capacity);
        std::uninitialized_move(data_, data_ + size_, new_data);
        std::destroy(data_, data_ + size_);
        release_heap();
        data_ = new_data;
        capacity_ = new_capacity;
    }

    template <class... Args>
    T& emplace_back_with_growth(Args&&... args) {
        // Construct the new element first: `args` may refer to an existing element that is about to move.
        const size_type new_capacity = grown_capacity(size_ + 1);
        T* new_data = std::allocator<T>().allocate(new_capacity);
        T* element = std::construct_at(new_data + size_, std::forward<Args>(args)...);
        std::uninitialized_move(data_, data_ + size_, new_data);
        std::destroy(data_, data_ + size_);
        release_heap();
        data_ = new_data;
        capacity_ = new_capacity;
        ++size_;
        return *element;
    }

    void release_heap() noexcept {
        if (!is_inline()) {
            std::allocator<T>().deallocate(data_, capacity_);
            data_ = inline_data();
            capacity_ = N;
        }
    }

    /// Takes the contents of `other` (this vector must be empty and inline). Leaves `other` empty.
    void take(SmallVector&& other) {
        if (!other.is_inline()) {
            data_ = std::exchange(other.data_, other.inline_data());
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, N);
            return;
        }
        std::uninitialized_move(other.data_, other.data_ + other.size_, data_);
        size_ = other.size_;
        other.clear();
    }

    alignas(T) std::byte storage_[N * sizeof(T)];
    T* data_;
    size_type size_ = 0;
    size_type capacity_ = N;
};

}  // namespace ii
