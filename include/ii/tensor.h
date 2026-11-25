// SPDX-License-Identifier: MIT
/// @file
/// Buffers and tensors.
///
/// A `Buffer` is a reference-counted block of memory; a `Tensor` is a typed, shaped, possibly strided view of a
/// buffer. Copying a `Tensor` is cheap and shares the memory. View operations (`reshape`, `permute`, `slice`,
/// `expand`) never copy data.
#pragma once

#include <ii/allocator.h>
#include <ii/dtype.h>
#include <ii/export.h>
#include <ii/shape.h>
#include <ii/status.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

II_BEGIN_DLL_EXPORTS

namespace ii {

/// Kind of device that owns a memory region.
enum class DeviceKind : std::uint8_t {
    kCpu,  ///< Host memory, directly addressable by the CPU.
};

/// Identifies a device: its kind and an index among devices of that kind.
struct Device {
    DeviceKind kind = DeviceKind::kCpu;
    int index = 0;

    static constexpr Device cpu() noexcept { return {}; }
    friend constexpr bool operator==(const Device&, const Device&) noexcept = default;
};

/// A reference-counted region of memory on a device.
///
/// Buffers are created through `allocate()` (owned memory) or `wrap()` (memory owned by someone else, e.g. the
/// application or a memory-mapped file) and are always held by `std::shared_ptr`.
class II_API Buffer {
    struct Key {};  // restricts construction to the factory functions

public:
    /// Allocates `size` bytes from `allocator`. The contents are uninitialized.
    static Result<std::shared_ptr<Buffer>> allocate(std::size_t size, Allocator& allocator = default_allocator(),
                                                    std::size_t alignment = kDefaultAlignment);

    /// Wraps external memory without copying. `keep_alive` (optional) is held as long as the buffer lives, e.g. the
    /// owner of a memory-mapped file. Without it the caller must keep the memory valid while the buffer is in use.
    static std::shared_ptr<Buffer> wrap(void* data, std::size_t size, std::shared_ptr<const void> keep_alive = nullptr,
                                        Device device = Device::cpu());

    Buffer(Key, void* data, std::size_t size, Device device, Allocator* allocator, std::size_t alignment,
           std::shared_ptr<const void> keep_alive) noexcept;
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    Device device() const noexcept { return device_; }

    /// True when the memory was allocated by ii (as opposed to wrapped).
    bool owns_memory() const noexcept { return allocator_ != nullptr; }

private:
    void* data_;
    std::size_t size_;
    Device device_;
    Allocator* allocator_;  // non-null for owned memory
    std::size_t alignment_;
    std::shared_ptr<const void> keep_alive_;
};

/// A typed, shaped view of a buffer.
///
/// Element `(i0, i1, ...)` is located at `data() + sum(ik * strides()[k])` elements. A default-constructed tensor is
/// empty (`!defined()`).
///
/// For block-quantized dtypes the innermost dimension must be contiguous and every view must start on a block
/// boundary; strides are still counted in elements.
class II_API Tensor {
public:
    Tensor() noexcept = default;

    /// Allocates an uninitialized contiguous tensor.
    static Result<Tensor> empty(DType dtype, const Shape& shape, Allocator& allocator = default_allocator());

    /// Allocates a zero-filled contiguous tensor.
    static Result<Tensor> zeros(DType dtype, const Shape& shape, Allocator& allocator = default_allocator());

    /// Wraps external memory holding a contiguous tensor (zero-copy). The memory must stay valid while the tensor
    /// (or any copy of it) is in use.
    static Tensor wrap(void* data, DType dtype, const Shape& shape);

    /// Creates a view of `buffer`. `offset` is in bytes. Fails if the view does not fit in the buffer.
    static Result<Tensor> view(std::shared_ptr<Buffer> buffer, DType dtype, const Shape& shape, const Strides& strides,
                               std::size_t offset = 0);

    /// Allocates a contiguous tensor and copies `values` (in row-major order) into it.
    template <class T>
    static Result<Tensor> from(std::span<const T> values, const Shape& shape,
                               Allocator& allocator = default_allocator()) {
        static_assert(kHasDType<T>, "T has no corresponding ii::DType");
        return from_bytes(dtype_of<T>(), shape, values.data(), values.size_bytes(), allocator);
    }
    template <class T>
    static Result<Tensor> from(std::initializer_list<T> values, const Shape& shape,
                               Allocator& allocator = default_allocator()) {
        return from(std::span<const T>(values.begin(), values.size()), shape, allocator);
    }

    /// A rank-0 tensor holding `value`.
    template <class T>
    static Result<Tensor> scalar(T value, Allocator& allocator = default_allocator()) {
        return from(std::span<const T>(&value, 1), Shape{}, allocator);
    }

    /// Allocates a contiguous tensor and copies `size` bytes of row-major data into it; `size` must match.
    static Result<Tensor> from_bytes(DType dtype, const Shape& shape, const void* data, std::size_t size,
                                     Allocator& allocator = default_allocator());

    bool defined() const noexcept { return buffer_ != nullptr; }
    DType dtype() const noexcept { return dtype_; }
    const Shape& shape() const noexcept { return shape_; }
    const Strides& strides() const noexcept { return strides_; }
    int rank() const noexcept { return shape_.rank(); }
    std::int64_t numel() const noexcept { return shape_.numel(); }
    Device device() const noexcept;

    /// Bytes occupied by the elements if stored contiguously.
    std::size_t nbytes() const noexcept { return storage_size(dtype_, numel()); }

    /// True when the elements are stored densely in row-major order (dimensions of size 1 may have any stride).
    bool is_contiguous() const noexcept;

    const std::shared_ptr<Buffer>& buffer() const noexcept { return buffer_; }

    /// Byte offset of the first element within the buffer.
    std::size_t offset() const noexcept { return offset_; }

    /// Address of the first element (nullptr for an empty tensor).
    void* data() const noexcept;

    /// Typed address of the first element. Terminates if `T` does not match `dtype()`.
    template <class T>
    T* data_as() const {
        check_dtype(dtype_of<T>());
        return static_cast<T*>(data());
    }

    /// The elements of a contiguous tensor. Terminates if `T` does not match or the tensor is not contiguous.
    template <class T>
    std::span<T> values() const {
        check_dtype(dtype_of<T>());
        check_contiguous();
        return {static_cast<T*>(data()), static_cast<std::size_t>(numel())};
    }

    // --- Views (no data is copied) ---

    /// Same elements with a different shape. One dimension may be -1 (inferred). Requires a contiguous tensor.
    Result<Tensor> reshape(std::span<const std::int64_t> shape) const;
    Result<Tensor> reshape(std::initializer_list<std::int64_t> shape) const {
        return reshape(std::span<const std::int64_t>(shape.begin(), shape.size()));
    }

    /// Reorders dimensions: dimension `i` of the result is dimension `perm[i]` of this tensor.
    Result<Tensor> permute(std::span<const int> perm) const;
    Result<Tensor> permute(std::initializer_list<int> perm) const {
        return permute(std::span<const int>(perm.begin(), perm.size()));
    }

    /// Elements `begin, begin + step, ...` (< `end`) along `axis`. Negative `axis`, `begin` and `end` count from the
    /// back; out-of-range bounds are clamped. `step` must be positive.
    Result<Tensor> slice(int axis, std::int64_t begin, std::int64_t end, std::int64_t step = 1) const;

    /// Broadcasts dimensions of size 1 to `shape` (numpy rules, new leading dimensions allowed) using zero strides.
    Result<Tensor> expand(const Shape& shape) const;

    // --- Copies ---

    /// A contiguous copy in newly allocated memory.
    Result<Tensor> clone(Allocator& allocator = default_allocator()) const;

    /// This tensor if it is already contiguous, otherwise a contiguous copy.
    Result<Tensor> contiguous(Allocator& allocator = default_allocator()) const;

    /// Copies the elements of `source` (same dtype and shape, any strides) into this tensor.
    Status copy_from(const Tensor& source) const;

    /// A short description, e.g. "f32[2, 3]", followed by up to `max_values` values.
    std::string to_string(int max_values = 16) const;

private:
    void check_dtype(DType requested) const;
    void check_contiguous() const;

    std::shared_ptr<Buffer> buffer_;
    std::size_t offset_ = 0;
    DType dtype_ = DType::kInvalid;
    Shape shape_;
    Strides strides_;
};

}  // namespace ii

II_END_DLL_EXPORTS
