// SPDX-License-Identifier: MIT
#include <ii/tensor.h>

#include "core/assert.h"
#include "tensor/quant.h"
#include "tensor/strided.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <vector>

namespace ii {

namespace detail {
void shape_rank_overflow(std::size_t rank) noexcept {
    fatal_error(__FILE__, __LINE__, std::format("shape rank {} exceeds the maximum of {}", rank, kMaxRank));
}
}  // namespace detail

std::string Shape::to_string() const {
    std::string result = "[";
    for (int i = 0; i < rank_; ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(dims_[i]);
    }
    result += ']';
    return result;
}

// ------------------------------------------------------------------------------------------------------------------
// Buffer

Buffer::Buffer(Key, void* data, std::size_t size, Device device, Allocator* allocator, std::size_t alignment,
               std::shared_ptr<const void> keep_alive) noexcept
    : data_(data),
      size_(size),
      device_(device),
      allocator_(allocator),
      alignment_(alignment),
      keep_alive_(std::move(keep_alive)) {}

Buffer::~Buffer() {
    if (allocator_ != nullptr) {
        allocator_->deallocate(data_, size_, alignment_);
    }
}

Result<std::shared_ptr<Buffer>> Buffer::allocate(std::size_t size, Allocator& allocator, std::size_t alignment) {
    void* data = allocator.allocate(size, alignment);
    if (data == nullptr) {
        return errors::out_of_memory("cannot allocate a buffer of {} bytes", size);
    }
    return std::make_shared<Buffer>(Key{}, data, size, Device::cpu(), &allocator, alignment, nullptr);
}

std::shared_ptr<Buffer> Buffer::wrap(void* data, std::size_t size, std::shared_ptr<const void> keep_alive,
                                     Device device) {
    return std::make_shared<Buffer>(Key{}, data, size, device, nullptr, 0, std::move(keep_alive));
}

// ------------------------------------------------------------------------------------------------------------------
// Tensor

namespace {

Status validate_shape(DType dtype, const Shape& shape) {
    if (dtype == DType::kInvalid || dtype == DType::kCount) {
        return errors::invalid_argument("invalid dtype");
    }
    std::int64_t numel = 1;
    for (const std::int64_t dim : shape) {
        if (dim < 0) {
            return errors::invalid_argument("negative dimension in shape {}", shape.to_string());
        }
        if (dim != 0 && numel > std::numeric_limits<std::int64_t>::max() / dim) {
            return errors::out_of_range("shape {} has too many elements", shape.to_string());
        }
        numel *= dim;
    }
    const std::int64_t block = dtype_info(dtype).block_size;
    if (block > 1 && (shape.empty() || shape[shape.rank() - 1] % block != 0)) {
        return errors::invalid_argument("the innermost dimension of a {} tensor must be a multiple of {}, got {}",
                                        ii::to_string(dtype), block, shape.to_string());
    }
    return ok_status();
}

/// Copies elements between two strided tensors of the same dtype and shape.
void strided_copy(const Tensor& dst, const Tensor& src) {
    const Shape& shape = dst.shape();
    if (shape.numel() == 0) {
        return;
    }
    auto* dst_bytes = static_cast<std::byte*>(dst.data());
    const auto* src_bytes = static_cast<const std::byte*>(src.data());
    if (dst.is_contiguous() && src.is_contiguous()) {
        std::memmove(dst_bytes, src_bytes, dst.nbytes());
        return;
    }
    const DType dtype = dst.dtype();
    const DTypeInfo& info = dtype_info(dtype);
    const int rank = shape.rank();
    const std::int64_t inner = rank == 0 ? 1 : shape[rank - 1];
    const std::int64_t dst_inner_stride = rank == 0 ? 1 : dst.strides()[rank - 1];
    const std::int64_t src_inner_stride = rank == 0 ? 1 : src.strides()[rank - 1];
    const Strides* const strides[2] = {&dst.strides(), &src.strides()};

    // Element offsets are converted to bytes; for block types they are always block aligned.
    auto to_bytes = [&](std::int64_t elements) {
        return static_cast<std::ptrdiff_t>(elements / info.block_size * info.block_bytes);
    };
    const bool rows_contiguous = (dst_inner_stride == 1 && src_inner_stride == 1) || inner == 1;
    const std::size_t row_bytes = storage_size(dtype, inner);
    const std::size_t element_bytes = info.block_bytes;
    for_each_row<2>(shape, strides, [&](const std::int64_t* offsets) {
        std::byte* d = dst_bytes + to_bytes(offsets[0]);
        const std::byte* s = src_bytes + to_bytes(offsets[1]);
        if (rows_contiguous) {
            std::memmove(d, s, row_bytes);
            return;
        }
        for (std::int64_t i = 0; i < inner; ++i) {
            std::memcpy(d + to_bytes(i * dst_inner_stride), s + to_bytes(i * src_inner_stride), element_bytes);
        }
    });
}

}  // namespace

Result<Tensor> Tensor::empty(DType dtype, const Shape& shape, Allocator& allocator) {
    II_TRY(validate_shape(dtype, shape));
    II_ASSIGN_OR_RETURN(std::shared_ptr<Buffer> buffer,
                        Buffer::allocate(storage_size(dtype, shape.numel()), allocator));
    Tensor tensor;
    tensor.buffer_ = std::move(buffer);
    tensor.dtype_ = dtype;
    tensor.shape_ = shape;
    tensor.strides_ = contiguous_strides(shape);
    return tensor;
}

Result<Tensor> Tensor::zeros(DType dtype, const Shape& shape, Allocator& allocator) {
    II_ASSIGN_OR_RETURN(Tensor tensor, empty(dtype, shape, allocator));
    std::memset(tensor.data(), 0, tensor.nbytes());
    return tensor;
}

Tensor Tensor::wrap(void* data, DType dtype, const Shape& shape) {
    Tensor tensor;
    tensor.buffer_ = Buffer::wrap(data, storage_size(dtype, shape.numel()));
    tensor.dtype_ = dtype;
    tensor.shape_ = shape;
    tensor.strides_ = contiguous_strides(shape);
    return tensor;
}

Result<Tensor> Tensor::view(std::shared_ptr<Buffer> buffer, DType dtype, const Shape& shape, const Strides& strides,
                            std::size_t offset) {
    if (buffer == nullptr) {
        return errors::invalid_argument("null buffer");
    }
    II_TRY(validate_shape(dtype, shape));
    if (strides.rank() != shape.rank()) {
        return errors::invalid_argument("strides {} do not match shape {}", strides.to_string(), shape.to_string());
    }
    const DTypeInfo& info = dtype_info(dtype);
    if (info.block_size > 1 && shape.rank() > 0 && strides[shape.rank() - 1] != 1) {
        return errors::invalid_argument("the innermost dimension of a {} tensor must be contiguous",
                                        ii::to_string(dtype));
    }
    // Compute the range of element offsets covered by the view.
    std::int64_t min_offset = 0;
    std::int64_t max_offset = 0;
    if (shape.numel() > 0) {
        for (int d = 0; d < shape.rank(); ++d) {
            const std::int64_t extent = (shape[d] - 1) * strides[d];
            (extent < 0 ? min_offset : max_offset) += extent;
        }
    }
    const auto min_byte = static_cast<std::int64_t>(offset) + min_offset / info.block_size * info.block_bytes;
    const auto end_byte = static_cast<std::int64_t>(offset) +
                          static_cast<std::int64_t>(storage_size(dtype, max_offset + (shape.numel() > 0 ? 1 : 0)));
    if (shape.numel() > 0 && (min_byte < 0 || end_byte > static_cast<std::int64_t>(buffer->size()))) {
        return errors::out_of_range("a {} view with shape {} and strides {} at offset {} exceeds a buffer of {} bytes",
                                    ii::to_string(dtype), shape.to_string(), strides.to_string(), offset,
                                    buffer->size());
    }
    Tensor tensor;
    tensor.buffer_ = std::move(buffer);
    tensor.offset_ = offset;
    tensor.dtype_ = dtype;
    tensor.shape_ = shape;
    tensor.strides_ = strides;
    return tensor;
}

Result<Tensor> Tensor::from_bytes(DType dtype, const Shape& shape, const void* data, std::size_t size,
                                  Allocator& allocator) {
    II_ASSIGN_OR_RETURN(Tensor tensor, empty(dtype, shape, allocator));
    if (size != tensor.nbytes()) {
        return errors::invalid_argument("{} bytes given for a {}{} tensor of {} bytes", size, ii::to_string(dtype),
                                        shape.to_string(), tensor.nbytes());
    }
    if (size > 0) {
        std::memcpy(tensor.data(), data, size);
    }
    return tensor;
}

Device Tensor::device() const noexcept {
    return buffer_ ? buffer_->device() : Device::cpu();
}

void* Tensor::data() const noexcept {
    if (!buffer_) {
        return nullptr;
    }
    return static_cast<std::byte*>(buffer_->data()) + offset_;
}

bool Tensor::is_contiguous() const noexcept {
    std::int64_t expected = 1;
    for (int d = shape_.rank() - 1; d >= 0; --d) {
        if (shape_[d] != 1 && strides_[d] != expected) {
            return false;
        }
        expected *= shape_[d];
    }
    return true;
}

void Tensor::check_dtype(DType requested) const {
    II_CHECK(requested == dtype_, "tensor element type is {}, but {} was requested", ii::to_string(dtype_),
             ii::to_string(requested));
}

void Tensor::check_contiguous() const {
    II_CHECK(is_contiguous(), "tensor with shape {} and strides {} is not contiguous", shape_.to_string(),
             strides_.to_string());
}

Result<Tensor> Tensor::reshape(std::span<const std::int64_t> new_dims) const {
    if (!defined()) {
        return errors::invalid_argument("reshape of an empty tensor");
    }
    if (new_dims.size() > static_cast<std::size_t>(kMaxRank)) {
        return errors::invalid_argument("rank {} exceeds the maximum of {}", new_dims.size(), kMaxRank);
    }
    if (!is_contiguous()) {
        return errors::invalid_argument("reshape requires a contiguous tensor; call contiguous() first");
    }
    Shape shape(new_dims);
    int inferred = -1;
    std::int64_t known = 1;
    for (int d = 0; d < shape.rank(); ++d) {
        if (shape[d] == -1) {
            if (inferred >= 0) {
                return errors::invalid_argument("reshape: more than one dimension is -1");
            }
            inferred = d;
        } else if (shape[d] < 0) {
            return errors::invalid_argument("reshape: invalid dimension {}", shape[d]);
        } else {
            known *= shape[d];
        }
    }
    if (inferred >= 0) {
        if (known == 0 || numel() % known != 0) {
            return errors::invalid_argument("cannot reshape {} into {}", shape_.to_string(), shape.to_string());
        }
        shape[inferred] = numel() / known;
    }
    if (shape.numel() != numel()) {
        return errors::invalid_argument("cannot reshape {} ({} elements) into {}", shape_.to_string(), numel(),
                                        shape.to_string());
    }
    return view(buffer_, dtype_, shape, contiguous_strides(shape), offset_);
}

Result<Tensor> Tensor::permute(std::span<const int> perm) const {
    if (static_cast<int>(perm.size()) != rank()) {
        return errors::invalid_argument("permutation of {} axes for a tensor of rank {}", perm.size(), rank());
    }
    bool seen[kMaxRank] = {};
    Shape shape = shape_;
    Strides strides = strides_;
    for (int i = 0; i < rank(); ++i) {
        const int axis = perm[i];
        if (axis < 0 || axis >= rank() || seen[axis]) {
            return errors::invalid_argument("invalid permutation");
        }
        seen[axis] = true;
        shape[i] = shape_[axis];
        strides[i] = strides_[axis];
    }
    if (is_block_quantized(dtype_) && rank() > 0 && perm[rank() - 1] != rank() - 1) {
        return errors::unsupported("cannot move the innermost dimension of a block-quantized tensor");
    }
    Tensor result = *this;
    result.shape_ = shape;
    result.strides_ = strides;
    return result;
}

Result<Tensor> Tensor::slice(int axis, std::int64_t begin, std::int64_t end, std::int64_t step) const {
    if (axis < 0) {
        axis += rank();
    }
    if (axis < 0 || axis >= rank()) {
        return errors::invalid_argument("slice axis out of range for rank {}", rank());
    }
    if (step <= 0) {
        return errors::invalid_argument("slice step must be positive, got {}", step);
    }
    const std::int64_t size = shape_[axis];
    auto clamp_index = [size](std::int64_t index) {
        if (index < 0) {
            index += size;
        }
        return std::clamp<std::int64_t>(index, 0, size);
    };
    begin = clamp_index(begin);
    end = std::max(clamp_index(end), begin);
    const std::int64_t count = (end - begin + step - 1) / step;
    const DTypeInfo& info = dtype_info(dtype_);
    const std::int64_t element_offset = begin * strides_[axis];
    if (info.block_size > 1 && (element_offset % info.block_size != 0 || (axis == rank() - 1 && step != 1) ||
                                (axis == rank() - 1 && count % info.block_size != 0))) {
        return errors::unsupported("slices of block-quantized tensors must cover whole blocks");
    }
    Tensor result = *this;
    result.shape_[axis] = count;
    result.strides_[axis] = strides_[axis] * step;
    if (count > 0) {
        result.offset_ = static_cast<std::size_t>(static_cast<std::int64_t>(offset_) +
                                                  element_offset / info.block_size * info.block_bytes);
    }
    return result;
}

Result<Tensor> Tensor::expand(const Shape& shape) const {
    if (shape.rank() < rank()) {
        return errors::invalid_argument("cannot expand {} to the lower rank shape {}", shape_.to_string(),
                                        shape.to_string());
    }
    if (is_block_quantized(dtype_)) {
        return errors::unsupported("cannot expand a block-quantized tensor");
    }
    const int lead = shape.rank() - rank();
    Strides strides = Shape::filled(shape.rank(), 0);
    for (int d = 0; d < shape.rank(); ++d) {
        if (d < lead) {
            continue;
        }
        const std::int64_t from = shape_[d - lead];
        if (from == shape[d]) {
            strides[d] = strides_[d - lead];
        } else if (from != 1) {
            return errors::invalid_argument("cannot expand {} to {}", shape_.to_string(), shape.to_string());
        }
    }
    Tensor result = *this;
    result.shape_ = shape;
    result.strides_ = strides;
    return result;
}

Result<Tensor> Tensor::clone(Allocator& allocator) const {
    if (!defined()) {
        return Tensor();
    }
    II_ASSIGN_OR_RETURN(Tensor copy, empty(dtype_, shape_, allocator));
    strided_copy(copy, *this);
    return copy;
}

Result<Tensor> Tensor::to(DType dtype, Allocator& allocator) const {
    if (!defined()) {
        return Tensor();
    }
    if (dtype == dtype_) {
        return clone(allocator);
    }
    auto float_like = [](DType t) {
        return t == DType::kF32 || t == DType::kF16 || t == DType::kBF16 || t == DType::kF64;
    };
    if (!(float_like(dtype_) || is_block_quantized(dtype_)) || !(float_like(dtype) || is_block_quantized(dtype))) {
        return errors::unsupported("conversion from {} to {}", ii::to_string(dtype_), ii::to_string(dtype));
    }
    if (is_block_quantized(dtype) && !quant::can_quantize(dtype)) {
        return errors::unsupported("quantization to {}", ii::to_string(dtype));
    }
    const std::int64_t row = rank() == 0 ? 1 : shape_[rank() - 1];
    if (is_block_quantized(dtype)) {
        II_TRY(quant::check_row(dtype, row));
    }
    II_ASSIGN_OR_RETURN(const Tensor source, contiguous(allocator));
    II_ASSIGN_OR_RETURN(Tensor result, empty(dtype, shape_, allocator));
    const std::int64_t n = numel();
    // Through f32, one row at a time.
    std::vector<float> values(static_cast<std::size_t>(std::max<std::int64_t>(row, 1)));
    const std::int64_t rows = row == 0 ? 0 : n / row;
    for (std::int64_t r = 0; r < rows; ++r) {
        const auto* in = static_cast<const std::byte*>(source.data()) + storage_size(dtype_, r * row);
        auto* out = static_cast<std::byte*>(result.data()) + storage_size(dtype, r * row);
        switch (dtype_) {
        case DType::kF32:
            std::memcpy(values.data(), in, static_cast<std::size_t>(row) * 4);
            break;
        case DType::kF16:
            for (std::int64_t i = 0; i < row; ++i) {
                values[static_cast<std::size_t>(i)] = reinterpret_cast<const f16*>(in)[i].to_float();
            }
            break;
        case DType::kBF16:
            for (std::int64_t i = 0; i < row; ++i) {
                values[static_cast<std::size_t>(i)] = reinterpret_cast<const bf16*>(in)[i].to_float();
            }
            break;
        case DType::kF64:
            for (std::int64_t i = 0; i < row; ++i) {
                values[static_cast<std::size_t>(i)] = static_cast<float>(reinterpret_cast<const double*>(in)[i]);
            }
            break;
        default:
            quant::dequantize(dtype_, in, values.data(), row);
        }
        switch (dtype) {
        case DType::kF32:
            std::memcpy(out, values.data(), static_cast<std::size_t>(row) * 4);
            break;
        case DType::kF16:
            for (std::int64_t i = 0; i < row; ++i) {
                reinterpret_cast<f16*>(out)[i] = dtype_ == DType::kF64
                                                     ? f16(static_cast<float>(reinterpret_cast<const double*>(in)[i]))
                                                     : f16(values[static_cast<std::size_t>(i)]);
            }
            break;
        case DType::kBF16:
            for (std::int64_t i = 0; i < row; ++i) {
                reinterpret_cast<bf16*>(out)[i] = bf16(values[static_cast<std::size_t>(i)]);
            }
            break;
        case DType::kF64:
            for (std::int64_t i = 0; i < row; ++i) {
                reinterpret_cast<double*>(out)[i] = static_cast<double>(values[static_cast<std::size_t>(i)]);
            }
            break;
        default:
            quant::quantize(dtype, values.data(), out, row);
        }
    }
    return result;
}

Result<Tensor> Tensor::contiguous(Allocator& allocator) const {
    if (!defined() || is_contiguous()) {
        return *this;
    }
    return clone(allocator);
}

Status Tensor::copy_from(const Tensor& source) const {
    if (!defined() || !source.defined()) {
        return errors::invalid_argument("copy_from with an empty tensor");
    }
    if (source.dtype_ != dtype_ || !(source.shape_ == shape_)) {
        return errors::invalid_argument("cannot copy a {}{} tensor into a {}{} tensor", ii::to_string(source.dtype_),
                                        source.shape_.to_string(), ii::to_string(dtype_), shape_.to_string());
    }
    strided_copy(*this, source);
    return ok_status();
}

namespace {

std::string format_element(DType dtype, const void* p) {
    switch (dtype) {
    case DType::kF32:
        return std::format("{}", *static_cast<const float*>(p));
    case DType::kF16:
        return std::format("{}", static_cast<const f16*>(p)->to_float());
    case DType::kBF16:
        return std::format("{}", static_cast<const bf16*>(p)->to_float());
    case DType::kF64:
        return std::format("{}", *static_cast<const double*>(p));
    case DType::kI8:
        return std::format("{}", *static_cast<const std::int8_t*>(p));
    case DType::kU8:
        return std::format("{}", *static_cast<const std::uint8_t*>(p));
    case DType::kI16:
        return std::format("{}", *static_cast<const std::int16_t*>(p));
    case DType::kU16:
        return std::format("{}", *static_cast<const std::uint16_t*>(p));
    case DType::kI32:
        return std::format("{}", *static_cast<const std::int32_t*>(p));
    case DType::kU32:
        return std::format("{}", *static_cast<const std::uint32_t*>(p));
    case DType::kI64:
        return std::format("{}", *static_cast<const std::int64_t*>(p));
    case DType::kU64:
        return std::format("{}", *static_cast<const std::uint64_t*>(p));
    case DType::kBool:
        return *static_cast<const std::uint8_t*>(p) != 0 ? "true" : "false";
    default:
        return "?";
    }
}

}  // namespace

std::string Tensor::to_string(int max_values) const {
    if (!defined()) {
        return "<empty tensor>";
    }
    std::string result = std::format("{}{}", ii::to_string(dtype_), shape_.to_string());
    if (is_block_quantized(dtype_) || max_values <= 0) {
        return result;
    }
    result += " {";
    const std::int64_t count = std::min<std::int64_t>(numel(), max_values);
    std::int64_t index[kMaxRank] = {};
    const std::size_t size = element_size(dtype_);
    for (std::int64_t i = 0; i < count; ++i) {
        unravel_index(i, shape_, index);
        const auto* p =
            static_cast<const std::byte*>(data()) + offset_of(index, strides_) * static_cast<std::int64_t>(size);
        if (i > 0) {
            result += ", ";
        }
        result += format_element(dtype_, p);
    }
    if (numel() > count) {
        result += ", ...";
    }
    result += '}';
    return result;
}

}  // namespace ii
