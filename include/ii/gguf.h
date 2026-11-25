// SPDX-License-Identifier: MIT
/// @file
/// Reading and writing GGUF files (the model format of llama.cpp / ggml).
///
/// A GGUF file holds typed metadata (architecture, hyper-parameters, tokenizer) and tensors. `gguf::File` maps the file
/// into memory and returns tensors as zero-copy views: block-quantized weights (`q4_k`, `q8_0`, ...) are used directly
/// from the mapping, their layouts are byte-compatible with ii's block dtypes. `gguf::Writer` writes files (used by
/// `ii quantize`).
///
/// @code
/// auto file = ii::gguf::File::open("model.gguf").value();
/// std::string arch = file.get_string("general.architecture").value_or("?");
/// ii::Tensor w = file.tensor("blk.0.attn_q.weight").value();  // e.g. q4_k[2048, 2048], no copy
/// @endcode
///
/// GGUF stores dimensions innermost first; ii shapes are outermost first, so a GGUF tensor with ne = [4096, 32000] is
/// an ii tensor of shape [32000, 4096]. Files are untrusted input: every size and offset is validated.
#pragma once

#include <ii/dtype.h>
#include <ii/export.h>
#include <ii/shape.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

II_BEGIN_DLL_EXPORTS

namespace ii::gguf {

/// GGUF metadata value types (the numbering of the file format).
enum class ValueType : std::uint32_t {
    kU8 = 0,
    kI8 = 1,
    kU16 = 2,
    kI16 = 3,
    kU32 = 4,
    kI32 = 5,
    kF32 = 6,
    kBool = 7,
    kString = 8,
    kArray = 9,
    kU64 = 10,
    kI64 = 11,
    kF64 = 12,
};

/// Name of a value type, e.g. "u32".
II_API std::string_view to_string(ValueType type) noexcept;

/// A metadata value: a scalar, a string or an array of scalars or strings (nested arrays are kept as empty arrays).
class II_API Value {
public:
    using Storage = std::variant<std::int64_t, std::uint64_t, double, bool, std::string, std::vector<std::int64_t>,
                                 std::vector<double>, std::vector<std::string>>;

    Value() = default;
    Value(ValueType type, Storage storage, ValueType element_type = ValueType::kU8)
        : type_(type), element_type_(element_type), storage_(std::move(storage)) {}

    /// Convenience constructors with the natural GGUF type.
    static Value u32(std::uint32_t v) { return {ValueType::kU32, std::uint64_t{v}}; }
    static Value i32(std::int32_t v) { return {ValueType::kI32, std::int64_t{v}}; }
    static Value u64(std::uint64_t v) { return {ValueType::kU64, v}; }
    static Value f32(float v) { return {ValueType::kF32, static_cast<double>(v)}; }
    static Value boolean(bool v) { return {ValueType::kBool, v}; }
    static Value string(std::string v) { return {ValueType::kString, std::move(v)}; }
    static Value strings(std::vector<std::string> v) { return {ValueType::kArray, std::move(v), ValueType::kString}; }
    static Value ints(std::vector<std::int64_t> v, ValueType element = ValueType::kI32) {
        return {ValueType::kArray, std::move(v), element};
    }
    static Value floats(std::vector<double> v) { return {ValueType::kArray, std::move(v), ValueType::kF32}; }

    ValueType type() const noexcept { return type_; }
    /// Element type of an array.
    ValueType element_type() const noexcept { return element_type_; }
    const Storage& storage() const noexcept { return storage_; }

    /// An integer (any integer type, or a bool) that fits into int64.
    std::optional<std::int64_t> as_int() const noexcept;
    /// A number (integer or float).
    std::optional<double> as_float() const noexcept;
    std::optional<bool> as_bool() const noexcept;
    /// The string, or nullptr.
    const std::string* as_string() const noexcept;
    /// Array contents, or nullptr when the value is not an array of that kind (integer arrays include bools).
    const std::vector<std::int64_t>* as_ints() const noexcept;
    const std::vector<double>* as_floats() const noexcept;
    const std::vector<std::string>* as_strings() const noexcept;
    /// Number of elements of an array; 0 otherwise.
    std::size_t array_size() const noexcept;

    /// Short human-readable rendering ("2048", "\"llama\"", "[32000 x string]").
    std::string to_string() const;

private:
    ValueType type_ = ValueType::kU8;
    ValueType element_type_ = ValueType::kU8;
    Storage storage_ = std::uint64_t{0};
};

/// The ggml tensor type ids of GGUF (`ggml_type`).
enum class GgmlType : std::uint32_t {
    kF32 = 0,
    kF16 = 1,
    kQ4_0 = 2,
    kQ4_1 = 3,
    kQ5_0 = 6,
    kQ5_1 = 7,
    kQ8_0 = 8,
    kQ8_1 = 9,
    kQ2_K = 10,
    kQ3_K = 11,
    kQ4_K = 12,
    kQ5_K = 13,
    kQ6_K = 14,
    kQ8_K = 15,
    kI8 = 24,
    kI16 = 25,
    kI32 = 26,
    kI64 = 27,
    kF64 = 28,
    kBF16 = 30,
};

/// Name of a ggml type id, e.g. "q4_K", "iq2_xxs" ("type 123" for unknown ids).
II_API std::string ggml_type_name(std::uint32_t type);
/// The ii dtype of a ggml type id; kInvalid for types ii does not support (i-quants, ternary, fp4, ...).
II_API DType dtype_from_ggml(std::uint32_t type) noexcept;
/// The ggml type id of an ii dtype, if GGUF can store it.
II_API std::optional<GgmlType> ggml_from_dtype(DType dtype) noexcept;

/// Description of one tensor of a file.
struct TensorInfo {
    std::string name;
    std::uint32_t ggml_type = 0;    ///< as stored in the file
    DType dtype = DType::kInvalid;  ///< kInvalid when ii does not support the ggml type
    Shape shape;                    ///< outermost first (the reverse of GGUF's ne[])
    std::uint64_t offset = 0;       ///< from the start of the file
    std::size_t nbytes = 0;
};

/// A memory-mapped GGUF file (versions 2 and 3). Copies share the mapping.
class II_API File {
public:
    /// Maps and parses `path`.
    static Result<File> open(const std::filesystem::path& path);
    /// Parses a file held in memory; `keep_alive` owns `data` (may be null if `data` outlives the File and its
    /// tensors).
    static Result<File> parse(std::span<const std::byte> data, std::shared_ptr<const void> keep_alive = nullptr);

    File();
    ~File();
    File(const File&);
    File& operator=(const File&);
    File(File&&) noexcept;
    File& operator=(File&&) noexcept;

    std::uint32_t version() const noexcept;
    std::size_t alignment() const noexcept;

    /// Metadata in file order.
    const std::vector<std::pair<std::string, Value>>& metadata() const noexcept;
    /// The value of `key`, or nullptr.
    const Value* find(std::string_view key) const noexcept;
    std::optional<std::int64_t> get_int(std::string_view key) const noexcept;
    std::optional<double> get_float(std::string_view key) const noexcept;
    std::optional<std::string> get_string(std::string_view key) const;

    const std::vector<TensorInfo>& tensors() const noexcept;
    /// The tensor info named `name`, or nullptr.
    const TensorInfo* find_tensor(std::string_view name) const noexcept;
    /// A tensor as a view of the mapping (copied only if the file data is misaligned for its dtype).
    Result<Tensor> tensor(const TensorInfo& info) const;
    Result<Tensor> tensor(std::string_view name) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

/// Writes a GGUF (version 3) file: metadata, then tensor data in the order the tensors were added.
class II_API Writer {
public:
    /// Produces a tensor's data when the writer reaches it (lets `ii quantize` hold one tensor at a time).
    using Source = std::function<Result<Tensor>()>;

    Writer();
    ~Writer();
    Writer(Writer&&) noexcept;
    Writer& operator=(Writer&&) noexcept;

    /// Sets (or replaces) a metadata value; the order of first insertion is kept.
    void set(std::string key, Value value);
    /// Adds a tensor whose data is `tensor` (any dtype GGUF can store).
    void add_tensor(std::string name, Tensor tensor);
    /// Adds a tensor whose data comes from `source`, which must return a tensor of `dtype` and `shape`.
    void add_tensor(std::string name, DType dtype, Shape shape, Source source);
    /// Alignment of tensor data (default 32; written as general.alignment when different).
    void set_alignment(std::size_t alignment);

    /// Writes the file. `progress` (optional) is called after each tensor with its index.
    Status write(const std::filesystem::path& path, const std::function<void(std::size_t)>& progress = {}) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ii::gguf

II_END_DLL_EXPORTS
