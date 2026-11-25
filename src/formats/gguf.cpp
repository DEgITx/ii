// SPDX-License-Identifier: MIT
// GGUF reader and writer. The layout (version 3):
//   "GGUF" u32 version, u64 tensor_count, u64 metadata_count,
//   metadata: { string key, u32 type, value },
//   tensor infos: { string name, u32 n_dims, u64 ne[n_dims], u32 ggml_type, u64 offset },
//   padding to `general.alignment` (default 32), tensor data (each tensor aligned).
// Strings are u64 length + bytes (no terminator). All integers are little-endian.
#include <ii/gguf.h>

#include "core/platform/mapped_file.h"
#include "tensor/quant.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>

namespace ii::gguf {

namespace {

constexpr std::uint32_t kMagic = 0x46554747;  // "GGUF"
constexpr std::size_t kDefaultAlignment = 32;
constexpr int kMaxDims = 4;  // GGML_MAX_DIMS
constexpr int kMaxNesting = 4;

struct TypeName {
    std::uint32_t id;
    std::string_view name;
};

constexpr TypeName kGgmlTypeNames[] = {
    {0, "f32"},      {1, "f16"},     {2, "q4_0"},     {3, "q4_1"},   {6, "q5_0"},    {7, "q5_1"},   {8, "q8_0"},
    {9, "q8_1"},     {10, "q2_K"},   {11, "q3_K"},    {12, "q4_K"},  {13, "q5_K"},   {14, "q6_K"},  {15, "q8_K"},
    {16, "iq2_xxs"}, {17, "iq2_xs"}, {18, "iq3_xxs"}, {19, "iq1_s"}, {20, "iq4_nl"}, {21, "iq3_s"}, {22, "iq2_s"},
    {23, "iq4_xs"},  {24, "i8"},     {25, "i16"},     {26, "i32"},   {27, "i64"},    {28, "f64"},   {29, "iq1_m"},
    {30, "bf16"},    {34, "tq1_0"},  {35, "tq2_0"},   {39, "mxfp4"},
};

constexpr std::pair<GgmlType, DType> kTypeMap[] = {
    {GgmlType::kF32, DType::kF32},   {GgmlType::kF16, DType::kF16},   {GgmlType::kBF16, DType::kBF16},
    {GgmlType::kF64, DType::kF64},   {GgmlType::kI8, DType::kI8},     {GgmlType::kI16, DType::kI16},
    {GgmlType::kI32, DType::kI32},   {GgmlType::kI64, DType::kI64},   {GgmlType::kQ4_0, DType::kQ4_0},
    {GgmlType::kQ4_1, DType::kQ4_1}, {GgmlType::kQ5_0, DType::kQ5_0}, {GgmlType::kQ5_1, DType::kQ5_1},
    {GgmlType::kQ8_0, DType::kQ8_0}, {GgmlType::kQ8_1, DType::kQ8_1}, {GgmlType::kQ2_K, DType::kQ2_K},
    {GgmlType::kQ3_K, DType::kQ3_K}, {GgmlType::kQ4_K, DType::kQ4_K}, {GgmlType::kQ5_K, DType::kQ5_K},
    {GgmlType::kQ6_K, DType::kQ6_K}, {GgmlType::kQ8_K, DType::kQ8_K},
};

/// Bounds-checked little-endian reader.
class Reader {
public:
    explicit Reader(std::span<const std::byte> data) : data_(data) {}

    std::size_t position() const noexcept { return position_; }
    std::size_t remaining() const noexcept { return data_.size() - position_; }

    template <class T>
    Result<T> read() {
        if (remaining() < sizeof(T)) {
            return errors::data_corrupted("GGUF: unexpected end of file at offset {}", position_);
        }
        T value;
        std::memcpy(&value, data_.data() + position_, sizeof(T));
        position_ += sizeof(T);
        return value;
    }

    Result<std::string> read_string() {
        II_ASSIGN_OR_RETURN(const std::uint64_t length, read<std::uint64_t>());
        if (length > remaining()) {
            return errors::data_corrupted("GGUF: string of {} bytes at offset {} exceeds the file", length, position_);
        }
        std::string text(reinterpret_cast<const char*>(data_.data() + position_), static_cast<std::size_t>(length));
        position_ += static_cast<std::size_t>(length);
        return text;
    }

    Status skip(std::uint64_t bytes) {
        if (bytes > remaining()) {
            return errors::data_corrupted("GGUF: unexpected end of file at offset {}", position_);
        }
        position_ += static_cast<std::size_t>(bytes);
        return ok_status();
    }

private:
    std::span<const std::byte> data_;
    std::size_t position_ = 0;
};

std::size_t scalar_size(ValueType type) noexcept {
    switch (type) {
    case ValueType::kU8:
    case ValueType::kI8:
    case ValueType::kBool:
        return 1;
    case ValueType::kU16:
    case ValueType::kI16:
        return 2;
    case ValueType::kU32:
    case ValueType::kI32:
    case ValueType::kF32:
        return 4;
    case ValueType::kU64:
    case ValueType::kI64:
    case ValueType::kF64:
        return 8;
    default:
        return 0;
    }
}

bool is_float(ValueType type) noexcept {
    return type == ValueType::kF32 || type == ValueType::kF64;
}

/// Reads one scalar of `type` as int64 / uint64 / double / bool storage.
Result<Value::Storage> read_scalar(Reader& in, ValueType type) {
    switch (type) {
    case ValueType::kU8: {
        II_ASSIGN_OR_RETURN(const std::uint8_t v, in.read<std::uint8_t>());
        return Value::Storage{std::uint64_t{v}};
    }
    case ValueType::kI8: {
        II_ASSIGN_OR_RETURN(const std::int8_t v, in.read<std::int8_t>());
        return Value::Storage{std::int64_t{v}};
    }
    case ValueType::kU16: {
        II_ASSIGN_OR_RETURN(const std::uint16_t v, in.read<std::uint16_t>());
        return Value::Storage{std::uint64_t{v}};
    }
    case ValueType::kI16: {
        II_ASSIGN_OR_RETURN(const std::int16_t v, in.read<std::int16_t>());
        return Value::Storage{std::int64_t{v}};
    }
    case ValueType::kU32: {
        II_ASSIGN_OR_RETURN(const std::uint32_t v, in.read<std::uint32_t>());
        return Value::Storage{std::uint64_t{v}};
    }
    case ValueType::kI32: {
        II_ASSIGN_OR_RETURN(const std::int32_t v, in.read<std::int32_t>());
        return Value::Storage{std::int64_t{v}};
    }
    case ValueType::kU64: {
        II_ASSIGN_OR_RETURN(const std::uint64_t v, in.read<std::uint64_t>());
        return Value::Storage{v};
    }
    case ValueType::kI64: {
        II_ASSIGN_OR_RETURN(const std::int64_t v, in.read<std::int64_t>());
        return Value::Storage{v};
    }
    case ValueType::kF32: {
        II_ASSIGN_OR_RETURN(const float v, in.read<float>());
        return Value::Storage{static_cast<double>(v)};
    }
    case ValueType::kF64: {
        II_ASSIGN_OR_RETURN(const double v, in.read<double>());
        return Value::Storage{v};
    }
    case ValueType::kBool: {
        II_ASSIGN_OR_RETURN(const std::uint8_t v, in.read<std::uint8_t>());
        if (v > 1) {
            return errors::data_corrupted("GGUF: bool value {} at offset {}", v, in.position() - 1);
        }
        return Value::Storage{v != 0};
    }
    default:
        return errors::data_corrupted("GGUF: value type {} is not a scalar", static_cast<std::uint32_t>(type));
    }
}

Status skip_value(Reader& in, ValueType type, int depth);

Status skip_array(Reader& in, int depth) {
    II_ASSIGN_OR_RETURN(const std::uint32_t raw_type, in.read<std::uint32_t>());
    II_ASSIGN_OR_RETURN(const std::uint64_t count, in.read<std::uint64_t>());
    const auto element = static_cast<ValueType>(raw_type);
    if (const std::size_t size = scalar_size(element); size > 0) {
        if (count > in.remaining() / size) {
            return errors::data_corrupted("GGUF: array of {} elements exceeds the file", count);
        }
        return in.skip(count * size);
    }
    if (count > in.remaining() / 8) {  // every string or array takes at least 8 bytes
        return errors::data_corrupted("GGUF: array of {} elements exceeds the file", count);
    }
    for (std::uint64_t i = 0; i < count; ++i) {
        II_TRY(skip_value(in, element, depth + 1));
    }
    return ok_status();
}

Status skip_value(Reader& in, ValueType type, int depth) {
    if (depth > kMaxNesting) {
        return errors::data_corrupted("GGUF: arrays nested too deeply");
    }
    if (type == ValueType::kString) {
        return in.read_string().status();
    }
    if (type == ValueType::kArray) {
        return skip_array(in, depth);
    }
    if (const std::size_t size = scalar_size(type); size > 0) {
        return in.skip(size);
    }
    return errors::data_corrupted("GGUF: unknown value type {}", static_cast<std::uint32_t>(type));
}

Result<Value> read_value(Reader& in, ValueType type) {
    if (type == ValueType::kString) {
        II_ASSIGN_OR_RETURN(std::string text, in.read_string());
        return Value(type, std::move(text));
    }
    if (type != ValueType::kArray) {
        if (scalar_size(type) == 0) {
            return errors::data_corrupted("GGUF: unknown value type {}", static_cast<std::uint32_t>(type));
        }
        II_ASSIGN_OR_RETURN(Value::Storage storage, read_scalar(in, type));
        return Value(type, std::move(storage));
    }
    II_ASSIGN_OR_RETURN(const std::uint32_t raw_type, in.read<std::uint32_t>());
    II_ASSIGN_OR_RETURN(const std::uint64_t count, in.read<std::uint64_t>());
    const auto element = static_cast<ValueType>(raw_type);
    if (element == ValueType::kString) {
        if (count > in.remaining() / 8) {
            return errors::data_corrupted("GGUF: array of {} strings exceeds the file", count);
        }
        std::vector<std::string> strings;
        strings.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            II_ASSIGN_OR_RETURN(std::string text, in.read_string());
            strings.push_back(std::move(text));
        }
        return Value(type, std::move(strings), element);
    }
    if (element == ValueType::kArray) {
        // Nested arrays are skipped (no known model needs them); every inner array takes at least 12 bytes.
        if (count > in.remaining() / 12) {
            return errors::data_corrupted("GGUF: array of {} arrays exceeds the file", count);
        }
        for (std::uint64_t i = 0; i < count; ++i) {
            II_TRY(skip_array(in, 1));
        }
        return Value(type, std::vector<std::int64_t>{}, element);
    }
    const std::size_t size = scalar_size(element);
    if (size == 0) {
        return errors::data_corrupted("GGUF: array of unknown type {}", raw_type);
    }
    if (count > in.remaining() / size) {
        return errors::data_corrupted("GGUF: array of {} elements exceeds the file", count);
    }
    if (is_float(element)) {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            II_ASSIGN_OR_RETURN(const Value::Storage v, read_scalar(in, element));
            values.push_back(std::get<double>(v));
        }
        return Value(type, std::move(values), element);
    }
    std::vector<std::int64_t> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        II_ASSIGN_OR_RETURN(const Value::Storage v, read_scalar(in, element));
        if (const auto* s = std::get_if<std::int64_t>(&v)) {
            values.push_back(*s);
        } else if (const auto* u = std::get_if<std::uint64_t>(&v)) {
            values.push_back(static_cast<std::int64_t>(*u));  // u64 above 2^63 wraps (never used for arrays)
        } else {
            values.push_back(std::get<bool>(v) ? 1 : 0);
        }
    }
    return Value(type, std::move(values), element);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Names and types

std::string_view to_string(ValueType type) noexcept {
    switch (type) {
    case ValueType::kU8:
        return "u8";
    case ValueType::kI8:
        return "i8";
    case ValueType::kU16:
        return "u16";
    case ValueType::kI16:
        return "i16";
    case ValueType::kU32:
        return "u32";
    case ValueType::kI32:
        return "i32";
    case ValueType::kF32:
        return "f32";
    case ValueType::kBool:
        return "bool";
    case ValueType::kString:
        return "string";
    case ValueType::kArray:
        return "array";
    case ValueType::kU64:
        return "u64";
    case ValueType::kI64:
        return "i64";
    case ValueType::kF64:
        return "f64";
    }
    return "unknown";
}

std::string ggml_type_name(std::uint32_t type) {
    for (const TypeName& entry : kGgmlTypeNames) {
        if (entry.id == type) {
            return std::string(entry.name);
        }
    }
    return std::format("type {}", type);
}

DType dtype_from_ggml(std::uint32_t type) noexcept {
    for (const auto& [ggml, dtype] : kTypeMap) {
        if (static_cast<std::uint32_t>(ggml) == type) {
            return dtype;
        }
    }
    return DType::kInvalid;
}

std::optional<GgmlType> ggml_from_dtype(DType dtype) noexcept {
    for (const auto& [ggml, mapped] : kTypeMap) {
        if (mapped == dtype) {
            return ggml;
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------------------------------------------
// Value

std::optional<std::int64_t> Value::as_int() const noexcept {
    if (const auto* s = std::get_if<std::int64_t>(&storage_)) {
        return *s;
    }
    if (const auto* u = std::get_if<std::uint64_t>(&storage_)) {
        if (*u <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return static_cast<std::int64_t>(*u);
        }
        return std::nullopt;
    }
    if (const auto* b = std::get_if<bool>(&storage_)) {
        return *b ? 1 : 0;
    }
    return std::nullopt;
}

std::optional<double> Value::as_float() const noexcept {
    if (const auto* d = std::get_if<double>(&storage_)) {
        return *d;
    }
    if (const auto* s = std::get_if<std::int64_t>(&storage_)) {
        return static_cast<double>(*s);
    }
    if (const auto* u = std::get_if<std::uint64_t>(&storage_)) {
        return static_cast<double>(*u);
    }
    return std::nullopt;
}

std::optional<bool> Value::as_bool() const noexcept {
    if (const auto* b = std::get_if<bool>(&storage_)) {
        return *b;
    }
    if (const std::optional<std::int64_t> i = as_int()) {
        return *i != 0;
    }
    return std::nullopt;
}

const std::string* Value::as_string() const noexcept {
    return std::get_if<std::string>(&storage_);
}

const std::vector<std::int64_t>* Value::as_ints() const noexcept {
    return type_ == ValueType::kArray ? std::get_if<std::vector<std::int64_t>>(&storage_) : nullptr;
}

const std::vector<double>* Value::as_floats() const noexcept {
    return type_ == ValueType::kArray ? std::get_if<std::vector<double>>(&storage_) : nullptr;
}

const std::vector<std::string>* Value::as_strings() const noexcept {
    return type_ == ValueType::kArray ? std::get_if<std::vector<std::string>>(&storage_) : nullptr;
}

std::size_t Value::array_size() const noexcept {
    if (const auto* i = as_ints()) {
        return i->size();
    }
    if (const auto* f = as_floats()) {
        return f->size();
    }
    if (const auto* s = as_strings()) {
        return s->size();
    }
    return 0;
}

std::string Value::to_string() const {
    if (type_ == ValueType::kArray) {
        return std::format("[{} x {}]", array_size(), gguf::to_string(element_type_));
    }
    if (const std::string* s = as_string()) {
        std::string text = s->size() > 60 ? s->substr(0, 57) + "..." : *s;
        for (char& c : text) {
            if (static_cast<unsigned char>(c) < 0x20) {
                c = ' ';
            }
        }
        return "\"" + text + "\"";
    }
    if (const auto* b = std::get_if<bool>(&storage_)) {
        return *b ? "true" : "false";
    }
    if (const auto* d = std::get_if<double>(&storage_)) {
        return std::format("{}", *d);
    }
    if (const auto* s = std::get_if<std::int64_t>(&storage_)) {
        return std::format("{}", *s);
    }
    return std::format("{}", std::get<std::uint64_t>(storage_));
}

// ---------------------------------------------------------------------------------------------------------------
// File

struct File::Impl {
    std::shared_ptr<const void> keep_alive;
    std::span<const std::byte> data;
    std::uint32_t version = 0;
    std::size_t alignment = kDefaultAlignment;
    std::vector<std::pair<std::string, Value>> metadata;
    std::vector<TensorInfo> tensors;
};

File::File() = default;
File::~File() = default;
File::File(const File&) = default;
File& File::operator=(const File&) = default;
File::File(File&&) noexcept = default;
File& File::operator=(File&&) noexcept = default;

Result<File> File::open(const std::filesystem::path& path) {
    II_ASSIGN_OR_RETURN(platform::MappedFile mapped, platform::MappedFile::open(path));
    auto owner = std::make_shared<platform::MappedFile>(std::move(mapped));
    const std::span<const std::byte> bytes = owner->bytes();
    Result<File> file = parse(bytes, std::move(owner));
    if (!file.ok()) {
        return Status(file.status()).add_context(path.string());
    }
    return file;
}

Result<File> File::parse(std::span<const std::byte> data, std::shared_ptr<const void> keep_alive) {
    auto impl = std::make_shared<Impl>();
    impl->keep_alive = std::move(keep_alive);
    impl->data = data;
    Reader in(data);
    II_ASSIGN_OR_RETURN(const std::uint32_t magic, in.read<std::uint32_t>());
    if (magic != kMagic) {
        return errors::data_corrupted("not a GGUF file (bad magic)");
    }
    II_ASSIGN_OR_RETURN(impl->version, in.read<std::uint32_t>());
    if (impl->version != 2 && impl->version != 3) {
        return errors::unsupported("GGUF version {} is not supported (2 and 3 are)", impl->version);
    }
    II_ASSIGN_OR_RETURN(const std::uint64_t tensor_count, in.read<std::uint64_t>());
    II_ASSIGN_OR_RETURN(const std::uint64_t metadata_count, in.read<std::uint64_t>());
    // Every metadata entry takes at least 12 bytes, every tensor info at least 24.
    if (metadata_count > in.remaining() / 12 || tensor_count > in.remaining() / 24) {
        return errors::data_corrupted("GGUF: {} metadata entries and {} tensors exceed the file", metadata_count,
                                      tensor_count);
    }
    impl->metadata.reserve(static_cast<std::size_t>(metadata_count));
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        II_ASSIGN_OR_RETURN(std::string key, in.read_string());
        II_ASSIGN_OR_RETURN(const std::uint32_t type, in.read<std::uint32_t>());
        Result<Value> value = read_value(in, static_cast<ValueType>(type));
        if (!value.ok()) {
            return Status(value.status()).add_context(std::format("metadata '{}'", key));
        }
        impl->metadata.emplace_back(std::move(key), std::move(*value));
    }
    for (const auto& [key, value] : impl->metadata) {
        if (key == "general.alignment") {
            const std::optional<std::int64_t> alignment = value.as_int();
            if (!alignment || *alignment <= 0 || *alignment > 1 << 20 ||
                !std::has_single_bit(static_cast<std::uint64_t>(*alignment))) {
                return errors::data_corrupted("GGUF: general.alignment {} is not a power of two", value.to_string());
            }
            impl->alignment = static_cast<std::size_t>(*alignment);
        }
    }

    impl->tensors.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t t = 0; t < tensor_count; ++t) {
        TensorInfo info;
        II_ASSIGN_OR_RETURN(info.name, in.read_string());
        II_ASSIGN_OR_RETURN(const std::uint32_t n_dims, in.read<std::uint32_t>());
        if (n_dims > kMaxDims) {
            return errors::data_corrupted("GGUF: tensor '{}' has {} dimensions", info.name, n_dims);
        }
        std::uint64_t ne[kMaxDims] = {1, 1, 1, 1};
        std::uint64_t count = 1;
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            II_ASSIGN_OR_RETURN(ne[d], in.read<std::uint64_t>());
            if (ne[d] > (std::uint64_t{1} << 40) || (ne[d] != 0 && count > (std::uint64_t{1} << 46) / ne[d])) {
                return errors::data_corrupted("GGUF: tensor '{}' is too large", info.name);
            }
            count *= ne[d];
        }
        for (std::uint32_t d = n_dims; d-- > 0;) {
            info.shape.push_back(static_cast<std::int64_t>(ne[d]));
        }
        II_ASSIGN_OR_RETURN(info.ggml_type, in.read<std::uint32_t>());
        II_ASSIGN_OR_RETURN(const std::uint64_t offset, in.read<std::uint64_t>());
        info.dtype = dtype_from_ggml(info.ggml_type);
        if (info.dtype != DType::kInvalid) {
            const std::uint64_t row = n_dims == 0 ? 1 : ne[0];
            if (row % dtype_info(info.dtype).block_size != 0) {
                return errors::data_corrupted("GGUF: tensor '{}' ({}) has rows of {} elements, not whole blocks",
                                              info.name, ggml_type_name(info.ggml_type), row);
            }
            info.nbytes = storage_size(info.dtype, static_cast<std::int64_t>(count));
        }
        info.offset = offset;  // relative to the data section for now
        impl->tensors.push_back(std::move(info));
    }
    // Tensor data starts at the next multiple of the alignment.
    const std::size_t header_end = in.position();
    const std::size_t data_start = (header_end + impl->alignment - 1) / impl->alignment * impl->alignment;
    if (data_start > data.size() && !impl->tensors.empty()) {
        return errors::data_corrupted("GGUF: the file ends before its tensor data");
    }
    for (TensorInfo& info : impl->tensors) {
        if (info.offset % impl->alignment != 0) {
            return errors::data_corrupted("GGUF: tensor '{}' data offset {} is not aligned to {}", info.name,
                                          info.offset, impl->alignment);
        }
        if (info.offset > data.size() || data_start > data.size() - info.offset ||
            info.nbytes > data.size() - data_start - info.offset) {
            return errors::data_corrupted("GGUF: tensor '{}' data [{}, +{}) is outside the file", info.name,
                                          info.offset, info.nbytes);
        }
        info.offset += data_start;
    }
    File file;
    file.impl_ = std::move(impl);
    return file;
}

std::uint32_t File::version() const noexcept {
    return impl_ ? impl_->version : 0;
}

std::size_t File::alignment() const noexcept {
    return impl_ ? impl_->alignment : kDefaultAlignment;
}

const std::vector<std::pair<std::string, Value>>& File::metadata() const noexcept {
    static const std::vector<std::pair<std::string, Value>> empty;
    return impl_ ? impl_->metadata : empty;
}

const Value* File::find(std::string_view key) const noexcept {
    for (const auto& [name, value] : metadata()) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

std::optional<std::int64_t> File::get_int(std::string_view key) const noexcept {
    const Value* value = find(key);
    return value != nullptr ? value->as_int() : std::nullopt;
}

std::optional<double> File::get_float(std::string_view key) const noexcept {
    const Value* value = find(key);
    return value != nullptr ? value->as_float() : std::nullopt;
}

std::optional<std::string> File::get_string(std::string_view key) const {
    const Value* value = find(key);
    if (value == nullptr || value->as_string() == nullptr) {
        return std::nullopt;
    }
    return *value->as_string();
}

const std::vector<TensorInfo>& File::tensors() const noexcept {
    static const std::vector<TensorInfo> empty;
    return impl_ ? impl_->tensors : empty;
}

const TensorInfo* File::find_tensor(std::string_view name) const noexcept {
    for (const TensorInfo& info : tensors()) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

Result<Tensor> File::tensor(const TensorInfo& info) const {
    if (info.dtype == DType::kInvalid) {
        return errors::unsupported("tensor '{}' has ggml type {}, which ii does not support", info.name,
                                   ggml_type_name(info.ggml_type));
    }
    const std::byte* data = impl_->data.data() + info.offset;
    const std::size_t alignment = std::max<std::size_t>(element_size(info.dtype), 4);
    if (impl_->keep_alive != nullptr && reinterpret_cast<std::uintptr_t>(data) % alignment == 0 && info.nbytes > 0) {
        auto buffer = Buffer::wrap(const_cast<std::byte*>(data), info.nbytes, impl_);
        return Tensor::view(std::move(buffer), info.dtype, info.shape, contiguous_strides(info.shape));
    }
    return Tensor::from_bytes(info.dtype, info.shape, data, info.nbytes);
}

Result<Tensor> File::tensor(std::string_view name) const {
    const TensorInfo* info = find_tensor(name);
    if (info == nullptr) {
        return errors::not_found("the GGUF file has no tensor '{}'", name);
    }
    return tensor(*info);
}

// ---------------------------------------------------------------------------------------------------------------
// Writer

namespace {

class Output {
public:
    void u8(std::uint8_t v) { bytes_.push_back(static_cast<char>(v)); }
    template <class T>
    void raw(T v) {
        char buffer[sizeof(T)];
        std::memcpy(buffer, &v, sizeof(T));
        bytes_.append(buffer, sizeof(T));
    }
    void string(std::string_view text) {
        raw<std::uint64_t>(text.size());
        bytes_.append(text);
    }
    std::string& bytes() noexcept { return bytes_; }

private:
    std::string bytes_;
};

void write_scalar(Output& out, ValueType type, const Value::Storage& storage) {
    std::int64_t i = 0;
    double d = 0;
    if (const auto* s = std::get_if<std::int64_t>(&storage)) {
        i = *s;
        d = static_cast<double>(*s);
    } else if (const auto* u = std::get_if<std::uint64_t>(&storage)) {
        i = static_cast<std::int64_t>(*u);
        d = static_cast<double>(*u);
    } else if (const auto* f = std::get_if<double>(&storage)) {
        d = *f;
        i = static_cast<std::int64_t>(*f);
    } else if (const auto* b = std::get_if<bool>(&storage)) {
        i = *b ? 1 : 0;
    }
    switch (type) {
    case ValueType::kU8:
    case ValueType::kI8:
    case ValueType::kBool:
        out.u8(static_cast<std::uint8_t>(i));
        break;
    case ValueType::kU16:
    case ValueType::kI16:
        out.raw(static_cast<std::uint16_t>(i));
        break;
    case ValueType::kU32:
    case ValueType::kI32:
        out.raw(static_cast<std::uint32_t>(i));
        break;
    case ValueType::kU64:
    case ValueType::kI64:
        out.raw(static_cast<std::uint64_t>(i));
        break;
    case ValueType::kF32:
        out.raw(static_cast<float>(d));
        break;
    case ValueType::kF64:
        out.raw(d);
        break;
    default:
        break;
    }
}

void write_value(Output& out, const Value& value) {
    out.raw(static_cast<std::uint32_t>(value.type()));
    if (value.type() == ValueType::kString) {
        out.string(*value.as_string());
        return;
    }
    if (value.type() != ValueType::kArray) {
        write_scalar(out, value.type(), value.storage());
        return;
    }
    out.raw(static_cast<std::uint32_t>(value.element_type()));
    out.raw(static_cast<std::uint64_t>(value.array_size()));
    if (const auto* strings = value.as_strings()) {
        for (const std::string& s : *strings) {
            out.string(s);
        }
    } else if (const auto* ints = value.as_ints()) {
        for (const std::int64_t v : *ints) {
            write_scalar(out, value.element_type(), Value::Storage{v});
        }
    } else if (const auto* floats = value.as_floats()) {
        for (const double v : *floats) {
            write_scalar(out, value.element_type(), Value::Storage{v});
        }
    }
}

}  // namespace

struct Writer::Impl {
    struct Entry {
        std::string name;
        DType dtype = DType::kInvalid;
        Shape shape;
        Source source;
    };
    std::vector<std::pair<std::string, Value>> metadata;
    std::vector<Entry> tensors;
    std::size_t alignment = kDefaultAlignment;
};

Writer::Writer() : impl_(std::make_unique<Impl>()) {}
Writer::~Writer() = default;
Writer::Writer(Writer&&) noexcept = default;
Writer& Writer::operator=(Writer&&) noexcept = default;

void Writer::set(std::string key, Value value) {
    for (auto& [name, existing] : impl_->metadata) {
        if (name == key) {
            existing = std::move(value);
            return;
        }
    }
    impl_->metadata.emplace_back(std::move(key), std::move(value));
}

void Writer::add_tensor(std::string name, Tensor tensor) {
    const DType dtype = tensor.dtype();
    const Shape shape = tensor.shape();
    impl_->tensors.push_back(Impl::Entry{std::move(name), dtype, shape,
                                         [tensor = std::move(tensor)]() -> Result<Tensor> { return tensor; }});
}

void Writer::add_tensor(std::string name, DType dtype, Shape shape, Source source) {
    impl_->tensors.push_back(Impl::Entry{std::move(name), dtype, shape, std::move(source)});
}

void Writer::set_alignment(std::size_t alignment) {
    impl_->alignment = alignment;
}

Status Writer::write(const std::filesystem::path& path, const std::function<void(std::size_t)>& progress) const {
    const std::size_t alignment = impl_->alignment;
    if (alignment == 0 || !std::has_single_bit(alignment)) {
        return errors::invalid_argument("GGUF alignment {} is not a power of two", alignment);
    }
    std::vector<std::pair<std::string, Value>> metadata = impl_->metadata;
    if (alignment != kDefaultAlignment) {
        metadata.emplace_back("general.alignment", Value::u32(static_cast<std::uint32_t>(alignment)));
    }
    Output header;
    header.raw(kMagic);
    header.raw(std::uint32_t{3});
    header.raw(static_cast<std::uint64_t>(impl_->tensors.size()));
    header.raw(static_cast<std::uint64_t>(metadata.size()));
    for (const auto& [key, value] : metadata) {
        header.string(key);
        write_value(header, value);
    }
    std::uint64_t offset = 0;
    for (const Impl::Entry& entry : impl_->tensors) {
        const std::optional<GgmlType> type = ggml_from_dtype(entry.dtype);
        if (!type) {
            return errors::unsupported("GGUF cannot store tensor '{}' of dtype {}", entry.name,
                                       ii::to_string(entry.dtype));
        }
        if (entry.shape.rank() > kMaxDims) {
            return errors::unsupported("GGUF tensors have at most 4 dimensions; '{}' has {}", entry.name,
                                       entry.shape.rank());
        }
        header.string(entry.name);
        header.raw(static_cast<std::uint32_t>(entry.shape.rank()));
        for (int d = entry.shape.rank(); d-- > 0;) {
            header.raw(static_cast<std::uint64_t>(entry.shape[d]));
        }
        header.raw(static_cast<std::uint32_t>(*type));
        header.raw(offset);
        const std::uint64_t size = storage_size(entry.dtype, entry.shape.numel());
        offset += (size + alignment - 1) / alignment * alignment;
    }
    std::string& bytes = header.bytes();
    bytes.resize((bytes.size() + alignment - 1) / alignment * alignment, '\0');

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return errors::io_error("cannot create '{}'", path.string());
    }
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    const std::string padding(alignment, '\0');
    for (std::size_t t = 0; t < impl_->tensors.size(); ++t) {
        const Impl::Entry& entry = impl_->tensors[t];
        II_ASSIGN_OR_RETURN(Tensor tensor, entry.source());
        if (tensor.dtype() != entry.dtype || !(tensor.shape() == entry.shape)) {
            return errors::invalid_argument("GGUF tensor '{}' was declared {}{} but is {}{}", entry.name,
                                            ii::to_string(entry.dtype), entry.shape.to_string(),
                                            ii::to_string(tensor.dtype()), tensor.shape().to_string());
        }
        II_ASSIGN_OR_RETURN(tensor, tensor.contiguous());
        const std::size_t size = tensor.nbytes();
        file.write(static_cast<const char*>(tensor.data()), static_cast<std::streamsize>(size));
        file.write(padding.data(), static_cast<std::streamsize>((alignment - size % alignment) % alignment));
        if (!file) {
            return errors::io_error("writing '{}' failed", path.string());
        }
        if (progress) {
            progress(t);
        }
    }
    file.close();
    if (!file) {
        return errors::io_error("writing '{}' failed", path.string());
    }
    return ok_status();
}

}  // namespace ii::gguf
