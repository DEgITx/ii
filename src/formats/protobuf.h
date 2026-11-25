// SPDX-License-Identifier: MIT
/// @file
/// A minimal decoder of the protobuf wire format (no generated code, no libprotobuf dependency).
///
/// The input is untrusted: every length is checked against the remaining bytes, varints are limited to 10 bytes,
/// and nothing is ever read outside of the given span. Groups (wire types 3 and 4, deprecated) are rejected.
///
/// @code
/// pb::Reader reader(bytes);
/// pb::Field field;
/// while (reader.next(field)) {
///     switch (field.number) {
///     case 1: name = field.string(); break;
///     case 2: II_TRY(pb::append_varints(field, dims)); break;
///     default: break;  // unknown fields are skipped
///     }
/// }
/// II_TRY(reader.status());
/// @endcode
#pragma once

#include <ii/status.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ii::formats::pb {

enum class WireType : std::uint8_t {
    kVarint = 0,
    kFixed64 = 1,
    kLength = 2,
    kStartGroup = 3,
    kEndGroup = 4,
    kFixed32 = 5,
};

/// One decoded field: its number, wire type and value.
struct Field {
    std::uint32_t number = 0;
    WireType wire = WireType::kVarint;
    std::uint64_t scalar = 0;          ///< kVarint, kFixed32, kFixed64
    std::span<const std::byte> bytes;  ///< kLength

    std::int64_t int64() const noexcept { return static_cast<std::int64_t>(scalar); }
    std::int32_t int32() const noexcept { return static_cast<std::int32_t>(static_cast<std::uint32_t>(scalar)); }
    float float32() const noexcept {
        const auto bits = static_cast<std::uint32_t>(scalar);
        float value;
        std::memcpy(&value, &bits, sizeof value);
        return value;
    }
    double float64() const noexcept {
        double value;
        std::memcpy(&value, &scalar, sizeof value);
        return value;
    }
    std::string_view string_view() const noexcept {
        return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
    }
    std::string string() const { return std::string(string_view()); }
};

/// Sequential reader of the fields of one message.
class Reader {
public:
    explicit Reader(std::span<const std::byte> data) noexcept : data_(data) {}

    /// Reads the next field. Returns false at the end of the message or on malformed input (see `status()`).
    bool next(Field& field);

    /// OK unless malformed input was encountered.
    const Status& status() const noexcept { return status_; }

private:
    bool fail(std::string_view what);

    std::span<const std::byte> data_;
    std::size_t pos_ = 0;
    Status status_;
};

/// Decodes a varint from the front of `data`; returns the number of bytes consumed or 0 on malformed input.
std::size_t decode_varint(std::span<const std::byte> data, std::uint64_t& value) noexcept;

/// Appends the values of a repeated integer field (packed or not) to `out`.
Status append_varints(const Field& field, std::vector<std::int64_t>& out);
Status append_varints(const Field& field, std::vector<std::int32_t>& out);
Status append_varints(const Field& field, std::vector<std::uint64_t>& out);

/// Appends the values of a repeated float / double field (packed or not) to `out`.
Status append_floats(const Field& field, std::vector<float>& out);
Status append_doubles(const Field& field, std::vector<double>& out);

/// Writer of the protobuf wire format; used by tests (building models in memory) and tools.
class Writer {
public:
    void varint(std::uint32_t number, std::uint64_t value);
    void fixed32(std::uint32_t number, std::uint32_t value);
    void float32(std::uint32_t number, float value);
    void bytes(std::uint32_t number, std::span<const std::byte> value);
    void string(std::uint32_t number, std::string_view value);
    void message(std::uint32_t number, const Writer& message) { bytes(number, message.data()); }
    void packed_varints(std::uint32_t number, std::span<const std::int64_t> values);
    void packed_floats(std::uint32_t number, std::span<const float> values);

    std::span<const std::byte> data() const noexcept { return buffer_; }
    const std::vector<std::byte>& buffer() const noexcept { return buffer_; }

private:
    void key(std::uint32_t number, WireType wire);
    void raw_varint(std::uint64_t value);
    std::vector<std::byte> buffer_;
};

}  // namespace ii::formats::pb
