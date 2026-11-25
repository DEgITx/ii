// SPDX-License-Identifier: MIT
#include "formats/protobuf.h"

#include <bit>

namespace ii::formats::pb {

static_assert(std::endian::native == std::endian::little, "the protobuf decoder assumes a little-endian host");

std::size_t decode_varint(std::span<const std::byte> data, std::uint64_t& value) noexcept {
    value = 0;
    for (std::size_t i = 0; i < data.size() && i < 10; ++i) {
        const auto byte = static_cast<std::uint8_t>(data[i]);
        if (i == 9 && byte > 1) {
            return 0;  // more than 64 bits
        }
        value |= static_cast<std::uint64_t>(byte & 0x7F) << (7 * i);
        if ((byte & 0x80) == 0) {
            return i + 1;
        }
    }
    return 0;
}

bool Reader::fail(std::string_view what) {
    if (status_.ok()) {
        status_ = errors::data_corrupted("malformed protobuf data at byte {}: {}", pos_, what);
    }
    pos_ = data_.size();
    return false;
}

bool Reader::next(Field& field) {
    if (pos_ >= data_.size() || !status_.ok()) {
        return false;
    }
    std::uint64_t key = 0;
    std::size_t used = decode_varint(data_.subspan(pos_), key);
    if (used == 0) {
        return fail("bad field key");
    }
    pos_ += used;
    const std::uint64_t number = key >> 3;
    if (number == 0 || number > 0x1FFFFFFF) {
        return fail("bad field number");
    }
    field.number = static_cast<std::uint32_t>(number);
    field.wire = static_cast<WireType>(key & 7);
    field.scalar = 0;
    field.bytes = {};
    const std::size_t remaining = data_.size() - pos_;
    switch (field.wire) {
    case WireType::kVarint:
        used = decode_varint(data_.subspan(pos_), field.scalar);
        if (used == 0) {
            return fail("bad varint");
        }
        pos_ += used;
        return true;
    case WireType::kFixed64:
        if (remaining < 8) {
            return fail("truncated fixed64");
        }
        std::memcpy(&field.scalar, data_.data() + pos_, 8);
        pos_ += 8;
        return true;
    case WireType::kFixed32: {
        if (remaining < 4) {
            return fail("truncated fixed32");
        }
        std::uint32_t value = 0;
        std::memcpy(&value, data_.data() + pos_, 4);
        field.scalar = value;
        pos_ += 4;
        return true;
    }
    case WireType::kLength: {
        std::uint64_t length = 0;
        used = decode_varint(data_.subspan(pos_), length);
        if (used == 0) {
            return fail("bad length");
        }
        pos_ += used;
        if (length > data_.size() - pos_) {
            return fail("length exceeds the data");
        }
        field.bytes = data_.subspan(pos_, static_cast<std::size_t>(length));
        pos_ += static_cast<std::size_t>(length);
        return true;
    }
    case WireType::kStartGroup:
    case WireType::kEndGroup:
        return fail("groups are not supported");
    }
    return fail("unknown wire type");
}

namespace {

template <class T>
Status append_varints_impl(const Field& field, std::vector<T>& out) {
    if (field.wire == WireType::kVarint) {
        out.push_back(static_cast<T>(field.scalar));
        return ok_status();
    }
    if (field.wire != WireType::kLength) {
        return errors::data_corrupted("field {}: expected integers, got wire type {}", field.number,
                                      static_cast<int>(field.wire));
    }
    std::span<const std::byte> data = field.bytes;
    while (!data.empty()) {
        std::uint64_t value = 0;
        const std::size_t used = decode_varint(data, value);
        if (used == 0) {
            return errors::data_corrupted("field {}: bad packed varint", field.number);
        }
        out.push_back(static_cast<T>(value));
        data = data.subspan(used);
    }
    return ok_status();
}

template <class T, WireType kWire>
Status append_fixed_impl(const Field& field, std::vector<T>& out) {
    if (field.wire == kWire) {
        if constexpr (sizeof(T) == 4) {
            const auto bits = static_cast<std::uint32_t>(field.scalar);
            T value;
            std::memcpy(&value, &bits, sizeof value);
            out.push_back(value);
        } else {
            T value;
            std::memcpy(&value, &field.scalar, sizeof value);
            out.push_back(value);
        }
        return ok_status();
    }
    if (field.wire != WireType::kLength || field.bytes.size() % sizeof(T) != 0) {
        return errors::data_corrupted("field {}: malformed packed {}-byte values", field.number, sizeof(T));
    }
    const std::size_t count = field.bytes.size() / sizeof(T);
    const std::size_t start = out.size();
    out.resize(start + count);
    if (count != 0) {
        std::memcpy(out.data() + start, field.bytes.data(), field.bytes.size());
    }
    return ok_status();
}

}  // namespace

Status append_varints(const Field& field, std::vector<std::int64_t>& out) {
    return append_varints_impl(field, out);
}

Status append_varints(const Field& field, std::vector<std::int32_t>& out) {
    return append_varints_impl(field, out);
}

Status append_varints(const Field& field, std::vector<std::uint64_t>& out) {
    return append_varints_impl(field, out);
}

Status append_floats(const Field& field, std::vector<float>& out) {
    return append_fixed_impl<float, WireType::kFixed32>(field, out);
}

Status append_doubles(const Field& field, std::vector<double>& out) {
    return append_fixed_impl<double, WireType::kFixed64>(field, out);
}

// ---------------------------------------------------------------------------------------------------------------
// Writer

void Writer::raw_varint(std::uint64_t value) {
    while (value >= 0x80) {
        buffer_.push_back(static_cast<std::byte>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    buffer_.push_back(static_cast<std::byte>(value));
}

void Writer::key(std::uint32_t number, WireType wire) {
    raw_varint((static_cast<std::uint64_t>(number) << 3) | static_cast<std::uint64_t>(wire));
}

void Writer::varint(std::uint32_t number, std::uint64_t value) {
    key(number, WireType::kVarint);
    raw_varint(value);
}

void Writer::fixed32(std::uint32_t number, std::uint32_t value) {
    key(number, WireType::kFixed32);
    for (int i = 0; i < 4; ++i) {
        buffer_.push_back(static_cast<std::byte>((value >> (8 * i)) & 0xFF));
    }
}

void Writer::float32(std::uint32_t number, float value) {
    fixed32(number, std::bit_cast<std::uint32_t>(value));
}

void Writer::bytes(std::uint32_t number, std::span<const std::byte> value) {
    key(number, WireType::kLength);
    raw_varint(value.size());
    buffer_.insert(buffer_.end(), value.begin(), value.end());
}

void Writer::string(std::uint32_t number, std::string_view value) {
    bytes(number, std::as_bytes(std::span<const char>(value.data(), value.size())));
}

void Writer::packed_varints(std::uint32_t number, std::span<const std::int64_t> values) {
    Writer packed;
    for (const std::int64_t value : values) {
        packed.raw_varint(static_cast<std::uint64_t>(value));
    }
    bytes(number, packed.data());
}

void Writer::packed_floats(std::uint32_t number, std::span<const float> values) {
    bytes(number, std::as_bytes(values));
}

}  // namespace ii::formats::pb
