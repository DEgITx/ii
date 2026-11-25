// SPDX-License-Identifier: MIT
/// @file
/// A small, strict JSON (RFC 8259) parser producing an immutable document tree. Used for model metadata
/// (Hugging Face `config.json`, `tokenizer.json`), test reference data and tool output.
///
/// Safe for untrusted input: nesting depth is limited and every error is reported as a `Status`.
#pragma once

#include <ii/status.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ii {

class Json {
public:
    enum class Kind : std::uint8_t { kNull, kBool, kNumber, kString, kArray, kObject };
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;  ///< members in document order

    /// Parses a complete document (surrounding whitespace allowed).
    static Result<Json> parse(std::string_view text, int max_depth = 128);

    Json() = default;

    Kind kind() const noexcept;
    bool is_null() const noexcept { return kind() == Kind::kNull; }
    bool is_bool() const noexcept { return kind() == Kind::kBool; }
    bool is_number() const noexcept { return kind() == Kind::kNumber; }
    bool is_string() const noexcept { return kind() == Kind::kString; }
    bool is_array() const noexcept { return kind() == Kind::kArray; }
    bool is_object() const noexcept { return kind() == Kind::kObject; }

    /// Accessors return a fallback (false, 0, empty) for values of another kind.
    bool as_bool() const noexcept;
    double as_double() const noexcept;
    /// The number as an exact integer, if it is one (no fraction or exponent and within range).
    std::optional<std::int64_t> as_int() const noexcept;
    const std::string& as_string() const noexcept;
    const Array& as_array() const noexcept;
    const Object& as_object() const noexcept;

    /// The member `key` of an object, or nullptr.
    const Json* find(std::string_view key) const noexcept;

    std::size_t size() const noexcept;  ///< elements of an array or members of an object; 0 otherwise

private:
    struct Number {
        double value = 0;
        std::optional<std::int64_t> integer;
    };
    friend class JsonParser;
    std::variant<std::monostate, bool, Number, std::string, Array, Object> value_;
};

/// Decodes standard base64 (RFC 4648, with or without padding). Returns nullopt on invalid input.
std::optional<std::vector<std::uint8_t>> base64_decode(std::string_view text);

}  // namespace ii
