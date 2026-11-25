// SPDX-License-Identifier: MIT
#include "core/json.h"

#include "core/parse.h"

#include <format>

namespace ii {

class JsonParser {
public:
    JsonParser(std::string_view text, int max_depth) : text_(text), max_depth_(max_depth) {}

    Result<Json> parse_document() {
        skip_whitespace();
        II_ASSIGN_OR_RETURN(Json value, parse_value(0));
        skip_whitespace();
        if (pos_ != text_.size()) {
            return error("unexpected text after the document");
        }
        return value;
    }

private:
    Status error(std::string_view message) const {
        return errors::data_corrupted("JSON at offset {}: {}", pos_, message);
    }

    void skip_whitespace() {
        while (pos_ < text_.size() &&
               (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' || text_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool consume(std::string_view literal) {
        if (text_.substr(pos_, literal.size()) == literal) {
            pos_ += literal.size();
            return true;
        }
        return false;
    }

    Result<Json> parse_value(int depth) {
        if (depth > max_depth_) {
            return error("nesting is too deep");
        }
        if (pos_ >= text_.size()) {
            return error("unexpected end of input");
        }
        Json result;
        switch (text_[pos_]) {
        case 'n':
            if (!consume("null")) {
                return error("invalid literal");
            }
            return result;
        case 't':
            if (!consume("true")) {
                return error("invalid literal");
            }
            result.value_ = true;
            return result;
        case 'f':
            if (!consume("false")) {
                return error("invalid literal");
            }
            result.value_ = false;
            return result;
        case '"': {
            II_ASSIGN_OR_RETURN(std::string text, parse_string());
            result.value_ = std::move(text);
            return result;
        }
        case '[':
            return parse_array(depth);
        case '{':
            return parse_object(depth);
        default:
            return parse_number();
        }
    }

    Result<Json> parse_array(int depth) {
        ++pos_;  // '['
        Json::Array elements;
        skip_whitespace();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            ++pos_;
            Json result;
            result.value_ = std::move(elements);
            return result;
        }
        for (;;) {
            skip_whitespace();
            II_ASSIGN_OR_RETURN(Json element, parse_value(depth + 1));
            elements.push_back(std::move(element));
            skip_whitespace();
            if (pos_ < text_.size() && text_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < text_.size() && text_[pos_] == ']') {
                ++pos_;
                break;
            }
            return error("expected ',' or ']'");
        }
        Json result;
        result.value_ = std::move(elements);
        return result;
    }

    Result<Json> parse_object(int depth) {
        ++pos_;  // '{'
        Json::Object members;
        skip_whitespace();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            ++pos_;
            Json result;
            result.value_ = std::move(members);
            return result;
        }
        for (;;) {
            skip_whitespace();
            if (pos_ >= text_.size() || text_[pos_] != '"') {
                return error("expected a member name");
            }
            II_ASSIGN_OR_RETURN(std::string key, parse_string());
            skip_whitespace();
            if (pos_ >= text_.size() || text_[pos_] != ':') {
                return error("expected ':'");
            }
            ++pos_;
            skip_whitespace();
            II_ASSIGN_OR_RETURN(Json value, parse_value(depth + 1));
            members.emplace_back(std::move(key), std::move(value));
            skip_whitespace();
            if (pos_ < text_.size() && text_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < text_.size() && text_[pos_] == '}') {
                ++pos_;
                break;
            }
            return error("expected ',' or '}'");
        }
        Json result;
        result.value_ = std::move(members);
        return result;
    }

    std::optional<std::uint32_t> hex4() {
        if (pos_ + 4 > text_.size()) {
            return std::nullopt;
        }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<std::uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<std::uint32_t>(c - 'A' + 10);
            } else {
                return std::nullopt;
            }
        }
        return value;
    }

    static void append_utf8(std::string& out, std::uint32_t code) {
        if (code < 0x80) {
            out += static_cast<char>(code);
        } else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else if (code < 0x10000) {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code >> 18));
            out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        }
    }

    Result<std::string> parse_string() {
        ++pos_;  // opening quote
        std::string out;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                return error("control character in string");
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= text_.size()) {
                break;
            }
            const char e = text_[pos_++];
            switch (e) {
            case '"':
            case '\\':
            case '/':
                out += e;
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                std::optional<std::uint32_t> code = hex4();
                if (!code) {
                    return error("invalid \\u escape");
                }
                if (*code >= 0xD800 && *code <= 0xDBFF) {
                    // A high surrogate must be followed by a low surrogate.
                    if (!consume("\\u")) {
                        return error("unpaired surrogate");
                    }
                    const std::optional<std::uint32_t> low = hex4();
                    if (!low || *low < 0xDC00 || *low > 0xDFFF) {
                        return error("unpaired surrogate");
                    }
                    *code = 0x10000 + ((*code - 0xD800) << 10) + (*low - 0xDC00);
                } else if (*code >= 0xDC00 && *code <= 0xDFFF) {
                    return error("unpaired surrogate");
                }
                append_utf8(out, *code);
                break;
            }
            default:
                return error("invalid escape");
            }
        }
        return error("unterminated string");
    }

    Result<Json> parse_number() {
        const std::size_t start = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') {
            ++pos_;
        }
        auto digits = [&] {
            const std::size_t begin = pos_;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                ++pos_;
            }
            return pos_ - begin;
        };
        const std::size_t int_start = pos_;
        const std::size_t int_digits = digits();
        if (int_digits == 0 || (int_digits > 1 && text_[int_start] == '0')) {
            pos_ = start;
            return error("invalid number");
        }
        bool integral = true;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            integral = false;
            if (digits() == 0) {
                return error("invalid number");
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            integral = false;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            if (digits() == 0) {
                return error("invalid number");
            }
        }
        const std::string_view literal = text_.substr(start, pos_ - start);
        const std::optional<double> value = parse_double(literal);
        if (!value) {
            return error("invalid number");
        }
        Json result;
        Json::Number number{*value, integral ? parse_integer(literal) : std::nullopt};
        result.value_ = number;
        return result;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
    int max_depth_;
};

Result<Json> Json::parse(std::string_view text, int max_depth) {
    return JsonParser(text, max_depth).parse_document();
}

Json::Kind Json::kind() const noexcept {
    return static_cast<Kind>(value_.index());
}

bool Json::as_bool() const noexcept {
    const bool* b = std::get_if<bool>(&value_);
    return b != nullptr && *b;
}

double Json::as_double() const noexcept {
    const Number* n = std::get_if<Number>(&value_);
    return n != nullptr ? n->value : 0.0;
}

std::optional<std::int64_t> Json::as_int() const noexcept {
    const Number* n = std::get_if<Number>(&value_);
    return n != nullptr ? n->integer : std::nullopt;
}

const std::string& Json::as_string() const noexcept {
    static const std::string kEmpty;
    const std::string* s = std::get_if<std::string>(&value_);
    return s != nullptr ? *s : kEmpty;
}

const Json::Array& Json::as_array() const noexcept {
    static const Array kEmpty;
    const Array* a = std::get_if<Array>(&value_);
    return a != nullptr ? *a : kEmpty;
}

const Json::Object& Json::as_object() const noexcept {
    static const Object kEmpty;
    const Object* o = std::get_if<Object>(&value_);
    return o != nullptr ? *o : kEmpty;
}

const Json* Json::find(std::string_view key) const noexcept {
    for (const auto& [name, value] : as_object()) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

std::size_t Json::size() const noexcept {
    if (const Array* a = std::get_if<Array>(&value_)) {
        return a->size();
    }
    if (const Object* o = std::get_if<Object>(&value_)) {
        return o->size();
    }
    return 0;
}

std::optional<std::vector<std::uint8_t>> base64_decode(std::string_view text) {
    auto value_of = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') {
            return c - 'A';
        }
        if (c >= 'a' && c <= 'z') {
            return c - 'a' + 26;
        }
        if (c >= '0' && c <= '9') {
            return c - '0' + 52;
        }
        if (c == '+') {
            return 62;
        }
        if (c == '/') {
            return 63;
        }
        return -1;
    };
    while (!text.empty() && text.back() == '=') {
        text.remove_suffix(1);
    }
    if (text.size() % 4 == 1) {
        return std::nullopt;
    }
    std::vector<std::uint8_t> out;
    out.reserve(text.size() * 3 / 4);
    std::uint32_t buffer = 0;
    int bits = 0;
    for (const char c : text) {
        const int v = value_of(c);
        if (v < 0) {
            return std::nullopt;
        }
        buffer = (buffer << 6) | static_cast<std::uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>((buffer >> bits) & 0xFF));
        }
    }
    return out;
}

}  // namespace ii
