// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace ii::cli {

/// A minimal streaming JSON writer producing indented output.
class JsonWriter {
public:
    void begin_object() { open('{'); }
    void end_object() { close('}'); }
    void begin_array() { open('['); }
    void end_array() { close(']'); }

    /// Writes an object key; must be followed by a value or begin_object/begin_array.
    void key(std::string_view name) {
        separator();
        write_string(name);
        out_ += ": ";
        after_key_ = true;
    }

    void value(std::string_view text) {
        separator();
        write_string(text);
    }
    void value(const char* text) { value(std::string_view(text)); }
    void value(bool flag) {
        separator();
        out_ += flag ? "true" : "false";
    }
    void value(std::int64_t number) {
        separator();
        out_ += std::to_string(number);
    }
    void value(std::uint64_t number) {
        separator();
        out_ += std::to_string(number);
    }
    void value(int number) { value(static_cast<std::int64_t>(number)); }
    void value(double number) {
        separator();
        out_ += std::format("{}", number);
    }

    template <class T>
    void field(std::string_view name, const T& v) {
        key(name);
        value(v);
    }

    const std::string& str() const { return out_; }

private:
    void open(char bracket) {
        separator();
        out_ += bracket;
        first_.push_back(true);
    }

    void close(char bracket) {
        const bool empty = first_.back();
        first_.pop_back();
        if (!empty) {
            newline();
        }
        out_ += bracket;
    }

    void separator() {
        if (after_key_) {
            after_key_ = false;
            return;
        }
        if (first_.empty()) {
            return;
        }
        if (!first_.back()) {
            out_ += ',';
        }
        first_.back() = false;
        newline();
    }

    void newline() {
        out_ += '\n';
        out_.append(first_.size() * 2, ' ');
    }

    void write_string(std::string_view text) {
        out_ += '"';
        for (const char c : text) {
            switch (c) {
            case '"':
                out_ += "\\\"";
                break;
            case '\\':
                out_ += "\\\\";
                break;
            case '\n':
                out_ += "\\n";
                break;
            case '\r':
                out_ += "\\r";
                break;
            case '\t':
                out_ += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out_ += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
                } else {
                    out_ += c;
                }
            }
        }
        out_ += '"';
    }

    std::string out_;
    std::vector<bool> first_;
    bool after_key_ = false;
};

}  // namespace ii::cli
