// SPDX-License-Identifier: MIT
/// @file
/// An interpreter for the subset of Jinja2 used by chat templates (as rendered by Hugging Face transformers:
/// trim_blocks and lstrip_blocks on, Python-like values, `raise_exception`, `namespace`, `tojson`, loop controls).
///
/// Supported: {{ }}, {% %}, {# #}, whitespace control (-), if / elif / else, for (with loop.*, an `if` filter and
/// else), set (also `set ns.x = ...` and block set), macro / call, break / continue, generation blocks, filter blocks;
/// literals, lists, dicts, attribute and item access, slices, calls with keyword arguments, filters, tests, the
/// usual operators and the conditional expression; common string, list and dict methods.
///
/// Templates are untrusted input: recursion depth, loop iterations and output size are limited.
#pragma once

#include <ii/status.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ii::llm::detail::jinja {

class Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;  ///< insertion ordered
using Kwargs = std::vector<std::pair<std::string, Value>>;
using Function = std::function<Result<Value>(std::vector<Value>& args, Kwargs& kwargs)>;

struct Undefined {};

class Value {
public:
    using Storage = std::variant<Undefined, std::nullptr_t, bool, std::int64_t, double, std::string,
                                 std::shared_ptr<Array>, std::shared_ptr<Object>, std::shared_ptr<Function>>;

    Value() = default;
    Value(std::nullptr_t) : v_(nullptr) {}
    Value(bool b) : v_(b) {}
    Value(int i) : v_(static_cast<std::int64_t>(i)) {}
    Value(std::int64_t i) : v_(i) {}
    Value(double d) : v_(d) {}
    Value(const char* s) : v_(std::string(s)) {}
    Value(std::string s) : v_(std::move(s)) {}
    Value(std::string_view s) : v_(std::string(s)) {}
    Value(Array a) : v_(std::make_shared<Array>(std::move(a))) {}
    Value(Object o) : v_(std::make_shared<Object>(std::move(o))) {}
    static Value function(Function f) {
        Value v;
        v.v_ = std::make_shared<Function>(std::move(f));
        return v;
    }

    bool is_undefined() const noexcept { return std::holds_alternative<Undefined>(v_); }
    bool is_none() const noexcept { return std::holds_alternative<std::nullptr_t>(v_); }
    bool is_bool() const noexcept { return std::holds_alternative<bool>(v_); }
    bool is_int() const noexcept { return std::holds_alternative<std::int64_t>(v_); }
    bool is_float() const noexcept { return std::holds_alternative<double>(v_); }
    bool is_number() const noexcept { return is_int() || is_float(); }
    bool is_string() const noexcept { return std::holds_alternative<std::string>(v_); }
    bool is_array() const noexcept { return std::holds_alternative<std::shared_ptr<Array>>(v_); }
    bool is_object() const noexcept { return std::holds_alternative<std::shared_ptr<Object>>(v_); }
    bool is_function() const noexcept { return std::holds_alternative<std::shared_ptr<Function>>(v_); }

    bool as_bool() const noexcept { return std::get<bool>(v_); }
    std::int64_t as_int() const noexcept { return std::get<std::int64_t>(v_); }
    double as_number() const noexcept { return is_int() ? static_cast<double>(as_int()) : std::get<double>(v_); }
    const std::string& as_string() const noexcept { return std::get<std::string>(v_); }
    Array& array() const noexcept { return *std::get<std::shared_ptr<Array>>(v_); }
    Object& object() const noexcept { return *std::get<std::shared_ptr<Object>>(v_); }
    Function& function() const noexcept { return *std::get<std::shared_ptr<Function>>(v_); }

    /// Member `key` of an object (nullptr when absent or not an object).
    const Value* find(std::string_view key) const;
    void set(const std::string& key, Value value) const;

    /// Python truthiness.
    bool truthy() const;
    /// str(): Python-like rendering (None, True, 1.5, ['a'], {'k': 1}); undefined renders as "".
    std::string to_string() const;
    /// JSON (tojson); `indent` < 0: single line with ", " and ": " separators.
    std::string to_json(int indent = -1, int depth = 0) const;
    /// Python repr (strings quoted).
    std::string repr() const;

    const Storage& storage() const noexcept { return v_; }

private:
    Storage v_;
};

bool operator==(const Value& a, const Value& b);

/// Renders `source` with the global variables of `context` (an object).
Result<std::string> render(std::string_view source, const Value& context);

}  // namespace ii::llm::detail::jinja
