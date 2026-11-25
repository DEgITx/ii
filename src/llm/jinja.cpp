// SPDX-License-Identifier: MIT
#include "llm/jinja.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <ctime>
#include <format>
#include <map>
#include <optional>

namespace ii::llm::detail::jinja {

namespace {

constexpr std::size_t kMaxOutput = std::size_t{16} << 20;
constexpr std::int64_t kMaxIterations = 1'000'000;
constexpr int kMaxDepth = 64;

Status template_error(std::string message) {
    return errors::invalid_argument("chat template: {}", message);
}

std::string format_float(double d) {
    if (std::isnan(d)) {
        return "nan";
    }
    if (std::isinf(d)) {
        return d > 0 ? "inf" : "-inf";
    }
    std::string text = std::format("{}", d);
    if (text.find_first_of(".en") == std::string::npos) {
        text += ".0";
    }
    return text;
}

std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += std::format("\\u{:04x}", static_cast<unsigned>(c));
            } else {
                out += c;
            }
        }
    }
    return out + "\"";
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Value

const Value* Value::find(std::string_view key) const {
    if (!is_object()) {
        return nullptr;
    }
    for (const auto& [name, value] : object()) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

void Value::set(const std::string& key, Value value) const {
    for (auto& [name, existing] : object()) {
        if (name == key) {
            existing = std::move(value);
            return;
        }
    }
    object().emplace_back(key, std::move(value));
}

bool Value::truthy() const {
    return std::visit(
        [](const auto& v) -> bool {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Undefined> || std::is_same_v<T, std::nullptr_t>) {
                return false;
            } else if constexpr (std::is_same_v<T, bool>) {
                return v;
            } else if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, double>) {
                return v != 0;
            } else if constexpr (std::is_same_v<T, std::string>) {
                return !v.empty();
            } else if constexpr (std::is_same_v<T, std::shared_ptr<Array>> ||
                                 std::is_same_v<T, std::shared_ptr<Object>>) {
                return !v->empty();
            } else {
                return true;
            }
        },
        v_);
}

std::string Value::repr() const {
    if (is_string()) {
        const std::string& s = as_string();
        const char quote = s.find('\'') != std::string::npos && s.find('"') == std::string::npos ? '"' : '\'';
        std::string out(1, quote);
        for (const char c : s) {
            if (c == quote || c == '\\') {
                out += '\\';
                out += c;
            } else if (c == '\n') {
                out += "\\n";
            } else if (c == '\t') {
                out += "\\t";
            } else if (c == '\r') {
                out += "\\r";
            } else {
                out += c;
            }
        }
        return out + quote;
    }
    return to_string();
}

std::string Value::to_string() const {
    return std::visit(
        [this](const auto& v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, Undefined>) {
                return "";
            } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
                return "None";
            } else if constexpr (std::is_same_v<T, bool>) {
                return v ? "True" : "False";
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return std::to_string(v);
            } else if constexpr (std::is_same_v<T, double>) {
                return format_float(v);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return v;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<Array>>) {
                std::string out = "[";
                for (std::size_t i = 0; i < v->size(); ++i) {
                    out += (i > 0 ? ", " : "") + (*v)[i].repr();
                }
                return out + "]";
            } else if constexpr (std::is_same_v<T, std::shared_ptr<Object>>) {
                std::string out = "{";
                for (std::size_t i = 0; i < v->size(); ++i) {
                    out += (i > 0 ? ", " : "") + Value((*v)[i].first).repr() + ": " + (*v)[i].second.repr();
                }
                return out + "}";
            } else {
                (void)this;
                return "<function>";
            }
        },
        v_);
}

std::string Value::to_json(int indent, int depth) const {
    const std::string pad = indent >= 0 ? "\n" + std::string(static_cast<std::size_t>(indent * (depth + 1)), ' ') : "";
    const std::string close_pad = indent >= 0 ? "\n" + std::string(static_cast<std::size_t>(indent * depth), ' ') : "";
    const std::string separator = indent >= 0 ? "," : ", ";
    if (is_undefined() || is_none()) {
        return "null";
    }
    if (is_bool()) {
        return as_bool() ? "true" : "false";
    }
    if (is_int()) {
        return std::to_string(as_int());
    }
    if (is_float()) {
        return format_float(std::get<double>(v_));
    }
    if (is_string()) {
        return json_string(as_string());
    }
    if (is_array()) {
        if (array().empty()) {
            return "[]";
        }
        std::string out = "[";
        for (std::size_t i = 0; i < array().size(); ++i) {
            out += (i > 0 ? separator : "") + pad + array()[i].to_json(indent, depth + 1);
        }
        return out + close_pad + "]";
    }
    if (is_object()) {
        if (object().empty()) {
            return "{}";
        }
        std::string out = "{";
        for (std::size_t i = 0; i < object().size(); ++i) {
            out += (i > 0 ? separator : "") + pad + json_string(object()[i].first) + ": " +
                   object()[i].second.to_json(indent, depth + 1);
        }
        return out + close_pad + "}";
    }
    return "null";
}

bool operator==(const Value& a, const Value& b) {
    if (a.is_number() && b.is_number()) {
        return a.as_number() == b.as_number();
    }
    if (a.is_bool() && b.is_bool()) {
        return a.as_bool() == b.as_bool();
    }
    if (a.is_string() && b.is_string()) {
        return a.as_string() == b.as_string();
    }
    if ((a.is_none() || a.is_undefined()) && (b.is_none() || b.is_undefined())) {
        return a.is_none() == b.is_none();
    }
    if (a.is_array() && b.is_array()) {
        return a.array() == b.array();
    }
    if (a.is_object() && b.is_object()) {
        if (a.object().size() != b.object().size()) {
            return false;
        }
        for (const auto& [key, value] : a.object()) {
            const Value* other = b.find(key);
            if (other == nullptr || !(*other == value)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

namespace {

// ---------------------------------------------------------------------------------------------------------------
// Lexer: template text into text segments and tags

enum class TagKind { kText, kExpr, kStmt };

struct Segment {
    TagKind kind = TagKind::kText;
    std::string text;  ///< text, or the tag's content
    std::size_t line = 1;
};

Result<std::vector<Segment>> split_template(std::string_view source) {
    struct Raw {
        TagKind kind;
        bool comment = false;
        std::string text;
        bool trim_left = false;   // {%- / {{- / {#-
        bool trim_right = false;  // -%} / -}} / -#}
        bool plus = false;        // {%+
        std::size_t line;
    };
    std::vector<Raw> raw;
    std::size_t i = 0;
    std::size_t line = 1;
    std::string text;
    while (i < source.size()) {
        const bool open = source[i] == '{' && i + 1 < source.size() &&
                          (source[i + 1] == '{' || source[i + 1] == '%' || source[i + 1] == '#');
        if (!open) {
            line += source[i] == '\n' ? 1 : 0;
            text += source[i++];
            continue;
        }
        raw.push_back(Raw{TagKind::kText, false, std::move(text), false, false, false, line});
        text.clear();
        const char type = source[i + 1];
        Raw tag{type == '{' ? TagKind::kExpr : TagKind::kStmt, type == '#', {}, false, false, false, line};
        std::size_t j = i + 2;
        if (j < source.size() && source[j] == '-') {
            tag.trim_left = true;
            ++j;
        } else if (j < source.size() && source[j] == '+') {
            tag.plus = true;
            ++j;
        }
        const char close = type == '{' ? '}' : type;
        char quote = 0;
        bool closed = false;
        while (j < source.size()) {
            const char c = source[j];
            if (quote != 0) {
                if (c == '\\' && j + 1 < source.size()) {
                    tag.text += c;
                    tag.text += source[j + 1];
                    j += 2;
                    continue;
                }
                if (c == quote) {
                    quote = 0;
                }
            } else if (!tag.comment && (c == '\'' || c == '"')) {
                quote = c;
            } else if (c == close && j + 1 < source.size() && source[j + 1] == '}') {
                closed = true;
                break;
            } else if (c == '-' && j + 2 < source.size() && source[j + 1] == close && source[j + 2] == '}') {
                tag.trim_right = true;
                ++j;
                closed = true;
                break;
            }
            line += c == '\n' ? 1 : 0;
            tag.text += c;
            ++j;
        }
        if (!closed) {
            return template_error(std::format("unclosed tag at line {}", tag.line));
        }
        raw.push_back(std::move(tag));
        i = j + 2;
    }
    raw.push_back(Raw{TagKind::kText, false, std::move(text), false, false, false, line});

    // Whitespace control: '-' trims all whitespace; lstrip_blocks / trim_blocks for statements and comments.
    for (std::size_t r = 0; r < raw.size(); ++r) {
        Raw& tag = raw[r];
        if (tag.kind == TagKind::kText) {
            continue;
        }
        const bool block = tag.kind == TagKind::kStmt;
        if (r > 0 && raw[r - 1].kind == TagKind::kText) {
            std::string& before = raw[r - 1].text;
            if (tag.trim_left) {
                before.erase(before.find_last_not_of(" \t\r\n") + 1);
            } else if (block && !tag.plus) {
                const std::size_t line_start = before.find_last_of('\n');
                const std::size_t from = line_start == std::string::npos ? 0 : line_start + 1;
                if (before.find_first_not_of(" \t", from) == std::string::npos &&
                    (line_start != std::string::npos || r == 1)) {
                    before.erase(from);
                }
            }
        }
        if (r + 1 < raw.size() && raw[r + 1].kind == TagKind::kText) {
            std::string& after = raw[r + 1].text;
            if (tag.trim_right) {
                after.erase(0, after.find_first_not_of(" \t\r\n") == std::string::npos
                                   ? after.size()
                                   : after.find_first_not_of(" \t\r\n"));
            } else if (block) {
                if (after.rfind("\r\n", 0) == 0) {
                    after.erase(0, 2);
                } else if (after.rfind('\n', 0) == 0) {
                    after.erase(0, 1);
                }
            }
        }
    }
    std::vector<Segment> segments;
    for (Raw& r : raw) {
        if (r.comment || (r.kind == TagKind::kText && r.text.empty())) {
            continue;
        }
        segments.push_back(Segment{r.kind, std::move(r.text), r.line});
    }
    return segments;
}

// ---------------------------------------------------------------------------------------------------------------
// Expression tokens

enum class Tok { kName, kInt, kFloat, kString, kOp, kEnd };

struct Token {
    Tok kind = Tok::kEnd;
    std::string text;
    std::int64_t i = 0;
    double f = 0;
};

Result<std::vector<Token>> tokenize(const std::string& s) {
    std::vector<Token> tokens;
    std::size_t i = 0;
    while (i < s.size()) {
        const char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++i;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) {
                ++j;
            }
            tokens.push_back(Token{Tok::kName, s.substr(i, j - i)});
            i = j;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            std::size_t j = i;
            bool is_float = false;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '_')) {
                ++j;
            }
            if (j + 1 < s.size() && s[j] == '.' && std::isdigit(static_cast<unsigned char>(s[j + 1]))) {
                is_float = true;
                ++j;
                while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) {
                    ++j;
                }
            }
            if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
                std::size_t k = j + 1;
                if (k < s.size() && (s[k] == '+' || s[k] == '-')) {
                    ++k;
                }
                if (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) {
                    is_float = true;
                    j = k;
                    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) {
                        ++j;
                    }
                }
            }
            std::string number = s.substr(i, j - i);
            number.erase(std::remove(number.begin(), number.end(), '_'), number.end());
            Token t;
            if (is_float) {
                t.kind = Tok::kFloat;
                t.f = std::strtod(number.c_str(), nullptr);
            } else {
                t.kind = Tok::kInt;
                t.i = INT64_MAX;
                std::from_chars(number.data(), number.data() + number.size(), t.i);
            }
            tokens.push_back(t);
            i = j;
            continue;
        }
        if (c == '\'' || c == '"') {
            std::string value;
            std::size_t j = i + 1;
            while (j < s.size() && s[j] != c) {
                if (s[j] == '\\' && j + 1 < s.size()) {
                    const char e = s[j + 1];
                    switch (e) {
                    case 'n':
                        value += '\n';
                        break;
                    case 't':
                        value += '\t';
                        break;
                    case 'r':
                        value += '\r';
                        break;
                    case '\\':
                    case '\'':
                    case '"':
                        value += e;
                        break;
                    default:
                        value += '\\';
                        value += e;
                    }
                    j += 2;
                    continue;
                }
                value += s[j++];
            }
            if (j >= s.size()) {
                return template_error("unterminated string literal");
            }
            tokens.push_back(Token{Tok::kString, std::move(value)});
            i = j + 1;
            continue;
        }
        static const char* const kOps[] = {"//", "**", "==", "!=", "<=", ">=", "(", ")", "[", "]", "{", "}", ",",
                                           ":",  ".",  "|",  "+",  "-",  "*",  "/", "%", "~", "<", ">", "="};
        bool matched = false;
        for (const char* op : kOps) {
            const std::size_t length = std::char_traits<char>::length(op);
            if (s.compare(i, length, op) == 0) {
                tokens.push_back(Token{Tok::kOp, op});
                i += length;
                matched = true;
                break;
            }
        }
        if (!matched) {
            return template_error(std::format("unexpected character '{}'", c));
        }
    }
    tokens.push_back(Token{Tok::kEnd, ""});
    return tokens;
}

// ---------------------------------------------------------------------------------------------------------------
// AST

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

enum class ExprKind {
    kLiteral,
    kName,
    kAttr,
    kItem,
    kSlice,
    kCall,
    kFilter,
    kTest,
    kUnary,
    kBinary,
    kAnd,
    kOr,
    kConditional,
    kList,
    kDict,
};

struct Expr {
    ExprKind kind = ExprKind::kLiteral;
    Value literal;
    std::string name;           ///< variable, attribute, filter, test or operator
    std::vector<ExprPtr> args;  ///< operands / positional arguments (kSlice: object, start, stop, step; null = absent)
    std::vector<std::pair<std::string, ExprPtr>> kwargs;
    bool negated = false;
};

struct Node;
using NodePtr = std::shared_ptr<Node>;
using Body = std::vector<NodePtr>;

enum class NodeKind { kText, kOutput, kIf, kFor, kSet, kSetBlock, kMacro, kBreak, kContinue, kFilterBlock, kScope };

struct Node {
    NodeKind kind = NodeKind::kText;
    std::string text;
    ExprPtr expr;
    std::vector<std::pair<ExprPtr, Body>> branches;  ///< if: (condition or null for else, body)
    std::vector<std::string> targets;                ///< for / set targets; set: "ns.attr" form uses `attr`
    std::string attr;
    ExprPtr filter;  ///< for: the `if` filter
    Body body;
    Body else_body;
    std::vector<std::pair<std::string, ExprPtr>> params;  ///< macro parameters with defaults
};

// ---------------------------------------------------------------------------------------------------------------
// Parser

class ExprParser {
public:
    explicit ExprParser(std::vector<Token> tokens) : t_(std::move(tokens)) {}

    const Token& peek(std::size_t ahead = 0) const { return t_[std::min(p_ + ahead, t_.size() - 1)]; }
    bool at_end() const { return peek().kind == Tok::kEnd; }
    bool is_op(const char* op, std::size_t ahead = 0) const {
        return peek(ahead).kind == Tok::kOp && peek(ahead).text == op;
    }
    bool is_name(const char* name, std::size_t ahead = 0) const {
        return peek(ahead).kind == Tok::kName && peek(ahead).text == name;
    }
    Token next() { return t_[std::min(p_++, t_.size() - 1)]; }
    Status expect_op(const char* op) {
        if (!is_op(op)) {
            return template_error(std::format("expected '{}', found '{}'", op, peek().text));
        }
        ++p_;
        return ok_status();
    }
    Result<std::string> expect_name() {
        if (peek().kind != Tok::kName) {
            return template_error(std::format("expected a name, found '{}'", peek().text));
        }
        return next().text;
    }

    Result<ExprPtr> expression() { return conditional(); }
    Result<ExprPtr> or_expression() { return logical_or(); }

private:
    static ExprPtr make(ExprKind kind, std::string name = {}, std::vector<ExprPtr> args = {}) {
        auto e = std::make_shared<Expr>();
        e->kind = kind;
        e->name = std::move(name);
        e->args = std::move(args);
        return e;
    }

    Result<ExprPtr> conditional() {
        II_ASSIGN_OR_RETURN(ExprPtr value, logical_or());
        if (is_name("if")) {
            ++p_;
            II_ASSIGN_OR_RETURN(ExprPtr condition, logical_or());
            ExprPtr otherwise;
            if (is_name("else")) {
                ++p_;
                II_ASSIGN_OR_RETURN(otherwise, conditional());
            } else {
                otherwise = make(ExprKind::kLiteral);
            }
            return make(ExprKind::kConditional, {}, {condition, value, otherwise});
        }
        return value;
    }

    Result<ExprPtr> logical_or() {
        II_ASSIGN_OR_RETURN(ExprPtr left, logical_and());
        while (is_name("or")) {
            ++p_;
            II_ASSIGN_OR_RETURN(ExprPtr right, logical_and());
            left = make(ExprKind::kOr, {}, {left, right});
        }
        return left;
    }

    Result<ExprPtr> logical_and() {
        II_ASSIGN_OR_RETURN(ExprPtr left, logical_not());
        while (is_name("and")) {
            ++p_;
            II_ASSIGN_OR_RETURN(ExprPtr right, logical_not());
            left = make(ExprKind::kAnd, {}, {left, right});
        }
        return left;
    }

    Result<ExprPtr> logical_not() {
        if (is_name("not")) {
            ++p_;
            II_ASSIGN_OR_RETURN(ExprPtr operand, logical_not());
            return make(ExprKind::kUnary, "not", {operand});
        }
        return comparison();
    }

    Result<ExprPtr> comparison() {
        II_ASSIGN_OR_RETURN(ExprPtr left, concat());
        while (true) {
            std::string op;
            if (peek().kind == Tok::kOp && (peek().text == "==" || peek().text == "!=" || peek().text == "<" ||
                                            peek().text == "<=" || peek().text == ">" || peek().text == ">=")) {
                op = next().text;
            } else if (is_name("in")) {
                ++p_;
                op = "in";
            } else if (is_name("not") && is_name("in", 1)) {
                p_ += 2;
                op = "not in";
            } else if (is_name("is")) {
                ++p_;
                auto test = make(ExprKind::kTest);
                if (is_name("not")) {
                    ++p_;
                    test->negated = true;
                }
                II_ASSIGN_OR_RETURN(test->name, expect_name());
                test->args.push_back(left);
                if (is_op("(")) {
                    II_TRY(call_arguments(*test));
                } else if (peek().kind == Tok::kInt || peek().kind == Tok::kString || peek().kind == Tok::kFloat) {
                    II_ASSIGN_OR_RETURN(ExprPtr argument, primary());
                    test->args.push_back(argument);
                }
                left = test;
                continue;
            } else {
                break;
            }
            II_ASSIGN_OR_RETURN(ExprPtr right, concat());
            left = make(ExprKind::kBinary, op, {left, right});
        }
        return left;
    }

    Result<ExprPtr> concat() {
        II_ASSIGN_OR_RETURN(ExprPtr left, additive());
        while (is_op("~")) {
            ++p_;
            II_ASSIGN_OR_RETURN(ExprPtr right, additive());
            left = make(ExprKind::kBinary, "~", {left, right});
        }
        return left;
    }

    Result<ExprPtr> additive() {
        II_ASSIGN_OR_RETURN(ExprPtr left, multiplicative());
        while (is_op("+") || is_op("-")) {
            const std::string op = next().text;
            II_ASSIGN_OR_RETURN(ExprPtr right, multiplicative());
            left = make(ExprKind::kBinary, op, {left, right});
        }
        return left;
    }

    Result<ExprPtr> multiplicative() {
        II_ASSIGN_OR_RETURN(ExprPtr left, unary());
        while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
            const std::string op = next().text;
            II_ASSIGN_OR_RETURN(ExprPtr right, unary());
            left = make(ExprKind::kBinary, op, {left, right});
        }
        return left;
    }

    Result<ExprPtr> unary() {
        if (is_op("-") || is_op("+")) {
            const std::string op = next().text;
            II_ASSIGN_OR_RETURN(ExprPtr operand, unary());
            return make(ExprKind::kUnary, op, {operand});
        }
        return power();
    }

    Result<ExprPtr> power() {
        II_ASSIGN_OR_RETURN(ExprPtr left, postfix());
        if (is_op("**")) {
            ++p_;
            II_ASSIGN_OR_RETURN(ExprPtr right, unary());
            return make(ExprKind::kBinary, "**", {left, right});
        }
        return left;
    }

    Status call_arguments(Expr& e) {
        II_TRY(expect_op("("));
        while (!is_op(")")) {
            if (peek().kind == Tok::kName && is_op("=", 1)) {
                std::string key = next().text;
                ++p_;
                II_ASSIGN_OR_RETURN(ExprPtr value, expression());
                e.kwargs.emplace_back(std::move(key), value);
            } else {
                II_ASSIGN_OR_RETURN(ExprPtr value, expression());
                e.args.push_back(value);
            }
            if (!is_op(",")) {
                break;
            }
            ++p_;
        }
        return expect_op(")");
    }

    Result<ExprPtr> postfix() {
        II_ASSIGN_OR_RETURN(ExprPtr e, primary());
        while (true) {
            if (is_op(".")) {
                ++p_;
                II_ASSIGN_OR_RETURN(std::string name, expect_name());
                e = make(ExprKind::kAttr, name, {e});
            } else if (is_op("[")) {
                ++p_;
                ExprPtr parts[3];
                bool slice = false;
                for (int part = 0; part < 3; ++part) {
                    if (!is_op(":") && !is_op("]")) {
                        II_ASSIGN_OR_RETURN(parts[part], expression());
                    }
                    if (is_op(":") && part < 2) {
                        ++p_;
                        slice = true;
                        continue;
                    }
                    break;
                }
                II_TRY(expect_op("]"));
                e = slice ? make(ExprKind::kSlice, {}, {e, parts[0], parts[1], parts[2]})
                          : make(ExprKind::kItem, {}, {e, parts[0]});
                if (!slice && parts[0] == nullptr) {
                    return template_error("empty subscript");
                }
            } else if (is_op("(")) {
                auto call = make(ExprKind::kCall, {}, {e});
                II_TRY(call_arguments(*call));
                e = call;
            } else if (is_op("|")) {
                ++p_;
                auto filter = make(ExprKind::kFilter);
                II_ASSIGN_OR_RETURN(filter->name, expect_name());
                filter->args.push_back(e);
                if (is_op("(")) {
                    II_TRY(call_arguments(*filter));
                }
                e = filter;
            } else {
                return e;
            }
        }
    }

    Result<ExprPtr> primary() {
        const Token t = next();
        switch (t.kind) {
        case Tok::kInt: {
            auto e = make(ExprKind::kLiteral);
            e->literal = Value(t.i);
            return e;
        }
        case Tok::kFloat: {
            auto e = make(ExprKind::kLiteral);
            e->literal = Value(t.f);
            return e;
        }
        case Tok::kString: {
            auto e = make(ExprKind::kLiteral);
            std::string text = t.text;
            while (peek().kind == Tok::kString) {  // adjacent literals concatenate
                text += next().text;
            }
            e->literal = Value(std::move(text));
            return e;
        }
        case Tok::kName: {
            auto e = make(ExprKind::kLiteral);
            if (t.text == "true" || t.text == "True") {
                e->literal = Value(true);
            } else if (t.text == "false" || t.text == "False") {
                e->literal = Value(false);
            } else if (t.text == "none" || t.text == "None") {
                e->literal = Value(nullptr);
            } else {
                return make(ExprKind::kName, t.text);
            }
            return e;
        }
        case Tok::kOp:
            if (t.text == "(") {
                II_ASSIGN_OR_RETURN(ExprPtr inner, expression());
                if (is_op(",")) {  // tuple: a list
                    auto list = make(ExprKind::kList, {}, {inner});
                    while (is_op(",")) {
                        ++p_;
                        if (is_op(")")) {
                            break;
                        }
                        II_ASSIGN_OR_RETURN(ExprPtr item, expression());
                        list->args.push_back(item);
                    }
                    inner = list;
                }
                II_TRY(expect_op(")"));
                return inner;
            }
            if (t.text == "[") {
                auto list = make(ExprKind::kList);
                while (!is_op("]")) {
                    II_ASSIGN_OR_RETURN(ExprPtr item, expression());
                    list->args.push_back(item);
                    if (!is_op(",")) {
                        break;
                    }
                    ++p_;
                }
                II_TRY(expect_op("]"));
                return list;
            }
            if (t.text == "{") {
                auto dict = make(ExprKind::kDict);
                while (!is_op("}")) {
                    II_ASSIGN_OR_RETURN(ExprPtr key, expression());
                    II_TRY(expect_op(":"));
                    II_ASSIGN_OR_RETURN(ExprPtr value, expression());
                    dict->args.push_back(key);
                    dict->args.push_back(value);
                    if (!is_op(",")) {
                        break;
                    }
                    ++p_;
                }
                II_TRY(expect_op("}"));
                return dict;
            }
            break;
        case Tok::kEnd:
            break;
        }
        return template_error(std::format("unexpected '{}'", t.text));
    }

    std::vector<Token> t_;
    std::size_t p_ = 0;
};

class TemplateParser {
public:
    explicit TemplateParser(std::vector<Segment> segments) : s_(std::move(segments)) {}

    Result<Body> parse() {
        std::string end;
        II_ASSIGN_OR_RETURN(Body body, block({}, end));
        return body;
    }

private:
    /// Parses nodes until a statement whose keyword is in `ends` (stored in `end`, its parser in `stmt_`).
    Result<Body> block(const std::vector<std::string>& ends, std::string& end) {
        Body body;
        while (p_ < s_.size()) {
            const Segment& seg = s_[p_++];
            if (seg.kind == TagKind::kText) {
                auto node = std::make_shared<Node>();
                node->kind = NodeKind::kText;
                node->text = seg.text;
                body.push_back(node);
                continue;
            }
            II_ASSIGN_OR_RETURN(std::vector<Token> tokens, tokenize(seg.text));
            auto parser = std::make_shared<ExprParser>(std::move(tokens));
            if (seg.kind == TagKind::kExpr) {
                auto node = std::make_shared<Node>();
                node->kind = NodeKind::kOutput;
                II_ASSIGN_OR_RETURN(node->expr, parser->expression());
                II_TRY(finish(*parser, seg));
                body.push_back(node);
                continue;
            }
            II_ASSIGN_OR_RETURN(const std::string keyword, parser->expect_name());
            if (std::find(ends.begin(), ends.end(), keyword) != ends.end()) {
                end = keyword;
                stmt_ = parser;
                line_ = seg.line;
                return body;
            }
            Result<NodePtr> node = statement(keyword, *parser, seg);
            if (!node.ok()) {
                return Status(node.status()).add_context(std::format("line {}", seg.line));
            }
            if (*node != nullptr) {
                body.push_back(*node);
            }
        }
        if (!ends.empty()) {
            return template_error(std::format("missing {{% {} %}}", ends.back()));
        }
        return body;
    }

    static Status finish(ExprParser& parser, const Segment& seg) {
        if (!parser.at_end()) {
            return template_error(std::format("unexpected '{}' at line {}", parser.peek().text, seg.line));
        }
        return ok_status();
    }

    Result<NodePtr> statement(const std::string& keyword, ExprParser& parser, const Segment& seg) {
        auto node = std::make_shared<Node>();
        if (keyword == "if") {
            node->kind = NodeKind::kIf;
            II_ASSIGN_OR_RETURN(ExprPtr condition, parser.expression());
            II_TRY(finish(parser, seg));
            while (true) {
                std::string end;
                II_ASSIGN_OR_RETURN(Body body, block({"elif", "else", "endif"}, end));
                node->branches.emplace_back(condition, std::move(body));
                if (end == "endif") {
                    break;
                }
                if (end == "else") {
                    condition = nullptr;
                } else {
                    II_ASSIGN_OR_RETURN(condition, stmt_->expression());
                }
                II_TRY(finish(*stmt_, seg));
            }
            return node;
        }
        if (keyword == "for") {
            node->kind = NodeKind::kFor;
            do {
                II_ASSIGN_OR_RETURN(std::string target, parser.expect_name());
                node->targets.push_back(std::move(target));
            } while (parser.is_op(",") && (parser.next(), true));
            if (!parser.is_name("in")) {
                return template_error("expected 'in' in for");
            }
            parser.next();
            // The iterable is an or-expression, so that a trailing `if` is the loop filter.
            II_ASSIGN_OR_RETURN(node->expr, parser.or_expression());
            if (parser.is_name("if")) {
                parser.next();
                II_ASSIGN_OR_RETURN(node->filter, parser.expression());
            }
            if (parser.is_name("recursive")) {
                parser.next();
            }
            II_TRY(finish(parser, seg));
            std::string end;
            II_ASSIGN_OR_RETURN(node->body, block({"else", "endfor"}, end));
            if (end == "else") {
                II_ASSIGN_OR_RETURN(node->else_body, block({"endfor"}, end));
            }
            return node;
        }
        if (keyword == "set") {
            II_ASSIGN_OR_RETURN(std::string target, parser.expect_name());
            if (parser.is_op(".")) {
                parser.next();
                II_ASSIGN_OR_RETURN(node->attr, parser.expect_name());
            }
            node->targets.push_back(target);
            while (parser.is_op(",")) {
                parser.next();
                II_ASSIGN_OR_RETURN(std::string more, parser.expect_name());
                node->targets.push_back(std::move(more));
            }
            if (parser.is_op("=")) {
                parser.next();
                node->kind = NodeKind::kSet;
                II_ASSIGN_OR_RETURN(node->expr, parser.expression());
                if (parser.is_op(",")) {  // set a, b = x, y
                    auto list = std::make_shared<Expr>();
                    list->kind = ExprKind::kList;
                    list->args.push_back(node->expr);
                    while (parser.is_op(",")) {
                        parser.next();
                        II_ASSIGN_OR_RETURN(ExprPtr item, parser.expression());
                        list->args.push_back(item);
                    }
                    node->expr = list;
                }
                II_TRY(finish(parser, seg));
                return node;
            }
            node->kind = NodeKind::kSetBlock;
            if (parser.is_op("|")) {
                // {% set x | filter %}: apply to the block's output.
                auto filter = std::make_shared<Expr>();
                filter->kind = ExprKind::kFilter;
                parser.next();
                II_ASSIGN_OR_RETURN(filter->name, parser.expect_name());
                node->filter = filter;
            }
            II_TRY(finish(parser, seg));
            std::string end;
            II_ASSIGN_OR_RETURN(node->body, block({"endset"}, end));
            return node;
        }
        if (keyword == "macro") {
            node->kind = NodeKind::kMacro;
            II_ASSIGN_OR_RETURN(node->attr, parser.expect_name());
            II_TRY(parser.expect_op("("));
            while (!parser.is_op(")")) {
                II_ASSIGN_OR_RETURN(std::string name, parser.expect_name());
                ExprPtr fallback;
                if (parser.is_op("=")) {
                    parser.next();
                    II_ASSIGN_OR_RETURN(fallback, parser.expression());
                }
                node->params.emplace_back(std::move(name), fallback);
                if (!parser.is_op(",")) {
                    break;
                }
                parser.next();
            }
            II_TRY(parser.expect_op(")"));
            II_TRY(finish(parser, seg));
            std::string end;
            II_ASSIGN_OR_RETURN(node->body, block({"endmacro"}, end));
            return node;
        }
        if (keyword == "break" || keyword == "continue") {
            node->kind = keyword == "break" ? NodeKind::kBreak : NodeKind::kContinue;
            II_TRY(finish(parser, seg));
            return node;
        }
        if (keyword == "filter") {
            node->kind = NodeKind::kFilterBlock;
            auto filter = std::make_shared<Expr>();
            filter->kind = ExprKind::kFilter;
            II_ASSIGN_OR_RETURN(filter->name, parser.expect_name());
            node->filter = filter;
            II_TRY(finish(parser, seg));
            std::string end;
            II_ASSIGN_OR_RETURN(node->body, block({"endfilter"}, end));
            return node;
        }
        if (keyword == "generation") {  // Hugging Face extension marking assistant text: render the body
            node->kind = NodeKind::kScope;
            std::string end;
            II_ASSIGN_OR_RETURN(node->body, block({"endgeneration"}, end));
            return node;
        }
        if (keyword == "do") {  // {% do expr %}: evaluate for side effects
            node->kind = NodeKind::kSet;
            II_ASSIGN_OR_RETURN(node->expr, parser.expression());
            II_TRY(finish(parser, seg));
            return node;
        }
        return template_error(std::format("unsupported statement '{}'", keyword));
    }

    std::vector<Segment> s_;
    std::size_t p_ = 0;
    std::shared_ptr<ExprParser> stmt_;
    std::size_t line_ = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// Interpreter

enum class Flow { kNormal, kBreak, kContinue };

Value make_list(Array items) {
    return Value(std::move(items));
}

std::string lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string upper(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string strip(const std::string& s, const std::string& chars, bool left, bool right) {
    std::size_t begin = 0;
    std::size_t end = s.size();
    while (left && begin < end && chars.find(s[begin]) != std::string::npos) {
        ++begin;
    }
    while (right && end > begin && chars.find(s[end - 1]) != std::string::npos) {
        --end;
    }
    return s.substr(begin, end - begin);
}

/// Normalizes a Python index (negative counts from the end); -1 when out of range.
std::int64_t python_index(std::int64_t index, std::size_t size) {
    const auto n = static_cast<std::int64_t>(size);
    if (index < 0) {
        index += n;
    }
    return index >= 0 && index < n ? index : -1;
}

/// Python slice indices.
std::vector<std::size_t> slice_indices(std::size_t size, const Value& start, const Value& stop, const Value& step) {
    const auto n = static_cast<std::int64_t>(size);
    const std::int64_t s = step.is_int() ? step.as_int() : 1;
    std::vector<std::size_t> result;
    if (s == 0) {
        return result;
    }
    auto clamp = [&](const Value& v, std::int64_t fallback) {
        if (!v.is_int()) {
            return fallback;
        }
        std::int64_t i = v.as_int();
        if (i < 0) {
            i += n;
        }
        return s > 0 ? std::clamp<std::int64_t>(i, 0, n) : std::clamp<std::int64_t>(i, -1, n - 1);
    };
    const std::int64_t begin = clamp(start, s > 0 ? 0 : n - 1);
    const std::int64_t end = clamp(stop, s > 0 ? n : -1);
    for (std::int64_t i = begin; s > 0 ? i < end : i > end; i += s) {
        result.push_back(static_cast<std::size_t>(i));
    }
    return result;
}

Array items_of(const Value& v) {
    if (v.is_array()) {
        return v.array();
    }
    Array items;
    if (v.is_object()) {
        for (const auto& [key, value] : v.object()) {
            items.emplace_back(key);
        }
    } else if (v.is_string()) {
        for (const char c : v.as_string()) {
            items.emplace_back(std::string(1, c));
        }
    }
    return items;
}

int compare(const Value& a, const Value& b) {
    if (a.is_number() && b.is_number()) {
        return a.as_number() < b.as_number() ? -1 : (a.as_number() > b.as_number() ? 1 : 0);
    }
    if (a.is_string() && b.is_string()) {
        return a.as_string().compare(b.as_string()) < 0 ? -1 : (a.as_string() == b.as_string() ? 0 : 1);
    }
    return a.to_string() < b.to_string() ? -1 : (a.to_string() == b.to_string() ? 0 : 1);
}

const Value* kwarg(const Kwargs& kwargs, std::string_view name) {
    for (const auto& [key, value] : kwargs) {
        if (key == name) {
            return &value;
        }
    }
    return nullptr;
}

/// Positional argument `index`, else keyword `name`, else nullptr.
const Value* argument(const std::vector<Value>& args, const Kwargs& kwargs, std::size_t index, std::string_view name) {
    if (index < args.size()) {
        return &args[index];
    }
    return kwarg(kwargs, name);
}

class Interpreter {
public:
    explicit Interpreter(const Value& globals) { scopes_.push_back(globals.is_object() ? globals.object() : Object{}); }

    Result<std::string> run(const Body& body) {
        std::string out;
        II_ASSIGN_OR_RETURN(const Flow flow, exec(body, out));
        (void)flow;
        return out;
    }

private:
    // --- scopes ---
    Value lookup(const std::string& name) const {
        for (auto scope = scopes_.rbegin(); scope != scopes_.rend(); ++scope) {
            for (const auto& [key, value] : *scope) {
                if (key == name) {
                    return value;
                }
            }
        }
        return builtin(name);
    }

    void assign(const std::string& name, Value value) {
        Object& scope = scopes_.back();
        for (auto& [key, existing] : scope) {
            if (key == name) {
                existing = std::move(value);
                return;
            }
        }
        scope.emplace_back(name, std::move(value));
    }

    // --- statements ---
    Result<Flow> exec(const Body& body, std::string& out) {
        for (const NodePtr& node : body) {
            II_ASSIGN_OR_RETURN(const Flow flow, exec(*node, out));
            if (flow != Flow::kNormal) {
                return flow;
            }
            if (out.size() > kMaxOutput) {
                return template_error("output too large");
            }
        }
        return Flow::kNormal;
    }

    Result<Flow> exec(const Node& node, std::string& out) {
        switch (node.kind) {
        case NodeKind::kText:
            out += node.text;
            return Flow::kNormal;
        case NodeKind::kOutput: {
            II_ASSIGN_OR_RETURN(const Value v, eval(*node.expr));
            out += v.to_string();
            return Flow::kNormal;
        }
        case NodeKind::kIf:
            for (const auto& [condition, body] : node.branches) {
                bool take = condition == nullptr;
                if (!take) {
                    II_ASSIGN_OR_RETURN(const Value v, eval(*condition));
                    take = v.truthy();
                }
                if (take) {
                    return exec(body, out);
                }
            }
            return Flow::kNormal;
        case NodeKind::kFor:
            return exec_for(node, out);
        case NodeKind::kSet: {
            II_ASSIGN_OR_RETURN(const Value v, eval(*node.expr));
            II_TRY(set_targets(node, v));
            return Flow::kNormal;
        }
        case NodeKind::kSetBlock: {
            std::string captured;
            II_ASSIGN_OR_RETURN(const Flow flow, exec(node.body, captured));
            (void)flow;
            Value v(captured);
            if (node.filter != nullptr) {
                std::vector<Value> args{v};
                Kwargs kwargs;
                II_ASSIGN_OR_RETURN(v, apply_filter(node.filter->name, args, kwargs));
            }
            II_TRY(set_targets(node, v));
            return Flow::kNormal;
        }
        case NodeKind::kMacro:
            define_macro(node);
            return Flow::kNormal;
        case NodeKind::kBreak:
            return Flow::kBreak;
        case NodeKind::kContinue:
            return Flow::kContinue;
        case NodeKind::kFilterBlock: {
            std::string captured;
            II_ASSIGN_OR_RETURN(const Flow flow, exec(node.body, captured));
            std::vector<Value> args{Value(captured)};
            Kwargs kwargs;
            II_ASSIGN_OR_RETURN(const Value v, apply_filter(node.filter->name, args, kwargs));
            out += v.to_string();
            return flow;
        }
        case NodeKind::kScope:
            return exec(node.body, out);
        }
        return Flow::kNormal;
    }

    Status set_targets(const Node& node, const Value& v) {
        if (node.targets.empty()) {
            return ok_status();  // {% do %}
        }
        if (!node.attr.empty()) {
            const Value target = lookup(node.targets[0]);
            if (!target.is_object()) {
                return template_error(std::format("cannot set attribute '{}' of '{}'", node.attr, node.targets[0]));
            }
            target.set(node.attr, v);
            return ok_status();
        }
        if (node.targets.size() == 1) {
            assign(node.targets[0], v);
            return ok_status();
        }
        const Array items = items_of(v);
        if (items.size() != node.targets.size()) {
            return template_error("cannot unpack the value into the targets of set");
        }
        for (std::size_t i = 0; i < items.size(); ++i) {
            assign(node.targets[i], items[i]);
        }
        return ok_status();
    }

    Result<Flow> exec_for(const Node& node, std::string& out) {
        II_ASSIGN_OR_RETURN(const Value iterable, eval(*node.expr));
        Array items;
        if (iterable.is_object() && node.targets.size() == 2) {
            for (const auto& [key, value] : iterable.object()) {
                items.emplace_back(Array{Value(key), value});
            }
        } else {
            items = items_of(iterable);
        }
        // Apply the loop filter first: loop.* counts the filtered items.
        std::vector<Value> selected;
        for (const Value& item : items) {
            if (node.filter != nullptr) {
                scopes_.emplace_back();
                bind_targets(node, item);
                Result<Value> keep = eval(*node.filter);
                scopes_.pop_back();
                if (!keep.ok()) {
                    return keep.status();
                }
                if (!keep->truthy()) {
                    continue;
                }
            }
            selected.push_back(item);
        }
        if (selected.empty()) {
            return exec(node.else_body, out);
        }
        const auto n = static_cast<std::int64_t>(selected.size());
        for (std::int64_t i = 0; i < n; ++i) {
            if (++iterations_ > kMaxIterations) {
                return template_error("too many loop iterations");
            }
            scopes_.emplace_back();
            bind_targets(node, selected[static_cast<std::size_t>(i)]);
            Object loop;
            loop.emplace_back("index", Value(i + 1));
            loop.emplace_back("index0", Value(i));
            loop.emplace_back("revindex", Value(n - i));
            loop.emplace_back("revindex0", Value(n - i - 1));
            loop.emplace_back("first", Value(i == 0));
            loop.emplace_back("last", Value(i == n - 1));
            loop.emplace_back("length", Value(n));
            loop.emplace_back("previtem", i > 0 ? selected[static_cast<std::size_t>(i - 1)] : Value());
            loop.emplace_back("nextitem", i + 1 < n ? selected[static_cast<std::size_t>(i + 1)] : Value());
            assign("loop", Value(std::move(loop)));
            Result<Flow> flow = exec(node.body, out);
            scopes_.pop_back();
            if (!flow.ok()) {
                return flow.status();
            }
            if (*flow == Flow::kBreak) {
                break;
            }
        }
        return Flow::kNormal;
    }

    void bind_targets(const Node& node, const Value& item) {
        if (node.targets.size() == 1) {
            assign(node.targets[0], item);
            return;
        }
        const Array parts = items_of(item);
        for (std::size_t t = 0; t < node.targets.size(); ++t) {
            assign(node.targets[t], t < parts.size() ? parts[t] : Value());
        }
    }

    void define_macro(const Node& node) {
        const Node* macro = &node;
        assign(node.attr, Value::function([this, macro](std::vector<Value>& args, Kwargs& kwargs) -> Result<Value> {
                   if (++depth_ > kMaxDepth) {
                       --depth_;
                       return template_error("macro recursion too deep");
                   }
                   scopes_.emplace_back();
                   for (std::size_t i = 0; i < macro->params.size(); ++i) {
                       const auto& [name, fallback] = macro->params[i];
                       Value v;
                       if (const Value* given = argument(args, kwargs, i, name)) {
                           v = *given;
                       } else if (fallback != nullptr) {
                           Result<Value> d = eval(*fallback);
                           if (!d.ok()) {
                               scopes_.pop_back();
                               --depth_;
                               return d.status();
                           }
                           v = *d;
                       }
                       assign(name, v);
                   }
                   std::string out;
                   Result<Flow> flow = exec(macro->body, out);
                   scopes_.pop_back();
                   --depth_;
                   if (!flow.ok()) {
                       return flow.status();
                   }
                   return Value(std::move(out));
               }));
    }

    // --- expressions ---
    Result<Value> eval(const Expr& e) {
        if (++depth_ > kMaxDepth * 4) {
            --depth_;
            return template_error("expression nesting too deep");
        }
        Result<Value> result = eval_inner(e);
        --depth_;
        return result;
    }

    Result<Value> eval_inner(const Expr& e) {
        switch (e.kind) {
        case ExprKind::kLiteral:
            return e.literal;
        case ExprKind::kName:
            return lookup(e.name);
        case ExprKind::kList: {
            Array items;
            for (const ExprPtr& item : e.args) {
                II_ASSIGN_OR_RETURN(Value v, eval(*item));
                items.push_back(std::move(v));
            }
            return make_list(std::move(items));
        }
        case ExprKind::kDict: {
            Value dict{Object{}};
            for (std::size_t i = 0; i + 1 < e.args.size(); i += 2) {
                II_ASSIGN_OR_RETURN(const Value key, eval(*e.args[i]));
                II_ASSIGN_OR_RETURN(Value value, eval(*e.args[i + 1]));
                dict.set(key.to_string(), std::move(value));
            }
            return dict;
        }
        case ExprKind::kAttr: {
            II_ASSIGN_OR_RETURN(const Value object, eval(*e.args[0]));
            return attribute(object, e.name);
        }
        case ExprKind::kItem: {
            II_ASSIGN_OR_RETURN(const Value object, eval(*e.args[0]));
            II_ASSIGN_OR_RETURN(const Value key, eval(*e.args[1]));
            return item(object, key);
        }
        case ExprKind::kSlice: {
            II_ASSIGN_OR_RETURN(const Value object, eval(*e.args[0]));
            Value bounds[3];
            for (int i = 0; i < 3; ++i) {
                if (e.args[static_cast<std::size_t>(i + 1)] != nullptr) {
                    II_ASSIGN_OR_RETURN(bounds[i], eval(*e.args[static_cast<std::size_t>(i + 1)]));
                }
            }
            if (object.is_string()) {
                std::string out;
                for (const std::size_t i : slice_indices(object.as_string().size(), bounds[0], bounds[1], bounds[2])) {
                    out += object.as_string()[i];
                }
                return Value(std::move(out));
            }
            Array items;
            if (object.is_array()) {
                for (const std::size_t i : slice_indices(object.array().size(), bounds[0], bounds[1], bounds[2])) {
                    items.push_back(object.array()[i]);
                }
            }
            return make_list(std::move(items));
        }
        case ExprKind::kCall: {
            II_ASSIGN_OR_RETURN(const Value callee, eval(*e.args[0]));
            std::vector<Value> args;
            Kwargs kwargs;
            for (std::size_t i = 1; i < e.args.size(); ++i) {
                II_ASSIGN_OR_RETURN(Value v, eval(*e.args[i]));
                args.push_back(std::move(v));
            }
            for (const auto& [key, value] : e.kwargs) {
                II_ASSIGN_OR_RETURN(Value v, eval(*value));
                kwargs.emplace_back(key, std::move(v));
            }
            if (!callee.is_function()) {
                return template_error("call of a value that is not a function");
            }
            return callee.function()(args, kwargs);
        }
        case ExprKind::kFilter: {
            std::vector<Value> args;
            Kwargs kwargs;
            for (const ExprPtr& arg : e.args) {
                II_ASSIGN_OR_RETURN(Value v, eval(*arg));
                args.push_back(std::move(v));
            }
            for (const auto& [key, value] : e.kwargs) {
                II_ASSIGN_OR_RETURN(Value v, eval(*value));
                kwargs.emplace_back(key, std::move(v));
            }
            return apply_filter(e.name, args, kwargs);
        }
        case ExprKind::kTest: {
            std::vector<Value> args;
            for (const ExprPtr& arg : e.args) {
                II_ASSIGN_OR_RETURN(Value v, eval(*arg));
                args.push_back(std::move(v));
            }
            II_ASSIGN_OR_RETURN(const bool result, apply_test(e.name, args));
            return Value(result != e.negated);
        }
        case ExprKind::kUnary: {
            II_ASSIGN_OR_RETURN(const Value v, eval(*e.args[0]));
            if (e.name == "not") {
                return Value(!v.truthy());
            }
            if (v.is_int()) {
                return Value(e.name == "-" ? -v.as_int() : v.as_int());
            }
            if (v.is_float()) {
                return Value(e.name == "-" ? -v.as_number() : v.as_number());
            }
            return template_error("unary operator on a non-number");
        }
        case ExprKind::kAnd: {
            II_ASSIGN_OR_RETURN(const Value a, eval(*e.args[0]));
            if (!a.truthy()) {
                return a;
            }
            return eval(*e.args[1]);
        }
        case ExprKind::kOr: {
            II_ASSIGN_OR_RETURN(const Value a, eval(*e.args[0]));
            if (a.truthy()) {
                return a;
            }
            return eval(*e.args[1]);
        }
        case ExprKind::kConditional: {
            II_ASSIGN_OR_RETURN(const Value condition, eval(*e.args[0]));
            return eval(*e.args[condition.truthy() ? 1 : 2]);
        }
        case ExprKind::kBinary: {
            II_ASSIGN_OR_RETURN(const Value a, eval(*e.args[0]));
            II_ASSIGN_OR_RETURN(const Value b, eval(*e.args[1]));
            return binary(e.name, a, b);
        }
        }
        return template_error("bad expression");
    }

    static Result<Value> binary(const std::string& op, const Value& a, const Value& b) {
        if (op == "==") {
            return Value(a == b);
        }
        if (op == "!=") {
            return Value(!(a == b));
        }
        if (op == "<" || op == "<=" || op == ">" || op == ">=") {
            const int c = compare(a, b);
            return Value(op == "<" ? c < 0 : op == "<=" ? c <= 0 : op == ">" ? c > 0 : c >= 0);
        }
        if (op == "in" || op == "not in") {
            bool found = false;
            if (b.is_string()) {
                found = b.as_string().find(a.to_string()) != std::string::npos;
            } else if (b.is_array()) {
                found = std::find(b.array().begin(), b.array().end(), a) != b.array().end();
            } else if (b.is_object()) {
                found = b.find(a.to_string()) != nullptr;
            }
            return Value(op == "in" ? found : !found);
        }
        if (op == "~") {
            return Value(a.to_string() + b.to_string());
        }
        if (op == "+") {
            if (a.is_string() && b.is_string()) {
                return Value(a.as_string() + b.as_string());
            }
            if (a.is_array() && b.is_array()) {
                Array items = a.array();
                items.insert(items.end(), b.array().begin(), b.array().end());
                return make_list(std::move(items));
            }
        }
        if (op == "*" && a.is_string() && b.is_int()) {
            std::string out;
            for (std::int64_t i = 0; i < std::min<std::int64_t>(b.as_int(), 1 << 20); ++i) {
                out += a.as_string();
            }
            return Value(std::move(out));
        }
        if (op == "%" && a.is_string()) {  // printf-style formatting with %s / %d
            std::string out;
            const Array values = b.is_array() ? b.array() : Array{b};
            std::size_t next = 0;
            const std::string& f = a.as_string();
            for (std::size_t i = 0; i < f.size(); ++i) {
                if (f[i] == '%' && i + 1 < f.size()) {
                    const char spec = f[++i];
                    if (spec == '%') {
                        out += '%';
                    } else if (next < values.size()) {
                        out += values[next++].to_string();
                    }
                } else {
                    out += f[i];
                }
            }
            return Value(std::move(out));
        }
        if (!a.is_number() || !b.is_number()) {
            return template_error(std::format("operator '{}' needs numbers", op));
        }
        const bool ints = a.is_int() && b.is_int();
        if (op == "+") {
            return ints ? Value(a.as_int() + b.as_int()) : Value(a.as_number() + b.as_number());
        }
        if (op == "-") {
            return ints ? Value(a.as_int() - b.as_int()) : Value(a.as_number() - b.as_number());
        }
        if (op == "*") {
            return ints ? Value(a.as_int() * b.as_int()) : Value(a.as_number() * b.as_number());
        }
        if (op == "/") {
            if (b.as_number() == 0) {
                return template_error("division by zero");
            }
            return Value(a.as_number() / b.as_number());
        }
        if (op == "//" || op == "%") {
            if (b.as_number() == 0) {
                return template_error("division by zero");
            }
            if (ints) {
                std::int64_t q = a.as_int() / b.as_int();
                if ((a.as_int() % b.as_int() != 0) && ((a.as_int() < 0) != (b.as_int() < 0))) {
                    --q;
                }
                return op == "//" ? Value(q) : Value(a.as_int() - q * b.as_int());
            }
            const double q = std::floor(a.as_number() / b.as_number());
            return op == "//" ? Value(q) : Value(a.as_number() - q * b.as_number());
        }
        if (op == "**") {
            return ints && b.as_int() >= 0 ? Value(static_cast<std::int64_t>(std::pow(a.as_number(), b.as_number())))
                                           : Value(std::pow(a.as_number(), b.as_number()));
        }
        return template_error(std::format("unknown operator '{}'", op));
    }

    static Result<Value> item(const Value& object, const Value& key) {
        if (object.is_array() && key.is_int()) {
            const std::int64_t i = python_index(key.as_int(), object.array().size());
            return i < 0 ? Value() : object.array()[static_cast<std::size_t>(i)];
        }
        if (object.is_string() && key.is_int()) {
            const std::int64_t i = python_index(key.as_int(), object.as_string().size());
            return i < 0 ? Value() : Value(std::string(1, object.as_string()[static_cast<std::size_t>(i)]));
        }
        if (object.is_object()) {
            const Value* v = object.find(key.to_string());
            return v != nullptr ? *v : Value();
        }
        return Value();
    }

    Result<Value> attribute(const Value& object, const std::string& name) {
        if (object.is_object()) {
            if (const Value* v = object.find(name)) {
                return *v;
            }
        }
        if (Value method = bound_method(object, name); method.is_function()) {
            return method;
        }
        return Value();
    }

    // --- methods of strings, lists and dicts ---
    static Value bound_method(const Value& self, const std::string& name) {
        if (self.is_string()) {
            const std::string s = self.as_string();
            static const char* const kNames[] = {"strip", "lstrip",     "rstrip",  "split", "startswith", "endswith",
                                                 "upper", "lower",      "replace", "title", "capitalize", "find",
                                                 "count", "splitlines", "join",    "format"};
            if (std::find(std::begin(kNames), std::end(kNames), name) == std::end(kNames)) {
                return Value();
            }
            return Value::function([s, name](std::vector<Value>& args, Kwargs& kwargs) -> Result<Value> {
                (void)kwargs;
                const std::string ws = " \t\n\r\f\v";
                if (name == "strip" || name == "lstrip" || name == "rstrip") {
                    const std::string chars = !args.empty() && args[0].is_string() ? args[0].as_string() : ws;
                    return Value(strip(s, chars, name != "rstrip", name != "lstrip"));
                }
                if (name == "upper") {
                    return Value(upper(s));
                }
                if (name == "lower") {
                    return Value(lower(s));
                }
                if (name == "title" || name == "capitalize") {
                    std::string out = name == "capitalize" ? lower(s) : s;
                    bool start = true;
                    for (char& c : out) {
                        if (start && std::isalpha(static_cast<unsigned char>(c))) {
                            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                        } else if (name == "title" && std::isalpha(static_cast<unsigned char>(c))) {
                            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                        }
                        start = name == "title" ? !std::isalpha(static_cast<unsigned char>(c)) : false;
                    }
                    return Value(std::move(out));
                }
                if (name == "startswith" || name == "endswith") {
                    const Array prefixes = !args.empty() && args[0].is_array()
                                               ? args[0].array()
                                               : Array{args.empty() ? Value("") : args[0]};
                    for (const Value& p : prefixes) {
                        const std::string& t = p.to_string();
                        if (t.size() <= s.size() &&
                            (name == "startswith" ? s.compare(0, t.size(), t) == 0
                                                  : s.compare(s.size() - t.size(), t.size(), t) == 0)) {
                            return Value(true);
                        }
                    }
                    return Value(false);
                }
                if (name == "replace") {
                    if (args.size() < 2) {
                        return template_error("replace() needs two arguments");
                    }
                    const std::string from = args[0].to_string();
                    const std::string to = args[1].to_string();
                    std::int64_t limit = args.size() > 2 && args[2].is_int() ? args[2].as_int() : -1;
                    if (from.empty()) {
                        return Value(s);
                    }
                    std::string out;
                    std::size_t start = 0;
                    while (limit != 0) {
                        const std::size_t at = s.find(from, start);
                        if (at == std::string::npos) {
                            break;
                        }
                        out.append(s, start, at - start);
                        out += to;
                        start = at + from.size();
                        --limit;
                    }
                    out.append(s, start);
                    return Value(std::move(out));
                }
                if (name == "split" || name == "splitlines") {
                    Array parts;
                    if (name == "splitlines") {
                        std::size_t start = 0;
                        for (std::size_t i = 0; i <= s.size(); ++i) {
                            if (i == s.size() || s[i] == '\n') {
                                if (i < s.size() || start < s.size()) {
                                    std::string line = s.substr(start, i - start);
                                    if (!line.empty() && line.back() == '\r') {
                                        line.pop_back();
                                    }
                                    parts.emplace_back(std::move(line));
                                }
                                start = i + 1;
                            }
                        }
                        return make_list(std::move(parts));
                    }
                    std::int64_t limit = args.size() > 1 && args[1].is_int() ? args[1].as_int() : -1;
                    if (args.empty() || args[0].is_none()) {
                        std::size_t i = 0;
                        while (i < s.size()) {
                            while (i < s.size() && ws.find(s[i]) != std::string::npos) {
                                ++i;
                            }
                            if (i >= s.size()) {
                                break;
                            }
                            std::size_t j = i;
                            while (j < s.size() && ws.find(s[j]) == std::string::npos) {
                                ++j;
                            }
                            if (limit == 0) {
                                parts.emplace_back(s.substr(i));
                                break;
                            }
                            parts.emplace_back(s.substr(i, j - i));
                            --limit;
                            i = j;
                        }
                        return make_list(std::move(parts));
                    }
                    const std::string sep = args[0].to_string();
                    if (sep.empty()) {
                        return template_error("empty separator");
                    }
                    std::size_t start = 0;
                    while (limit != 0) {
                        const std::size_t at = s.find(sep, start);
                        if (at == std::string::npos) {
                            break;
                        }
                        parts.emplace_back(s.substr(start, at - start));
                        start = at + sep.size();
                        --limit;
                    }
                    parts.emplace_back(s.substr(start));
                    return make_list(std::move(parts));
                }
                if (name == "find") {
                    const std::size_t at = args.empty() ? std::string::npos : s.find(args[0].to_string());
                    return Value(at == std::string::npos ? std::int64_t{-1} : static_cast<std::int64_t>(at));
                }
                if (name == "count") {
                    const std::string needle = args.empty() ? std::string() : args[0].to_string();
                    std::int64_t count = 0;
                    for (std::size_t at = 0; !needle.empty() && (at = s.find(needle, at)) != std::string::npos;
                         at += needle.size()) {
                        ++count;
                    }
                    return Value(count);
                }
                if (name == "join") {
                    std::string out;
                    const Array items = args.empty() ? Array{} : items_of(args[0]);
                    for (std::size_t i = 0; i < items.size(); ++i) {
                        out += (i > 0 ? s : "") + items[i].to_string();
                    }
                    return Value(std::move(out));
                }
                // format: positional {} replacement
                std::string out;
                std::size_t next = 0;
                for (std::size_t i = 0; i < s.size(); ++i) {
                    if (s[i] == '{' && i + 1 < s.size() && s[i + 1] == '}') {
                        out += next < args.size() ? args[next++].to_string() : "";
                        ++i;
                    } else {
                        out += s[i];
                    }
                }
                return Value(std::move(out));
            });
        }
        if (self.is_object()) {
            if (name == "items" || name == "keys" || name == "values" || name == "get" || name == "update" ||
                name == "pop") {
                return Value::function([self, name](std::vector<Value>& args, Kwargs& kwargs) -> Result<Value> {
                    Array out;
                    if (name == "get") {
                        const Value* v = args.empty() ? nullptr : self.find(args[0].to_string());
                        return v != nullptr ? *v : (args.size() > 1 ? args[1] : Value(nullptr));
                    }
                    if (name == "pop") {
                        if (args.empty()) {
                            return template_error("pop() needs a key");
                        }
                        Object& object = self.object();
                        for (auto it = object.begin(); it != object.end(); ++it) {
                            if (it->first == args[0].to_string()) {
                                Value v = it->second;
                                object.erase(it);
                                return v;
                            }
                        }
                        return args.size() > 1 ? args[1] : Value(nullptr);
                    }
                    if (name == "update") {
                        if (!args.empty() && args[0].is_object()) {
                            for (const auto& [key, value] : args[0].object()) {
                                self.set(key, value);
                            }
                        }
                        for (const auto& [key, value] : kwargs) {
                            self.set(key, value);
                        }
                        return Value(nullptr);
                    }
                    for (const auto& [key, value] : self.object()) {
                        if (name == "items") {
                            out.emplace_back(Array{Value(key), value});
                        } else if (name == "keys") {
                            out.emplace_back(key);
                        } else {
                            out.push_back(value);
                        }
                    }
                    return make_list(std::move(out));
                });
            }
        }
        if (self.is_array()) {
            if (name == "append" || name == "pop" || name == "index" || name == "count" || name == "extend" ||
                name == "insert") {
                return Value::function([self, name](std::vector<Value>& args, Kwargs&) -> Result<Value> {
                    Array& items = self.array();
                    if (name == "append") {
                        if (!args.empty()) {
                            items.push_back(args[0]);
                        }
                        return Value(nullptr);
                    }
                    if (name == "extend") {
                        if (!args.empty()) {
                            const Array more = items_of(args[0]);
                            items.insert(items.end(), more.begin(), more.end());
                        }
                        return Value(nullptr);
                    }
                    if (name == "insert") {
                        if (args.size() >= 2 && args[0].is_int()) {
                            const auto at =
                                std::clamp<std::int64_t>(args[0].as_int(), 0, static_cast<std::int64_t>(items.size()));
                            items.insert(items.begin() + at, args[1]);
                        }
                        return Value(nullptr);
                    }
                    if (name == "pop") {
                        if (items.empty()) {
                            return template_error("pop from an empty list");
                        }
                        const std::int64_t i =
                            python_index(args.empty() || !args[0].is_int() ? -1 : args[0].as_int(), items.size());
                        if (i < 0) {
                            return template_error("pop index out of range");
                        }
                        Value v = items[static_cast<std::size_t>(i)];
                        items.erase(items.begin() + i);
                        return v;
                    }
                    std::int64_t count = 0;
                    for (std::size_t i = 0; i < items.size(); ++i) {
                        if (!args.empty() && items[i] == args[0]) {
                            if (name == "index") {
                                return Value(static_cast<std::int64_t>(i));
                            }
                            ++count;
                        }
                    }
                    if (name == "index") {
                        return template_error("value not in list");
                    }
                    return Value(count);
                });
            }
        }
        return Value();
    }

    // --- filters ---
    Result<Value> apply_filter(const std::string& name, std::vector<Value>& args, const Kwargs& kwargs) {
        const Value v = args.empty() ? Value() : args[0];
        auto arg = [&](std::size_t index, std::string_view key) { return argument(args, kwargs, index, key); };
        if (name == "length" || name == "count") {
            if (v.is_string()) {
                return Value(static_cast<std::int64_t>(v.as_string().size()));
            }
            if (v.is_array()) {
                return Value(static_cast<std::int64_t>(v.array().size()));
            }
            if (v.is_object()) {
                return Value(static_cast<std::int64_t>(v.object().size()));
            }
            return Value(std::int64_t{0});
        }
        if (name == "trim") {
            return Value(strip(v.to_string(), " \t\n\r\f\v", true, true));
        }
        if (name == "upper") {
            return Value(upper(v.to_string()));
        }
        if (name == "lower") {
            return Value(lower(v.to_string()));
        }
        if (name == "capitalize" || name == "title") {
            Value method = bound_method(Value(v.to_string()), name);
            std::vector<Value> none;
            Kwargs no_kwargs;
            return method.function()(none, no_kwargs);
        }
        if (name == "string") {
            return Value(v.to_string());
        }
        if (name == "safe" || name == "escape" || name == "e" || name == "forceescape") {
            if (name == "safe") {
                return v;
            }
            std::string out;
            for (const char c : v.to_string()) {
                switch (c) {
                case '&':
                    out += "&amp;";
                    break;
                case '<':
                    out += "&lt;";
                    break;
                case '>':
                    out += "&gt;";
                    break;
                case '"':
                    out += "&#34;";
                    break;
                case '\'':
                    out += "&#39;";
                    break;
                default:
                    out += c;
                }
            }
            return Value(std::move(out));
        }
        if (name == "tojson") {
            const Value* indent = arg(1, "indent");
            return Value(v.to_json(indent != nullptr && indent->is_int() ? static_cast<int>(indent->as_int()) : -1));
        }
        if (name == "int") {
            if (v.is_number()) {
                return Value(static_cast<std::int64_t>(v.as_number()));
            }
            if (v.is_string()) {
                const std::string text = strip(v.as_string(), " \t\n\r", true, true);
                std::int64_t parsed = 0;
                std::from_chars(text.data(), text.data() + text.size(), parsed);
                return Value(parsed);
            }
            return Value(std::int64_t{0});
        }
        if (name == "float") {
            if (v.is_number()) {
                return Value(v.as_number());
            }
            return Value(v.is_string() ? std::strtod(v.as_string().c_str(), nullptr) : 0.0);
        }
        if (name == "abs") {
            return v.is_int() ? Value(v.as_int() < 0 ? -v.as_int() : v.as_int()) : Value(std::fabs(v.as_number()));
        }
        if (name == "round") {
            const Value* precision = arg(1, "precision");
            const double scale = std::pow(10.0, precision != nullptr && precision->is_int() ? precision->as_int() : 0);
            return Value(std::round(v.as_number() * scale) / scale);
        }
        if (name == "default" || name == "d") {
            const Value* fallback = arg(1, "default_value");
            const Value* boolean = arg(2, "boolean");
            const bool use_fallback = v.is_undefined() || (boolean != nullptr && boolean->truthy() && !v.truthy());
            return use_fallback ? (fallback != nullptr ? *fallback : Value("")) : v;
        }
        if (name == "first" || name == "last") {
            const Array items = items_of(v);
            if (items.empty()) {
                return Value();
            }
            return name == "first" ? items.front() : items.back();
        }
        if (name == "list") {
            return make_list(items_of(v));
        }
        if (name == "reverse") {
            if (v.is_string()) {
                return Value(std::string(v.as_string().rbegin(), v.as_string().rend()));
            }
            Array items = items_of(v);
            std::reverse(items.begin(), items.end());
            return make_list(std::move(items));
        }
        if (name == "join") {
            const Value* separator = arg(1, "d");
            const Value* attribute_name = arg(2, "attribute");
            std::string out;
            const Array items = items_of(v);
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (i > 0 && separator != nullptr) {
                    out += separator->to_string();
                }
                const Value item = attribute_name != nullptr ? (items[i].find(attribute_name->to_string()) != nullptr
                                                                    ? *items[i].find(attribute_name->to_string())
                                                                    : Value())
                                                             : items[i];
                out += item.to_string();
            }
            return Value(std::move(out));
        }
        if (name == "items") {
            Array out;
            if (v.is_object()) {
                for (const auto& [key, value] : v.object()) {
                    out.emplace_back(Array{Value(key), value});
                }
            }
            return make_list(std::move(out));
        }
        if (name == "dictsort") {
            Array out;
            if (v.is_object()) {
                for (const auto& [key, value] : v.object()) {
                    out.emplace_back(Array{Value(key), value});
                }
            }
            std::stable_sort(out.begin(), out.end(),
                             [](const Value& a, const Value& b) { return compare(a.array()[0], b.array()[0]) < 0; });
            return make_list(std::move(out));
        }
        if (name == "unique") {
            Array out;
            for (const Value& item : items_of(v)) {
                if (std::find(out.begin(), out.end(), item) == out.end()) {
                    out.push_back(item);
                }
            }
            return make_list(std::move(out));
        }
        if (name == "sort") {
            Array items = items_of(v);
            const Value* reverse = arg(1, "reverse");
            const Value* attribute_name = arg(3, "attribute");
            auto key = [&](const Value& item) {
                if (attribute_name != nullptr) {
                    const Value* field = item.find(attribute_name->to_string());
                    return field != nullptr ? *field : Value();
                }
                return item;
            };
            std::stable_sort(items.begin(), items.end(),
                             [&](const Value& a, const Value& b) { return compare(key(a), key(b)) < 0; });
            if (reverse != nullptr && reverse->truthy()) {
                std::reverse(items.begin(), items.end());
            }
            return make_list(std::move(items));
        }
        if (name == "sum") {
            double total = 0;
            bool all_int = true;
            for (const Value& item : items_of(v)) {
                total += item.is_number() ? item.as_number() : 0;
                all_int = all_int && item.is_int();
            }
            return all_int ? Value(static_cast<std::int64_t>(total)) : Value(total);
        }
        if (name == "replace") {
            Value method = bound_method(Value(v.to_string()), "replace");
            std::vector<Value> rest(args.begin() + 1, args.end());
            Kwargs no_kwargs;
            return method.function()(rest, no_kwargs);
        }
        if (name == "indent") {
            const Value* width = arg(1, "width");
            const Value* first = arg(2, "first");
            const std::string pad(static_cast<std::size_t>(width != nullptr && width->is_int() ? width->as_int() : 4),
                                  ' ');
            std::string out;
            const std::string s = v.to_string();
            bool line_start = first != nullptr && first->truthy();
            for (std::size_t i = 0; i < s.size(); ++i) {
                if (line_start && s[i] != '\n') {
                    out += pad;
                }
                out += s[i];
                line_start = s[i] == '\n';
            }
            return Value(std::move(out));
        }
        if (name == "map" || name == "selectattr" || name == "rejectattr" || name == "select" || name == "reject") {
            Array out;
            const Array items = items_of(v);
            if (name == "map") {
                const Value* attribute_name = kwarg(kwargs, "attribute");
                for (const Value& item : items) {
                    if (attribute_name != nullptr) {
                        const Value* field = item.find(attribute_name->to_string());
                        out.push_back(field != nullptr ? *field : Value());
                    } else if (args.size() > 1) {
                        std::vector<Value> call{item};
                        call.insert(call.end(), args.begin() + 2, args.end());
                        Kwargs none;
                        II_ASSIGN_OR_RETURN(Value mapped, apply_filter(args[1].to_string(), call, none));
                        out.push_back(std::move(mapped));
                    }
                }
                return make_list(std::move(out));
            }
            const bool by_attribute = name == "selectattr" || name == "rejectattr";
            const bool keep_matching = name == "selectattr" || name == "select";
            for (const Value& item : items) {
                Value subject = item;
                std::size_t test_index = 1;
                if (by_attribute) {
                    const Value* field = args.size() > 1 ? item.find(args[1].to_string()) : nullptr;
                    subject = field != nullptr ? *field : Value();
                    test_index = 2;
                }
                bool match = subject.truthy();
                if (args.size() > test_index) {
                    std::vector<Value> test_args{subject};
                    test_args.insert(test_args.end(), args.begin() + static_cast<std::ptrdiff_t>(test_index) + 1,
                                     args.end());
                    II_ASSIGN_OR_RETURN(match, apply_test(args[test_index].to_string(), test_args));
                }
                if (match == keep_matching) {
                    out.push_back(item);
                }
            }
            return make_list(std::move(out));
        }
        if (name == "wordcount") {
            std::int64_t count = 0;
            bool in_word = false;
            for (const char c : v.to_string()) {
                const bool space = std::isspace(static_cast<unsigned char>(c)) != 0;
                count += !space && !in_word ? 1 : 0;
                in_word = !space;
            }
            return Value(count);
        }
        return template_error(std::format("unknown filter '{}'", name));
    }

    // --- tests ---
    Result<bool> apply_test(const std::string& name, const std::vector<Value>& args) {
        const Value v = args.empty() ? Value() : args[0];
        if (name == "defined") {
            return !v.is_undefined();
        }
        if (name == "undefined") {
            return v.is_undefined();
        }
        if (name == "none") {
            return v.is_none();
        }
        if (name == "string") {
            return v.is_string();
        }
        if (name == "number") {
            return v.is_number();
        }
        if (name == "integer") {
            return v.is_int();
        }
        if (name == "float") {
            return v.is_float();
        }
        if (name == "boolean") {
            return v.is_bool();
        }
        if (name == "true") {
            return v.is_bool() && v.as_bool();
        }
        if (name == "false") {
            return v.is_bool() && !v.as_bool();
        }
        if (name == "mapping") {
            return v.is_object();
        }
        if (name == "sequence" || name == "iterable") {
            return v.is_array() || v.is_string() || (name == "iterable" && v.is_object());
        }
        if (name == "callable") {
            return v.is_function();
        }
        if (name == "even" || name == "odd") {
            return v.is_int() && (v.as_int() % 2 == 0) == (name == "even");
        }
        if (name == "divisibleby") {
            return v.is_int() && args.size() > 1 && args[1].is_int() && args[1].as_int() != 0 &&
                   v.as_int() % args[1].as_int() == 0;
        }
        if (name == "equalto" || name == "eq" || name == "sameas" || name == "==") {
            return args.size() > 1 && v == args[1];
        }
        if (name == "ne" || name == "!=") {
            return args.size() > 1 && !(v == args[1]);
        }
        if (name == "in") {
            II_ASSIGN_OR_RETURN(const Value found, binary("in", v, args.size() > 1 ? args[1] : Value()));
            return found.truthy();
        }
        if (name == "lower") {
            return v.is_string() && lower(v.as_string()) == v.as_string();
        }
        if (name == "upper") {
            return v.is_string() && upper(v.as_string()) == v.as_string();
        }
        return template_error(std::format("unknown test '{}'", name));
    }

    // --- global functions ---
    static Value builtin(const std::string& name) {
        if (name == "raise_exception") {
            return Value::function([](std::vector<Value>& args, Kwargs&) -> Result<Value> {
                return template_error(args.empty() ? std::string("raise_exception") : args[0].to_string());
            });
        }
        if (name == "range") {
            return Value::function([](std::vector<Value>& args, Kwargs&) -> Result<Value> {
                std::int64_t start = 0;
                std::int64_t stop = 0;
                std::int64_t step = 1;
                if (args.size() == 1 && args[0].is_int()) {
                    stop = args[0].as_int();
                } else if (args.size() >= 2 && args[0].is_int() && args[1].is_int()) {
                    start = args[0].as_int();
                    stop = args[1].as_int();
                    step = args.size() > 2 && args[2].is_int() ? args[2].as_int() : 1;
                }
                if (step == 0) {
                    return template_error("range() step must not be zero");
                }
                Array out;
                for (std::int64_t i = start; step > 0 ? i < stop : i > stop; i += step) {
                    if (out.size() > static_cast<std::size_t>(kMaxIterations)) {
                        return template_error("range too large");
                    }
                    out.emplace_back(i);
                }
                return make_list(std::move(out));
            });
        }
        if (name == "namespace" || name == "dict") {
            return Value::function([](std::vector<Value>& args, Kwargs& kwargs) -> Result<Value> {
                Value object{Object{}};
                if (!args.empty() && args[0].is_object()) {
                    for (const auto& [key, value] : args[0].object()) {
                        object.set(key, value);
                    }
                }
                for (const auto& [key, value] : kwargs) {
                    object.set(key, value);
                }
                return object;
            });
        }
        if (name == "strftime_now") {
            return Value::function([](std::vector<Value>& args, Kwargs&) -> Result<Value> {
                const std::time_t now = std::time(nullptr);
                std::tm tm{};
#if defined(_WIN32)
                localtime_s(&tm, &now);
#else
                localtime_r(&now, &tm);
#endif
                // The common strftime specifiers (templates use e.g. "%d %b %Y").
                static const char* const kMonths[] = {"January",   "February", "March",    "April",
                                                      "May",       "June",     "July",     "August",
                                                      "September", "October",  "November", "December"};
                static const char* const kDays[] = {"Sunday",   "Monday", "Tuesday", "Wednesday",
                                                    "Thursday", "Friday", "Saturday"};
                const std::string format = args.empty() ? std::string("%Y-%m-%d") : args[0].to_string();
                std::string out;
                for (std::size_t i = 0; i < format.size(); ++i) {
                    if (format[i] != '%' || i + 1 == format.size()) {
                        out += format[i];
                        continue;
                    }
                    const std::string month = kMonths[tm.tm_mon % 12];
                    const std::string day = kDays[tm.tm_wday % 7];
                    switch (format[++i]) {
                    case 'Y':
                        out += std::to_string(tm.tm_year + 1900);
                        break;
                    case 'y':
                        out += std::format("{:02}", (tm.tm_year + 1900) % 100);
                        break;
                    case 'm':
                        out += std::format("{:02}", tm.tm_mon + 1);
                        break;
                    case 'd':
                        out += std::format("{:02}", tm.tm_mday);
                        break;
                    case 'H':
                        out += std::format("{:02}", tm.tm_hour);
                        break;
                    case 'M':
                        out += std::format("{:02}", tm.tm_min);
                        break;
                    case 'S':
                        out += std::format("{:02}", tm.tm_sec);
                        break;
                    case 'B':
                        out += month;
                        break;
                    case 'b':
                        out += month.substr(0, 3);
                        break;
                    case 'A':
                        out += day;
                        break;
                    case 'a':
                        out += day.substr(0, 3);
                        break;
                    case '%':
                        out += '%';
                        break;
                    default:
                        out += '%';
                        out += format[i];
                    }
                }
                return Value(std::move(out));
            });
        }
        return Value();
    }

    std::vector<Object> scopes_;
    std::int64_t iterations_ = 0;
    int depth_ = 0;
};

}  // namespace

Result<std::string> render(std::string_view source, const Value& context) {
    II_ASSIGN_OR_RETURN(std::vector<Segment> segments, split_template(source));
    TemplateParser parser(std::move(segments));
    II_ASSIGN_OR_RETURN(const Body body, parser.parse());
    Interpreter interpreter(context);
    return interpreter.run(body);
}

}  // namespace ii::llm::detail::jinja
