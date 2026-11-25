// SPDX-License-Identifier: MIT
#include "ir/text.h"

#include "core/parse.h"
#include "ir/tensor_literal.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <format>
#include <unordered_map>
#include <unordered_set>

namespace ii::ir {

// ==================================================================================================================
// Printing
// ==================================================================================================================

namespace {

bool is_name_start(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
}
bool is_name_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.';
}

/// True when `name` can be printed without quotes.
bool is_plain_name(std::string_view name) {
    return !name.empty() && is_name_start(name[0]) && std::all_of(name.begin(), name.end(), is_name_char);
}

std::string quoted(std::string_view text) {
    return to_string(AttrValue(std::string(text)));
}

std::string format_name(std::string_view name) {
    return is_plain_name(name) ? std::string(name) : quoted(name);
}

class Printer {
public:
    Printer(const Graph& graph, const PrintOptions& options) : graph_(graph), options_(options) { assign_names(); }

    std::string print() {
        out_ += graph_.name().empty() ? "graph {\n" : std::format("graph {} {{\n", format_name(graph_.name()));
        for (const Value* input : graph_.inputs()) {
            out_ += std::format("  input {} : {}\n", ref(input), input->type().to_string());
        }
        for (const StateBinding& binding : graph_.states()) {
            const Value* state = binding.state;
            out_ += std::format("  state {} : {}", ref(state), state->type().to_string());
            if (state->data().defined()) {
                out_ += " = " + literal(state->data());
            }
            out_ += '\n';
        }
        for (const auto& value : graph_.values()) {
            if (value->kind() == ValueKind::kConstant) {
                out_ += std::format("  const {} : {} = {}\n", ref(value.get()), value->type().to_string(),
                                    literal(value->data()));
            }
        }
        for (const auto& node : graph_.nodes()) {
            print_node(*node);
        }
        for (const StateBinding& binding : graph_.states()) {
            if (binding.update != nullptr) {
                out_ += std::format("  update {} = {}\n", ref(binding.state), ref(binding.update));
            }
        }
        for (const GraphOutput& output : graph_.outputs()) {
            out_ += std::format("  output {} = {}\n", format_name(output.name), ref(output.value));
        }
        out_ += "}\n";
        return std::move(out_);
    }

private:
    void assign_names() {
        std::unordered_set<std::string> used;
        int counter = 0;
        auto assign = [&](const Value* value) {
            const std::string& name = value->name();
            if (!name.empty() &&
                !(std::isdigit(static_cast<unsigned char>(name[0])) != 0 &&
                  std::all_of(name.begin(), name.end(),
                              [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; })) &&
                used.insert(name).second) {
                names_[value] = "%" + format_name(name);
            } else {
                names_[value] = std::format("%{}", counter++);
            }
        };
        for (const Value* input : graph_.inputs()) {
            assign(input);
        }
        for (const StateBinding& binding : graph_.states()) {
            assign(binding.state);
        }
        for (const auto& value : graph_.values()) {
            if (value->kind() == ValueKind::kConstant) {
                assign(value.get());
            }
        }
        for (const auto& node : graph_.nodes()) {
            for (const Value* output : node->outputs()) {
                assign(output);
            }
        }
    }

    std::string ref(const Value* value) const {
        if (value == nullptr) {
            return "_";
        }
        const auto it = names_.find(value);
        return it != names_.end() ? it->second : std::format("%<unknown #{}>", value->id());
    }

    std::string literal(const Tensor& tensor) const {
        return format_tensor_literal(tensor, options_.max_constant_elements);
    }

    void print_node(const Node& node) {
        out_ += "  ";
        for (int i = 0; i < node.num_outputs(); ++i) {
            if (i > 0) {
                out_ += ", ";
            }
            out_ += std::format("{} : {}", ref(node.output(i)), node.output(i)->type().to_string());
        }
        if (node.num_outputs() == 0) {
            out_ += "()";
        }
        out_ += std::format(" = {}(", node.op_name());
        for (int i = 0; i < node.num_inputs(); ++i) {
            if (i > 0) {
                out_ += ", ";
            }
            out_ += ref(node.input(i));
        }
        out_ += ')';
        if (!node.attrs().empty()) {
            out_ += ' ';
            out_ += node.attrs().to_string();
        }
        out_ += '\n';
    }

    const Graph& graph_;
    const PrintOptions& options_;
    std::unordered_map<const Value*, std::string> names_;
    std::string out_;
};

}  // namespace

std::string print_graph(const Graph& graph, const PrintOptions& options) {
    return Printer(graph, options).print();
}

std::string Graph::to_text(const PrintOptions& options) const {
    return print_graph(*this, options);
}

// ==================================================================================================================
// Parsing
// ==================================================================================================================

namespace {

class Parser {
public:
    Parser(std::string_view text, const ParseOptions& options)
        : text_(text), options_(options), registry_(options.registry ? *options.registry : default_op_registry()) {}

    Result<std::unique_ptr<Graph>> parse() {
        graph_ = std::make_unique<Graph>();
        II_TRY(parse_graph());
        skip_space();
        if (!at_end()) {
            return error("unexpected text after the graph");
        }
        if (options_.verify) {
            if (Status status = graph_->verify(); !status.ok()) {
                return status.add_context("parsed graph is invalid");
            }
        }
        return std::move(graph_);
    }

private:
    // --- Low-level scanning ---

    bool at_end() const { return pos_ >= text_.size(); }
    char peek() const { return at_end() ? '\0' : text_[pos_]; }

    void skip_space() {
        while (!at_end()) {
            const char c = text_[pos_];
            if (c == '#') {
                while (!at_end() && text_[pos_] != '\n') {
                    ++pos_;
                }
            } else if (std::isspace(static_cast<unsigned char>(c)) != 0) {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool accept(char c) {
        skip_space();
        if (peek() == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    bool accept_word(std::string_view word) {
        skip_space();
        if (text_.substr(pos_, word.size()) == word &&
            (pos_ + word.size() >= text_.size() || !is_name_char(text_[pos_ + word.size()]))) {
            pos_ += word.size();
            return true;
        }
        return false;
    }

    Status expect(char c) {
        if (!accept(c)) {
            return error(std::format("expected '{}'", c));
        }
        return ok_status();
    }

    Status error(std::string_view message) const {
        std::size_t line = 1;
        std::size_t column = 1;
        for (std::size_t i = 0; i < pos_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        return errors::invalid_argument("{}:{}: {}", line, column, message);
    }

    std::string_view identifier() {
        skip_space();
        const std::size_t start = pos_;
        if (!is_name_start(peek())) {
            return {};
        }
        while (!at_end() && is_name_char(text_[pos_])) {
            ++pos_;
        }
        return text_.substr(start, pos_ - start);
    }

    Result<std::string> string_literal() {
        skip_space();
        if (peek() != '"') {
            return error("expected a string");
        }
        ++pos_;
        std::string result;
        while (!at_end() && text_[pos_] != '"') {
            char c = text_[pos_++];
            if (c == '\\') {
                if (at_end()) {
                    break;
                }
                const char e = text_[pos_++];
                switch (e) {
                case 'n':
                    c = '\n';
                    break;
                case 't':
                    c = '\t';
                    break;
                case 'x': {
                    const std::optional<int> code =
                        pos_ + 2 <= text_.size() ? hex_byte(text_.substr(pos_, 2)) : std::nullopt;
                    if (!code) {
                        return error("invalid \\x escape");
                    }
                    pos_ += 2;
                    c = static_cast<char>(*code);
                    break;
                }
                default:
                    c = e;
                }
            }
            result += c;
        }
        if (at_end()) {
            return error("unterminated string");
        }
        ++pos_;
        return result;
    }

    static std::optional<int> hex_digit(char c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        return std::nullopt;
    }

    static std::optional<int> hex_byte(std::string_view two) {
        const std::optional<int> hi = hex_digit(two[0]);
        const std::optional<int> lo = hex_digit(two[1]);
        if (!hi || !lo) {
            return std::nullopt;
        }
        return *hi * 16 + *lo;
    }

    /// Raw text of one list element: up to a ',' or the closing bracket at nesting depth 0.
    std::string_view list_element() {
        skip_space();
        const std::size_t start = pos_;
        int depth = 0;
        while (!at_end()) {
            const char c = text_[pos_];
            if (c == '"') {
                ++pos_;
                while (!at_end() && text_[pos_] != '"') {
                    pos_ += text_[pos_] == '\\' ? 2 : 1;
                }
                ++pos_;
                continue;
            }
            if (c == '(' || c == '[' || c == '{') {
                ++depth;
            } else if (c == ')' || c == ']' || c == '}') {
                if (depth == 0) {
                    break;
                }
                --depth;
            } else if (c == ',' && depth == 0) {
                break;
            }
            ++pos_;
        }
        std::string_view element = text_.substr(start, pos_ - start);
        while (!element.empty() && std::isspace(static_cast<unsigned char>(element.back())) != 0) {
            element.remove_suffix(1);
        }
        return element;
    }

    /// Parses "[e1, e2, ...]" calling `fn(element_text)` per element.
    template <class Fn>
    Status bracket_list(char open, char close, Fn&& fn) {
        II_TRY(expect(open));
        if (accept(close)) {
            return ok_status();
        }
        for (;;) {
            const std::size_t element_pos = pos_;
            const std::string_view element = list_element();
            if (element.empty()) {
                return error("empty list element");
            }
            if (Status status = fn(element); !status.ok()) {
                pos_ = element_pos;
                return error(status.message());
            }
            if (accept(close)) {
                return ok_status();
            }
            II_TRY(expect(','));
        }
    }

    // --- Types ---

    Result<Dims> dims() {
        Dims result;
        II_TRY(bracket_list('[', ']', [&](std::string_view element) -> Status {
            const std::optional<Dim> dim = Dim::parse(element);
            if (!dim) {
                return errors::invalid_argument("invalid dimension '{}'", element);
            }
            result.push_back(*dim);
            return ok_status();
        }));
        return result;
    }

    Result<DType> dtype() {
        const std::size_t start = pos_;
        const std::string_view name = identifier();
        const std::optional<DType> parsed = parse_dtype(name);
        if (!parsed) {
            pos_ = start;
            return error(std::format("expected a dtype, got '{}'", name));
        }
        return *parsed;
    }

    Result<TensorType> tensor_type() {
        II_ASSIGN_OR_RETURN(const DType element, dtype());
        II_ASSIGN_OR_RETURN(Dims shape, dims());
        return TensorType{element, std::move(shape)};
    }

    // --- Literals ---

    /// Parses a tensor literal after its type: "{v, ...}", "{...}" (elided) or "0x...".
    /// Returns an empty tensor for elided data.
    Result<Tensor> literal_data(const TensorType& type) {
        const std::optional<Shape> shape = type.static_shape();
        if (!shape) {
            return error("tensor literals need a static shape");
        }
        skip_space();
        if (text_.substr(pos_, 2) == "0x") {
            pos_ += 2;
            if (!is_block_quantized(type.dtype)) {
                return error("hex literals are only used for block-quantized types");
            }
            II_ASSIGN_OR_RETURN(Tensor tensor, Tensor::empty(type.dtype, *shape));
            auto* bytes = static_cast<std::uint8_t*>(tensor.data());
            for (std::size_t i = 0; i < tensor.nbytes(); ++i) {
                const std::optional<int> byte =
                    pos_ + 2 <= text_.size() ? hex_byte(text_.substr(pos_, 2)) : std::nullopt;
                if (!byte) {
                    return error("invalid or too short hex literal");
                }
                bytes[i] = static_cast<std::uint8_t>(*byte);
                pos_ += 2;
            }
            return tensor;
        }
        if (is_block_quantized(type.dtype)) {
            II_TRY(expect('{'));
            if (text_.substr(pos_, 4) == "...}") {
                pos_ += 4;
                return Tensor();
            }
            return error("block-quantized literals use hex: 0x...");
        }
        skip_space();
        if (text_.substr(pos_, 5) == "{...}") {
            pos_ += 5;
            return Tensor();
        }
        II_ASSIGN_OR_RETURN(Tensor tensor, Tensor::empty(type.dtype, *shape));
        std::int64_t count = 0;
        const std::int64_t numel = tensor.numel();
        const std::size_t element_bytes = element_size(type.dtype);
        auto* base = static_cast<std::byte*>(tensor.data());
        II_TRY(bracket_list('{', '}', [&](std::string_view element) -> Status {
            if (count >= numel) {
                return errors::invalid_argument("too many values for {}", type.to_string());
            }
            II_TRY(store_element(type.dtype, element, base + count * static_cast<std::int64_t>(element_bytes)));
            ++count;
            return ok_status();
        }));
        if (count != numel) {
            return error(std::format("{} values given for {} ({} elements)", count, type.to_string(), numel));
        }
        return tensor;
    }

    static Status store_element(DType dtype, std::string_view text, void* out) {
        if (dtype == DType::kBool) {
            if (text == "true" || text == "1") {
                *static_cast<std::uint8_t*>(out) = 1;
            } else if (text == "false" || text == "0") {
                *static_cast<std::uint8_t*>(out) = 0;
            } else {
                return errors::invalid_argument("invalid bool '{}'", text);
            }
            return ok_status();
        }
        if (is_float(dtype)) {
            const std::optional<double> value = parse_double(text);
            if (!value) {
                return errors::invalid_argument("invalid number '{}'", text);
            }
            switch (dtype) {
            case DType::kF32:
                *static_cast<float*>(out) = static_cast<float>(*value);
                break;
            case DType::kF64:
                *static_cast<double*>(out) = *value;
                break;
            case DType::kF16:
                *static_cast<f16*>(out) = f16(static_cast<float>(*value));
                break;
            default:
                *static_cast<bf16*>(out) = bf16(static_cast<float>(*value));
                break;
            }
            return ok_status();
        }
        auto store = [&]<class T>(T*) -> Status {
            const std::optional<T> value = parse_integer<T>(text);
            if (!value) {
                return errors::invalid_argument("invalid {} value '{}'", to_string(dtype), text);
            }
            std::memcpy(out, &*value, sizeof(T));
            return ok_status();
        };
        switch (dtype) {
        case DType::kI8:
            return store(static_cast<std::int8_t*>(nullptr));
        case DType::kU8:
            return store(static_cast<std::uint8_t*>(nullptr));
        case DType::kI16:
            return store(static_cast<std::int16_t*>(nullptr));
        case DType::kU16:
            return store(static_cast<std::uint16_t*>(nullptr));
        case DType::kI32:
            return store(static_cast<std::int32_t*>(nullptr));
        case DType::kU32:
            return store(static_cast<std::uint32_t*>(nullptr));
        case DType::kI64:
            return store(static_cast<std::int64_t*>(nullptr));
        case DType::kU64:
            return store(static_cast<std::uint64_t*>(nullptr));
        default:
            return errors::invalid_argument("unsupported literal type {}", to_string(dtype));
        }
    }

    // --- Attribute values ---

    Result<AttrValue> attr_value() {
        skip_space();
        const char c = peek();
        if (c == '"') {
            II_ASSIGN_OR_RETURN(std::string text, string_literal());
            return AttrValue(std::move(text));
        }
        if (c == '[') {
            return attr_list();
        }
        if (is_name_start(c)) {
            const std::size_t start = pos_;
            const std::string_view word = identifier();
            if (word == "true" || word == "false") {
                return AttrValue(word == "true");
            }
            if (word == "inf" || word == "nan") {
                return AttrValue(*parse_double(word));
            }
            if (const std::optional<DType> parsed = parse_dtype(word)) {
                skip_space();
                if (peek() == '[') {
                    pos_ = start;
                    II_ASSIGN_OR_RETURN(const TensorType type, tensor_type());
                    II_ASSIGN_OR_RETURN(Tensor data, literal_data(type));
                    if (!data.defined()) {
                        return error("tensor attributes cannot have elided data");
                    }
                    return AttrValue(std::move(data));
                }
                return AttrValue(*parsed);
            }
            pos_ = start;
            return error(std::format("unexpected '{}' in an attribute value", word));
        }
        // A number: scan until a delimiter.
        const std::size_t start = pos_;
        while (!at_end() && text_[pos_] != ',' && text_[pos_] != '}' && text_[pos_] != ']' &&
               std::isspace(static_cast<unsigned char>(text_[pos_])) == 0) {
            ++pos_;
        }
        const std::string_view number = text_.substr(start, pos_ - start);
        if (const std::optional<std::int64_t> i = parse_integer(number)) {
            return AttrValue(*i);
        }
        if (const std::optional<double> d = parse_double(number)) {
            return AttrValue(*d);
        }
        pos_ = start;
        return error(std::format("invalid attribute value '{}'", number));
    }

    Result<AttrValue> attr_list() {
        std::vector<std::int64_t> ints;
        std::vector<double> floats;
        std::vector<std::string> strings;
        Dims dims;
        bool any_float = false;
        bool any_dim = false;
        bool any_string = false;
        std::size_t count = 0;
        II_TRY(bracket_list('[', ']', [&](std::string_view element) -> Status {
            ++count;
            if (element.front() == '"') {
                Parser sub(element, options_);
                II_ASSIGN_OR_RETURN(std::string text, sub.string_literal());
                strings.push_back(std::move(text));
                any_string = true;
                return ok_status();
            }
            if (const std::optional<std::int64_t> i = parse_integer(element)) {
                ints.push_back(*i);
                floats.push_back(static_cast<double>(*i));
                dims.emplace_back(*i);
                return ok_status();
            }
            if (const std::optional<double> d = parse_double(element)) {
                floats.push_back(*d);
                any_float = true;
                return ok_status();
            }
            if (const std::optional<Dim> dim = Dim::parse(element)) {
                dims.push_back(*dim);
                any_dim = true;
                return ok_status();
            }
            return errors::invalid_argument("invalid list element '{}'", element);
        }));
        if (any_string) {
            if (strings.size() != count) {
                return error("lists cannot mix strings and numbers");
            }
            return AttrValue(std::move(strings));
        }
        if (any_float && any_dim) {
            return error("lists cannot mix floats and dimensions");
        }
        if (any_float) {
            return AttrValue(std::move(floats));
        }
        if (any_dim) {
            return AttrValue(std::move(dims));
        }
        return AttrValue(std::move(ints));
    }

    Result<Attributes> attributes() {
        Attributes attrs;
        if (!accept('{')) {
            return attrs;
        }
        if (accept('}')) {
            return attrs;
        }
        for (;;) {
            const std::string_view name = identifier();
            if (name.empty()) {
                return error("expected an attribute name");
            }
            if (attrs.contains(name)) {
                return error(std::format("attribute '{}' is repeated", name));
            }
            II_TRY(expect('='));
            II_ASSIGN_OR_RETURN(AttrValue value, attr_value());
            attrs.set(name, std::move(value));
            if (accept('}')) {
                return attrs;
            }
            II_TRY(expect(','));
        }
    }

    // --- Names and statements ---

    /// Parses "%name" / "%\"text\"" / "%123"; returns the printable key and whether the name is numeric.
    Result<std::string> value_name(bool* numeric = nullptr) {
        II_TRY(expect('%'));
        if (numeric != nullptr) {
            *numeric = false;
        }
        if (peek() == '"') {
            return string_literal();
        }
        if (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
            const std::size_t start = pos_;
            while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
                ++pos_;
            }
            if (numeric != nullptr) {
                *numeric = true;
            }
            return std::string(text_.substr(start, pos_ - start));
        }
        const std::string_view name = identifier();
        if (name.empty()) {
            return error("expected a value name after '%'");
        }
        return std::string(name);
    }

    Result<std::string> output_name() {
        skip_space();
        if (peek() == '"') {
            return string_literal();
        }
        const std::string_view name = identifier();
        if (name.empty()) {
            return error("expected an output name");
        }
        return std::string(name);
    }

    Status define(const std::string& key, Value* value, bool numeric) {
        if (!values_.emplace(key, value).second) {
            return error(std::format("value %{} is defined twice", key));
        }
        if (!numeric) {
            value->set_name(key);
        }
        return ok_status();
    }

    Result<Value*> use() {
        skip_space();
        if (peek() == '_') {
            const std::size_t start = pos_;
            ++pos_;
            if (!is_name_char(peek())) {
                return static_cast<Value*>(nullptr);
            }
            pos_ = start;
        }
        const std::size_t start = pos_;
        II_ASSIGN_OR_RETURN(const std::string key, value_name());
        const auto it = values_.find(key);
        if (it == values_.end()) {
            pos_ = start;
            return error(std::format("undefined value %{}", key));
        }
        return it->second;
    }

    Status parse_graph() {
        if (!accept_word("graph")) {
            return error("expected 'graph'");
        }
        skip_space();
        if (peek() == '"') {
            II_ASSIGN_OR_RETURN(std::string name, string_literal());
            graph_->set_name(std::move(name));
        } else if (const std::string_view name = identifier(); !name.empty()) {
            graph_->set_name(std::string(name));
        }
        II_TRY(expect('{'));
        while (!accept('}')) {
            if (at_end()) {
                return error("unexpected end of text, expected '}'");
            }
            II_TRY(statement());
        }
        return ok_status();
    }

    Status statement() {
        if (accept_word("input")) {
            bool numeric = false;
            II_ASSIGN_OR_RETURN(const std::string key, value_name(&numeric));
            II_TRY(expect(':'));
            II_ASSIGN_OR_RETURN(TensorType type, tensor_type());
            Value* value = graph_->add_input(numeric ? key : key, std::move(type));
            return define(key, value, false);
        }
        if (accept_word("state")) {
            bool numeric = false;
            II_ASSIGN_OR_RETURN(const std::string key, value_name(&numeric));
            II_TRY(expect(':'));
            II_ASSIGN_OR_RETURN(TensorType type, tensor_type());
            Tensor initial;
            if (accept('=')) {
                II_ASSIGN_OR_RETURN(const TensorType literal_type, tensor_type());
                II_ASSIGN_OR_RETURN(initial, literal_data(literal_type));
                if (!initial.defined()) {
                    return error("state initial values cannot be elided");
                }
            }
            Value* value = graph_->add_state(key, std::move(type), std::move(initial));
            return define(key, value, false);
        }
        if (accept_word("const")) {
            bool numeric = false;
            II_ASSIGN_OR_RETURN(const std::string key, value_name(&numeric));
            II_TRY(expect(':'));
            II_ASSIGN_OR_RETURN(const TensorType type, tensor_type());
            II_TRY(expect('='));
            II_ASSIGN_OR_RETURN(const TensorType literal_type, tensor_type());
            if (!(literal_type == type)) {
                return error(std::format("constant type {} does not match its literal {}", type.to_string(),
                                         literal_type.to_string()));
            }
            II_ASSIGN_OR_RETURN(Tensor data, literal_data(literal_type));
            if (!data.defined()) {
                if (!options_.resolve_constant) {
                    return error(std::format("constant %{} has elided data", key));
                }
                II_ASSIGN_OR_RETURN(data, options_.resolve_constant(key, type));
                if (!data.defined() || !(type_of(data) == type)) {
                    return error(
                        std::format("resolved data for constant %{} does not match {}", key, type.to_string()));
                }
            }
            return define(key, graph_->add_constant(std::move(data)), numeric);
        }
        if (accept_word("update")) {
            II_ASSIGN_OR_RETURN(Value * state, use());
            if (state == nullptr || state->kind() != ValueKind::kState) {
                return error("'update' needs a state");
            }
            II_TRY(expect('='));
            II_ASSIGN_OR_RETURN(Value * value, use());
            if (value == nullptr) {
                return error("'update' needs a value");
            }
            graph_->set_state_update(state, value);
            return ok_status();
        }
        if (accept_word("output")) {
            II_ASSIGN_OR_RETURN(std::string name, output_name());
            II_TRY(expect('='));
            II_ASSIGN_OR_RETURN(Value * value, use());
            if (value == nullptr) {
                return error("'output' needs a value");
            }
            graph_->add_output(std::move(name), value);
            return ok_status();
        }
        return node_statement();
    }

    Status node_statement() {
        struct Declared {
            std::string key;
            bool numeric;
            TensorType type;
        };
        std::vector<Declared> results;
        skip_space();
        if (peek() == '(') {
            II_TRY(expect('('));
            II_TRY(expect(')'));
        } else {
            do {
                bool numeric = false;
                II_ASSIGN_OR_RETURN(std::string key, value_name(&numeric));
                II_TRY(expect(':'));
                II_ASSIGN_OR_RETURN(TensorType type, tensor_type());
                results.push_back(Declared{std::move(key), numeric, std::move(type)});
            } while (accept(','));
        }
        II_TRY(expect('='));
        const std::size_t op_pos = pos_;
        const std::string_view op_name = identifier();
        const OpSchema* op = registry_.find(op_name);
        if (op == nullptr) {
            pos_ = op_pos;
            return error(std::format("unknown operation '{}'", op_name));
        }
        std::vector<Value*> inputs;
        II_TRY(expect('('));
        if (!accept(')')) {
            for (;;) {
                II_ASSIGN_OR_RETURN(Value * input, use());
                inputs.push_back(input);
                if (accept(')')) {
                    break;
                }
                II_TRY(expect(','));
            }
        }
        II_ASSIGN_OR_RETURN(Attributes attrs, attributes());
        if (Status status = canonicalize_attributes(*op, attrs); !status.ok()) {
            pos_ = op_pos;
            return error(status.message());
        }
        std::vector<TensorType> types;
        for (const Declared& result : results) {
            types.push_back(result.type);
        }
        Node* node = graph_->create_node(*op, std::move(inputs), std::move(attrs), types);
        for (std::size_t i = 0; i < results.size(); ++i) {
            II_TRY(define(results[i].key, node->output(static_cast<int>(i)), results[i].numeric));
        }
        return ok_status();
    }

    std::string_view text_;
    std::size_t pos_ = 0;
    const ParseOptions& options_;
    const OpRegistry& registry_;
    std::unique_ptr<Graph> graph_;
    std::unordered_map<std::string, Value*> values_;
};

}  // namespace

Result<std::unique_ptr<Graph>> parse_graph(std::string_view text, const ParseOptions& options) {
    return Parser(text, options).parse();
}

}  // namespace ii::ir
