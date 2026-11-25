// SPDX-License-Identifier: MIT
#include <ii/attribute.h>

#include "ir/tensor_literal.h"
#include "tensor/strided.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>

namespace ii {

namespace {

std::string format_double(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value < 0 ? "-inf" : "inf";
    }
    std::string text = std::format("{}", value);
    // Keep floats distinguishable from integers in the text format.
    if (text.find_first_of(".eE") == std::string::npos) {
        text += ".0";
    }
    return text;
}

std::string quote(std::string_view text) {
    std::string result = "\"";
    for (const char c : text) {
        switch (c) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                result += std::format("\\x{:02x}", static_cast<unsigned char>(c));
            } else {
                result += c;
            }
        }
    }
    result += '"';
    return result;
}

template <class T, class F>
std::string format_list(const std::vector<T>& values, F&& format_one) {
    std::string result = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += format_one(values[i]);
    }
    result += ']';
    return result;
}

bool tensor_equal(const Tensor& a, const Tensor& b) noexcept {
    if (a.defined() != b.defined()) {
        return false;
    }
    if (!a.defined()) {
        return true;
    }
    if (a.dtype() != b.dtype() || !(a.shape() == b.shape())) {
        return false;
    }
    const Result<Tensor> ca = a.contiguous();
    const Result<Tensor> cb = b.contiguous();
    if (!ca.ok() || !cb.ok()) {
        return false;
    }
    return std::memcmp(ca->data(), cb->data(), ca->nbytes()) == 0;
}

}  // namespace

std::string_view to_string(AttrKind kind) noexcept {
    switch (kind) {
    case AttrKind::kBool:
        return "bool";
    case AttrKind::kInt:
        return "int";
    case AttrKind::kFloat:
        return "float";
    case AttrKind::kString:
        return "string";
    case AttrKind::kInts:
        return "ints";
    case AttrKind::kFloats:
        return "floats";
    case AttrKind::kStrings:
        return "strings";
    case AttrKind::kDType:
        return "dtype";
    case AttrKind::kDims:
        return "dims";
    case AttrKind::kTensor:
        return "tensor";
    }
    return "unknown";
}

std::string to_string(const AttrValue& value) {
    return std::visit(
        [](const auto& v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, bool>) {
                return v ? "true" : "false";
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return std::to_string(v);
            } else if constexpr (std::is_same_v<T, double>) {
                return format_double(v);
            } else if constexpr (std::is_same_v<T, std::string>) {
                return quote(v);
            } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                return format_list(v, [](std::int64_t x) { return std::to_string(x); });
            } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                return format_list(v, [](double x) { return format_double(x); });
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                return format_list(v, [](const std::string& x) { return quote(x); });
            } else if constexpr (std::is_same_v<T, DType>) {
                return std::string(ii::to_string(v));
            } else if constexpr (std::is_same_v<T, Dims>) {
                // A dims list prints like an integer list when it is all constant; the parser accepts both.
                return ii::to_string(v);
            } else {
                return ir::format_tensor_literal(v);
            }
        },
        value);
}

bool attr_equal(const AttrValue& a, const AttrValue& b) noexcept {
    if (a.index() != b.index()) {
        return false;
    }
    if (const Tensor* ta = std::get_if<Tensor>(&a)) {
        return tensor_equal(*ta, std::get<Tensor>(b));
    }
    if (const double* da = std::get_if<double>(&a)) {
        const double db = std::get<double>(b);
        return *da == db || (std::isnan(*da) && std::isnan(db));
    }
    return std::visit(
        [&](const auto& va) -> bool {
            using T = std::decay_t<decltype(va)>;
            if constexpr (std::is_same_v<T, Tensor>) {
                return false;  // handled above
            } else {
                return va == std::get<T>(b);
            }
        },
        a);
}

Attributes::Attributes(std::initializer_list<Entry> entries) {
    for (const Entry& entry : entries) {
        set(entry.first, entry.second);
    }
}

void Attributes::set(std::string_view name, AttrValue value) {
    const auto it = std::lower_bound(entries_.begin(), entries_.end(), name,
                                     [](const Entry& entry, std::string_view key) { return entry.first < key; });
    if (it != entries_.end() && it->first == name) {
        it->second = std::move(value);
    } else {
        entries_.insert(it, Entry(std::string(name), std::move(value)));
    }
}

void Attributes::erase(std::string_view name) {
    std::erase_if(entries_, [&](const Entry& entry) { return entry.first == name; });
}

const AttrValue* Attributes::find(std::string_view name) const noexcept {
    const auto it = std::lower_bound(entries_.begin(), entries_.end(), name,
                                     [](const Entry& entry, std::string_view key) { return entry.first < key; });
    return it != entries_.end() && it->first == name ? &it->second : nullptr;
}

bool Attributes::get_bool(std::string_view name, bool fallback) const noexcept {
    const AttrValue* value = find(name);
    if (value == nullptr) {
        return fallback;
    }
    if (const bool* b = std::get_if<bool>(value)) {
        return *b;
    }
    if (const std::int64_t* i = std::get_if<std::int64_t>(value)) {
        return *i != 0;
    }
    return fallback;
}

std::int64_t Attributes::get_int(std::string_view name, std::int64_t fallback) const noexcept {
    const AttrValue* value = find(name);
    if (value == nullptr) {
        return fallback;
    }
    if (const std::int64_t* i = std::get_if<std::int64_t>(value)) {
        return *i;
    }
    if (const bool* b = std::get_if<bool>(value)) {
        return *b ? 1 : 0;
    }
    return fallback;
}

double Attributes::get_float(std::string_view name, double fallback) const noexcept {
    const AttrValue* value = find(name);
    if (value == nullptr) {
        return fallback;
    }
    if (const double* d = std::get_if<double>(value)) {
        return *d;
    }
    if (const std::int64_t* i = std::get_if<std::int64_t>(value)) {
        return static_cast<double>(*i);
    }
    return fallback;
}

std::string_view Attributes::get_string(std::string_view name, std::string_view fallback) const noexcept {
    const std::string* s = get_if<std::string>(name);
    return s != nullptr ? std::string_view(*s) : fallback;
}

std::vector<std::int64_t> Attributes::get_ints(std::string_view name) const {
    const AttrValue* value = find(name);
    if (value == nullptr) {
        return {};
    }
    if (const auto* ints = std::get_if<std::vector<std::int64_t>>(value)) {
        return *ints;
    }
    if (const auto* dims = std::get_if<Dims>(value)) {
        std::vector<std::int64_t> result;
        for (const Dim& dim : *dims) {
            if (!dim.is_constant()) {
                return {};
            }
            result.push_back(dim.value());
        }
        return result;
    }
    if (const std::int64_t* i = std::get_if<std::int64_t>(value)) {
        return {*i};
    }
    return {};
}

std::vector<double> Attributes::get_floats(std::string_view name) const {
    const AttrValue* value = find(name);
    if (value == nullptr) {
        return {};
    }
    if (const auto* floats = std::get_if<std::vector<double>>(value)) {
        return *floats;
    }
    if (const auto* ints = std::get_if<std::vector<std::int64_t>>(value)) {
        std::vector<double> floats;
        floats.reserve(ints->size());
        for (const std::int64_t i : *ints) {
            floats.push_back(static_cast<double>(i));
        }
        return floats;
    }
    return {};
}

DType Attributes::get_dtype(std::string_view name, DType fallback) const noexcept {
    const DType* dtype = get_if<DType>(name);
    return dtype != nullptr ? *dtype : fallback;
}

Dims Attributes::get_dims(std::string_view name) const {
    const AttrValue* value = find(name);
    if (value == nullptr) {
        return {};
    }
    if (const auto* dims = std::get_if<Dims>(value)) {
        return *dims;
    }
    if (const auto* ints = std::get_if<std::vector<std::int64_t>>(value)) {
        return Dims(ints->begin(), ints->end());
    }
    return {};
}

Tensor Attributes::get_tensor(std::string_view name) const {
    const Tensor* tensor = get_if<Tensor>(name);
    return tensor != nullptr ? *tensor : Tensor();
}

std::string Attributes::to_string() const {
    if (entries_.empty()) {
        return {};
    }
    std::string result = "{";
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += entries_[i].first;
        result += " = ";
        result += ii::to_string(entries_[i].second);
    }
    result += '}';
    return result;
}

bool operator==(const Attributes& a, const Attributes& b) noexcept {
    return a.entries_.size() == b.entries_.size() &&
           std::equal(a.entries_.begin(), a.entries_.end(), b.entries_.begin(), [](const auto& x, const auto& y) {
               return x.first == y.first && attr_equal(x.second, y.second);
           });
}

namespace ir {

namespace {

template <class T>
std::string format_value(T value) {
    if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(value)) {
            return "nan";
        }
        if (std::isinf(value)) {
            return value < 0 ? "-inf" : "inf";
        }
        return std::format("{}", value);
    } else {
        return std::format("{}", value);
    }
}

}  // namespace

std::string format_tensor_literal(const Tensor& tensor, std::int64_t max_elements) {
    if (!tensor.defined()) {
        return "<empty>";
    }
    const DType dtype = tensor.dtype();
    std::string result = std::format("{}{}", ii::to_string(dtype), tensor.shape().to_string());
    if (max_elements >= 0 && tensor.numel() > max_elements) {
        result += " {...}";
        return result;
    }
    const Result<Tensor> dense = tensor.contiguous();
    if (!dense.ok()) {
        result += " {?}";
        return result;
    }
    if (is_block_quantized(dtype)) {
        result += " 0x";
        const auto* bytes = static_cast<const std::uint8_t*>(dense->data());
        for (std::size_t i = 0; i < dense->nbytes(); ++i) {
            result += std::format("{:02x}", bytes[i]);
        }
        return result;
    }
    result += " {";
    const std::int64_t count = dense->numel();
    for (std::int64_t i = 0; i < count; ++i) {
        if (i > 0) {
            result += ", ";
        }
        const void* p =
            static_cast<const std::byte*>(dense->data()) + i * static_cast<std::int64_t>(element_size(dtype));
        switch (dtype) {
        case DType::kF32:
            result += format_value(*static_cast<const float*>(p));
            break;
        case DType::kF16:
            result += format_value(static_cast<const f16*>(p)->to_float());
            break;
        case DType::kBF16:
            result += format_value(static_cast<const bf16*>(p)->to_float());
            break;
        case DType::kF64:
            result += format_value(*static_cast<const double*>(p));
            break;
        case DType::kI8:
            result += format_value(static_cast<int>(*static_cast<const std::int8_t*>(p)));
            break;
        case DType::kU8:
            result += format_value(static_cast<unsigned>(*static_cast<const std::uint8_t*>(p)));
            break;
        case DType::kI16:
            result += format_value(*static_cast<const std::int16_t*>(p));
            break;
        case DType::kU16:
            result += format_value(*static_cast<const std::uint16_t*>(p));
            break;
        case DType::kI32:
            result += format_value(*static_cast<const std::int32_t*>(p));
            break;
        case DType::kU32:
            result += format_value(*static_cast<const std::uint32_t*>(p));
            break;
        case DType::kI64:
            result += format_value(*static_cast<const std::int64_t*>(p));
            break;
        case DType::kU64:
            result += format_value(*static_cast<const std::uint64_t*>(p));
            break;
        case DType::kBool:
            result += format_value(*static_cast<const std::uint8_t*>(p) != 0);
            break;
        default:
            result += "?";
        }
    }
    result += '}';
    return result;
}

}  // namespace ir

}  // namespace ii
