// SPDX-License-Identifier: MIT
/// @file
/// Operation attributes: typed, named constant parameters of a model operation (`axis = 1`, `epsilon = 1e-5`,
/// `perm = [0, 2, 1]`, ...).
#pragma once

#include <ii/dim.h>
#include <ii/dtype.h>
#include <ii/export.h>
#include <ii/tensor.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

II_BEGIN_DLL_EXPORTS

namespace ii {

/// The value of an attribute.
///
/// Integer lists and dimension lists are distinct: `Dims` may contain symbolic dimensions (e.g. the target shape of
/// a reshape); accessors convert an integer list to `Dims` where a `Dims` is expected.
using AttrValue = std::variant<bool, std::int64_t, double, std::string, std::vector<std::int64_t>, std::vector<double>,
                               std::vector<std::string>, DType, Dims, Tensor>;

/// Kind of an attribute value; the order matches the alternatives of `AttrValue`.
enum class AttrKind : std::uint8_t {
    kBool,
    kInt,
    kFloat,
    kString,
    kInts,
    kFloats,
    kStrings,
    kDType,
    kDims,
    kTensor,
};

/// The kind of a value.
inline AttrKind kind_of(const AttrValue& value) noexcept {
    return static_cast<AttrKind>(value.index());
}

/// Name of an attribute kind, e.g. "int", "ints".
II_API std::string_view to_string(AttrKind kind) noexcept;

/// Formats a value in the IR text syntax, e.g. `true`, `3`, `1e-05`, `"tanh"`, `[0, 2, 1]`, `f16`, `[N, 3]`.
II_API std::string to_string(const AttrValue& value);

/// An ordered set of named attributes. Names are unique; iteration is in name order.
class II_API Attributes {
public:
    using Entry = std::pair<std::string, AttrValue>;

    Attributes() = default;
    Attributes(std::initializer_list<Entry> entries);

    /// Sets (adds or replaces) an attribute.
    void set(std::string_view name, AttrValue value);

    /// Removes an attribute; does nothing if absent.
    void erase(std::string_view name);

    /// The attribute value or nullptr.
    const AttrValue* find(std::string_view name) const noexcept;

    bool contains(std::string_view name) const noexcept { return find(name) != nullptr; }
    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }
    auto begin() const noexcept { return entries_.begin(); }
    auto end() const noexcept { return entries_.end(); }

    /// The value as `T` if present with exactly that kind, otherwise nullptr.
    template <class T>
    const T* get_if(std::string_view name) const noexcept {
        const AttrValue* value = find(name);
        return value != nullptr ? std::get_if<T>(value) : nullptr;
    }

    // Typed accessors with fallbacks. Integer accessors accept bools; float accessors accept integers; dims
    // accessors accept integer lists.
    bool get_bool(std::string_view name, bool fallback = false) const noexcept;
    std::int64_t get_int(std::string_view name, std::int64_t fallback = 0) const noexcept;
    double get_float(std::string_view name, double fallback = 0.0) const noexcept;
    std::string_view get_string(std::string_view name, std::string_view fallback = {}) const noexcept;
    std::vector<std::int64_t> get_ints(std::string_view name) const;
    std::vector<double> get_floats(std::string_view name) const;
    DType get_dtype(std::string_view name, DType fallback = DType::kInvalid) const noexcept;
    Dims get_dims(std::string_view name) const;
    Tensor get_tensor(std::string_view name) const;

    /// "{a = 1, b = [2, 3]}" or "" when empty.
    std::string to_string() const;

    friend bool operator==(const Attributes& a, const Attributes& b) noexcept;

private:
    std::vector<Entry> entries_;  // sorted by name
};

/// Equality of attribute values. Tensors compare by dtype, shape and contents.
II_API bool attr_equal(const AttrValue& a, const AttrValue& b) noexcept;

}  // namespace ii

II_END_DLL_EXPORTS
