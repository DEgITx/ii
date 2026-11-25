// SPDX-License-Identifier: MIT
/// @file
/// Symbolic tensor dimensions.
///
/// A `Dim` is either a constant or an integer expression over named symbols, e.g. `N`, `seq + past`, `N * seq`,
/// `(H - 3) / 2 + 1`. Models with dynamic shapes are described and compiled once with symbolic dimensions; concrete
/// sizes are computed from the symbol values when the model runs.
///
/// Expressions are kept in a canonical form — a polynomial with integer coefficients whose variables are symbols and
/// floor divisions of polynomials by positive constants — so structurally equal expressions compare equal:
/// `N * 2 + N == 3 * N`, `(4 * N + 2) / 2 == 2 * N + 1`.
#pragma once

#include <ii/export.h>
#include <ii/function_ref.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ii {

namespace detail {
struct DimPoly;
}  // namespace detail

/// A tensor dimension: a constant or a symbolic integer expression. Cheap to copy (constants do not allocate;
/// symbolic expressions are shared and immutable).
class Dim {
public:
    /// The constant 0.
    constexpr Dim() noexcept = default;

    /// A constant dimension.
    constexpr Dim(std::int64_t value) noexcept : value_(value) {}  // NOLINT(google-explicit-constructor)
    constexpr Dim(int value) noexcept : value_(value) {}           // NOLINT(google-explicit-constructor)

    /// A dimension consisting of one symbol, e.g. `Dim("N")`. The name must be a valid identifier
    /// (`[A-Za-z_][A-Za-z0-9_]*`); invalid names terminate the program (use `parse()` for untrusted input).
    II_API Dim(const char* symbol);         // NOLINT(google-explicit-constructor)
    II_API Dim(std::string_view symbol);    // NOLINT(google-explicit-constructor)
    II_API Dim(const std::string& symbol);  // NOLINT(google-explicit-constructor)

    /// A dimension consisting of one symbol.
    II_API static Dim symbol(std::string_view name);

    /// Parses an expression such as "N", "seq + past", "2 * N", "(H - 1) / 2 + 1". Returns nullopt on syntax errors.
    II_API static std::optional<Dim> parse(std::string_view text);

    /// True for a constant dimension.
    bool is_constant() const noexcept { return poly_ == nullptr; }

    /// The constant value, or nullopt for a symbolic dimension.
    std::optional<std::int64_t> constant() const noexcept {
        return is_constant() ? std::optional<std::int64_t>(value_) : std::nullopt;
    }

    /// The constant value. Must only be called on constant dimensions (returns 0 otherwise).
    std::int64_t value() const noexcept { return is_constant() ? value_ : 0; }

    /// The symbol name if the dimension is exactly one symbol (`N`), otherwise empty.
    II_API std::string_view symbol_name() const noexcept;

    /// Names of all symbols the expression depends on, sorted and unique.
    II_API std::vector<std::string> symbols() const;

    /// Evaluates the expression. `lookup` returns the value of a symbol or nullopt if it is unknown; the result is
    /// nullopt when a symbol is unknown or a division by a non-positive divisor would be required.
    II_API std::optional<std::int64_t>
    evaluate(FunctionRef<std::optional<std::int64_t>(std::string_view symbol)> lookup) const;

    /// Replaces symbols by expressions: `lookup` returns the replacement or nullopt to keep a symbol.
    II_API Dim substitute(FunctionRef<std::optional<Dim>(std::string_view symbol)> lookup) const;

    /// Human-readable canonical form, e.g. "2*N + 1". Parsable by `parse()`.
    II_API std::string to_string() const;

    /// Structural hash, consistent with `operator==`.
    II_API std::size_t hash() const noexcept;

    II_API friend Dim operator+(const Dim& a, const Dim& b);
    II_API friend Dim operator-(const Dim& a, const Dim& b);
    II_API friend Dim operator*(const Dim& a, const Dim& b);
    II_API friend Dim operator-(const Dim& a);

    /// Floor division by a positive constant (the divisor must be > 0; otherwise the result is the constant 0 and
    /// `valid_divisor()` is false — shape inference checks divisors before dividing).
    II_API friend Dim floor_div(const Dim& a, std::int64_t divisor);
    /// Ceiling division by a positive constant.
    II_API friend Dim ceil_div(const Dim& a, std::int64_t divisor);

    /// Structural equality of the canonical forms. `N == N` is true; `N == M` is false even though the values may
    /// coincide at run time.
    II_API friend bool operator==(const Dim& a, const Dim& b) noexcept;

    /// Exact division by a constant if every coefficient is divisible (`(4*N) / 2 == 2*N`), otherwise nullopt.
    II_API std::optional<Dim> exact_div(std::int64_t divisor) const;

    /// Exact division by another dimension when it is a monomial dividing every term (`(N*S*4) / (S*2) == 2*N`).
    II_API std::optional<Dim> exact_div(const Dim& divisor) const;

    // Implementation detail: constructs a symbolic dimension from its canonical polynomial (or a constant if the
    // polynomial is constant).
    II_API static Dim from_poly(std::shared_ptr<const detail::DimPoly> poly);
    const detail::DimPoly* poly() const noexcept { return poly_.get(); }

private:
    std::int64_t value_ = 0;                       // value of a constant dimension
    std::shared_ptr<const detail::DimPoly> poly_;  // null for constants
};

/// A list of dimensions: the (possibly symbolic) shape of a tensor in a model.
using Dims = std::vector<Dim>;

/// Formats dimensions as "[N, 3, 224, 224]".
II_API std::string to_string(const Dims& dims);

}  // namespace ii
