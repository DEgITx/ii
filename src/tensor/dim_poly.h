// SPDX-License-Identifier: MIT
/// @file
/// The canonical polynomial representation behind symbolic `Dim`s.
///
/// A dimension expression is a sum of terms `coefficient * atom1^p1 * atom2^p2 * ...`. An atom is either a symbol
/// (`N`) or a floor division of a polynomial by a positive constant (`floor((H - 1) / 2)`) whose numerator could not
/// be divided exactly. Terms are sorted and merged, atoms inside a term are sorted, zero terms are dropped — so two
/// structurally equal expressions have identical representations.
#pragma once

#include <ii/dim.h>

#include <compare>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ii::detail {

struct DimAtom;
using DimAtomPtr = std::shared_ptr<const DimAtom>;

/// One factor of a monomial: an atom raised to a positive power.
struct DimFactor {
    DimAtomPtr atom;
    int power = 1;
};

/// A monomial with its coefficient. A term without factors is the constant term.
struct DimTerm {
    std::int64_t coefficient = 0;
    std::vector<DimFactor> factors;  // sorted by atom, unique atoms
};

/// A canonical polynomial: terms sorted by their monomial, no zero coefficients, no duplicate monomials.
/// The constant term (if non-zero) is last.
struct DimPoly {
    std::vector<DimTerm> terms;
};

/// A variable of a polynomial.
struct DimAtom {
    enum class Kind : std::uint8_t { kSymbol, kFloorDiv };
    Kind kind = Kind::kSymbol;
    std::string name;                          // kSymbol
    std::shared_ptr<const DimPoly> numerator;  // kFloorDiv
    std::int64_t divisor = 1;                  // kFloorDiv, > 1
};

std::strong_ordering compare(const DimAtom& a, const DimAtom& b) noexcept;
std::strong_ordering compare(const DimPoly& a, const DimPoly& b) noexcept;

}  // namespace ii::detail
