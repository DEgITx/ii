// SPDX-License-Identifier: MIT
#include "tensor/dim_poly.h"

#include "core/assert.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <format>
#include <functional>
#include <limits>
#include <numeric>

namespace ii {

namespace detail {

namespace {

std::strong_ordering compare_factors(const std::vector<DimFactor>& a, const std::vector<DimFactor>& b) noexcept {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (const auto c = compare(*a[i].atom, *b[i].atom); c != 0) {
            return c;
        }
        if (const auto c = a[i].power <=> b[i].power; c != 0) {
            return c;
        }
    }
    // Longer monomials (higher degree) first, so that the constant term (no factors) is always last.
    return b.size() <=> a.size();
}

}  // namespace

std::strong_ordering compare(const DimAtom& a, const DimAtom& b) noexcept {
    if (&a == &b) {
        return std::strong_ordering::equal;
    }
    if (const auto c = a.kind <=> b.kind; c != 0) {
        return c;
    }
    if (a.kind == DimAtom::Kind::kSymbol) {
        const int c = a.name.compare(b.name);
        return c < 0 ? std::strong_ordering::less
                     : (c > 0 ? std::strong_ordering::greater : std::strong_ordering::equal);
    }
    if (const auto c = a.divisor <=> b.divisor; c != 0) {
        return c;
    }
    return compare(*a.numerator, *b.numerator);
}

std::strong_ordering compare(const DimPoly& a, const DimPoly& b) noexcept {
    const std::size_t n = std::min(a.terms.size(), b.terms.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (const auto c = compare_factors(a.terms[i].factors, b.terms[i].factors); c != 0) {
            return c;
        }
        if (const auto c = a.terms[i].coefficient <=> b.terms[i].coefficient; c != 0) {
            return c;
        }
    }
    return a.terms.size() <=> b.terms.size();
}

}  // namespace detail

namespace {

using detail::DimAtom;
using detail::DimAtomPtr;

// Coefficient arithmetic wraps around instead of overflowing (signed overflow is undefined behavior). Dimensions
// that large are invalid anyway and are rejected where shapes are validated.
std::int64_t wrapping_add(std::int64_t a, std::int64_t b) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
}
std::int64_t wrapping_sub(std::int64_t a, std::int64_t b) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b));
}
std::int64_t wrapping_mul(std::int64_t a, std::int64_t b) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b));
}
using detail::DimFactor;
using detail::DimPoly;
using detail::DimTerm;

// Polynomial arithmetic works on plain DimPoly values; constants are polynomials with a single factor-less term.

DimPoly constant_poly(std::int64_t value) {
    DimPoly poly;
    if (value != 0) {
        poly.terms.push_back(DimTerm{value, {}});
    }
    return poly;
}

DimPoly to_poly(const Dim& dim) {
    if (dim.is_constant()) {
        return constant_poly(dim.value());
    }
    return *dim.poly();
}

DimPoly atom_poly(DimAtomPtr atom) {
    DimPoly poly;
    poly.terms.push_back(DimTerm{1, {DimFactor{std::move(atom), 1}}});
    return poly;
}

/// Sorts terms, merges equal monomials and drops zero terms.
void normalize(DimPoly& poly) {
    std::stable_sort(poly.terms.begin(), poly.terms.end(), [](const DimTerm& a, const DimTerm& b) {
        return detail::compare_factors(a.factors, b.factors) < 0;
    });
    std::vector<DimTerm> merged;
    merged.reserve(poly.terms.size());
    for (DimTerm& term : poly.terms) {
        if (!merged.empty()) {
            DimTerm& last = merged.back();
            if (detail::compare_factors(last.factors, term.factors) == 0) {
                last.coefficient = wrapping_add(last.coefficient, term.coefficient);
                continue;
            }
        }
        merged.push_back(std::move(term));
    }
    std::erase_if(merged, [](const DimTerm& term) { return term.coefficient == 0; });
    poly.terms = std::move(merged);
}

DimPoly add(const DimPoly& a, const DimPoly& b, std::int64_t b_scale = 1) {
    DimPoly result = a;
    for (const DimTerm& term : b.terms) {
        result.terms.push_back(DimTerm{wrapping_mul(term.coefficient, b_scale), term.factors});
    }
    normalize(result);
    return result;
}

std::vector<DimFactor> multiply_monomials(const std::vector<DimFactor>& a, const std::vector<DimFactor>& b) {
    std::vector<DimFactor> result;
    result.reserve(a.size() + b.size());
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() || j < b.size()) {
        if (j == b.size()) {
            result.push_back(a[i++]);
        } else if (i == a.size()) {
            result.push_back(b[j++]);
        } else {
            const auto c = compare(*a[i].atom, *b[j].atom);
            if (c < 0) {
                result.push_back(a[i++]);
            } else if (c > 0) {
                result.push_back(b[j++]);
            } else {
                result.push_back(DimFactor{a[i].atom, a[i].power + b[j].power});
                ++i;
                ++j;
            }
        }
    }
    return result;
}

DimPoly multiply(const DimPoly& a, const DimPoly& b) {
    DimPoly result;
    result.terms.reserve(a.terms.size() * b.terms.size());
    for (const DimTerm& x : a.terms) {
        for (const DimTerm& y : b.terms) {
            result.terms.push_back(
                DimTerm{wrapping_mul(x.coefficient, y.coefficient), multiply_monomials(x.factors, y.factors)});
        }
    }
    normalize(result);
    return result;
}

std::int64_t floor_div_int(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}

std::int64_t floor_mod_int(std::int64_t a, std::int64_t b) {
    return a - floor_div_int(a, b) * b;
}

DimPoly floor_div_poly(const DimPoly& numerator, std::int64_t divisor) {
    II_ASSERT(divisor > 0);
    if (divisor == 1) {
        return numerator;
    }
    // floor((q*d + r) / d) = q + floor(r / d) for integer q: split off the exactly divisible parts of every term.
    DimPoly quotient;
    DimPoly remainder;
    for (const DimTerm& term : numerator.terms) {
        const std::int64_t q = floor_div_int(term.coefficient, divisor);
        const std::int64_t r = floor_mod_int(term.coefficient, divisor);
        if (q != 0) {
            quotient.terms.push_back(DimTerm{q, term.factors});
        }
        if (r != 0) {
            remainder.terms.push_back(DimTerm{r, term.factors});
        }
    }
    if (!remainder.terms.empty()) {
        const bool constant_only = remainder.terms.size() == 1 && remainder.terms[0].factors.empty();
        if (constant_only) {
            // 0 <= r < divisor, so floor(r / divisor) == 0.
        } else {
            // Keep a constant remainder inside the atom so that, e.g., floor((N + 1) / 2) stays one atom.
            auto atom = std::make_shared<DimAtom>();
            atom->kind = DimAtom::Kind::kFloorDiv;
            atom->numerator = std::make_shared<const DimPoly>(std::move(remainder));
            atom->divisor = divisor;
            DimPoly atom_term = atom_poly(std::move(atom));
            quotient.terms.insert(quotient.terms.end(), atom_term.terms.begin(), atom_term.terms.end());
        }
    }
    normalize(quotient);
    return quotient;
}

Dim make_dim(DimPoly poly) {
    if (poly.terms.empty()) {
        return Dim(0);
    }
    if (poly.terms.size() == 1 && poly.terms[0].factors.empty()) {
        return Dim(poly.terms[0].coefficient);
    }
    return Dim::from_poly(std::make_shared<const DimPoly>(std::move(poly)));
}

bool is_identifier(std::string_view name) {
    if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) != 0 || name[0] == '_')) {
        return false;
    }
    return std::all_of(name.begin(), name.end(),
                       [](char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_'; });
}

// --- Evaluation ---

std::optional<std::int64_t> evaluate_poly(const DimPoly& poly,
                                          FunctionRef<std::optional<std::int64_t>(std::string_view)> lookup);

std::optional<std::int64_t> evaluate_atom(const DimAtom& atom,
                                          FunctionRef<std::optional<std::int64_t>(std::string_view)> lookup) {
    if (atom.kind == DimAtom::Kind::kSymbol) {
        return lookup(atom.name);
    }
    const std::optional<std::int64_t> numerator = evaluate_poly(*atom.numerator, lookup);
    if (!numerator) {
        return std::nullopt;
    }
    return floor_div_int(*numerator, atom.divisor);
}

std::optional<std::int64_t> evaluate_poly(const DimPoly& poly,
                                          FunctionRef<std::optional<std::int64_t>(std::string_view)> lookup) {
    std::int64_t sum = 0;
    for (const DimTerm& term : poly.terms) {
        std::int64_t product = term.coefficient;
        for (const DimFactor& factor : term.factors) {
            const std::optional<std::int64_t> value = evaluate_atom(*factor.atom, lookup);
            if (!value) {
                return std::nullopt;
            }
            for (int p = 0; p < factor.power; ++p) {
                product = wrapping_mul(product, *value);
            }
        }
        sum = wrapping_add(sum, product);
    }
    return sum;
}

// --- Substitution ---

DimPoly substitute_poly(const DimPoly& poly, FunctionRef<std::optional<Dim>(std::string_view)> lookup);

DimPoly substitute_atom(const DimAtomPtr& atom, FunctionRef<std::optional<Dim>(std::string_view)> lookup) {
    if (atom->kind == DimAtom::Kind::kSymbol) {
        if (std::optional<Dim> replacement = lookup(atom->name)) {
            return to_poly(*replacement);
        }
        return atom_poly(atom);
    }
    return floor_div_poly(substitute_poly(*atom->numerator, lookup), atom->divisor);
}

DimPoly substitute_poly(const DimPoly& poly, FunctionRef<std::optional<Dim>(std::string_view)> lookup) {
    DimPoly result;
    for (const DimTerm& term : poly.terms) {
        DimPoly product = constant_poly(term.coefficient);
        for (const DimFactor& factor : term.factors) {
            const DimPoly value = substitute_atom(factor.atom, lookup);
            for (int p = 0; p < factor.power; ++p) {
                product = multiply(product, value);
            }
        }
        result = add(result, product);
    }
    return result;
}

// --- Printing ---

std::string poly_to_string(const DimPoly& poly);

std::string atom_to_string(const DimAtom& atom) {
    if (atom.kind == DimAtom::Kind::kSymbol) {
        return atom.name;
    }
    return std::format("({}) / {}", poly_to_string(*atom.numerator), atom.divisor);
}

std::string poly_to_string(const DimPoly& poly) {
    if (poly.terms.empty()) {
        return "0";
    }
    std::string result;
    bool first = true;
    for (const DimTerm& term : poly.terms) {
        std::int64_t coefficient = term.coefficient;
        if (first) {
            if (coefficient < 0) {
                result += '-';
                coefficient = -coefficient;
            }
        } else {
            result += coefficient < 0 ? " - " : " + ";
            coefficient = coefficient < 0 ? -coefficient : coefficient;
        }
        first = false;
        std::string monomial;
        for (const DimFactor& factor : term.factors) {
            if (!monomial.empty()) {
                monomial += '*';
            }
            std::string atom = atom_to_string(*factor.atom);
            if (factor.atom->kind == DimAtom::Kind::kFloorDiv) {
                atom = "(" + atom + ")";
            }
            for (int p = 0; p < factor.power; ++p) {
                if (p > 0) {
                    monomial += '*';
                }
                monomial += atom;
            }
        }
        if (monomial.empty()) {
            result += std::to_string(coefficient);
        } else if (coefficient == 1) {
            result += monomial;
        } else {
            result += std::to_string(coefficient) + "*" + monomial;
        }
    }
    return result;
}

// --- Hashing ---

std::size_t hash_combine(std::size_t seed, std::size_t value) noexcept {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

std::size_t hash_poly(const DimPoly& poly) noexcept;

std::size_t hash_atom(const DimAtom& atom) noexcept {
    if (atom.kind == DimAtom::Kind::kSymbol) {
        return std::hash<std::string>{}(atom.name);
    }
    return hash_combine(hash_poly(*atom.numerator), static_cast<std::size_t>(atom.divisor));
}

std::size_t hash_poly(const DimPoly& poly) noexcept {
    std::size_t seed = 0x51ed27;
    for (const DimTerm& term : poly.terms) {
        seed = hash_combine(seed, static_cast<std::size_t>(term.coefficient));
        for (const DimFactor& factor : term.factors) {
            seed = hash_combine(seed, hash_atom(*factor.atom));
            seed = hash_combine(seed, static_cast<std::size_t>(factor.power));
        }
    }
    return seed;
}

void collect_symbols(const DimPoly& poly, std::vector<std::string>& out) {
    for (const DimTerm& term : poly.terms) {
        for (const DimFactor& factor : term.factors) {
            if (factor.atom->kind == DimAtom::Kind::kSymbol) {
                out.push_back(factor.atom->name);
            } else {
                collect_symbols(*factor.atom->numerator, out);
            }
        }
    }
}

// --- Parsing: expr := term (('+' | '-') term)*; term := unary (('*' | '/') unary)*; unary := '-' unary | primary;
//     primary := integer | identifier | '(' expr ')'. Division requires a positive constant divisor.

class DimParser {
public:
    explicit DimParser(std::string_view text) : text_(text) {}

    std::optional<Dim> parse() {
        std::optional<Dim> result = expression();
        if (too_large_) {
            return std::nullopt;
        }
        skip_spaces();
        if (!result || pos_ != text_.size()) {
            return std::nullopt;
        }
        return result;
    }

private:
    static constexpr int kMaxDepth = 64;
    static constexpr std::size_t kMaxTerms = 256;

    /// Rejects expressions whose canonical form would be unreasonably large, e.g. (a+b)*(c+d)*(e+f)*...
    std::optional<Dim> checked(Dim dim) {
        if (dim.poly() != nullptr && dim.poly()->terms.size() > kMaxTerms) {
            too_large_ = true;
            return std::nullopt;
        }
        return dim;
    }

    void skip_spaces() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
            ++pos_;
        }
    }

    bool accept(char c) {
        skip_spaces();
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    std::optional<Dim> expression() {
        std::optional<Dim> left = term();
        while (left) {
            if (accept('+')) {
                std::optional<Dim> right = term();
                if (!right) {
                    return std::nullopt;
                }
                left = checked(*left + *right);
            } else if (accept('-')) {
                std::optional<Dim> right = term();
                if (!right) {
                    return std::nullopt;
                }
                left = checked(*left - *right);
            } else {
                break;
            }
        }
        return left;
    }

    std::optional<Dim> term() {
        std::optional<Dim> left = unary();
        while (left) {
            if (accept('*')) {
                std::optional<Dim> right = unary();
                if (!right) {
                    return std::nullopt;
                }
                left = checked(*left * *right);
            } else if (accept('/')) {
                std::optional<Dim> right = unary();
                if (!right || !right->is_constant() || right->value() <= 0) {
                    return std::nullopt;
                }
                left = floor_div(*left, right->value());
            } else {
                break;
            }
        }
        return left;
    }

    std::optional<Dim> unary() {
        if (accept('-')) {
            if (++depth_ > kMaxDepth) {
                too_large_ = true;
                return std::nullopt;
            }
            std::optional<Dim> operand = unary();
            --depth_;
            return operand ? checked(-*operand) : std::nullopt;
        }
        return primary();
    }

    std::optional<Dim> primary() {
        skip_spaces();
        if (pos_ >= text_.size()) {
            return std::nullopt;
        }
        if (accept('(')) {
            if (++depth_ > kMaxDepth) {
                too_large_ = true;
                return std::nullopt;
            }
            std::optional<Dim> inner = expression();
            --depth_;
            if (!inner || !accept(')')) {
                return std::nullopt;
            }
            return inner;
        }
        const char c = text_[pos_];
        if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
            std::int64_t value = 0;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
                const int digit = text_[pos_] - '0';
                if (value > (std::numeric_limits<std::int64_t>::max() - digit) / 10) {
                    return std::nullopt;
                }
                value = value * 10 + digit;
                ++pos_;
            }
            return Dim(value);
        }
        if (std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_') {
            const std::size_t start = pos_;
            while (pos_ < text_.size() &&
                   (std::isalnum(static_cast<unsigned char>(text_[pos_])) != 0 || text_[pos_] == '_')) {
                ++pos_;
            }
            return Dim::symbol(text_.substr(start, pos_ - start));
        }
        return std::nullopt;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
    int depth_ = 0;
    bool too_large_ = false;
};

}  // namespace

Dim::Dim(const char* symbol) : Dim(Dim::symbol(std::string_view(symbol))) {}
Dim::Dim(std::string_view symbol) : Dim(Dim::symbol(symbol)) {}
Dim::Dim(const std::string& symbol) : Dim(Dim::symbol(std::string_view(symbol))) {}

Dim Dim::symbol(std::string_view name) {
    II_CHECK(is_identifier(name), "invalid dimension symbol '{}'", name);
    auto atom = std::make_shared<DimAtom>();
    atom->kind = DimAtom::Kind::kSymbol;
    atom->name = std::string(name);
    return make_dim(atom_poly(std::move(atom)));
}

std::optional<Dim> Dim::parse(std::string_view text) {
    return DimParser(text).parse();
}

Dim Dim::from_poly(std::shared_ptr<const detail::DimPoly> poly) {
    Dim dim;
    if (poly->terms.empty()) {
        return dim;
    }
    if (poly->terms.size() == 1 && poly->terms[0].factors.empty()) {
        dim.value_ = poly->terms[0].coefficient;
        return dim;
    }
    dim.poly_ = std::move(poly);
    return dim;
}

std::string_view Dim::symbol_name() const noexcept {
    if (poly_ && poly_->terms.size() == 1) {
        const DimTerm& term = poly_->terms[0];
        if (term.coefficient == 1 && term.factors.size() == 1 && term.factors[0].power == 1 &&
            term.factors[0].atom->kind == DimAtom::Kind::kSymbol) {
            return term.factors[0].atom->name;
        }
    }
    return {};
}

std::vector<std::string> Dim::symbols() const {
    std::vector<std::string> result;
    if (poly_) {
        collect_symbols(*poly_, result);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::optional<std::int64_t>
Dim::evaluate(FunctionRef<std::optional<std::int64_t>(std::string_view symbol)> lookup) const {
    if (!poly_) {
        return value_;
    }
    return evaluate_poly(*poly_, lookup);
}

Dim Dim::substitute(FunctionRef<std::optional<Dim>(std::string_view symbol)> lookup) const {
    if (!poly_) {
        return *this;
    }
    return make_dim(substitute_poly(*poly_, lookup));
}

std::string Dim::to_string() const {
    if (!poly_) {
        return std::to_string(value_);
    }
    return poly_to_string(*poly_);
}

std::size_t Dim::hash() const noexcept {
    if (!poly_) {
        return std::hash<std::int64_t>{}(value_);
    }
    return hash_poly(*poly_);
}

Dim operator+(const Dim& a, const Dim& b) {
    if (a.is_constant() && b.is_constant()) {
        return Dim(wrapping_add(a.value(), b.value()));
    }
    return make_dim(add(to_poly(a), to_poly(b)));
}

Dim operator-(const Dim& a, const Dim& b) {
    if (a.is_constant() && b.is_constant()) {
        return Dim(wrapping_sub(a.value(), b.value()));
    }
    return make_dim(add(to_poly(a), to_poly(b), -1));
}

Dim operator-(const Dim& a) {
    return Dim(0) - a;
}

Dim operator*(const Dim& a, const Dim& b) {
    if (a.is_constant() && b.is_constant()) {
        return Dim(wrapping_mul(a.value(), b.value()));
    }
    return make_dim(multiply(to_poly(a), to_poly(b)));
}

Dim floor_div(const Dim& a, std::int64_t divisor) {
    if (divisor <= 0) {
        return Dim(0);
    }
    if (a.is_constant()) {
        return Dim(floor_div_int(a.value(), divisor));
    }
    return make_dim(floor_div_poly(to_poly(a), divisor));
}

Dim ceil_div(const Dim& a, std::int64_t divisor) {
    if (divisor <= 0) {
        return Dim(0);
    }
    return floor_div(a + Dim(divisor - 1), divisor);
}

bool operator==(const Dim& a, const Dim& b) noexcept {
    if (a.is_constant() != b.is_constant()) {
        return false;
    }
    if (a.is_constant()) {
        return a.value() == b.value();
    }
    return a.poly_ == b.poly_ || compare(*a.poly_, *b.poly_) == 0;
}

std::optional<Dim> Dim::exact_div(std::int64_t divisor) const {
    if (divisor == 0) {
        return std::nullopt;
    }
    if (!poly_) {
        if (value_ % divisor != 0) {
            return std::nullopt;
        }
        return Dim(value_ / divisor);
    }
    DimPoly result = *poly_;
    for (DimTerm& term : result.terms) {
        if (term.coefficient % divisor != 0) {
            return std::nullopt;
        }
        term.coefficient /= divisor;
    }
    return make_dim(std::move(result));
}

std::optional<Dim> Dim::exact_div(const Dim& divisor) const {
    if (divisor.is_constant()) {
        return exact_div(divisor.value());
    }
    // The divisor must be a monomial c * x1^p1 * ...; every term of the dividend must contain it.
    const DimPoly& d = *divisor.poly_;
    if (d.terms.size() != 1 || !poly_) {
        return std::nullopt;
    }
    const DimTerm& dt = d.terms[0];
    DimPoly result;
    for (const DimTerm& term : poly_->terms) {
        if (term.coefficient % dt.coefficient != 0) {
            return std::nullopt;
        }
        DimTerm quotient{term.coefficient / dt.coefficient, {}};
        std::size_t j = 0;
        for (const DimFactor& factor : term.factors) {
            if (j < dt.factors.size() && compare(*factor.atom, *dt.factors[j].atom) == 0) {
                const int power = factor.power - dt.factors[j].power;
                if (power < 0) {
                    return std::nullopt;
                }
                if (power > 0) {
                    quotient.factors.push_back(DimFactor{factor.atom, power});
                }
                ++j;
            } else {
                quotient.factors.push_back(factor);
            }
        }
        if (j != dt.factors.size()) {
            return std::nullopt;
        }
        result.terms.push_back(std::move(quotient));
    }
    normalize(result);
    return make_dim(std::move(result));
}

std::string to_string(const Dims& dims) {
    std::string result = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += dims[i].to_string();
    }
    result += ']';
    return result;
}

}  // namespace ii
