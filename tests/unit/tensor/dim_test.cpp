// SPDX-License-Identifier: MIT
#include <ii/dim.h>

#include <gtest/gtest.h>

#include <map>
#include <string>

namespace ii {
namespace {

std::optional<std::int64_t> eval(const Dim& dim, const std::map<std::string, std::int64_t, std::less<>>& values) {
    return dim.evaluate([&](std::string_view name) -> std::optional<std::int64_t> {
        const auto it = values.find(name);
        return it == values.end() ? std::nullopt : std::optional<std::int64_t>(it->second);
    });
}

TEST(DimTest, Constants) {
    const Dim a = 3;
    EXPECT_TRUE(a.is_constant());
    EXPECT_EQ(a.value(), 3);
    EXPECT_EQ(a + Dim(4), Dim(7));
    EXPECT_EQ(a * Dim(4), Dim(12));
    EXPECT_EQ(floor_div(Dim(7), 2), Dim(3));
    EXPECT_EQ(floor_div(Dim(-7), 2), Dim(-4));
    EXPECT_EQ(ceil_div(Dim(7), 2), Dim(4));
    EXPECT_EQ(a.to_string(), "3");
}

TEST(DimTest, Symbols) {
    const Dim n = "N";
    EXPECT_FALSE(n.is_constant());
    EXPECT_EQ(n.symbol_name(), "N");
    EXPECT_EQ(n, Dim::symbol("N"));
    EXPECT_FALSE(n == Dim("M"));
    EXPECT_FALSE(n == Dim(1));
    EXPECT_EQ(n.to_string(), "N");
    EXPECT_TRUE((n * 2).symbol_name().empty());
}

TEST(DimTest, CanonicalForm) {
    const Dim n = "N";
    const Dim s = "S";
    EXPECT_EQ(n * 2 + n, n * 3);
    EXPECT_EQ(n + s, s + n);
    EXPECT_EQ(n * s, s * n);
    EXPECT_EQ((n + 1) * (n + 1), n * n + n * 2 + 1);
    EXPECT_EQ(n - n, Dim(0));
    EXPECT_TRUE((n - n).is_constant());
    EXPECT_EQ((n + s) - s, n);
    EXPECT_EQ((n * 2 + 1).to_string(), "2*N + 1");
    EXPECT_EQ((s * n * 4).to_string(), "4*N*S");
    EXPECT_EQ((n * n).to_string(), "N*N");
    EXPECT_EQ((Dim(1) - n).to_string(), "-N + 1");
    EXPECT_EQ((n + s).hash(), (s + n).hash());
}

TEST(DimTest, Division) {
    const Dim n = "N";
    EXPECT_EQ(floor_div(n * 4 + 2, 2), n * 2 + 1);
    EXPECT_EQ(floor_div(n * 4 + 3, 2), n * 2 + 1);  // floor(3 / 2) == 1
    const Dim half = floor_div(n, 2);
    EXPECT_FALSE(half.is_constant());
    EXPECT_EQ(half, floor_div(n, 2));
    EXPECT_FALSE(half == floor_div(n, 3));
    EXPECT_EQ(floor_div(n * 4 + 1, 2), n * 2 + floor_div(Dim(1) + Dim(0) * n, 2));
    // Conv output size: (H + 2*1 - 3) / 2 + 1
    const Dim h = "H";
    const Dim out = floor_div(h + 2 - 3, 2) + 1;
    EXPECT_EQ(eval(out, {{"H", 224}}), 112);
    EXPECT_EQ(eval(out, {{"H", 7}}), 4);
    EXPECT_EQ(eval(floor_div(n - 1, 2), {{"N", 0}}), -1);
    EXPECT_EQ(ceil_div(n, 4), floor_div(n + 3, 4));
}

TEST(DimTest, ExactDivision) {
    const Dim n = "N";
    const Dim s = "S";
    EXPECT_EQ((n * 4 + 8).exact_div(4), n + 2);
    EXPECT_FALSE((n * 4 + 2).exact_div(4).has_value());
    EXPECT_EQ((n * s * 4).exact_div(s * 2), n * 2);
    EXPECT_FALSE((n * 4).exact_div(s).has_value());
    EXPECT_FALSE((n + s).exact_div(n).has_value());
    EXPECT_EQ(Dim(12).exact_div(Dim(4)), Dim(3));
}

TEST(DimTest, Evaluate) {
    const Dim n = "N";
    const Dim s = "seq";
    const Dim e = n * s * 2 + floor_div(s + 1, 2) - 3;
    EXPECT_EQ(eval(e, {{"N", 3}, {"seq", 5}}), 3 * 5 * 2 + 3 - 3);
    EXPECT_EQ(eval(e, {{"N", 3}}), std::nullopt);
    EXPECT_EQ(eval(Dim(5), {}), 5);
}

TEST(DimTest, SymbolsAndSubstitute) {
    const Dim e = Dim("b") * Dim("a") + floor_div(Dim("c"), 2) + Dim("a");
    EXPECT_EQ(e.symbols(), (std::vector<std::string>{"a", "b", "c"}));
    const Dim replaced = e.substitute([](std::string_view name) -> std::optional<Dim> {
        if (name == "a") {
            return Dim(2);
        }
        if (name == "c") {
            return Dim("b") * 2;
        }
        return std::nullopt;
    });
    EXPECT_EQ(replaced, Dim("b") * 3 + 2);
}

TEST(DimTest, Parse) {
    EXPECT_EQ(Dim::parse("N"), Dim("N"));
    EXPECT_EQ(Dim::parse("42"), Dim(42));
    EXPECT_EQ(Dim::parse("2*N + 1"), Dim("N") * 2 + 1);
    EXPECT_EQ(Dim::parse(" ( H - 1 ) / 2 + 1 "), floor_div(Dim("H") - 1, 2) + 1);
    EXPECT_EQ(Dim::parse("-N"), -Dim("N"));
    EXPECT_EQ(Dim::parse("N*N*S"), Dim("N") * Dim("N") * Dim("S"));
    EXPECT_FALSE(Dim::parse("").has_value());
    EXPECT_FALSE(Dim::parse("N +").has_value());
    EXPECT_FALSE(Dim::parse("N / M").has_value());
    EXPECT_FALSE(Dim::parse("N / 0").has_value());
    EXPECT_FALSE(Dim::parse("(N").has_value());
    EXPECT_FALSE(Dim::parse("99999999999999999999").has_value());
    EXPECT_FALSE(Dim::parse("N $").has_value());
}

TEST(DimTest, ParseLimitsUntrustedInput) {
    EXPECT_TRUE(Dim::parse(std::string(32, '(') + "N" + std::string(32, ')')).has_value());
    EXPECT_FALSE(Dim::parse(std::string(100, '(') + "N" + std::string(100, ')')).has_value());
    EXPECT_FALSE(Dim::parse(std::string(100, '-') + "N").has_value());
    // (a0 + b0) * (a1 + b1) * ... has 2^k terms in canonical form.
    std::string blowup = "(a0 + b0)";
    for (int i = 1; i < 12; ++i) {
        blowup += " * (a" + std::to_string(i) + " + b" + std::to_string(i) + ")";
    }
    EXPECT_FALSE(Dim::parse(blowup).has_value());
    // Constant overflow wraps around instead of being undefined behavior.
    EXPECT_TRUE(Dim::parse("9223372036854775807 + 1").has_value());
    EXPECT_TRUE(Dim::parse("4611686018427387904 * 4").has_value());
}

TEST(DimTest, ToStringRoundTrips) {
    const Dim n = "N";
    const Dim h = "H";
    for (const Dim& dim : {Dim(0), Dim(-5), n, n * 2 + 1, Dim(1) - n, n * h * 3 - h, floor_div(h - 1, 2) + 1,
                           floor_div(n * h + 1, 4) * n, ceil_div(n, 8) * 8}) {
        const std::string text = dim.to_string();
        const std::optional<Dim> parsed = Dim::parse(text);
        ASSERT_TRUE(parsed.has_value()) << text;
        EXPECT_EQ(*parsed, dim) << text << " vs " << parsed->to_string();
    }
}

TEST(DimTest, DimsToString) {
    EXPECT_EQ(to_string(Dims{Dim("N"), 3, 224}), "[N, 3, 224]");
    EXPECT_EQ(to_string(Dims{}), "[]");
}

}  // namespace
}  // namespace ii
