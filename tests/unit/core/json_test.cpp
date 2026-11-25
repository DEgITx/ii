// SPDX-License-Identifier: MIT
#include "core/json.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

namespace ii {
namespace {

using ::testing::HasSubstr;

TEST(JsonTest, ParsesDocuments) {
    const Json doc =
        Json::parse(R"( {"a": [1, -2.5e3, true, false, null], "b": {"c": "d\n\u00e9\ud83d\ude00"}, "e": []} )").value();
    ASSERT_TRUE(doc.is_object());
    EXPECT_EQ(doc.size(), 3u);
    const Json& a = *doc.find("a");
    ASSERT_EQ(a.size(), 5u);
    EXPECT_EQ(a.as_array()[0].as_int(), 1);
    EXPECT_EQ(a.as_array()[1].as_double(), -2500.0);
    EXPECT_FALSE(a.as_array()[1].as_int().has_value());
    EXPECT_TRUE(a.as_array()[2].as_bool());
    EXPECT_TRUE(a.as_array()[4].is_null());
    EXPECT_EQ(doc.find("b")->find("c")->as_string(), "d\n\xC3\xA9\xF0\x9F\x98\x80");
    EXPECT_TRUE(doc.find("e")->is_array());
    EXPECT_EQ(doc.find("missing"), nullptr);
    EXPECT_EQ(doc.as_object()[0].first, "a");  // document order is preserved
}

TEST(JsonTest, LargeIntegersStayExact) {
    const Json doc = Json::parse("[9007199254740993, -9223372036854775808, 18446744073709551616]").value();
    EXPECT_EQ(doc.as_array()[0].as_int(), 9007199254740993LL);
    EXPECT_EQ(doc.as_array()[1].as_int(), INT64_MIN);
    EXPECT_FALSE(doc.as_array()[2].as_int().has_value());  // out of range: double only
    EXPECT_EQ(doc.as_array()[2].as_double(), 18446744073709551616.0);
}

TEST(JsonTest, RejectsInvalidDocuments) {
    for (const char* text : {"", "{", "[1,]", "{\"a\" 1}", "01", "1.", "-", "\"\\x\"", "\"\\ud800\"", "nul", "[1] 2",
                             "\"a\nb\"", "{1: 2}", "+1", ".5", "1e"}) {
        EXPECT_FALSE(Json::parse(text).ok()) << text;
    }
    EXPECT_THAT(Json::parse("[1, }").status().to_string(), HasSubstr("offset"));
}

TEST(JsonTest, LimitsNesting) {
    // Kept small: the parser is recursive, and debug builds have large stack frames (1 MiB stacks on Windows).
    const std::string deep = std::string(200, '[') + std::string(200, ']');
    EXPECT_THAT(Json::parse(deep).status().to_string(), HasSubstr("too deep"));
    EXPECT_TRUE(Json::parse(deep, 256).ok());
}

TEST(Base64Test, Decodes) {
    EXPECT_EQ(base64_decode("").value(), std::vector<std::uint8_t>{});
    EXPECT_EQ(base64_decode("TWFu").value(), (std::vector<std::uint8_t>{'M', 'a', 'n'}));
    EXPECT_EQ(base64_decode("TWE=").value(), (std::vector<std::uint8_t>{'M', 'a'}));
    EXPECT_EQ(base64_decode("TQ==").value(), (std::vector<std::uint8_t>{'M'}));
    EXPECT_EQ(base64_decode("TQ").value(), (std::vector<std::uint8_t>{'M'}));
    EXPECT_EQ(base64_decode("/+8=").value(), (std::vector<std::uint8_t>{0xFF, 0xEF}));
    EXPECT_FALSE(base64_decode("T").has_value());
    EXPECT_FALSE(base64_decode("TW!u").has_value());
}

}  // namespace
}  // namespace ii
