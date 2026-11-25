// SPDX-License-Identifier: MIT
// The Jinja interpreter: real chat templates rendered like transformers (tests/data/llm/chat_templates.json, from
// scripts/gen_chat_template_cases.py), and the language features one by one.
#include "core/json.h"
#include "llm/jinja.h"
#include "support/test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

namespace ii::llm::detail::jinja {
namespace {

using ::testing::HasSubstr;

Value to_value(const Json& json) {
    switch (json.kind()) {
    case Json::Kind::kNull:
        return Value(nullptr);
    case Json::Kind::kBool:
        return Value(json.as_bool());
    case Json::Kind::kNumber:
        if (const auto i = json.as_int()) {
            return Value(*i);
        }
        return Value(json.as_double());
    case Json::Kind::kString:
        return Value(json.as_string());
    case Json::Kind::kArray: {
        Array items;
        for (const Json& item : json.as_array()) {
            items.push_back(to_value(item));
        }
        return Value(std::move(items));
    }
    case Json::Kind::kObject: {
        Object object;
        for (const auto& [key, value] : json.as_object()) {
            object.emplace_back(key, to_value(value));
        }
        return Value(std::move(object));
    }
    }
    return Value();
}

const Json& template_cases() {
    static const Json cases = [] {
        std::ifstream file(ii_test::test_data_dir() + "/llm/chat_templates.json", std::ios::binary);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return Json::parse(buffer.str()).value();
    }();
    return cases;
}

TEST(JinjaTest, RendersRealChatTemplatesLikeTransformers) {
    const Json& data = template_cases();
    ASSERT_GE(data.find("cases")->size(), 60u);
    int errors_expected = 0;
    for (const Json& c : data.find("cases")->as_array()) {
        Value context = to_value(*data.find("context"));
        context.set("messages", to_value(*c.find("messages")));
        context.set("add_generation_prompt", Value(c.find("add_generation_prompt")->as_bool()));
        const std::string& name = c.find("name")->as_string();
        const Result<std::string> rendered = render(c.find("template")->as_string(), context);
        if (const Json* error = c.find("error")) {
            ++errors_expected;
            ASSERT_FALSE(rendered.ok()) << name << ": expected the error '" << error->as_string() << "'";
            EXPECT_THAT(rendered.status().to_string(), HasSubstr(error->as_string())) << name;
            continue;
        }
        ASSERT_TRUE(rendered.ok()) << name << ": " << rendered.status().to_string();
        EXPECT_EQ(*rendered, c.find("expected")->as_string()) << name;
    }
    EXPECT_GT(errors_expected, 0);  // gemma rejects system messages
}

std::string render_text(const std::string& source, Object variables = {}) {
    const Result<std::string> result = render(source, Value(std::move(variables)));
    return result.ok() ? *result : "ERROR: " + result.status().to_string();
}

TEST(JinjaTest, Expressions) {
    EXPECT_EQ(render_text("{{ 1 + 2 * 3 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ 2 ** 10 }} {{ 7 / 2 }}"),
              "7 3 -4 1 1024 3.5");
    EXPECT_EQ(render_text("{{ 'a' ~ 1 ~ none }} {{ 'ab' + 'cd' }} {{ [1, 2] + [3] }} {{ 'x' * 3 }}"),
              "a1None abcd [1, 2, 3] xxx");
    EXPECT_EQ(render_text("{{ 'yes' if 1 < 2 and not false else 'no' }} {{ 3 in [1, 3] }} {{ 'b' not in 'abc' }}"),
              "yes True False");
    EXPECT_EQ(render_text("{{ x.a }} {{ x['b'][1] }} {{ x.missing is defined }} {{ y is undefined }}",
                          {{"x", Value(Object{{"a", Value(1)}, {"b", Value(Array{Value(5), Value(6)})}})}}),
              "1 6 False True");
    EXPECT_EQ(render_text("{{ 'hello'[1:4] }} {{ [1, 2, 3, 4][::-1] }} {{ [1, 2, 3][-1] }}"), "ell [4, 3, 2, 1] 3");
    EXPECT_EQ(render_text("{{ {'k': 'v', 'n': 1} }} {{ (1, 2) }} {{ 'it''s' }}"), "{'k': 'v', 'n': 1} [1, 2] its");
}

TEST(JinjaTest, FiltersTestsAndMethods) {
    EXPECT_EQ(
        render_text("{{ '  hi  ' | trim }}|{{ 'ab' | upper }}|{{ [3, 1, 2] | sort | join(',') }}|{{ 'x' | length }}"),
        "hi|AB|1,2,3|1");
    EXPECT_EQ(render_text("{{ {'a': [1, 'é']} | tojson }} {{ {'a': 1} | tojson(indent=2) }}"),
              "{\"a\": [1, \"é\"]} {\n  \"a\": 1\n}");
    EXPECT_EQ(render_text("{{ x | selectattr('role', 'equalto', 'user') | map(attribute='c') | list }}",
                          {{"x", Value(Array{Value(Object{{"role", Value("user")}, {"c", Value(1)}}),
                                             Value(Object{{"role", Value("bot")}, {"c", Value(2)}})})}}),
              "[1]");
    EXPECT_EQ(render_text("{{ none | default('d') }} {{ u | default('d') }} {{ 5 is odd }} {{ 'a' is string }}"),
              "None d True True");
    EXPECT_EQ(render_text("{{ ' a b '.strip() }}|{{ 'a,b'.split(',') }}|{{ 'abc'.startswith('ab') }}|"
                          "{{ 'x{}y'.format(1) }}|{{ 'Hi'.lower().upper() }}"),
              "a b|['a', 'b']|True|x1y|HI");
    EXPECT_EQ(render_text("{% set d = {'a': 1} %}{% for k, v in d.items() %}{{ k }}={{ v }}{% endfor %}"
                          "{{ d.get('b', 2) }}{{ d.keys() | list }}"),
              "a=12['a']");
}

TEST(JinjaTest, Statements) {
    EXPECT_EQ(render_text("{% for x in [1, 2, 3] if x != 2 %}{{ loop.index }}:{{ x }}{% if not loop.last %},{% endif %}"
                          "{% endfor %}"),
              "1:1,2:3");
    EXPECT_EQ(render_text("{% for x in [] %}a{% else %}empty{% endfor %}"), "empty");
    EXPECT_EQ(render_text("{% for x in range(5) %}{% if x == 3 %}{% break %}{% endif %}{% if x == 1 %}{% continue %}"
                          "{% endif %}{{ x }}{% endfor %}"),
              "02");
    // Assignments in a loop do not leak out of it; namespaces do.
    EXPECT_EQ(render_text("{% set n = 0 %}{% set ns = namespace(n=0) %}{% for x in [1, 2] %}{% set n = n + x %}"
                          "{% set ns.n = ns.n + x %}{% endfor %}{{ n }} {{ ns.n }}"),
              "0 3");
    EXPECT_EQ(render_text("{% macro greet(name, punct='!') %}Hi {{ name }}{{ punct }}{% endmacro %}"
                          "{{ greet('a') }} {{ greet('b', punct='?') }}"),
              "Hi a! Hi b?");
    EXPECT_EQ(render_text("{% set block %}x{{ 1 }}{% endset %}[{{ block }}]{% generation %}g{% endgeneration %}"),
              "[x1]g");
    EXPECT_THAT(render_text("{{ raise_exception('bad role') }}"), HasSubstr("bad role"));
    EXPECT_THAT(render_text("{% if %}"), HasSubstr("ERROR"));
    EXPECT_THAT(render_text("{% for x in [1] %}"), HasSubstr("endfor"));
    EXPECT_THAT(render_text("{{ unknown_filter_test | nope }}"), HasSubstr("unknown filter"));
}

TEST(JinjaTest, WhitespaceControl) {
    // trim_blocks and lstrip_blocks as in transformers, plus explicit '-'.
    EXPECT_EQ(render_text("a\n  {% if true %}\n  b\n  {% endif %}\nc"), "a\n  b\nc");
    EXPECT_EQ(render_text("a  {{- ' b ' -}}  c"), "a b c");
    EXPECT_EQ(render_text("{# comment #}x{#- other -#}  y"), "xy");
}

TEST(JinjaTest, LimitsProtectAgainstHostileTemplates) {
    EXPECT_THAT(render_text("{% for i in range(2000000) %}{% endfor %}"), HasSubstr("range too large"));
    EXPECT_THAT(render_text("{% macro f(n) %}{{ f(n) }}{% endmacro %}{{ f(1) }}"), HasSubstr("too deep"));
    EXPECT_THAT(render_text("{% set s = 'x' * 1000000 %}{% for i in range(100) %}{{ s }}{% endfor %}"),
                HasSubstr("too large"));
}

}  // namespace
}  // namespace ii::llm::detail::jinja
