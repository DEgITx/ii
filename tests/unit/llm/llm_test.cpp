// SPDX-License-Identifier: MIT
// Language models against transformers: tiny random models of every supported architecture
// (tests/data/llm, scripts/gen_llm_test_models.py), loaded from Hugging Face safetensors and from GGUF.
#include <ii/io.h>
#include <ii/llm.h>

#include "core/json.h"
#include "support/test_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ii {
namespace {

using ::testing::HasSubstr;

std::filesystem::path data(const std::string& name) {
    return std::filesystem::path(ii_test::test_data_dir()) / "llm" / name;
}

struct Expected {
    std::vector<llm::Token> tokens;
    std::vector<float> logits;  // [tokens, vocab]
};

Expected load_expected(const std::filesystem::path& path) {
    Expected e;
    auto arrays = load_npz(path);
    if (!arrays.ok()) {
        ADD_FAILURE() << arrays.status().to_string();
        return e;
    }
    for (const auto& [name, tensor] : *arrays) {
        if (name == "tokens") {
            const auto values = tensor.values<std::int32_t>();
            e.tokens.assign(values.begin(), values.end());
        } else if (name == "logits") {
            const auto values = tensor.values<float>();
            e.logits.assign(values.begin(), values.end());
        }
    }
    return e;
}

Context make_context() {
    ContextOptions options;
    options.num_threads = 2;
    return Context::create(std::move(options)).value();
}

/// Max |a - b| relative to the largest |b| of the row.
double row_error(std::span<const float> a, std::span<const float> b) {
    double max_ref = 0;
    double max_diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(b[i])));
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return max_diff / std::max(max_ref, 1e-6);
}

struct ModelCase {
    std::string name;      ///< test name
    std::string path;      ///< under tests/data/llm
    std::string expected;  ///< expected.npz under tests/data/llm
    double tolerance;
    std::vector<std::string> backends;
};

class LlmLogitsTest : public ::testing::TestWithParam<ModelCase> {};

TEST_P(LlmLogitsTest, MatchesTransformers) {
    const ModelCase& c = GetParam();
    const Context context = make_context();
    llm::LoadOptions options;
    options.context_length = 32;
    options.all_logits = true;
    options.compile.backends = c.backends;
    const auto loaded = llm::Llm::load(context, data(c.path), options);
    ASSERT_TRUE(loaded.ok()) << loaded.status().to_string();
    const Expected expected = load_expected(data(c.expected));
    const std::int64_t vocab = loaded->info().vocab_size;
    ASSERT_EQ(static_cast<std::int64_t>(expected.logits.size()),
              vocab * static_cast<std::int64_t>(expected.tokens.size()));

    // All tokens at once, then the same sequence in chunks through the KV cache.
    for (const std::vector<std::size_t>& chunks :
         {std::vector<std::size_t>{12}, std::vector<std::size_t>{5, 1, 1, 3, 1, 1}}) {
        auto sequence = loaded->new_sequence();
        ASSERT_TRUE(sequence.ok()) << sequence.status().to_string();
        std::size_t offset = 0;
        for (const std::size_t length : chunks) {
            const auto logits = sequence->eval(std::span<const llm::Token>(expected.tokens).subspan(offset, length));
            ASSERT_TRUE(logits.ok()) << logits.status().to_string();
            ASSERT_EQ(logits->size(), length * static_cast<std::size_t>(vocab));
            for (std::size_t t = 0; t < length; ++t) {
                const auto row = logits->subspan(t * static_cast<std::size_t>(vocab), static_cast<std::size_t>(vocab));
                const auto want =
                    std::span<const float>(expected.logits)
                        .subspan((offset + t) * static_cast<std::size_t>(vocab), static_cast<std::size_t>(vocab));
                EXPECT_LT(row_error(row, want), c.tolerance) << c.name << " position " << offset + t;
            }
            offset += length;
        }
        EXPECT_EQ(sequence->position(), 12);
    }
}

std::vector<ModelCase> model_cases() {
    std::vector<ModelCase> cases;
    for (const char* arch : {"llama", "qwen2", "qwen3", "gemma2", "gemma3", "phi3"}) {
        const std::string a(arch);
        cases.push_back({a + "_safetensors", a, a + "/expected.npz", 2e-4, {}});
        cases.push_back({a + "_gguf", a + ".gguf", a + "/expected.npz", 2e-4, {}});
    }
    // ii-cpu quantizes the activations of block-quantized products to 8 bits (as llama.cpp): ~1% of the logits.
    cases.push_back({"llama_q8_0_gguf", "llama-q8_0.gguf", "llama-q8_0.expected.npz", 3e-2, {}});
    cases.push_back(
        {"llama_q8_0_gguf_reference_backend", "llama-q8_0.gguf", "llama-q8_0.expected.npz", 2e-4, {"ii-ref"}});
    cases.push_back({"llama_gguf_reference_backend", "llama.gguf", "llama/expected.npz", 2e-4, {"ii-ref"}});
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Models, LlmLogitsTest, ::testing::ValuesIn(model_cases()),
                         [](const auto& param_info) { return param_info.param.name; });

TEST(LlmTest, ReportsModelInfo) {
    const Context context = make_context();
    const auto llm = llm::Llm::load(context, data("qwen2.gguf"));
    ASSERT_TRUE(llm.ok()) << llm.status().to_string();
    const llm::ModelInfo& info = llm->info();
    EXPECT_EQ(info.architecture, "qwen2");
    EXPECT_EQ(info.format, "gguf");
    EXPECT_EQ(info.layers, 2);
    EXPECT_EQ(info.heads, 4);
    EXPECT_EQ(info.kv_heads, 2);
    EXPECT_EQ(info.context_length, 128);  // the trained context (below the 4096 default cap)
    EXPECT_EQ(info.weight_type, "f16");
    EXPECT_GT(info.parameters, 30000);
    EXPECT_FALSE(llm::Llm::load(context, data("missing.gguf")).ok());
}

TEST(LlmTest, ContextOverflowIsAnError) {
    const Context context = make_context();
    llm::LoadOptions options;
    options.context_length = 8;
    const auto llm = llm::Llm::load(context, data("llama.gguf"), options);
    ASSERT_TRUE(llm.ok());
    auto sequence = llm->new_sequence().value();
    const std::vector<llm::Token> tokens(6, 5);
    ASSERT_TRUE(sequence.eval(tokens).ok());
    EXPECT_THAT(sequence.eval(tokens).status().to_string(), HasSubstr("context is full"));
    sequence.rewind(2);
    EXPECT_EQ(sequence.position(), 2);
    const std::vector<llm::Token> bad = {100000};
    EXPECT_THAT(sequence.eval(bad).status().to_string(), HasSubstr("vocabulary"));
    EXPECT_TRUE(sequence.eval(tokens).ok());
}

TEST(LlmTest, GreedyGenerationIsDeterministic) {
    const Context context = make_context();
    const auto llm = llm::Llm::load(context, data("qwen3.gguf"));
    ASSERT_TRUE(llm.ok());
    llm::SamplingOptions greedy;
    greedy.temperature = 0;
    std::vector<std::string> outputs;
    for (int run = 0; run < 2; ++run) {
        auto sequence = llm->new_sequence().value();
        llm::Sampler sampler(greedy);
        std::string text;
        const std::vector<llm::Token> prompt = llm->tokenizer().encode("Hello world");
        const auto stats = sequence.generate(prompt, sampler, 10, [&](std::string_view piece) {
            text += piece;
            return true;
        });
        ASSERT_TRUE(stats.ok()) << stats.status().to_string();
        EXPECT_EQ(stats->prompt_tokens, static_cast<std::int64_t>(prompt.size()));
        EXPECT_LE(stats->generated_tokens, 10);
        outputs.push_back(text);
    }
    EXPECT_EQ(outputs[0], outputs[1]);
}

TEST(LlmTest, ChatUsesTheTemplate) {
    const Context context = make_context();
    const auto llm = llm::Llm::load(context, data("gemma2"));
    ASSERT_TRUE(llm.ok()) << llm.status().to_string();
    const std::vector<llm::Message> messages = {{"system", "Be brief."}, {"user", "Hi"}};
    const auto text = llm->format_chat(messages);
    ASSERT_TRUE(text.ok()) << text.status().to_string();
    EXPECT_EQ(*text,
              "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    llm::ChatOptions options;
    options.temperature = 0;
    options.max_tokens = 5;
    auto chat = llm->new_chat(options).value();
    const auto first = chat.send("Hello");
    ASSERT_TRUE(first.ok()) << first.status().to_string();
    const auto second = chat.send("Again");
    ASSERT_TRUE(second.ok()) << second.status().to_string();
    ASSERT_EQ(chat.messages().size(), 4u);
    // The second turn reuses the cached first turn: only its new tokens are evaluated.
    const std::vector<llm::Message> asked(chat.messages().begin(), chat.messages().begin() + 3);
    const auto full = llm->tokenizer().encode(llm->format_chat(asked).value(), true, true);
    EXPECT_LT(chat.last_stats().prompt_tokens, static_cast<std::int64_t>(full.size()) / 2);
}

// --- tokenizer ---

TEST(TokenizerTest, MatchesHuggingFaceTokenizers) {
    std::ifstream file(data("tokenizer_cases.json"), std::ios::binary);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const Json cases = Json::parse(buffer.str()).value();
    const Context context = make_context();
    for (const char* model : {"llama", "llama.gguf"}) {
        const auto llm = llm::Llm::load(context, data(model));
        ASSERT_TRUE(llm.ok()) << llm.status().to_string();
        const llm::Tokenizer& tokenizer = llm->tokenizer();
        EXPECT_EQ(tokenizer.kind(), "bpe");
        for (const Json& c : cases.as_array()) {
            const std::string& text = c.find("text")->as_string();
            std::vector<llm::Token> expected;
            for (const Json& id : c.find("ids")->as_array()) {
                expected.push_back(static_cast<llm::Token>(*id.as_int()));
            }
            EXPECT_EQ(tokenizer.encode(text, false, true), expected) << model << ": '" << text << "'";
            EXPECT_EQ(tokenizer.decode(expected, true), text) << model;
        }
        // add_special prepends BOS; special tokens are only parsed on request.
        const std::vector<llm::Token> with_bos = tokenizer.encode("Hi", true);
        ASSERT_FALSE(with_bos.empty());
        EXPECT_EQ(with_bos[0], tokenizer.bos());
        EXPECT_GT(tokenizer.encode("<|im_end|>", false, false).size(), 1u);
        EXPECT_EQ(tokenizer.encode("<|im_end|>", false, true).size(), 1u);
        EXPECT_TRUE(tokenizer.is_end_of_generation(tokenizer.encode("<|im_end|>", false, true)[0]));
        EXPECT_EQ(tokenizer.decode(tokenizer.encode("<|im_end|>x", false, true)), "x");
    }
}

}  // namespace
}  // namespace ii
