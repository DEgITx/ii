// SPDX-License-Identifier: MIT
// Fuzz target: tokenizers and the chat template interpreter on arbitrary input. The first byte selects the target:
// a tokenizer of the test models (encode, with and without special tokens, then decode), or a Jinja template rendered
// with a small conversation. Nothing may crash, hang or allocate without bound.
#include <ii/gguf.h>

#include "llm/jinja.h"
#include "llm/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace {

std::shared_ptr<ii::llm::detail::TokenizerImpl> load(const char* name) {
    const std::string path = std::string(II_FUZZ_DATA_DIR) + "/llm/" + name;
    auto file = ii::gguf::File::open(path);
    if (!file.ok()) {
        return nullptr;
    }
    auto tokenizer = ii::llm::detail::tokenizer_from_gguf(*file);
    return tokenizer.ok() ? *tokenizer : nullptr;
}

/// A small SentencePiece vocabulary (scores) or a BPE-over-characters one with byte fallback.
std::shared_ptr<ii::llm::detail::TokenizerImpl> spm(ii::llm::detail::Algorithm algorithm) {
    using namespace ii::llm::detail;
    auto t = std::make_shared<TokenizerImpl>();
    t->algorithm = algorithm;
    t->texts = {"<unk>", "<s>", "</s>", "<|im_end|>"};
    t->types = {TokenType::kUnknown, TokenType::kControl, TokenType::kControl, TokenType::kControl};
    for (int b = 0; b < 256; ++b) {
        static const char* const kHex = "0123456789ABCDEF";
        t->texts.push_back(std::string("<0x") + kHex[b >> 4] + kHex[b & 15] + ">");
        t->types.push_back(TokenType::kByte);
    }
    // "\xE2\x96\x81" is SentencePiece's space (U+2581).
    const char* const pieces[] = {"\xE2\x96\x81", "\xE2\x96\x81t",    "h",        "e",
                                  "he",           "\xE2\x96\x81the",  "a",        "b",
                                  "ab",           "\xE2\x96\x81\x61", "\xD0\xBF", "\xD1\x80\xD0\xB8"};
    for (const char* piece : pieces) {
        t->texts.emplace_back(piece);
        t->types.push_back(TokenType::kNormal);
    }
    for (std::size_t i = 0; i < t->texts.size(); ++i) {
        t->scores.push_back(-static_cast<float>(i));
    }
    t->merge_ranks[{"\xE2\x96\x81", "t"}] = 0;
    t->merge_ranks[{"h", "e"}] = 1;
    t->merge_ranks[{"\xE2\x96\x81t", "he"}] = 2;
    t->merge_ranks[{"a", "b"}] = 3;
    t->bos = 1;
    t->eos = 2;
    t->unknown = 0;
    t->add_bos = true;
    t->finalize();
    return t;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size == 0) {
        return 0;
    }
    static const std::shared_ptr<ii::llm::detail::TokenizerImpl> tokenizers[] = {
        load("qwen2.gguf"),
        spm(ii::llm::detail::Algorithm::kSpmScores),
        spm(ii::llm::detail::Algorithm::kBpeChars),
    };
    const std::string_view text(reinterpret_cast<const char*>(data + 1), size - 1);
    switch (data[0] % 4) {
    case 0:
    case 1:
    case 2:
        if (const auto& tokenizer = tokenizers[data[0] % 4]; tokenizer != nullptr) {
            const auto tokens = tokenizer->encode(text, (data[0] & 4) != 0, (data[0] & 8) != 0);
            (void)tokenizer->decode(tokens, (data[0] & 16) != 0);
        }
        break;
    default: {
        using ii::llm::detail::jinja::Array;
        using ii::llm::detail::jinja::Object;
        using ii::llm::detail::jinja::Value;
        const Value context(Object{
            {"messages", Value(Array{Value(Object{{"role", Value("user")}, {"content", Value("Hi")}}),
                                     Value(Object{{"role", Value("assistant")}, {"content", Value("Hello")}})})},
            {"add_generation_prompt", Value(true)},
            {"bos_token", Value("<s>")},
        });
        (void)ii::llm::detail::jinja::render(text, context);
        break;
    }
    }
    return 0;
}
