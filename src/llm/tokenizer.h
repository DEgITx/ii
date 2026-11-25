// SPDX-License-Identifier: MIT
/// @file
/// Tokenizer implementation: SentencePiece-style merging by token scores (GGUF "llama" vocabularies), BPE merging by
/// merge rank over byte-level symbols (GPT-2 / Llama 3 / Qwen vocabularies) or over characters with byte fallback
/// (Hugging Face tokenizer.json of SentencePiece models), special tokens and decoding.
#pragma once

#include <ii/gguf.h>
#include <ii/llm.h>

#include "core/json.h"
#include "llm/unicode.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ii::llm::detail {

enum class TokenType : std::uint8_t {
    kNormal = 1,
    kUnknown = 2,
    kControl = 3,
    kUserDefined = 4,
    kUnused = 5,
    kByte = 6,
};

enum class Algorithm : std::uint8_t {
    kSpmScores,  ///< SentencePiece: merge the adjacent pair whose merged text is the highest-scored token
    kBpeBytes,   ///< byte-level BPE: bytes mapped to printable characters, merges by rank
    kBpeChars,   ///< BPE over characters with "▁" for spaces and <0xXX> byte fallback
};

struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        return std::hash<std::string>()(p.first) * 31 + std::hash<std::string>()(p.second);
    }
};

struct TokenizerImpl {
    Algorithm algorithm = Algorithm::kSpmScores;
    SplitPattern split = SplitPattern::kGpt2;
    std::vector<std::string> texts;  ///< token text as stored in the vocabulary
    std::vector<float> scores;
    std::vector<TokenType> types;
    std::unordered_map<std::string, Token> lookup;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks;
    Token bos = -1;
    Token eos = -1;
    Token unknown = -1;
    std::vector<Token> end_of_generation;
    bool add_bos = false;
    bool add_eos = false;
    bool add_space_prefix = true;  ///< SentencePiece: prepend "▁" to the text
    bool ignore_merges = false;    ///< Llama 3: a word that is a token is not merged
    Token byte_tokens[256];        ///< <0xXX> tokens (SentencePiece), or the byte-level tokens
    std::vector<Token> specials;   ///< control and user-defined tokens, longest text first

    // --- construction ---
    /// Finishes the tables after texts / types / scores / merges are set.
    void finalize();

    // --- encoding ---
    std::vector<Token> encode(std::string_view text, bool add_special, bool parse_special) const;
    void encode_fragment(std::string_view text, bool space_prefix, std::vector<Token>& out) const;
    void encode_spm(const std::string& text, std::vector<Token>& out) const;
    void encode_bpe_word(const std::string& word, std::vector<Token>& out) const;

    // --- decoding ---
    /// The bytes a token stands for (special tokens: their text if `render_special`, else nothing).
    std::string piece(Token token, bool render_special) const;
    std::string decode(std::span<const Token> tokens, bool render_special) const;

    bool is_special(Token token) const noexcept {
        return token >= 0 && static_cast<std::size_t>(token) < types.size() &&
               (types[static_cast<std::size_t>(token)] == TokenType::kControl ||
                types[static_cast<std::size_t>(token)] == TokenType::kUserDefined ||
                types[static_cast<std::size_t>(token)] == TokenType::kUnknown);
    }
};

/// Reads the tokenizer of a GGUF model.
Result<std::shared_ptr<TokenizerImpl>> tokenizer_from_gguf(const gguf::File& file);
/// Reads a Hugging Face tokenizer (tokenizer.json, with tokenizer_config.json for the special-token settings).
Result<std::shared_ptr<TokenizerImpl>> tokenizer_from_hf(const Json& tokenizer, const Json* config,
                                                         const Json* generation_config);

/// The GPT-2 byte-to-character map of byte-level BPE.
const std::string& byte_to_char(unsigned char byte);

}  // namespace ii::llm::detail
