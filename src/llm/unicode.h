// SPDX-License-Identifier: MIT
/// @file
/// Minimal Unicode support for tokenizers: UTF-8 decoding / encoding, the general categories L and N, the White_Space
/// property, and the pre-tokenizer splits of byte-level BPE models (GPT-2, Llama 3, Qwen 2).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ii::llm::detail {

struct CodepointRange {
    std::uint32_t first;
    std::uint32_t last;
};

extern const CodepointRange kLetterRanges[];
extern const std::size_t kLetterRangesCount;
extern const CodepointRange kNumberRanges[];
extern const std::size_t kNumberRangesCount;

bool is_letter(std::uint32_t cp) noexcept;
bool is_number(std::uint32_t cp) noexcept;
bool is_whitespace(std::uint32_t cp) noexcept;

/// Length of the UTF-8 sequence starting with `lead` (1 for invalid lead bytes).
std::size_t utf8_length(unsigned char lead) noexcept;

/// Decodes UTF-8; invalid bytes decode to U+FFFD each. `offsets` (optional) receives the byte offset of every code
/// point plus a final entry with the text size.
std::vector<std::uint32_t> decode_utf8(std::string_view text, std::vector<std::size_t>* offsets = nullptr);
void append_utf8(std::string& out, std::uint32_t cp);

/// Number of bytes at the end of `text` that form an incomplete UTF-8 sequence (0 when it ends on a boundary).
std::size_t incomplete_utf8_suffix(std::string_view text) noexcept;

/// Pre-tokenizer patterns of byte-level BPE tokenizers.
enum class SplitPattern {
    kGpt2,    ///< 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    kLlama3,  ///< case-insensitive contractions, [^\r\n\p{L}\p{N}]?\p{L}+, \p{N}{1,3}, ... (Llama 3)
    kQwen2,   ///< as kLlama3 with single digits (Qwen 2, Qwen 3)
};

/// Splits `text` into the pieces the pattern matches (byte ranges of `text`, in order, covering all of it).
std::vector<std::string_view> split_words(std::string_view text, SplitPattern pattern);

}  // namespace ii::llm::detail
