// SPDX-License-Identifier: MIT
#include "llm/unicode.h"

#include <algorithm>

namespace ii::llm::detail {

namespace {

bool in_ranges(const CodepointRange* ranges, std::size_t count, std::uint32_t cp) noexcept {
    const CodepointRange* end = ranges + count;
    const CodepointRange* it =
        std::upper_bound(ranges, end, cp, [](std::uint32_t value, const CodepointRange& r) { return value < r.first; });
    return it != ranges && cp <= (it - 1)->last;
}

constexpr std::uint32_t kReplacement = 0xFFFD;

}  // namespace

bool is_letter(std::uint32_t cp) noexcept {
    if (cp < 0x80) {
        return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    }
    return in_ranges(kLetterRanges, kLetterRangesCount, cp);
}

bool is_number(std::uint32_t cp) noexcept {
    if (cp < 0x80) {
        return cp >= '0' && cp <= '9';
    }
    return in_ranges(kNumberRanges, kNumberRangesCount, cp);
}

bool is_whitespace(std::uint32_t cp) noexcept {
    // The Unicode White_Space property (what \s matches in the tokenizers' regular expressions).
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

std::size_t utf8_length(unsigned char lead) noexcept {
    if (lead < 0x80) {
        return 1;
    }
    if ((lead & 0xE0) == 0xC0) {
        return 2;
    }
    if ((lead & 0xF0) == 0xE0) {
        return 3;
    }
    if ((lead & 0xF8) == 0xF0) {
        return 4;
    }
    return 1;
}

std::vector<std::uint32_t> decode_utf8(std::string_view text, std::vector<std::size_t>* offsets) {
    std::vector<std::uint32_t> result;
    result.reserve(text.size());
    if (offsets != nullptr) {
        offsets->clear();
        offsets->reserve(text.size() + 1);
    }
    std::size_t i = 0;
    while (i < text.size()) {
        if (offsets != nullptr) {
            offsets->push_back(i);
        }
        const auto lead = static_cast<unsigned char>(text[i]);
        const std::size_t length = utf8_length(lead);
        std::uint32_t cp = kReplacement;
        bool valid = length > 1 || lead < 0x80;
        if (length > 1 && i + length <= text.size()) {
            cp = lead & (0xFFu >> (length + 1));
            for (std::size_t k = 1; k < length; ++k) {
                const auto c = static_cast<unsigned char>(text[i + k]);
                if ((c & 0xC0) != 0x80) {
                    valid = false;
                    break;
                }
                cp = (cp << 6) | (c & 0x3F);
            }
            // Reject overlong encodings, surrogates and values above U+10FFFF.
            const std::uint32_t minimum = length == 2 ? 0x80 : length == 3 ? 0x800 : 0x10000;
            if (valid && (cp < minimum || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) {
                valid = false;
            }
        } else if (length > 1) {
            valid = false;
        } else {
            cp = lead;
        }
        if (valid) {
            result.push_back(cp);
            i += length;
        } else {
            result.push_back(kReplacement);
            i += 1;
        }
    }
    if (offsets != nullptr) {
        offsets->push_back(text.size());
    }
    return result;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

std::size_t incomplete_utf8_suffix(std::string_view text) noexcept {
    // Look at most 3 bytes back for a lead byte whose sequence runs past the end.
    for (std::size_t back = 1; back <= std::min<std::size_t>(3, text.size()); ++back) {
        const auto c = static_cast<unsigned char>(text[text.size() - back]);
        if ((c & 0xC0) == 0x80) {
            continue;  // continuation byte
        }
        const std::size_t length = utf8_length(c);
        return length > back ? back : 0;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// Pre-tokenizer splits (hand-written equivalents of the patterns, as in llama.cpp)

std::vector<std::string_view> split_words(std::string_view text, SplitPattern pattern) {
    std::vector<std::size_t> offsets;
    const std::vector<std::uint32_t> cps = decode_utf8(text, &offsets);
    const std::size_t n = cps.size();
    std::vector<std::string_view> words;
    auto at = [&](std::size_t i) -> std::uint32_t { return i < n ? cps[i] : 0xFFFFFFFFu; };
    auto letter = [&](std::size_t i) { return i < n && is_letter(cps[i]); };
    auto number = [&](std::size_t i) { return i < n && is_number(cps[i]); };
    auto space = [&](std::size_t i) { return i < n && is_whitespace(cps[i]); };
    auto newline = [&](std::size_t i) { return i < n && (cps[i] == '\r' || cps[i] == '\n'); };
    // "Other": not whitespace, letter or number ([^\s\p{L}\p{N}]).
    auto other = [&](std::size_t i) {
        return i < n && !is_whitespace(cps[i]) && !is_letter(cps[i]) && !is_number(cps[i]);
    };
    auto lower = [](std::uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };
    const bool gpt2 = pattern == SplitPattern::kGpt2;

    std::size_t i = 0;
    while (i < n) {
        std::size_t end = i;
        // Contractions: 's 't 're 've 'm 'll 'd (case-insensitive except for GPT-2).
        if (at(i) == '\'' && i + 1 < n) {
            const std::uint32_t c1 = gpt2 ? at(i + 1) : lower(at(i + 1));
            const std::uint32_t c2 = i + 2 < n ? (gpt2 ? at(i + 2) : lower(at(i + 2))) : 0;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                end = i + 2;
            } else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                end = i + 3;
            }
        }
        if (end == i) {
            if (gpt2) {
                // ' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+'
                const std::size_t start = at(i) == ' ' && (letter(i + 1) || number(i + 1) || other(i + 1)) ? i + 1 : i;
                if (letter(start)) {
                    end = start;
                    while (letter(end)) {
                        ++end;
                    }
                } else if (number(start)) {
                    end = start;
                    while (number(end)) {
                        ++end;
                    }
                } else if (other(start)) {
                    end = start;
                    while (other(end)) {
                        ++end;
                    }
                }
            } else {
                // [^\r\n\p{L}\p{N}]?\p{L}+
                if (!newline(i) && !letter(i) && !number(i) && letter(i + 1)) {
                    end = i + 1;
                } else if (letter(i)) {
                    end = i;
                }
                if (end != i || letter(i)) {
                    while (letter(end)) {
                        ++end;
                    }
                } else if (number(i)) {
                    // \p{N}{1,3} (Llama 3) or \p{N} (Qwen 2)
                    const std::size_t limit = pattern == SplitPattern::kLlama3 ? 3 : 1;
                    end = i;
                    while (number(end) && end - i < limit) {
                        ++end;
                    }
                } else {
                    // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
                    const std::size_t start = at(i) == ' ' && other(i + 1) ? i + 1 : i;
                    if (other(start)) {
                        end = start;
                        while (other(end)) {
                            ++end;
                        }
                        while (newline(end)) {
                            ++end;
                        }
                    }
                }
            }
        }
        if (end == i && space(i)) {
            std::size_t run = i;
            while (space(run)) {
                ++run;
            }
            // \s*[\r\n]+ (not GPT-2): up to the last newline of the run.
            if (!gpt2) {
                for (std::size_t k = run; k > i; --k) {
                    if (newline(k - 1)) {
                        end = k;
                        break;
                    }
                }
            }
            if (end == i) {
                // \s+(?!\S): leave the last space for the next word; \s+ otherwise.
                end = run == n || run - i < 2 ? run : run - 1;
            }
        }
        if (end == i) {
            end = i + 1;  // unreachable for valid patterns; guarantees progress
        }
        words.push_back(text.substr(offsets[i], offsets[end] - offsets[i]));
        i = end;
    }
    return words;
}

}  // namespace ii::llm::detail
