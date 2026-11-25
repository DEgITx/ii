// SPDX-License-Identifier: MIT
#include "llm/tokenizer.h"

#include <algorithm>
#include <array>
#include <format>
#include <queue>

namespace ii::llm::detail {

namespace {

constexpr std::string_view kSpace = "\xE2\x96\x81";  // U+2581 "▁", SentencePiece's space

/// Written forms of tokens that end a turn in common chat formats (llama.cpp adds them to the end-of-generation set
/// the same way).
constexpr std::string_view kEndOfTurnTexts[] = {
    "<|eot_id|>",      "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
    "<|end_of_text|>", "<|eom_id|>", "<EOT>",   "</s>",
};

struct ByteMap {
    std::array<std::string, 256> to_char;
    std::unordered_map<std::uint32_t, unsigned char> from_char;

    ByteMap() {
        // GPT-2 bytes_to_unicode: printable bytes map to themselves, the others to U+0100 onwards.
        std::uint32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            const std::uint32_t cp = printable ? static_cast<std::uint32_t>(b) : next++;
            append_utf8(to_char[static_cast<std::size_t>(b)], cp);
            from_char[cp] = static_cast<unsigned char>(b);
        }
    }
};

const ByteMap& byte_map() {
    static const ByteMap map;
    return map;
}

std::string replace_all(std::string_view text, std::string_view from, std::string_view to) {
    std::string out;
    std::size_t start = 0;
    while (true) {
        const std::size_t at = text.find(from, start);
        if (at == std::string_view::npos) {
            out.append(text.substr(start));
            return out;
        }
        out.append(text.substr(start, at - start));
        out.append(to);
        start = at + from.size();
    }
}

/// The byte of a "<0xXX>" token text, or -1.
int parse_byte_token(std::string_view text) {
    if (text.size() != 6 || text.substr(0, 3) != "<0x" || text[5] != '>') {
        return -1;
    }
    int value = 0;
    for (int i = 3; i < 5; ++i) {
        const char c = text[static_cast<std::size_t>(i)];
        value *= 16;
        if (c >= '0' && c <= '9') {
            value += c - '0';
        } else if (c >= 'A' && c <= 'F') {
            value += c - 'A' + 10;
        } else if (c >= 'a' && c <= 'f') {
            value += c - 'a' + 10;
        } else {
            return -1;
        }
    }
    return value;
}

struct Symbol {
    std::size_t start = 0;  ///< byte offset in the text
    std::size_t length = 0;
    int prev = -1;
    int next = -1;
};

}  // namespace

const std::string& byte_to_char(unsigned char byte) {
    return byte_map().to_char[byte];
}

// ---------------------------------------------------------------------------------------------------------------
// Construction

void TokenizerImpl::finalize() {
    const std::size_t n = texts.size();
    scores.resize(n, 0.0f);
    types.resize(n, TokenType::kNormal);
    lookup.clear();
    lookup.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        lookup.emplace(texts[i], static_cast<Token>(i));
    }
    for (int b = 0; b < 256; ++b) {
        Token token = -1;
        if (algorithm == Algorithm::kBpeBytes) {
            if (const auto it = lookup.find(byte_to_char(static_cast<unsigned char>(b))); it != lookup.end()) {
                token = it->second;
            }
        } else {
            if (const auto it = lookup.find(std::format("<0x{:02X}>", b)); it != lookup.end()) {
                token = it->second;
                types[static_cast<std::size_t>(token)] = TokenType::kByte;
            } else if (const auto single = lookup.find(std::string(1, static_cast<char>(b))); single != lookup.end()) {
                token = single->second;
            }
        }
        byte_tokens[b] = token;
    }
    specials.clear();
    for (std::size_t i = 0; i < n; ++i) {
        if ((types[i] == TokenType::kControl || types[i] == TokenType::kUserDefined) && !texts[i].empty()) {
            specials.push_back(static_cast<Token>(i));
        }
    }
    std::stable_sort(specials.begin(), specials.end(), [&](Token a, Token b) {
        return texts[static_cast<std::size_t>(a)].size() > texts[static_cast<std::size_t>(b)].size();
    });
    if (eos >= 0) {
        end_of_generation.push_back(eos);
    }
    for (const std::string_view text : kEndOfTurnTexts) {
        if (const auto it = lookup.find(std::string(text)); it != lookup.end() && is_special(it->second)) {
            end_of_generation.push_back(it->second);
        }
    }
    std::sort(end_of_generation.begin(), end_of_generation.end());
    end_of_generation.erase(std::unique(end_of_generation.begin(), end_of_generation.end()), end_of_generation.end());
}

// ---------------------------------------------------------------------------------------------------------------
// Encoding

std::vector<Token> TokenizerImpl::encode(std::string_view text, bool add_special, bool parse_special) const {
    std::vector<Token> out;
    if (add_special && add_bos && bos >= 0) {
        out.push_back(bos);
    }
    bool previous_special = true;  // SentencePiece: a space prefix at the start and after special tokens
    std::size_t start = 0;
    std::size_t i = 0;
    auto flush = [&](std::size_t end) {
        if (end > start) {
            encode_fragment(text.substr(start, end - start), previous_special, out);
            previous_special = false;
        }
    };
    while (i < text.size()) {
        Token matched = -1;
        for (const Token token : specials) {
            const std::string& s = texts[static_cast<std::size_t>(token)];
            if (!parse_special && types[static_cast<std::size_t>(token)] == TokenType::kControl) {
                continue;
            }
            if (text.compare(i, s.size(), s) == 0) {
                matched = token;
                break;
            }
        }
        if (matched < 0) {
            ++i;
            continue;
        }
        flush(i);
        out.push_back(matched);
        previous_special = true;
        i += texts[static_cast<std::size_t>(matched)].size();
        start = i;
    }
    flush(text.size());
    if (add_special && add_eos && eos >= 0) {
        out.push_back(eos);
    }
    return out;
}

void TokenizerImpl::encode_fragment(std::string_view text, bool space_prefix, std::vector<Token>& out) const {
    switch (algorithm) {
    case Algorithm::kSpmScores: {
        std::string escaped = add_space_prefix && space_prefix ? std::string(kSpace) : std::string();
        escaped += replace_all(text, " ", kSpace);
        encode_spm(escaped, out);
        break;
    }
    case Algorithm::kBpeChars: {
        std::string escaped = add_space_prefix && space_prefix ? std::string(kSpace) : std::string();
        escaped += replace_all(text, " ", kSpace);
        encode_bpe_word(escaped, out);
        break;
    }
    case Algorithm::kBpeBytes:
        for (const std::string_view word : split_words(text, split)) {
            std::string mapped;
            for (const char c : word) {
                mapped += byte_to_char(static_cast<unsigned char>(c));
            }
            encode_bpe_word(mapped, out);
        }
        break;
    }
}

void TokenizerImpl::encode_spm(const std::string& text, std::vector<Token>& out) const {
    std::vector<Symbol> symbols;
    for (std::size_t offset = 0; offset < text.size();) {
        Symbol s;
        s.start = offset;
        s.length = std::min(utf8_length(static_cast<unsigned char>(text[offset])), text.size() - offset);
        offset += s.length;
        s.prev = static_cast<int>(symbols.size()) - 1;
        s.next = offset == text.size() ? -1 : static_cast<int>(symbols.size()) + 1;
        symbols.push_back(s);
    }
    struct Bigram {
        int left;
        int right;
        float score;
        std::size_t size;
    };
    auto worse = [](const Bigram& a, const Bigram& b) {
        return a.score < b.score || (a.score == b.score && a.left > b.left);
    };
    std::priority_queue<Bigram, std::vector<Bigram>, decltype(worse)> queue(worse);
    auto try_add = [&](int left, int right) {
        if (left < 0 || right < 0) {
            return;
        }
        const std::size_t size =
            symbols[static_cast<std::size_t>(left)].length + symbols[static_cast<std::size_t>(right)].length;
        const auto it = lookup.find(text.substr(symbols[static_cast<std::size_t>(left)].start, size));
        if (it == lookup.end()) {
            return;
        }
        queue.push(Bigram{left, right, scores[static_cast<std::size_t>(it->second)], size});
    };
    for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
        try_add(i - 1, i);
    }
    while (!queue.empty()) {
        const Bigram bigram = queue.top();
        queue.pop();
        Symbol& left = symbols[static_cast<std::size_t>(bigram.left)];
        Symbol& right = symbols[static_cast<std::size_t>(bigram.right)];
        if (left.length == 0 || right.length == 0 || left.length + right.length != bigram.size) {
            continue;  // stale
        }
        left.length += right.length;
        right.length = 0;
        left.next = right.next;
        if (right.next >= 0) {
            symbols[static_cast<std::size_t>(right.next)].prev = bigram.left;
        }
        try_add(left.prev, bigram.left);
        try_add(bigram.left, left.next);
    }
    for (int i = symbols.empty() ? -1 : 0; i != -1; i = symbols[static_cast<std::size_t>(i)].next) {
        const Symbol& s = symbols[static_cast<std::size_t>(i)];
        const std::string piece = text.substr(s.start, s.length);
        if (const auto it = lookup.find(piece); it != lookup.end()) {
            out.push_back(it->second);
            continue;
        }
        for (const char c : piece) {
            const Token token = byte_tokens[static_cast<unsigned char>(c)];
            out.push_back(token >= 0 ? token : unknown);
        }
    }
}

void TokenizerImpl::encode_bpe_word(const std::string& word, std::vector<Token>& out) const {
    if (word.empty()) {
        return;
    }
    if (ignore_merges) {
        if (const auto it = lookup.find(word); it != lookup.end()) {
            out.push_back(it->second);
            return;
        }
    }
    std::vector<Symbol> symbols;
    for (std::size_t offset = 0; offset < word.size();) {
        Symbol s;
        s.start = offset;
        s.length = std::min(utf8_length(static_cast<unsigned char>(word[offset])), word.size() - offset);
        offset += s.length;
        s.prev = static_cast<int>(symbols.size()) - 1;
        s.next = offset == word.size() ? -1 : static_cast<int>(symbols.size()) + 1;
        symbols.push_back(s);
    }
    struct Bigram {
        int left;
        int right;
        int rank;
        std::size_t size;
    };
    auto worse = [](const Bigram& a, const Bigram& b) {
        return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    };
    std::priority_queue<Bigram, std::vector<Bigram>, decltype(worse)> queue(worse);
    std::pair<std::string, std::string> key;
    auto try_add = [&](int left, int right) {
        if (left < 0 || right < 0) {
            return;
        }
        const Symbol& l = symbols[static_cast<std::size_t>(left)];
        const Symbol& r = symbols[static_cast<std::size_t>(right)];
        key.first.assign(word, l.start, l.length);
        key.second.assign(word, r.start, r.length);
        const auto it = merge_ranks.find(key);
        if (it == merge_ranks.end()) {
            return;
        }
        queue.push(Bigram{left, right, it->second, l.length + r.length});
    };
    for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
        try_add(i - 1, i);
    }
    while (!queue.empty()) {
        const Bigram bigram = queue.top();
        queue.pop();
        Symbol& left = symbols[static_cast<std::size_t>(bigram.left)];
        Symbol& right = symbols[static_cast<std::size_t>(bigram.right)];
        if (left.length == 0 || right.length == 0 || left.length + right.length != bigram.size ||
            left.next != bigram.right) {
            continue;  // stale
        }
        left.length += right.length;
        right.length = 0;
        left.next = right.next;
        if (right.next >= 0) {
            symbols[static_cast<std::size_t>(right.next)].prev = bigram.left;
        }
        try_add(left.prev, bigram.left);
        try_add(bigram.left, left.next);
    }
    for (int i = 0; i != -1; i = symbols[static_cast<std::size_t>(i)].next) {
        const Symbol& s = symbols[static_cast<std::size_t>(i)];
        const std::string piece = word.substr(s.start, s.length);
        if (const auto it = lookup.find(piece); it != lookup.end()) {
            out.push_back(it->second);
            continue;
        }
        if (algorithm == Algorithm::kBpeBytes) {
            // Characters of the byte-level alphabet are tokens of their own.
            for (std::size_t offset = 0; offset < piece.size();) {
                const std::size_t length = utf8_length(static_cast<unsigned char>(piece[offset]));
                if (const auto it = lookup.find(piece.substr(offset, length)); it != lookup.end()) {
                    out.push_back(it->second);
                }
                offset += length;
            }
        } else {
            for (const char c : piece) {
                const Token token = byte_tokens[static_cast<unsigned char>(c)];
                out.push_back(token >= 0 ? token : unknown);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Decoding

std::string TokenizerImpl::piece(Token token, bool render_special) const {
    if (token < 0 || static_cast<std::size_t>(token) >= texts.size()) {
        return {};
    }
    const std::string& text = texts[static_cast<std::size_t>(token)];
    switch (types[static_cast<std::size_t>(token)]) {
    case TokenType::kControl:
    case TokenType::kUnknown:
    case TokenType::kUnused:
        return render_special ? text : std::string();
    case TokenType::kUserDefined:
        return text;
    case TokenType::kByte:
        if (const int byte = parse_byte_token(text); byte >= 0) {
            return std::string(1, static_cast<char>(byte));
        }
        return {};
    case TokenType::kNormal:
        break;
    }
    if (algorithm == Algorithm::kBpeBytes) {
        std::string bytes;
        const ByteMap& map = byte_map();
        for (const std::uint32_t cp : decode_utf8(text)) {
            if (const auto it = map.from_char.find(cp); it != map.from_char.end()) {
                bytes += static_cast<char>(it->second);
            } else {
                append_utf8(bytes, cp);
            }
        }
        return bytes;
    }
    return replace_all(text, kSpace, " ");
}

std::string TokenizerImpl::decode(std::span<const Token> tokens, bool render_special) const {
    std::string out;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        std::string p = piece(tokens[i], render_special);
        // SentencePiece: the space added in front of the text is not part of it.
        if (out.empty() && algorithm != Algorithm::kBpeBytes && add_space_prefix && !p.empty() && p[0] == ' ' &&
            !is_special(tokens[i])) {
            p.erase(0, 1);
        }
        out += p;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// GGUF

Result<std::shared_ptr<TokenizerImpl>> tokenizer_from_gguf(const gguf::File& file) {
    auto t = std::make_shared<TokenizerImpl>();
    const std::string model = file.get_string("tokenizer.ggml.model").value_or("");
    const std::string pre = file.get_string("tokenizer.ggml.pre").value_or("default");
    if (model == "llama") {
        t->algorithm = Algorithm::kSpmScores;
        t->add_bos = true;
        t->add_space_prefix = true;
    } else if (model == "gpt2") {
        t->algorithm = Algorithm::kBpeBytes;
        t->add_space_prefix = false;
        if (pre == "llama-bpe" || pre == "llama3" || pre == "llama-v3" || pre == "falcon3" || pre == "pixtral" ||
            pre == "smaug-bpe" || pre == "dbrx") {
            t->split = SplitPattern::kLlama3;
            t->ignore_merges = true;
        } else if (pre == "qwen2" || pre == "deepseek-r1-qwen" || pre == "megrez") {
            t->split = SplitPattern::kQwen2;
        } else {
            t->split = SplitPattern::kGpt2;
        }
    } else {
        return errors::unsupported("tokenizer model '{}' is not supported (llama and gpt2 are)", model);
    }
    const gguf::Value* tokens = file.find("tokenizer.ggml.tokens");
    if (tokens == nullptr || tokens->as_strings() == nullptr || tokens->as_strings()->empty()) {
        return errors::data_corrupted("GGUF model without tokenizer.ggml.tokens");
    }
    t->texts = *tokens->as_strings();
    const std::size_t n = t->texts.size();
    if (const gguf::Value* scores = file.find("tokenizer.ggml.scores"); scores != nullptr && scores->as_floats()) {
        if (scores->array_size() != n) {
            return errors::data_corrupted("tokenizer.ggml.scores has {} entries for {} tokens", scores->array_size(),
                                          n);
        }
        for (const double s : *scores->as_floats()) {
            t->scores.push_back(static_cast<float>(s));
        }
    }
    if (t->algorithm == Algorithm::kSpmScores &&
        (t->scores.empty() ||
         std::all_of(t->scores.begin(), t->scores.end(), [&](float s) { return s == t->scores[0]; }))) {
        // Some converted files have no scores (all equal): SentencePiece vocabularies are sorted by score, so the
        // token order is the merge priority.
        t->scores.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            t->scores[i] = -static_cast<float>(i);
        }
    }
    if (const gguf::Value* types = file.find("tokenizer.ggml.token_type"); types != nullptr && types->as_ints()) {
        if (types->array_size() != n) {
            return errors::data_corrupted("tokenizer.ggml.token_type has {} entries for {} tokens", types->array_size(),
                                          n);
        }
        for (const std::int64_t type : *types->as_ints()) {
            t->types.push_back(type >= 1 && type <= 6 ? static_cast<TokenType>(type) : TokenType::kNormal);
        }
    }
    if (const gguf::Value* merges = file.find("tokenizer.ggml.merges"); merges != nullptr && merges->as_strings()) {
        int rank = 0;
        for (const std::string& merge : *merges->as_strings()) {
            const std::size_t space = merge.find(' ', 1);
            if (space == std::string::npos) {
                return errors::data_corrupted("bad tokenizer merge '{}'", merge);
            }
            t->merge_ranks.emplace(std::make_pair(merge.substr(0, space), merge.substr(space + 1)), rank++);
        }
    } else if (t->algorithm == Algorithm::kBpeBytes) {
        return errors::data_corrupted("BPE tokenizer without tokenizer.ggml.merges");
    }
    auto id = [&](const char* key) -> Token {
        const std::optional<std::int64_t> v = file.get_int(key);
        return v && *v >= 0 && static_cast<std::size_t>(*v) < n ? static_cast<Token>(*v) : -1;
    };
    t->bos = id("tokenizer.ggml.bos_token_id");
    t->eos = id("tokenizer.ggml.eos_token_id");
    t->unknown = id("tokenizer.ggml.unknown_token_id");
    if (const gguf::Value* v = file.find("tokenizer.ggml.add_bos_token")) {
        t->add_bos = v->as_bool().value_or(t->add_bos);
    }
    if (const gguf::Value* v = file.find("tokenizer.ggml.add_eos_token")) {
        t->add_eos = v->as_bool().value_or(false);
    }
    if (const gguf::Value* v = file.find("tokenizer.ggml.add_space_prefix")) {
        t->add_space_prefix = v->as_bool().value_or(t->add_space_prefix);
    }
    t->finalize();
    for (const char* key : {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
        if (const Token token = id(key); token >= 0) {
            t->end_of_generation.push_back(token);
        }
    }
    return t;
}

// ---------------------------------------------------------------------------------------------------------------
// Hugging Face tokenizer.json

namespace {

/// "<s>" or {"content": "<s>", ...}.
std::string token_content(const Json* value) {
    if (value == nullptr) {
        return {};
    }
    if (value->is_string()) {
        return value->as_string();
    }
    if (const Json* content = value->find("content"); content != nullptr && content->is_string()) {
        return content->as_string();
    }
    return {};
}

/// Searches a pre-tokenizer / normalizer tree for a component of `type` (Sequence nodes are walked).
const Json* find_component(const Json* node, std::string_view type) {
    if (node == nullptr || !node->is_object()) {
        return nullptr;
    }
    if (const Json* t = node->find("type"); t != nullptr && t->as_string() == type) {
        return node;
    }
    for (const char* list : {"pretokenizers", "normalizers"}) {
        if (const Json* items = node->find(list); items != nullptr && items->is_array()) {
            for (const Json& item : items->as_array()) {
                if (const Json* found = find_component(&item, type)) {
                    return found;
                }
            }
        }
    }
    return nullptr;
}

}  // namespace

Result<std::shared_ptr<TokenizerImpl>> tokenizer_from_hf(const Json& tokenizer, const Json* config,
                                                         const Json* generation_config) {
    auto t = std::make_shared<TokenizerImpl>();
    const Json* model = tokenizer.find("model");
    if (model == nullptr || !model->is_object()) {
        return errors::data_corrupted("tokenizer.json without a model");
    }
    const Json* type = model->find("type");
    if (type == nullptr || type->as_string() != "BPE") {
        return errors::unsupported("tokenizer model '{}' is not supported (BPE is)",
                                   type != nullptr ? type->as_string() : std::string("?"));
    }
    const Json* vocab = model->find("vocab");
    if (vocab == nullptr || !vocab->is_object()) {
        return errors::data_corrupted("tokenizer.json without a vocabulary");
    }
    std::int64_t size = 0;
    for (const auto& [text, value] : vocab->as_object()) {
        size = std::max<std::int64_t>(size, value.as_int().value_or(-1) + 1);
    }
    const Json* added = tokenizer.find("added_tokens");
    if (added != nullptr && added->is_array()) {
        for (const Json& token : added->as_array()) {
            const Json* id = token.find("id");
            size = std::max<std::int64_t>(size, id != nullptr ? id->as_int().value_or(-1) + 1 : 0);
        }
    }
    if (size <= 0 || size > (1 << 24)) {
        return errors::data_corrupted("tokenizer.json vocabulary of {} tokens", size);
    }
    t->texts.assign(static_cast<std::size_t>(size), std::string());
    t->types.assign(static_cast<std::size_t>(size), TokenType::kUnused);
    for (const auto& [text, value] : vocab->as_object()) {
        const std::optional<std::int64_t> id = value.as_int();
        if (id && *id >= 0) {
            t->texts[static_cast<std::size_t>(*id)] = text;
            t->types[static_cast<std::size_t>(*id)] = TokenType::kNormal;
        }
    }
    if (added != nullptr && added->is_array()) {
        for (const Json& token : added->as_array()) {
            const std::optional<std::int64_t> id =
                token.find("id") != nullptr ? token.find("id")->as_int() : std::nullopt;
            if (!id || *id < 0) {
                continue;
            }
            t->texts[static_cast<std::size_t>(*id)] = token_content(&token);
            const Json* special = token.find("special");
            t->types[static_cast<std::size_t>(*id)] =
                special != nullptr && special->as_bool() ? TokenType::kControl : TokenType::kUserDefined;
        }
    }
    const Json* merges = model->find("merges");
    if (merges != nullptr && merges->is_array()) {
        int rank = 0;
        for (const Json& merge : merges->as_array()) {
            if (merge.is_string()) {
                const std::string& text = merge.as_string();
                const std::size_t space = text.find(' ', 1);
                if (space == std::string::npos) {
                    return errors::data_corrupted("bad tokenizer merge '{}'", text);
                }
                t->merge_ranks.emplace(std::make_pair(text.substr(0, space), text.substr(space + 1)), rank++);
            } else if (merge.is_array() && merge.size() == 2) {
                t->merge_ranks.emplace(std::make_pair(merge.as_array()[0].as_string(), merge.as_array()[1].as_string()),
                                       rank++);
            }
        }
    }
    if (const Json* ignore = model->find("ignore_merges")) {
        t->ignore_merges = ignore->as_bool();
    }
    const Json* pre = tokenizer.find("pre_tokenizer");
    const Json* normalizer = tokenizer.find("normalizer");
    if (find_component(pre, "ByteLevel") != nullptr) {
        t->algorithm = Algorithm::kBpeBytes;
        t->add_space_prefix = false;
        t->split = SplitPattern::kGpt2;
        if (const Json* split = find_component(pre, "Split")) {
            const Json* pattern = split->find("pattern");
            const Json* regex = pattern != nullptr ? pattern->find("Regex") : nullptr;
            const std::string text = regex != nullptr ? regex->as_string() : std::string();
            if (text.find("\\p{N}{1,3}") != std::string::npos) {
                t->split = SplitPattern::kLlama3;
            } else if (text.find("[^\\r\\n\\p{L}\\p{N}]?\\p{L}+") != std::string::npos) {
                t->split = SplitPattern::kQwen2;
            }
        }
    } else {
        t->algorithm = Algorithm::kBpeChars;
        t->add_space_prefix = false;
        if (const Json* prepend = find_component(normalizer, "Prepend")) {
            t->add_space_prefix = prepend->find("prepend") != nullptr;
        }
        if (const Json* metaspace = find_component(pre, "Metaspace")) {
            const Json* scheme = metaspace->find("prepend_scheme");
            const Json* add_prefix = metaspace->find("add_prefix_space");
            t->add_space_prefix = (scheme != nullptr && scheme->as_string() != "never") ||
                                  (add_prefix != nullptr && add_prefix->as_bool());
        }
    }

    auto find_id = [&](const std::string& text) -> Token {
        if (text.empty()) {
            return -1;
        }
        for (std::size_t i = 0; i < t->texts.size(); ++i) {
            if (t->texts[i] == text) {
                return static_cast<Token>(i);
            }
        }
        return -1;
    };
    if (config != nullptr) {
        t->bos = find_id(token_content(config->find("bos_token")));
        t->eos = find_id(token_content(config->find("eos_token")));
        t->unknown = find_id(token_content(config->find("unk_token")));
    }
    // add_bos: tokenizer_config.json's flag, else whether the post-processor inserts the BOS token.
    bool add_bos_known = false;
    if (config != nullptr) {
        if (const Json* flag = config->find("add_bos_token"); flag != nullptr && flag->is_bool()) {
            t->add_bos = flag->as_bool();
            add_bos_known = true;
        }
        if (const Json* flag = config->find("add_eos_token"); flag != nullptr && flag->is_bool()) {
            t->add_eos = flag->as_bool();
        }
    }
    if (!add_bos_known && t->bos >= 0) {
        if (const Json* processor = find_component(tokenizer.find("post_processor"), "TemplateProcessing")) {
            if (const Json* single = processor->find("single"); single != nullptr && single->is_array()) {
                for (const Json& item : single->as_array()) {
                    const Json* special = item.find("SpecialToken");
                    if (special != nullptr && find_id(token_content(special->find("id"))) == t->bos) {
                        t->add_bos = true;
                    }
                }
            }
        }
    }
    t->finalize();
    if (generation_config != nullptr) {
        if (const Json* eos = generation_config->find("eos_token_id")) {
            auto add = [&](const Json& value) {
                const std::optional<std::int64_t> id = value.as_int();
                if (id && *id >= 0 && *id < size) {
                    t->end_of_generation.push_back(static_cast<Token>(*id));
                }
            };
            if (eos->is_array()) {
                for (const Json& item : eos->as_array()) {
                    add(item);
                }
            } else {
                add(*eos);
            }
        }
    }
    return t;
}

}  // namespace ii::llm::detail
