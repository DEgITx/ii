// SPDX-License-Identifier: MIT
#include <ii/llm.h>

#include "core/json.h"
#include "formats/safetensors.h"
#include "llm/graph.h"
#include "llm/hparams.h"
#include "llm/jinja.h"
#include "llm/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <optional>
#include <sstream>

namespace ii::llm {

namespace detail {

struct LlmImpl {
    Context context;
    ModelInfo info;
    Tokenizer tokenizer;
    std::shared_ptr<const TokenizerImpl> tokens;
    std::string chat_template;
    LoadOptions options;
    std::optional<Program> program;
};

struct SequenceImpl {
    std::shared_ptr<const LlmImpl> llm;
    std::optional<Session> session;
    std::vector<Token> tokens;
    std::vector<float> logits;
    std::vector<std::int32_t> batch;
    Tensor pos;
};

struct ChatImpl {
    std::shared_ptr<const LlmImpl> llm;
    std::optional<Sequence> sequence;
    Sampler sampler;
    ChatOptions options;
    std::vector<Message> messages;
    GenerationStats stats;
};

}  // namespace detail

namespace {

using detail::LlmImpl;

Result<std::string> read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return errors::not_found("cannot open '{}'", path.string());
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

Result<Json> read_json(const std::filesystem::path& path) {
    II_ASSIGN_OR_RETURN(const std::string text, read_text(path));
    Result<Json> json = Json::parse(text);
    if (!json.ok()) {
        return Status(json.status()).add_context(path.string());
    }
    return json;
}

/// The chat template of a Hugging Face model: tokenizer_config.json's chat_template (a string or a list of named
/// templates), else chat_template.jinja.
std::string hf_chat_template(const std::filesystem::path& dir, const Json* config) {
    if (config != nullptr) {
        if (const Json* t = config->find("chat_template")) {
            if (t->is_string()) {
                return t->as_string();
            }
            if (t->is_array()) {
                for (const Json& item : t->as_array()) {
                    const Json* name = item.find("name");
                    const Json* text = item.find("template");
                    if (name != nullptr && name->as_string() == "default" && text != nullptr) {
                        return text->as_string();
                    }
                }
                if (t->size() > 0 && t->as_array()[0].find("template") != nullptr) {
                    return t->as_array()[0].find("template")->as_string();
                }
            }
        }
    }
    const Result<std::string> file = read_text(dir / "chat_template.jinja");
    return file.ok() ? *file : std::string();
}

/// A generic template for models without one (ChatML).
constexpr std::string_view kChatMl =
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + "
    "'\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Tokenizer

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(const Tokenizer&) = default;
Tokenizer& Tokenizer::operator=(const Tokenizer&) = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<Token> Tokenizer::encode(std::string_view text, bool add_special, bool parse_special) const {
    return impl_ ? impl_->encode(text, add_special, parse_special) : std::vector<Token>();
}

std::string Tokenizer::decode(std::span<const Token> tokens, bool render_special) const {
    return impl_ ? impl_->decode(tokens, render_special) : std::string();
}

std::string Tokenizer::token_text(Token token) const {
    return impl_ ? impl_->piece(token, true) : std::string();
}

std::int64_t Tokenizer::vocab_size() const noexcept {
    return impl_ ? static_cast<std::int64_t>(impl_->texts.size()) : 0;
}

Token Tokenizer::bos() const noexcept {
    return impl_ ? impl_->bos : -1;
}

Token Tokenizer::eos() const noexcept {
    return impl_ ? impl_->eos : -1;
}

bool Tokenizer::is_end_of_generation(Token token) const noexcept {
    return impl_ && std::binary_search(impl_->end_of_generation.begin(), impl_->end_of_generation.end(), token);
}

std::string_view Tokenizer::kind() const noexcept {
    if (!impl_) {
        return "";
    }
    return impl_->algorithm == detail::Algorithm::kBpeBytes ? "bpe" : "spm";
}

// ---------------------------------------------------------------------------------------------------------------
// Llm

Llm::Llm() = default;
Llm::~Llm() = default;
Llm::Llm(const Llm&) = default;
Llm& Llm::operator=(const Llm&) = default;
Llm::Llm(Llm&&) noexcept = default;
Llm& Llm::operator=(Llm&&) noexcept = default;
Llm::Llm(std::shared_ptr<const detail::LlmImpl> impl) noexcept : impl_(std::move(impl)) {}

Result<Llm> Llm::load(const Context& context, const std::filesystem::path& path, const LoadOptions& options) {
    auto impl = std::make_shared<LlmImpl>(LlmImpl{context, {}, {}, {}, {}, options, std::nullopt});
    detail::HParams hparams;
    std::unique_ptr<detail::WeightSource> weights;
    std::shared_ptr<detail::TokenizerImpl> tokenizer;
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) {
        II_ASSIGN_OR_RETURN(const Json config, read_json(path / "config.json"));
        II_ASSIGN_OR_RETURN(hparams, detail::hparams_from_config(config));
        II_ASSIGN_OR_RETURN(formats::SafetensorsModel model, formats::SafetensorsModel::open(path));
        weights = detail::safetensors_weights(std::move(model), hparams);
        II_ASSIGN_OR_RETURN(const Json tokenizer_json, read_json(path / "tokenizer.json"));
        std::optional<Json> tokenizer_config;
        std::optional<Json> generation_config;
        if (Result<Json> j = read_json(path / "tokenizer_config.json"); j.ok()) {
            tokenizer_config = std::move(*j);
        }
        if (Result<Json> j = read_json(path / "generation_config.json"); j.ok()) {
            generation_config = std::move(*j);
        }
        II_ASSIGN_OR_RETURN(tokenizer,
                            detail::tokenizer_from_hf(tokenizer_json, tokenizer_config ? &*tokenizer_config : nullptr,
                                                      generation_config ? &*generation_config : nullptr));
        impl->chat_template = hf_chat_template(path, tokenizer_config ? &*tokenizer_config : nullptr);
        impl->info.format = "safetensors";
        if (const Json* name = config.find("_name_or_path"); name != nullptr && name->is_string()) {
            impl->info.name = name->as_string();
        }
    } else {
        II_ASSIGN_OR_RETURN(gguf::File file, gguf::File::open(path));
        II_ASSIGN_OR_RETURN(hparams, detail::hparams_from_gguf(file));
        II_ASSIGN_OR_RETURN(tokenizer, detail::tokenizer_from_gguf(file));
        impl->chat_template = file.get_string("tokenizer.chat_template").value_or("");
        impl->info.format = "gguf";
        impl->info.name = hparams.name;
        weights = detail::gguf_weights(std::move(file));
    }
    if (static_cast<std::int64_t>(tokenizer->texts.size()) > hparams.vocab_size && hparams.vocab_size > 0) {
        // Some Hugging Face tokenizers list a few more tokens than the embedding has rows; ignore the extra ones.
        tokenizer->texts.resize(static_cast<std::size_t>(hparams.vocab_size));
        tokenizer->types.resize(static_cast<std::size_t>(hparams.vocab_size));
        tokenizer->scores.resize(static_cast<std::size_t>(hparams.vocab_size));
        tokenizer->finalize();
    }
    impl->tokens = tokenizer;
    impl->tokenizer = Tokenizer(tokenizer);

    detail::GraphOptions graph;
    graph.context =
        options.context_length > 0 ? options.context_length : std::min<std::int64_t>(hparams.trained_context, 4096);
    graph.cache_dtype = options.kv_cache_dtype;
    graph.all_logits = options.all_logits;
    if (graph.context <= 0) {
        return errors::invalid_argument("invalid context length {}", graph.context);
    }
    detail::WeightStats stats;
    II_ASSIGN_OR_RETURN(const Model model, detail::build_model(context, hparams, *weights, graph, &stats));
    II_ASSIGN_OR_RETURN(Program program, model.compile(options.compile));
    impl->program = std::move(program);

    ModelInfo& info = impl->info;
    info.architecture = hparams.architecture;
    info.vocab_size = hparams.vocab_size;
    info.embedding_size = hparams.embedding;
    info.layers = hparams.layers;
    info.heads = hparams.heads;
    info.kv_heads = hparams.kv_heads;
    info.head_size = hparams.head_size;
    info.feed_forward_size = hparams.feed_forward;
    info.trained_context = hparams.trained_context;
    info.context_length = graph.context;
    info.parameters = stats.parameters;
    info.weight_bytes = stats.bytes;
    info.weight_type = stats.main_dtype;
    return Llm(std::move(impl));
}

const ModelInfo& Llm::info() const noexcept {
    return impl_->info;
}

const Tokenizer& Llm::tokenizer() const noexcept {
    return impl_->tokenizer;
}

const Program& Llm::program() const noexcept {
    return *impl_->program;
}

const std::string& Llm::chat_template() const noexcept {
    return impl_->chat_template;
}

Result<std::string> Llm::format_chat(std::span<const Message> messages, bool add_generation_prompt) const {
    using detail::jinja::Array;
    using detail::jinja::Object;
    using detail::jinja::Value;
    Array list;
    for (const Message& m : messages) {
        list.emplace_back(Object{{"role", Value(m.role)}, {"content", Value(m.content)}});
    }
    const detail::TokenizerImpl& t = *impl_->tokens;
    auto text_of = [&](Token token) { return token >= 0 ? t.texts[static_cast<std::size_t>(token)] : std::string(); };
    Object context{
        {"messages", Value(std::move(list))},
        {"add_generation_prompt", Value(add_generation_prompt)},
        {"bos_token", Value(text_of(t.bos))},
        {"eos_token", Value(text_of(t.eos))},
        {"tools", Value(nullptr)},
    };
    const std::string_view source = impl_->chat_template.empty() ? kChatMl : std::string_view(impl_->chat_template);
    return detail::jinja::render(source, Value(std::move(context)));
}

Result<Sequence> Llm::new_sequence() const {
    auto impl = std::make_unique<detail::SequenceImpl>();
    impl->llm = impl_;
    II_ASSIGN_OR_RETURN(Session session, impl_->program->new_session());
    impl->session = std::move(session);
    II_ASSIGN_OR_RETURN(impl->pos, Tensor::scalar<std::int64_t>(0));
    return Sequence(std::move(impl));
}

Result<Chat> Llm::new_chat(const ChatOptions& options) const {
    auto impl =
        std::make_unique<detail::ChatImpl>(detail::ChatImpl{impl_, std::nullopt, Sampler(options), options, {}, {}});
    II_ASSIGN_OR_RETURN(Sequence sequence, new_sequence());
    impl->sequence = std::move(sequence);
    if (!options.system_prompt.empty()) {
        impl->messages.push_back(Message{"system", options.system_prompt});
    }
    return Chat(std::move(impl));
}

// ---------------------------------------------------------------------------------------------------------------
// Sequence

Sequence::Sequence(std::unique_ptr<detail::SequenceImpl> impl) noexcept : impl_(std::move(impl)) {}
Sequence::Sequence(Sequence&&) noexcept = default;
Sequence& Sequence::operator=(Sequence&&) noexcept = default;
Sequence::~Sequence() = default;

std::int64_t Sequence::position() const noexcept {
    return static_cast<std::int64_t>(impl_->tokens.size());
}

std::int64_t Sequence::capacity() const noexcept {
    return impl_->llm->info.context_length;
}

std::span<const Token> Sequence::tokens() const noexcept {
    return impl_->tokens;
}

Result<std::span<const float>> Sequence::eval(std::span<const Token> tokens, std::int64_t batch) {
    detail::SequenceImpl& s = *impl_;
    if (tokens.empty()) {
        return errors::invalid_argument("no tokens to evaluate");
    }
    if (position() + static_cast<std::int64_t>(tokens.size()) > capacity()) {
        return errors::out_of_range("the context is full: {} + {} tokens exceed {}", position(), tokens.size(),
                                    capacity());
    }
    const std::int64_t vocab = s.llm->info.vocab_size;
    for (const Token token : tokens) {
        if (token < 0 || token >= vocab) {
            return errors::invalid_argument("token {} is outside the vocabulary of {}", token, vocab);
        }
    }
    const bool all = s.llm->options.all_logits;
    if (batch <= 0) {
        batch = 512;
    }
    s.logits.clear();
    for (std::size_t offset = 0; offset < tokens.size(); offset += static_cast<std::size_t>(batch)) {
        const std::size_t count = std::min(static_cast<std::size_t>(batch), tokens.size() - offset);
        s.batch.assign(tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                       tokens.begin() + static_cast<std::ptrdiff_t>(offset + count));
        s.pos.data_as<std::int64_t>()[0] = position();
        II_TRY(s.session->set_input(
            "tokens", Tensor::wrap(s.batch.data(), DType::kI32, Shape{1, static_cast<std::int64_t>(count)})));
        II_TRY(s.session->set_input("pos", s.pos));
        II_TRY(s.session->run());
        II_ASSIGN_OR_RETURN(const Tensor logits, s.session->output("logits"));
        const auto values = logits.values<float>();
        if (all) {
            s.logits.insert(s.logits.end(), values.begin(), values.end());
        } else {
            s.logits.assign(values.end() - vocab, values.end());
        }
        s.tokens.insert(s.tokens.end(), s.batch.begin(), s.batch.end());
    }
    return std::span<const float>(s.logits);
}

Session& Sequence::session() noexcept {
    return *impl_->session;
}

void Sequence::rewind(std::int64_t position) {
    if (position >= 0 && position < this->position()) {
        impl_->tokens.resize(static_cast<std::size_t>(position));
    }
}

Result<GenerationStats> Sequence::generate(std::span<const Token> prompt, Sampler& sampler, std::int64_t max_tokens,
                                           const PieceCallback& on_piece) {
    using Clock = std::chrono::steady_clock;
    GenerationStats stats;
    const detail::TokenizerImpl& tokenizer = *impl_->llm->tokens;
    const std::int64_t vocab = impl_->llm->info.vocab_size;
    const Clock::time_point start = Clock::now();
    II_ASSIGN_OR_RETURN(std::span<const float> logits, eval(prompt));
    if (impl_->llm->options.all_logits) {
        logits = logits.subspan(logits.size() - static_cast<std::size_t>(vocab));
    }
    stats.prompt_tokens = static_cast<std::int64_t>(prompt.size());
    const Clock::time_point generation_start = Clock::now();
    stats.prompt_seconds = std::chrono::duration<double>(generation_start - start).count();
    std::string pending;
    bool stop = false;
    while (!stop && (max_tokens < 0 || stats.generated_tokens < max_tokens)) {
        const Token token = sampler.sample(logits, impl_->tokens);
        if (std::binary_search(tokenizer.end_of_generation.begin(), tokenizer.end_of_generation.end(), token)) {
            stats.stopped_by_end_token = true;
            break;
        }
        ++stats.generated_tokens;
        pending += tokenizer.piece(token, false);
        const std::size_t keep = detail::incomplete_utf8_suffix(pending);
        if (on_piece && pending.size() > keep) {
            stop = !on_piece(std::string_view(pending).substr(0, pending.size() - keep));
        }
        pending.erase(0, pending.size() - keep);
        if (stop || position() >= capacity()) {
            break;
        }
        const Token next[] = {token};
        II_ASSIGN_OR_RETURN(logits, eval(next));
        if (impl_->llm->options.all_logits) {
            logits = logits.subspan(logits.size() - static_cast<std::size_t>(vocab));
        }
    }
    if (on_piece && !pending.empty() && !stop) {
        on_piece(pending);
    }
    stats.generation_seconds = std::chrono::duration<double>(Clock::now() - generation_start).count();
    return stats;
}

// ---------------------------------------------------------------------------------------------------------------
// Chat

Chat::Chat(std::unique_ptr<detail::ChatImpl> impl) noexcept : impl_(std::move(impl)) {}
Chat::Chat(Chat&&) noexcept = default;
Chat& Chat::operator=(Chat&&) noexcept = default;
Chat::~Chat() = default;

Result<std::string> Chat::send(std::string_view message, const PieceCallback& on_piece) {
    detail::ChatImpl& c = *impl_;
    c.messages.push_back(Message{"user", std::string(message)});
    const Llm llm(c.llm);
    Result<std::string> text = llm.format_chat(c.messages, true);
    if (!text.ok()) {
        c.messages.pop_back();
        return text.status();
    }
    std::vector<Token> tokens = c.llm->tokenizer.encode(*text, true, true);
    const Token bos = c.llm->tokenizer.bos();
    if (tokens.size() >= 2 && tokens[0] == bos && tokens[1] == bos) {
        tokens.erase(tokens.begin());  // the template already starts with the BOS token
    }
    // Reuse the cached prefix; at least one token is evaluated to get logits.
    Sequence& sequence = *c.sequence;
    const std::span<const Token> cached = sequence.tokens();
    std::size_t common = 0;
    while (common < cached.size() && common + 1 < tokens.size() && cached[common] == tokens[common]) {
        ++common;
    }
    sequence.rewind(static_cast<std::int64_t>(common));
    if (static_cast<std::int64_t>(tokens.size()) >= sequence.capacity()) {
        c.messages.pop_back();
        return errors::out_of_range("the conversation ({} tokens) does not fit into the context of {} tokens",
                                    tokens.size(), sequence.capacity());
    }
    std::string reply;
    const std::int64_t room = sequence.capacity() - static_cast<std::int64_t>(tokens.size());
    const std::int64_t limit = c.options.max_tokens < 0 ? room : std::min(c.options.max_tokens, room);
    Result<GenerationStats> stats = sequence.generate(std::span<const Token>(tokens).subspan(common), c.sampler, limit,
                                                      [&](std::string_view piece) {
                                                          reply += piece;
                                                          return !on_piece || on_piece(piece);
                                                      });
    if (!stats.ok()) {
        c.messages.pop_back();
        return stats.status();
    }
    c.stats = *stats;
    c.messages.push_back(Message{"assistant", reply});
    return reply;
}

const std::vector<Message>& Chat::messages() const noexcept {
    return impl_->messages;
}

const GenerationStats& Chat::last_stats() const noexcept {
    return impl_->stats;
}

void Chat::reset() {
    impl_->messages.clear();
    if (!impl_->options.system_prompt.empty()) {
        impl_->messages.push_back(Message{"system", impl_->options.system_prompt});
    }
    impl_->sequence->rewind(0);
}

}  // namespace ii::llm
