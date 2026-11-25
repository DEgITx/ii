// SPDX-License-Identifier: MIT
/// @file
/// Large language models: loading (GGUF, Hugging Face safetensors), tokenization, chat templates, sampling and
/// generation.
///
/// @code
/// auto context = ii::Context::create().value();
/// auto llm  = ii::llm::Llm::load(context, "qwen2.5-0.5b-instruct-q4_k_m.gguf").value();
/// auto chat = llm.new_chat({.temperature = 0.7f, .top_p = 0.9f}).value();
/// chat.send("Hi! Who are you?", [](std::string_view piece) { std::cout << piece << std::flush; return true; });
/// @endcode
///
/// A model is an ordinary ii graph built from the architecture's hyper-parameters (llama, qwen2, qwen3, gemma,
/// gemma2, gemma3, phi3 and models with their layout); it is compiled once into a `Program` shared by all
/// sequences. Each `Sequence` owns a session whose KV cache is a preallocated state of `context_length` tokens:
/// prompt processing (many tokens per step) and decoding (one token per step) run the same program.
#pragma once

#include <ii/context.h>
#include <ii/export.h>
#include <ii/model.h>
#include <ii/status.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

II_BEGIN_DLL_EXPORTS

namespace ii::llm {

namespace detail {
struct LlmImpl;
struct TokenizerImpl;
struct SequenceImpl;
struct ChatImpl;
}  // namespace detail

/// Token id.
using Token = std::int32_t;

/// Architecture and size of a loaded model.
struct ModelInfo {
    std::string architecture;  ///< as in GGUF general.architecture / normalized from config.json ("llama", "qwen2")
    std::string name;          ///< general.name, if known
    std::string format;        ///< "gguf" or "safetensors"
    std::int64_t vocab_size = 0;
    std::int64_t embedding_size = 0;
    std::int64_t layers = 0;
    std::int64_t heads = 0;
    std::int64_t kv_heads = 0;
    std::int64_t head_size = 0;
    std::int64_t feed_forward_size = 0;
    std::int64_t trained_context = 0;  ///< context length the model was trained for
    std::int64_t context_length = 0;   ///< KV cache capacity of this instance
    std::int64_t parameters = 0;       ///< number of weights
    std::size_t weight_bytes = 0;      ///< bytes of all weights as stored
    std::string weight_type;           ///< dtype holding most weights, e.g. "q4_k", "bf16"
};

/// Options of `Llm::load`.
struct LoadOptions {
    /// KV cache capacity in tokens (prompt plus generated text). 0: the trained context length, at most 4096.
    std::int64_t context_length = 0;
    /// Element type of the KV cache: f32 (default) or f16 (half the memory).
    DType kv_cache_dtype = DType::kF32;
    /// Logits for every position of a step instead of the last one only (perplexity evaluation).
    bool all_logits = false;
    /// Backends and compiler options.
    CompileOptions compile;
};

/// A tokenizer (SentencePiece or byte-level BPE) with the model's special tokens.
class II_API Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(const Tokenizer&);
    Tokenizer& operator=(const Tokenizer&);
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    /// Splits text into tokens. `add_special` prepends BOS (and appends EOS) when the model asks for it;
    /// `parse_special` recognizes special tokens written in the text (e.g. "<|im_start|>").
    std::vector<Token> encode(std::string_view text, bool add_special = true, bool parse_special = false) const;
    /// The text of a token sequence. Special tokens are skipped unless `render_special`.
    std::string decode(std::span<const Token> tokens, bool render_special = false) const;
    /// The text of one token (special tokens: their written form, e.g. "</s>").
    std::string token_text(Token token) const;

    std::int64_t vocab_size() const noexcept;
    Token bos() const noexcept;  ///< -1 if none
    Token eos() const noexcept;  ///< -1 if none
    /// True for tokens that end generation (EOS, end-of-turn tokens such as <|im_end|> or <end_of_turn>).
    bool is_end_of_generation(Token token) const noexcept;
    /// "spm" (SentencePiece) or "bpe" (byte-level BPE).
    std::string_view kind() const noexcept;

    /// Implementation access (used by the loaders).
    explicit Tokenizer(std::shared_ptr<const detail::TokenizerImpl> impl) noexcept : impl_(std::move(impl)) {}

private:
    std::shared_ptr<const detail::TokenizerImpl> impl_;
};

/// A chat message.
struct Message {
    std::string role;  ///< "system", "user", "assistant"
    std::string content;
};

/// Sampling parameters. The chain is: repetition penalty, top-k, temperature, softmax, top-p, min-p, random draw.
/// temperature <= 0 selects the most likely token (greedy decoding).
struct SamplingOptions {
    float temperature = 0.8f;
    std::int32_t top_k = 40;      ///< 0: disabled
    float top_p = 0.95f;          ///< 1: disabled
    float min_p = 0.05f;          ///< 0: disabled
    float repeat_penalty = 1.0f;  ///< 1: disabled
    std::int32_t repeat_last_n = 64;
    std::uint64_t seed = 0;  ///< 0: a random seed
};

/// Picks tokens from logits; deterministic for a fixed seed.
class II_API Sampler {
public:
    explicit Sampler(const SamplingOptions& options = {});
    /// The next token given the logits of the last position and the tokens so far (for the repetition penalty).
    Token sample(std::span<const float> logits, std::span<const Token> history = {});
    const SamplingOptions& options() const noexcept { return options_; }
    /// The token returned by the last call of `sample` (-1 before).
    Token last_token() const noexcept { return last_; }

private:
    Token sample_impl(std::span<const float> logits, std::span<const Token> history);

    SamplingOptions options_;
    Token last_ = -1;
    std::uint64_t state_[2] = {};
    std::vector<std::pair<float, Token>> candidates_;
};

class Sequence;
class Chat;

/// Chat options: sampling, the length limit of one reply and an optional system prompt.
struct ChatOptions : SamplingOptions {
    std::int64_t max_tokens = -1;  ///< per reply; -1: until the end of the context
    std::string system_prompt;
};

/// A loaded and compiled model. Copies share it; use from many threads.
class II_API Llm {
public:
    /// Loads a GGUF file, or a Hugging Face model directory (config.json, tokenizer.json, *.safetensors).
    static Result<Llm> load(const Context& context, const std::filesystem::path& path, const LoadOptions& options = {});

    const ModelInfo& info() const noexcept;
    const Tokenizer& tokenizer() const noexcept;
    /// The compiled program (inputs "tokens" i32[1, S] and "pos" i64[]; output "logits").
    const Program& program() const noexcept;
    /// The chat template (Jinja) of the model, possibly empty.
    const std::string& chat_template() const noexcept;

    /// Renders a conversation with the model's chat template.
    Result<std::string> format_chat(std::span<const Message> messages, bool add_generation_prompt = true) const;

    /// A new sequence with an empty KV cache.
    Result<Sequence> new_sequence() const;
    /// A new conversation.
    Result<Chat> new_chat(const ChatOptions& options = {}) const;

    Llm();
    ~Llm();
    Llm(const Llm&);
    Llm& operator=(const Llm&);
    Llm(Llm&&) noexcept;
    Llm& operator=(Llm&&) noexcept;
    explicit Llm(std::shared_ptr<const detail::LlmImpl> impl) noexcept;

private:
    std::shared_ptr<const detail::LlmImpl> impl_;
};

/// Statistics of a generation.
struct GenerationStats {
    std::int64_t prompt_tokens = 0;
    std::int64_t generated_tokens = 0;
    double prompt_seconds = 0;
    double generation_seconds = 0;
    bool stopped_by_end_token = false;
};

/// Called with every generated piece of text; return false to stop.
using PieceCallback = std::function<bool(std::string_view piece)>;

/// One token sequence and its KV cache. Use from one thread at a time.
class II_API Sequence {
public:
    /// Tokens in the cache.
    std::int64_t position() const noexcept;
    /// Cache capacity.
    std::int64_t capacity() const noexcept;
    /// The tokens in the cache.
    std::span<const Token> tokens() const noexcept;

    /// Appends tokens to the cache and returns the logits of the last one ([vocab]), or of every new token
    /// ([n, vocab], with LoadOptions::all_logits). Long inputs are processed in batches of `batch` tokens.
    Result<std::span<const float>> eval(std::span<const Token> tokens, std::int64_t batch = 512);

    /// Drops the tokens after `position` (e.g. to reuse a common prompt prefix).
    void rewind(std::int64_t position);

    /// The session running the model (e.g. to enable profiling: the last eval step is profiled).
    Session& session() noexcept;

    /// Evaluates `prompt` (already tokenized) and samples up to `max_tokens` tokens (-1: until the end of the
    /// context), stopping at an end-of-generation token. `on_piece` receives the text as it is generated.
    Result<GenerationStats> generate(std::span<const Token> prompt, Sampler& sampler, std::int64_t max_tokens,
                                     const PieceCallback& on_piece = {});

    Sequence(Sequence&&) noexcept;
    Sequence& operator=(Sequence&&) noexcept;
    ~Sequence();
    explicit Sequence(std::unique_ptr<detail::SequenceImpl> impl) noexcept;

private:
    std::unique_ptr<detail::SequenceImpl> impl_;
};

/// A conversation: keeps the messages, applies the chat template and reuses the KV cache between turns.
class II_API Chat {
public:
    /// Adds a user message, generates the reply (streamed to `on_piece`) and returns it.
    Result<std::string> send(std::string_view message, const PieceCallback& on_piece = {});
    const std::vector<Message>& messages() const noexcept;
    /// Statistics of the last reply.
    const GenerationStats& last_stats() const noexcept;
    /// Forgets the conversation (keeps the system prompt).
    void reset();

    Chat(Chat&&) noexcept;
    Chat& operator=(Chat&&) noexcept;
    ~Chat();
    explicit Chat(std::unique_ptr<detail::ChatImpl> impl) noexcept;

private:
    std::unique_ptr<detail::ChatImpl> impl_;
};

}  // namespace ii::llm

II_END_DLL_EXPORTS
