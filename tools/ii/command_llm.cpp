// SPDX-License-Identifier: MIT
// Language model commands: `ii generate`, `ii chat`, `ii bench --llm`, `ii perplexity`, `ii quantize` and
// `ii inspect` of GGUF files.
#include <ii/gguf.h>
#include <ii/ii.h>
#include <ii/llm.h>

#include "args.h"
#include "commands.h"
#include "console.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ii::cli {

namespace {

constexpr const char* kModelOptions = R"(Model:
  --ctx N             Context length (KV cache capacity) in tokens (default: the model's, at most 4096)
  --kv-type TYPE      KV cache element type: f32 (default) or f16
  --threads N         Threads (default: all logical CPUs)
  --backends LIST     Comma-separated backends in order of preference (default: all)
  --isa NAME          Kernel variant of ii-cpu: generic, avx2, avx512, neon (default: the best supported)
  --device-memory LIST  Memory limits per device, e.g. cuda:0=4G (the rest of the model runs on the CPU)
)";

constexpr const char* kSamplingOptions = R"(Sampling:
  --temp X            Temperature (default 0.8; 0: greedy)
  --top-k N           Keep the N most likely tokens (default 40; 0: off)
  --top-p X           Nucleus sampling (default 0.95; 1: off)
  --min-p X           Drop tokens below X times the top probability (default 0.05; 0: off)
  --repeat-penalty X  Penalty for tokens of the last --repeat-last-n (default 1: off)
  --repeat-last-n N   Default 64
  --seed N            Random seed (default: random)
)";

constexpr const char* kGenerateUsage = R"(Usage: ii generate <model> -p TEXT [options]

Generates text with a language model (a GGUF file or a Hugging Face model directory).

  -p, --prompt TEXT   The prompt
  -f, --file FILE     Read the prompt from FILE
  -n N                Tokens to generate (default 128; -1: until the end of the context or an end token)
  --chat              Wrap the prompt as a user message with the model's chat template
  --system TEXT       System message (with --chat)
  --raw               Do not add the BOS token
  --verbose           Print the prompt tokens
  --profile           Time every operation of the prompt step and of the last generated token
  --trace FILE        With --profile: also write both steps as a Chrome trace (https://ui.perfetto.dev)
  --save-logits FILE  Write the prompt tokens and the logits of every prompt position to an .npz archive
                      ("tokens" i32[n], "logits" f32[n, vocab]) and stop
  --save-tokens FILE  Write the prompt and generated tokens to an .npz archive ("prompt", "generated" i32)
)";

constexpr const char* kChatUsage = R"(Usage: ii chat <model> [options]

Interactive chat with a language model. Commands: /reset (new conversation), /exit (or Ctrl+D / Ctrl+Z).

  --system TEXT       System message
  -n N                Maximum tokens per reply (default: until an end token)
)";

constexpr const char* kBenchUsage = R"(Usage: ii bench <model> --llm [options]

Measures prompt processing (pp) and token generation (tg) speed of a language model, like llama-bench:
pp processes P random tokens in one step, tg generates N tokens one at a time.

  -p N                Prompt tokens (default 512; 0: skip)
  -n N                Generated tokens (default 128; 0: skip)
  -r N                Repetitions (default 3)
)";

constexpr const char* kPerplexityUsage = R"(Usage: ii perplexity <model> -f FILE [options]

Perplexity of a text, computed like llama.cpp's llama-perplexity: the text is split into chunks of --ctx tokens
(the first token of each chunk replaced by BOS) and the second half of every chunk is scored.

  -f, --file FILE     The text (e.g. wikitext-2 wiki.test.raw)
  --tokens FILE       Use these tokens (an .npy of i32 / i64) instead of tokenizing a text
  --chunks N          Score at most N chunks (default: all)
  --batch N           Tokens per step (default 512)
)";

constexpr const char* kQuantizeUsage = R"(Usage: ii quantize <input.gguf> <output.gguf> --type TYPE

Converts the weights of a GGUF model. Matrices are converted, vectors (norms, biases) stay f32.

  --type TYPE         f32, f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, q2_k, q3_k, q4_k, q5_k, q6_k, or the
                      llama.cpp mixes q4_k_m and q5_k_m (more bits for attn_v / ffn_down of some layers and q6_k
                      for the output matrix)
)";

const std::set<std::string, std::less<>> kModelOptionNames = {"ctx",      "kv-type", "threads",
                                                              "backends", "isa",     "device-memory"};
const std::set<std::string, std::less<>> kSamplingOptionNames = {"temp",           "top-k",         "top-p", "min-p",
                                                                 "repeat-penalty", "repeat-last-n", "seed"};

std::set<std::string, std::less<>> merge(std::set<std::string, std::less<>> a,
                                         const std::set<std::string, std::less<>>& b) {
    a.insert(b.begin(), b.end());
    return a;
}

std::filesystem::path to_path(const std::string& utf8) {
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

std::vector<std::string> split_list(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    return parts;
}

/// "-p" / "-f" / "-n" / "-r" are accepted as aliases of the long options.
std::vector<std::string> expand_short(const std::vector<std::string>& args) {
    static const std::map<std::string, std::string> kShort = {
        {"-p", "--prompt"}, {"-f", "--file"}, {"-n", "--n"}, {"-r", "--r"}};
    std::vector<std::string> out;
    for (const std::string& arg : args) {
        const auto it = kShort.find(arg);
        out.push_back(it != kShort.end() ? it->second : arg);
    }
    return out;
}

int fail(const Status& status) {
    std::fprintf(stderr, "ii: %s\n", status.to_string().c_str());
    return 1;
}

struct Loaded {
    Context context;
    llm::Llm llm;
};

Result<Loaded> load_llm(const Args& args, bool all_logits = false) {
    if (args.positionals().empty()) {
        return errors::invalid_argument("missing the model path");
    }
    ContextOptions context_options;
    II_ASSIGN_OR_RETURN(context_options.num_threads, args.int_option("threads", 0));
    context_options.cpu_isa = args.option("isa").value_or("");
    II_ASSIGN_OR_RETURN(Context context, Context::create(std::move(context_options)));
    llm::LoadOptions options;
    II_ASSIGN_OR_RETURN(const int ctx, args.int_option("ctx", 0));
    options.context_length = ctx;
    options.all_logits = all_logits;
    const std::string kv = args.option("kv-type").value_or("f32");
    if (kv != "f32" && kv != "f16") {
        return errors::invalid_argument("--kv-type must be f32 or f16");
    }
    options.kv_cache_dtype = kv == "f16" ? DType::kF16 : DType::kF32;
    if (const auto backends = args.option("backends")) {
        options.compile.backends = split_list(*backends);
    }
    II_ASSIGN_OR_RETURN(options.compile.device_memory, parse_device_memory(args.option("device-memory").value_or("")));
    const auto start = std::chrono::steady_clock::now();
    II_ASSIGN_OR_RETURN(llm::Llm llm, llm::Llm::load(context, to_path(args.positionals()[0]), options));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const llm::ModelInfo& info = llm.info();
    std::fprintf(stderr, "ii: %s %s (%s), %.2fB parameters, %.2f GiB %s, %lld layers, context %lld, loaded in %.2f s\n",
                 info.architecture.c_str(), info.name.c_str(), info.format.c_str(),
                 static_cast<double>(info.parameters) / 1e9,
                 static_cast<double>(info.weight_bytes) / (1024.0 * 1024.0 * 1024.0), info.weight_type.c_str(),
                 static_cast<long long>(info.layers), static_cast<long long>(info.context_length), seconds);
    return Loaded{std::move(context), std::move(llm)};
}

Result<llm::SamplingOptions> sampling_options(const Args& args) {
    llm::SamplingOptions s;
    II_ASSIGN_OR_RETURN(const double temp, args.double_option("temp", 0.8));
    II_ASSIGN_OR_RETURN(s.top_k, args.int_option("top-k", 40));
    II_ASSIGN_OR_RETURN(const double top_p, args.double_option("top-p", 0.95));
    II_ASSIGN_OR_RETURN(const double min_p, args.double_option("min-p", 0.05));
    II_ASSIGN_OR_RETURN(const double penalty, args.double_option("repeat-penalty", 1.0));
    II_ASSIGN_OR_RETURN(s.repeat_last_n, args.int_option("repeat-last-n", 64));
    II_ASSIGN_OR_RETURN(const int seed, args.int_option("seed", 0));
    s.temperature = static_cast<float>(temp);
    s.top_p = static_cast<float>(top_p);
    s.min_p = static_cast<float>(min_p);
    s.repeat_penalty = static_cast<float>(penalty);
    s.seed = static_cast<std::uint64_t>(seed);
    return s;
}

Result<std::string> read_file_text(const std::string& path) {
    std::ifstream file(to_path(path), std::ios::binary);
    if (!file) {
        return errors::not_found("cannot open '{}'", path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void print_speed(const llm::GenerationStats& stats) {
    const double pp = stats.prompt_seconds > 0 ? static_cast<double>(stats.prompt_tokens) / stats.prompt_seconds : 0;
    const double tg =
        stats.generation_seconds > 0 ? static_cast<double>(stats.generated_tokens) / stats.generation_seconds : 0;
    std::fprintf(
        stderr, "\nii: prompt %lld tokens in %.2f s (%.1f tokens/s), generated %lld tokens in %.2f s (%.1f tokens/s)\n",
        static_cast<long long>(stats.prompt_tokens), stats.prompt_seconds, pp,
        static_cast<long long>(stats.generated_tokens), stats.generation_seconds, tg);
}

bool print_piece(std::string_view piece) {
    std::fwrite(piece.data(), 1, piece.size(), stdout);
    std::fflush(stdout);
    return true;
}

}  // namespace

bool is_llm_path(const std::string& path) {
    std::error_code error;
    const std::filesystem::path p = to_path(path);
    if (std::filesystem::is_directory(p, error)) {
        return std::filesystem::exists(p / "config.json", error);
    }
    std::ifstream file(p, std::ios::binary);
    char magic[4] = {};
    file.read(magic, 4);
    return file && std::string_view(magic, 4) == "GGUF";
}

// ---------------------------------------------------------------------------------------------------------------

int run_generate(const std::vector<std::string>& raw) {
    const auto parsed = Args::parse(
        expand_short(raw), {"help", "chat", "raw", "verbose", "profile"},
        merge(merge({"prompt", "file", "n", "system", "save-logits", "save-tokens", "trace"}, kModelOptionNames),
              kSamplingOptionNames));
    if (!parsed.ok()) {
        return fail(parsed.status());
    }
    const Args& args = *parsed;
    if (args.flag("help") || args.positionals().empty()) {
        std::printf("%s\n%s\n%s", kGenerateUsage, kModelOptions, kSamplingOptions);
        return args.flag("help") ? 0 : 2;
    }
    std::string prompt = args.option("prompt").value_or("");
    if (const auto file = args.option("file")) {
        const auto text = read_file_text(*file);
        if (!text.ok()) {
            return fail(text.status());
        }
        prompt = *text;
    }
    const auto sampling = sampling_options(args);
    const auto n = args.int_option("n", 128);
    if (!sampling.ok() || !n.ok()) {
        return fail(!sampling.ok() ? sampling.status() : n.status());
    }
    auto loaded = load_llm(args, args.option("save-logits").has_value());
    if (!loaded.ok()) {
        return fail(loaded.status());
    }
    const llm::Llm& model = loaded->llm;
    std::vector<llm::Token> tokens;
    if (args.flag("chat")) {
        std::vector<llm::Message> messages;
        if (const auto system = args.option("system")) {
            messages.push_back({"system", *system});
        }
        messages.push_back({"user", prompt});
        const auto text = model.format_chat(messages);
        if (!text.ok()) {
            return fail(text.status());
        }
        tokens = model.tokenizer().encode(*text, !args.flag("raw"), true);
        if (tokens.size() >= 2 && tokens[0] == tokens[1] && tokens[0] == model.tokenizer().bos()) {
            tokens.erase(tokens.begin());
        }
    } else {
        tokens = model.tokenizer().encode(prompt, !args.flag("raw"), false);
        std::printf("%s", prompt.c_str());
    }
    if (tokens.empty()) {
        return fail(errors::invalid_argument("the prompt is empty"));
    }
    if (args.flag("verbose")) {
        std::fprintf(stderr, "ii: %zu prompt tokens:", tokens.size());
        for (const llm::Token t : tokens) {
            std::fprintf(stderr, " %d", t);
        }
        std::fprintf(stderr, "\n");
    }
    auto sequence = model.new_sequence();
    if (!sequence.ok()) {
        return fail(sequence.status());
    }
    if (const auto path = args.option("save-logits")) {
        const auto logits = sequence->eval(tokens);
        if (!logits.ok()) {
            return fail(logits.status());
        }
        const auto n_tokens = static_cast<std::int64_t>(tokens.size());
        std::vector<NamedTensor> arrays;
        arrays.emplace_back("tokens", Tensor::from(std::span<const llm::Token>(tokens), Shape{n_tokens}).value());
        arrays.emplace_back("logits", Tensor::from(*logits, Shape{n_tokens, model.info().vocab_size}).value());
        if (Status status = save_npz(to_path(*path), arrays); !status.ok()) {
            return fail(status);
        }
        std::printf("\nii: logits of %lld positions written to %s\n", static_cast<long long>(n_tokens), path->c_str());
        return 0;
    }
    llm::Sampler sampler(*sampling);
    std::vector<Profile> prompt_profile;
    if (args.flag("profile")) {
        sequence->session().set_profiling(true);
        if (tokens.size() > 1) {
            const auto prefix = sequence->eval(std::span<const llm::Token>(tokens).first(tokens.size() - 1));
            if (!prefix.ok()) {
                return fail(prefix.status());
            }
            prompt_profile.push_back(sequence->session().profile().value());
            tokens.erase(tokens.begin(), tokens.end() - 1);
        }
    }
    const std::size_t prompt_end = tokens.size() + static_cast<std::size_t>(sequence->position());
    const auto stats = sequence->generate(tokens, sampler, *n, print_piece);
    if (!stats.ok()) {
        return fail(stats.status());
    }
    std::printf("\n");
    print_speed(*stats);
    if (const auto path = args.option("save-tokens")) {
        // The cache holds the prompt and every generated token but the last one.
        std::vector<llm::Token> all(sequence->tokens().begin(), sequence->tokens().end());
        const std::vector<llm::Token> prompt_tokens(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(prompt_end));
        std::vector<llm::Token> generated(all.begin() + static_cast<std::ptrdiff_t>(prompt_end), all.end());
        if (static_cast<std::int64_t>(generated.size()) < stats->generated_tokens) {
            generated.push_back(sampler.last_token());
        }
        std::vector<NamedTensor> arrays;
        arrays.emplace_back("prompt", Tensor::from(std::span<const llm::Token>(prompt_tokens),
                                                   Shape{static_cast<std::int64_t>(prompt_tokens.size())})
                                          .value());
        arrays.emplace_back("generated", Tensor::from(std::span<const llm::Token>(generated),
                                                      Shape{static_cast<std::int64_t>(generated.size())})
                                             .value());
        if (Status status = save_npz(to_path(*path), arrays); !status.ok()) {
            return fail(status);
        }
    }
    if (args.flag("profile")) {
        if (!prompt_profile.empty()) {
            std::printf("\nPrompt step (%.2f ms):", prompt_profile[0].total_us / 1000.0);
            print_profile_summary(prompt_profile);
        }
        const auto last = sequence->session().profile();
        if (const auto trace = args.option("trace"); last.ok() && trace) {
            std::vector<Profile> runs = prompt_profile;
            runs.push_back(*last);
            std::ofstream(to_path(*trace), std::ios::binary) << to_chrome_trace(runs);
        }
        if (last.ok()) {
            std::printf("\nOne generated token (%.2f ms):", last->total_us / 1000.0);
            print_profile_summary({*last});
        }
    }
    return 0;
}

int run_chat(const std::vector<std::string>& raw) {
    const auto parsed = Args::parse(expand_short(raw), {"help"},
                                    merge(merge({"system", "n"}, kModelOptionNames), kSamplingOptionNames));
    if (!parsed.ok()) {
        return fail(parsed.status());
    }
    const Args& args = *parsed;
    if (args.flag("help") || args.positionals().empty()) {
        std::printf("%s\n%s\n%s", kChatUsage, kModelOptions, kSamplingOptions);
        return args.flag("help") ? 0 : 2;
    }
    const auto sampling = sampling_options(args);
    const auto n = args.int_option("n", -1);
    if (!sampling.ok() || !n.ok()) {
        return fail(!sampling.ok() ? sampling.status() : n.status());
    }
    auto loaded = load_llm(args);
    if (!loaded.ok()) {
        return fail(loaded.status());
    }
    llm::ChatOptions options;
    static_cast<llm::SamplingOptions&>(options) = *sampling;
    options.max_tokens = *n;
    options.system_prompt = args.option("system").value_or("");
    auto chat = loaded->llm.new_chat(options);
    if (!chat.ok()) {
        return fail(chat.status());
    }
    std::fprintf(stderr, "ii: chat started; /reset starts over, /exit quits\n");
    std::string line;
    while (true) {
        std::printf("\n> ");
        std::fflush(stdout);
        if (!read_line(line) || line == "/exit" || line == "/quit") {
            break;
        }
        if (line == "/reset") {
            chat->reset();
            std::printf("(new conversation)\n");
            continue;
        }
        if (line.empty()) {
            continue;
        }
        const auto reply = chat->send(line, print_piece);
        if (!reply.ok()) {
            std::fprintf(stderr, "\nii: %s\n", reply.status().to_string().c_str());
            continue;
        }
        std::printf("\n");
        const llm::GenerationStats& stats = chat->last_stats();
        std::fprintf(stderr, "[%lld + %lld tokens, %.1f tokens/s]\n", static_cast<long long>(stats.prompt_tokens),
                     static_cast<long long>(stats.generated_tokens),
                     stats.generation_seconds > 0
                         ? static_cast<double>(stats.generated_tokens) / stats.generation_seconds
                         : 0.0);
    }
    return 0;
}

int run_llm_bench(const std::vector<std::string>& raw) {
    const auto parsed = Args::parse(expand_short(raw), {"help", "llm"}, merge({"n", "r", "prompt"}, kModelOptionNames));
    if (!parsed.ok()) {
        return fail(parsed.status());
    }
    const Args& args = *parsed;
    if (args.flag("help") || args.positionals().empty()) {
        std::printf("%s\n%s", kBenchUsage, kModelOptions);
        return args.flag("help") ? 0 : 2;
    }
    // -p is "--prompt" after expand_short: the number of prompt tokens here.
    const auto p = args.int_option("prompt", 512);
    const auto n = args.int_option("n", 128);
    const auto repetitions = args.int_option("r", 3);
    if (!p.ok() || !n.ok() || !repetitions.ok()) {
        return fail(errors::invalid_argument("-p, -n and -r take integers"));
    }
    auto loaded = load_llm(args);
    if (!loaded.ok()) {
        return fail(loaded.status());
    }
    const llm::Llm& model = loaded->llm;
    if (*p + *n + 1 > model.info().context_length) {
        return fail(
            errors::invalid_argument("-p + -n exceed the context length {}; use --ctx", model.info().context_length));
    }
    auto sequence = model.new_sequence();
    if (!sequence.ok()) {
        return fail(sequence.status());
    }
    std::mt19937 rng(1);
    std::uniform_int_distribution<llm::Token> token(0, static_cast<llm::Token>(model.info().vocab_size - 1));
    using Clock = std::chrono::steady_clock;
    auto measure = [&](int count, bool prompt) -> Result<std::vector<double>> {
        std::vector<double> speeds;
        for (int r = 0; r < *repetitions + 1; ++r) {  // the first run warms up
            sequence->rewind(0);
            std::vector<llm::Token> tokens(static_cast<std::size_t>(prompt ? count : 1));
            for (llm::Token& t : tokens) {
                t = token(rng);
            }
            const Clock::time_point start = Clock::now();
            if (prompt) {
                II_TRY(sequence->eval(tokens, count).status());
            } else {
                for (int i = 0; i < count; ++i) {
                    tokens[0] = token(rng);
                    II_TRY(sequence->eval(tokens).status());
                }
            }
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            if (r > 0) {
                speeds.push_back(static_cast<double>(count) / seconds);
            }
        }
        return speeds;
    };
    const llm::ModelInfo& info = model.info();
    std::printf("| model | size | params | threads | test | t/s |\n|---|---:|---:|---:|---:|---:|\n");
    for (const auto& [count, prompt] : {std::pair{*p, true}, std::pair{*n, false}}) {
        if (count <= 0) {
            continue;
        }
        const auto speeds = measure(count, prompt);
        if (!speeds.ok()) {
            return fail(speeds.status());
        }
        const double mean = std::accumulate(speeds->begin(), speeds->end(), 0.0) / static_cast<double>(speeds->size());
        double variance = 0;
        for (const double s : *speeds) {
            variance += (s - mean) * (s - mean);
        }
        const double deviation = std::sqrt(variance / static_cast<double>(speeds->size()));
        std::printf("| %s %s | %.2f GiB | %.2f B | %d | %s%d | %.2f ± %.2f |\n", info.architecture.c_str(),
                    info.weight_type.c_str(), static_cast<double>(info.weight_bytes) / (1024.0 * 1024.0 * 1024.0),
                    static_cast<double>(info.parameters) / 1e9, loaded->context.executor().concurrency(),
                    prompt ? "pp" : "tg", count, mean, deviation);
        std::fflush(stdout);
    }
    return 0;
}

int run_perplexity(const std::vector<std::string>& raw) {
    const auto parsed =
        Args::parse(expand_short(raw), {"help"}, merge({"file", "chunks", "batch", "tokens"}, kModelOptionNames));
    if (!parsed.ok()) {
        return fail(parsed.status());
    }
    const Args& args = *parsed;
    if (args.flag("help") || args.positionals().empty() || (!args.option("file") && !args.option("tokens"))) {
        std::printf("%s\n%s", kPerplexityUsage, kModelOptions);
        return args.flag("help") ? 0 : 2;
    }
    const auto text = args.option("file") ? read_file_text(*args.option("file")) : Result<std::string>(std::string());
    const auto chunks = args.int_option("chunks", -1);
    const auto batch = args.int_option("batch", 512);
    if (!text.ok() || !chunks.ok() || !batch.ok()) {
        return fail(!text.ok() ? text.status() : errors::invalid_argument("--chunks and --batch take integers"));
    }
    std::vector<std::string> with_ctx = raw;
    if (!args.option("ctx")) {
        with_ctx.push_back("--ctx");
        with_ctx.push_back("512");
    }
    const auto reparsed =
        Args::parse(expand_short(with_ctx), {"help"}, merge({"file", "chunks", "batch", "tokens"}, kModelOptionNames));
    auto loaded = load_llm(*reparsed, true);
    if (!loaded.ok()) {
        return fail(loaded.status());
    }
    const llm::Llm& model = loaded->llm;
    const std::int64_t n_ctx = model.info().context_length;
    const std::int64_t vocab = model.info().vocab_size;
    std::vector<llm::Token> tokens;
    if (const auto path = args.option("tokens")) {
        // Tokens from another tokenizer (e.g. the reference one), to measure the model alone.
        const auto array = load_npy(to_path(*path));
        if (!array.ok()) {
            return fail(array.status());
        }
        for (std::int64_t i = 0; i < array->numel(); ++i) {
            tokens.push_back(array->dtype() == DType::kI64 ? static_cast<llm::Token>(array->data_as<std::int64_t>()[i])
                                                           : array->data_as<std::int32_t>()[i]);
        }
    } else {
        tokens = model.tokenizer().encode(*text, true, false);
    }
    const std::int64_t available = static_cast<std::int64_t>(tokens.size()) / n_ctx;
    const std::int64_t n_chunks = *chunks < 0 ? available : std::min<std::int64_t>(*chunks, available);
    if (n_chunks <= 0) {
        return fail(
            errors::invalid_argument("the text has {} tokens, fewer than one chunk of {}", tokens.size(), n_ctx));
    }
    std::fprintf(stderr, "ii: %zu tokens, %lld chunks of %lld\n", tokens.size(), static_cast<long long>(n_chunks),
                 static_cast<long long>(n_ctx));
    auto sequence = model.new_sequence();
    if (!sequence.ok()) {
        return fail(sequence.status());
    }
    const std::int64_t first = n_ctx / 2;
    double nll = 0;
    std::int64_t count = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::int64_t c = 0; c < n_chunks; ++c) {
        std::vector<llm::Token> chunk(tokens.begin() + c * n_ctx, tokens.begin() + (c + 1) * n_ctx);
        if (model.tokenizer().bos() >= 0) {
            chunk[0] = model.tokenizer().bos();
        }
        sequence->rewind(0);
        const auto logits = sequence->eval(chunk, *batch);
        if (!logits.ok()) {
            return fail(logits.status());
        }
        for (std::int64_t j = first; j < n_ctx - 1; ++j) {
            const float* row = logits->data() + j * vocab;
            const float max = *std::max_element(row, row + vocab);
            double sum = 0;
            for (std::int64_t v = 0; v < vocab; ++v) {
                sum += std::exp(static_cast<double>(row[v] - max));
            }
            const llm::Token next = tokens[static_cast<std::size_t>(c * n_ctx + j + 1)];
            nll += -(static_cast<double>(row[next] - max) - std::log(sum));
            ++count;
        }
        std::printf("[%lld]%.4f ", static_cast<long long>(c + 1), std::exp(nll / static_cast<double>(count)));
        std::fflush(stdout);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("\nFinal estimate: PPL = %.4f (%lld tokens scored, %.1f s, %.1f tokens/s)\n",
                std::exp(nll / static_cast<double>(count)), static_cast<long long>(count), seconds,
                static_cast<double>(n_chunks * n_ctx) / seconds);
    return 0;
}

int run_tokenize(const std::vector<std::string>& raw) {
    const auto parsed = Args::parse(expand_short(raw), {"help", "special", "no-bos"},
                                    merge({"prompt", "file", "save"}, kModelOptionNames));
    if (!parsed.ok()) {
        return fail(parsed.status());
    }
    const Args& args = *parsed;
    if (args.flag("help") || args.positionals().empty() || (!args.option("prompt") && !args.option("file"))) {
        std::printf("%s", R"(Usage: ii tokenize <model> (-p TEXT | -f FILE) [options]

Prints the tokens of a text (count, then id and text of each).

  --special           Recognize special tokens written in the text (e.g. <|im_start|>)
  --no-bos            Do not add the BOS token
  --save FILE         Write the token ids to an .npy file (i32)
)");
        return args.flag("help") ? 0 : 2;
    }
    std::string text = args.option("prompt").value_or("");
    if (const auto file = args.option("file")) {
        const auto content = read_file_text(*file);
        if (!content.ok()) {
            return fail(content.status());
        }
        text = *content;
    }
    auto loaded = load_llm(args);
    if (!loaded.ok()) {
        return fail(loaded.status());
    }
    const llm::Tokenizer& tokenizer = loaded->llm.tokenizer();
    const std::vector<llm::Token> tokens = tokenizer.encode(text, !args.flag("no-bos"), args.flag("special"));
    std::printf("%zu tokens\n", tokens.size());
    if (const auto path = args.option("save")) {
        const auto tensor =
            Tensor::from(std::span<const llm::Token>(tokens), Shape{static_cast<std::int64_t>(tokens.size())});
        if (Status status = save_npy(to_path(*path), *tensor); !status.ok()) {
            return fail(status);
        }
        return 0;
    }
    for (const llm::Token token : tokens) {
        std::string piece = tokenizer.token_text(token);
        for (char& c : piece) {
            if (c == '\n') {
                c = ' ';
            }
        }
        std::printf("%8d  '%s'\n", token, piece.c_str());
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// quantize

namespace {

/// llama.cpp's "use more bits" layer rule of the K-quant mixes.
bool use_more_bits(std::int64_t layer, std::int64_t layers) {
    return layer < layers / 8 || layer >= 7 * layers / 8 || (layer - layers / 8) % 3 == 2;
}

/// llama.cpp file types (general.file_type).
int file_type(const std::string& type) {
    static const std::map<std::string, int> kTypes = {
        {"f32", 0},     {"f16", 1},   {"q4_0", 2},    {"q4_1", 3},  {"q8_0", 7},
        {"q5_0", 8},    {"q5_1", 9},  {"q2_k", 10},   {"q3_k", 12}, {"q4_k", 14},
        {"q4_k_m", 15}, {"q5_k", 16}, {"q5_k_m", 17}, {"q6_k", 18}, {"bf16", 32},
    };
    const auto it = kTypes.find(type);
    return it != kTypes.end() ? it->second : -1;
}

/// The dtype of a matrix in the target type, with llama.cpp's fallbacks for rows that are not whole blocks.
DType target_dtype(const std::string& type, const std::string& name, std::int64_t row, std::int64_t layers) {
    std::string base = type;
    if (type == "q4_k_m" || type == "q5_k_m") {
        base = type == "q4_k_m" ? "q4_k" : "q5_k";
        std::int64_t layer = -1;
        if (name.rfind("blk.", 0) == 0) {
            layer = std::strtoll(name.c_str() + 4, nullptr, 10);
        }
        const bool more = layer >= 0 && use_more_bits(layer, layers);
        if (name == "output.weight" ||
            ((name.find("attn_v.weight") != std::string::npos || name.find("ffn_down.weight") != std::string::npos) &&
             more)) {
            base = "q6_k";
        }
    } else if (name == "output.weight" && (type == "q2_k" || type == "q3_k" || type == "q4_k" || type == "q5_k")) {
        base = "q6_k";
    }
    DType dtype = *parse_dtype(base);
    if (row % dtype_info(dtype).block_size != 0) {
        // Rows that are not whole 256-blocks: llama.cpp's legacy fallbacks.
        static const std::map<DType, DType> kFallback = {{DType::kQ2_K, DType::kQ4_0},
                                                         {DType::kQ3_K, DType::kQ4_0},
                                                         {DType::kQ4_K, DType::kQ5_0},
                                                         {DType::kQ5_K, DType::kQ5_1},
                                                         {DType::kQ6_K, DType::kQ8_0}};
        const auto it = kFallback.find(dtype);
        dtype = it != kFallback.end() ? it->second : DType::kF16;
        if (row % dtype_info(dtype).block_size != 0) {
            dtype = DType::kF16;
        }
    }
    return dtype;
}

}  // namespace

int run_quantize(const std::vector<std::string>& raw) {
    const auto parsed = Args::parse(raw, {"help"}, {"type"});
    if (!parsed.ok()) {
        return fail(parsed.status());
    }
    const Args& args = *parsed;
    if (args.flag("help") || args.positionals().size() != 2 || !args.option("type")) {
        std::printf("%s", kQuantizeUsage);
        return args.flag("help") ? 0 : 2;
    }
    const std::string type = *args.option("type");
    if (file_type(type) < 0) {
        return fail(errors::invalid_argument("unknown type '{}'", type));
    }
    const auto input = gguf::File::open(to_path(args.positionals()[0]));
    if (!input.ok()) {
        return fail(input.status());
    }
    const std::string arch = input->get_string("general.architecture").value_or("");
    const std::int64_t layers = input->get_int(arch + ".block_count").value_or(0);
    gguf::Writer writer;
    for (const auto& [key, value] : input->metadata()) {
        if (key != "general.alignment") {
            writer.set(key, value);
        }
    }
    writer.set("general.file_type", gguf::Value::u32(static_cast<std::uint32_t>(file_type(type))));
    writer.set("general.quantization_version", gguf::Value::u32(2));
    std::size_t in_bytes = 0;
    std::size_t out_bytes = 0;
    for (const gguf::TensorInfo& info : input->tensors()) {
        if (info.dtype == DType::kInvalid) {
            return fail(
                errors::unsupported("tensor '{}' has ggml type {}", info.name, gguf::ggml_type_name(info.ggml_type)));
        }
        DType dtype = info.dtype;
        const bool matrix = info.shape.rank() >= 2 && info.name != "rope_freqs.weight";
        if (matrix) {
            dtype = target_dtype(type, info.name, info.shape[info.shape.rank() - 1], layers);
        }
        in_bytes += info.nbytes;
        out_bytes += storage_size(dtype, info.shape.numel());
        writer.add_tensor(info.name, dtype, info.shape, [file = *input, info, dtype]() -> Result<Tensor> {
            II_ASSIGN_OR_RETURN(Tensor tensor, file.tensor(info));
            if (tensor.dtype() == dtype) {
                return tensor;
            }
            return tensor.to(dtype);
        });
    }
    std::fprintf(stderr, "ii: %zu tensors, %.2f GiB -> %.2f GiB (%s)\n", input->tensors().size(),
                 static_cast<double>(in_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(out_bytes) / (1024.0 * 1024.0 * 1024.0), type.c_str());
    const std::size_t total = input->tensors().size();
    const Status status = writer.write(to_path(args.positionals()[1]), [&](std::size_t i) {
        if ((i + 1) % 16 == 0 || i + 1 == total) {
            std::fprintf(stderr, "\rii: %zu / %zu tensors", i + 1, total);
        }
    });
    std::fprintf(stderr, "\n");
    if (!status.ok()) {
        return fail(status);
    }
    return 0;
}

int inspect_llm_plan(const std::vector<std::string>& raw) {
    const auto parsed = Args::parse(raw, {"plan", "help"}, kModelOptionNames);
    if (!parsed.ok()) {
        return fail(parsed.status());
    }
    const Result<Loaded> loaded = load_llm(*parsed);
    if (!loaded.ok()) {
        return fail(loaded.status());
    }
    std::printf("Execution plan: %s\n%s", loaded->llm.program().describe().c_str(),
                loaded->llm.program().describe_plan().c_str());
    return 0;
}

int inspect_gguf(const std::string& path, bool all) {
    const auto file = gguf::File::open(to_path(path));
    if (!file.ok()) {
        return fail(file.status());
    }
    std::printf("GGUF version %u, %zu metadata entries, %zu tensors\n\n", file->version(), file->metadata().size(),
                file->tensors().size());
    for (const auto& [key, value] : file->metadata()) {
        std::printf("  %-48s %s\n", key.c_str(), value.to_string().c_str());
    }
    std::map<std::string, std::pair<std::size_t, std::size_t>> by_type;
    std::int64_t parameters = 0;
    for (const gguf::TensorInfo& info : file->tensors()) {
        auto& [count, bytes] = by_type[gguf::ggml_type_name(info.ggml_type)];
        ++count;
        bytes += info.nbytes;
        parameters += info.shape.numel();
    }
    std::printf("\nTensors: %.3fB parameters\n", static_cast<double>(parameters) / 1e9);
    for (const auto& [name, entry] : by_type) {
        std::printf("  %-8s %5zu tensors %10.1f MiB\n", name.c_str(), entry.first,
                    static_cast<double>(entry.second) / (1024.0 * 1024.0));
    }
    if (all) {
        std::printf("\n");
        for (const gguf::TensorInfo& info : file->tensors()) {
            std::printf("  %-40s %-8s %s\n", info.name.c_str(), gguf::ggml_type_name(info.ggml_type).c_str(),
                        info.shape.to_string().c_str());
        }
    }
    return 0;
}

}  // namespace ii::cli
