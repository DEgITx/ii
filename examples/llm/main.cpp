// SPDX-License-Identifier: MIT
// A language model in a few lines: load a GGUF file (or a Hugging Face model directory), chat, stream the reply.
//
//   ii_example_llm qwen2.5-0.5b-instruct-q4_k_m.gguf "What is the capital of France?"
#include <ii/llm.h>

#include <cstdio>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf | hf-model-dir> [question]\n", argv[0]);
        return 2;
    }
    const std::string question = argc > 2 ? argv[2] : "Hello! Who are you?";

    auto context = ii::Context::create();
    if (!context.ok()) {
        std::fprintf(stderr, "%s\n", context.status().to_string().c_str());
        return 1;
    }
    ii::llm::LoadOptions options;
    options.context_length = 1024;  // KV cache capacity in tokens
    auto llm = ii::llm::Llm::load(*context, argv[1], options);
    if (!llm.ok()) {
        std::fprintf(stderr, "%s\n", llm.status().to_string().c_str());
        return 1;
    }
    const ii::llm::ModelInfo& info = llm->info();
    std::printf("%s: %s, %lld layers, %.2fB parameters (%s)\n\n", info.architecture.c_str(), info.name.c_str(),
                static_cast<long long>(info.layers), static_cast<double>(info.parameters) / 1e9,
                info.weight_type.c_str());

    ii::llm::ChatOptions chat_options;
    chat_options.temperature = 0;  // greedy: the same answer every time
    chat_options.max_tokens = 64;
    auto chat = llm->new_chat(chat_options);
    if (!chat.ok()) {
        std::fprintf(stderr, "%s\n", chat.status().to_string().c_str());
        return 1;
    }
    std::printf("> %s\n", question.c_str());
    const auto reply = chat->send(question, [](std::string_view piece) {
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        return true;  // false would stop the generation
    });
    if (!reply.ok()) {
        std::fprintf(stderr, "\n%s\n", reply.status().to_string().c_str());
        return 1;
    }
    const ii::llm::GenerationStats& stats = chat->last_stats();
    std::printf("\n\n[%lld prompt tokens, %lld generated, %.1f tokens/s]\n",
                static_cast<long long>(stats.prompt_tokens), static_cast<long long>(stats.generated_tokens),
                stats.generation_seconds > 0 ? static_cast<double>(stats.generated_tokens) / stats.generation_seconds
                                             : 0.0);
    return 0;
}
