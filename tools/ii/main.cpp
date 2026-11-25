// SPDX-License-Identifier: MIT
// The ii command-line tool.
#include <ii/version.h>

#include "commands.h"
#include "console.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char* kUsage = R"(Usage: ii <command> [options]

Commands:
  info        Describe the machine: OS, CPU, memory, kernel variants, roofline
  inspect     Describe a model and its execution plan
  ops         List the operations of the OpSet
  run         Run a model (ONNX or ii text) on inputs from files or random ones; outputs, timings, profile
  bench       Benchmark a model: latency percentiles and throughput; language models: pp / tg tokens per second
  generate    Generate text with a language model (GGUF or Hugging Face directory)
  chat        Chat with a language model
  perplexity  Perplexity of a language model on a text file
  quantize    Convert the weights of a GGUF model (q4_k_m, q8_0, ...)
  tokenize    Show the tokens of a text for a language model
  version     Print the version
  help        Show this help

Run 'ii <command> --help' for the options of a command.
)";

int print_version() {
    const ii::BuildInfo& build = ii::build_info();
    std::printf("ii %s (git %s)\n", build.version.data(), build.git_hash.data());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = ii::cli::init_console(argc, argv);
    if (args.size() < 2) {
        std::printf("%s", kUsage);
        return 0;
    }

    const std::string_view command = args[1];
    const std::vector<std::string> rest(args.begin() + 2, args.end());

    if (command == "info") {
        return ii::cli::run_info(rest);
    }
    if (command == "inspect") {
        return ii::cli::run_inspect(rest);
    }
    if (command == "ops") {
        return ii::cli::run_ops(rest);
    }
    if (command == "run") {
        return ii::cli::run_run(rest);
    }
    if (command == "bench") {
        const bool llm = std::find(rest.begin(), rest.end(), "--llm") != rest.end() ||
                         (!rest.empty() && ii::cli::is_llm_path(rest.front()));
        return llm ? ii::cli::run_llm_bench(rest) : ii::cli::run_bench(rest);
    }
    if (command == "generate") {
        return ii::cli::run_generate(rest);
    }
    if (command == "chat") {
        return ii::cli::run_chat(rest);
    }
    if (command == "perplexity") {
        return ii::cli::run_perplexity(rest);
    }
    if (command == "quantize") {
        return ii::cli::run_quantize(rest);
    }
    if (command == "tokenize") {
        return ii::cli::run_tokenize(rest);
    }
    if (command == "version" || command == "--version" || command == "-V") {
        return print_version();
    }
    if (command == "help" || command == "--help" || command == "-h") {
        std::printf("%s", kUsage);
        return 0;
    }
    std::fprintf(stderr, "ii: unknown command '%s'\n\n%s", args[1].c_str(), kUsage);
    return 2;
}
