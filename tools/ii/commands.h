// SPDX-License-Identifier: MIT
#pragma once

#include <ii/model.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ii::cli {

/// Each command receives its arguments (without the command name) and returns the process exit code.
int run_info(const std::vector<std::string>& args);
int run_inspect(const std::vector<std::string>& args);
int run_ops(const std::vector<std::string>& args);
int run_run(const std::vector<std::string>& args);
int run_bench(const std::vector<std::string>& args);
int run_generate(const std::vector<std::string>& args);
int run_chat(const std::vector<std::string>& args);
int run_llm_bench(const std::vector<std::string>& args);
int run_perplexity(const std::vector<std::string>& args);
int run_quantize(const std::vector<std::string>& args);
int run_tokenize(const std::vector<std::string>& args);
int run_check_backends(const std::vector<std::string>& args);
/// `ii inspect` of a GGUF file: metadata and tensors (`all`: every tensor).
int inspect_gguf(const std::string& path, bool all);
/// `ii inspect <llm> --plan`: loads a language model and prints its execution plan.
int inspect_llm_plan(const std::vector<std::string>& args);
/// Prints the per-operation time summary of profiled runs.
void print_profile_summary(const std::vector<Profile>& profiles);
/// True if `path` is a GGUF file or a Hugging Face model directory (config.json).
bool is_llm_path(const std::string& path);
/// "cuda:0=4G,test:0=64M" as `CompileOptions::device_memory` (suffixes K, M, G).
Result<std::vector<std::pair<std::string, std::uint64_t>>> parse_device_memory(std::string_view text);

}  // namespace ii::cli
