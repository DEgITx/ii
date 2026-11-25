// SPDX-License-Identifier: MIT
// `ii inspect`: describes a model file and how ii would execute it.
#include <ii/ii.h>

#include "args.h"
#include "commands.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ii::cli {

namespace {

constexpr const char* kUsage = R"(Usage: ii inspect <model> [options]

Describes a model (inputs, outputs, states) and its execution plan. Supported formats: the ii IR text format.

Options:
  --graph             Print the optimized graph
  --backends LIST     Comma-separated backends to compile for (default: all)
  --no-fold           Disable constant folding
  --decompose         Decompose all composite operations
  --help              Show this help
)";

std::string describe(const std::vector<TensorInfo>& infos) {
    std::string text;
    for (const TensorInfo& info : infos) {
        text += "  " + info.name + " : " + std::string(to_string(info.dtype)) + to_string(info.shape) + "\n";
    }
    return text.empty() ? "  (none)\n" : text;
}

Result<std::string> read_file(const std::string& utf8_path) {
    const std::filesystem::path path(std::u8string(utf8_path.begin(), utf8_path.end()));
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return errors::io_error("cannot open '{}'", utf8_path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

}  // namespace

int run_inspect(const std::vector<std::string>& args) {
    auto parsed = Args::parse(args, {"graph", "no-fold", "decompose", "help"}, {"backends"});
    if (!parsed.ok()) {
        std::fprintf(stderr, "ii inspect: %s\n\n%s", std::string(parsed.status().message()).c_str(), kUsage);
        return 2;
    }
    if (parsed->flag("help")) {
        std::printf("%s", kUsage);
        return 0;
    }
    if (parsed->positionals().size() != 1) {
        std::fprintf(stderr, "ii inspect: expected one model file\n\n%s", kUsage);
        return 2;
    }
    const std::string& path = parsed->positionals().front();
    auto fail = [&](const Status& status) {
        std::fprintf(stderr, "ii inspect: %s: %s\n", path.c_str(), status.to_string().c_str());
        return 1;
    };
    const Result<std::string> text = read_file(path);
    if (!text.ok()) {
        return fail(text.status());
    }
    const Result<Context> context = Context::create();
    if (!context.ok()) {
        return fail(context.status());
    }
    const Result<Model> model = Model::from_text(*context, *text);
    if (!model.ok()) {
        return fail(model.status());
    }
    std::printf("Inputs:\n%sOutputs:\n%s", describe(model->inputs()).c_str(), describe(model->outputs()).c_str());
    if (!model->states().empty()) {
        std::printf("States:\n%s", describe(model->states()).c_str());
    }

    CompileOptions options;
    options.fold_constants = !parsed->flag("no-fold");
    options.decompose_composites = parsed->flag("decompose");
    if (const std::optional<std::string> list = parsed->option("backends")) {
        std::string_view rest = *list;
        while (!rest.empty()) {
            const std::size_t comma = rest.find(',');
            options.backends.emplace_back(rest.substr(0, comma));
            rest = comma == std::string_view::npos ? std::string_view() : rest.substr(comma + 1);
        }
    }
    const Result<Program> program = model->compile(options);
    if (!program.ok()) {
        return fail(program.status());
    }
    std::printf("\nExecution plan: %s", program->describe().c_str());
    if (parsed->flag("graph")) {
        std::printf("\nOptimized graph:\n%s", program->to_text().c_str());
    }
    return 0;
}

}  // namespace ii::cli
