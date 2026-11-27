// SPDX-License-Identifier: MIT
// `ii inspect`: describes a model file and how ii would execute it.
#include <ii/ii.h>

#include "args.h"
#include "commands.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace ii::cli {

namespace {

constexpr const char* kUsage = R"(Usage: ii inspect <model> [options]

Describes a model (inputs, outputs, states) and its execution plan. Formats: ONNX and the ii IR text format.

Options:
  --graph             Print the optimized graph
  --plan              Explain the plan: devices, partitions with estimated times, copies between devices,
                      memory per device and why backends declined operations
  --plan-dims LIST    Symbol values assumed for the estimates, e.g. N=1 (other symbols count as 1)
  --device-memory LIST  Memory limits per device, e.g. cuda:0=4G,test:0=64M (suffixes K, M, G)

Language models (GGUF or Hugging Face directory) print metadata and tensors; with --plan they are loaded (options
--ctx, --kv-type, --threads, --backends, --device-memory as for ii generate) and the plan is explained.
  --imported          Print the graph as loaded (before optimization)
  --dims LIST         Fix symbolic dimensions when loading, e.g. batch=1,seq=128
  --backends LIST     Comma-separated backends to compile for (default: all)
  --no-fold           Disable constant folding
  --no-fuse           Disable fusion (common subexpressions, patterns, elementwise chains)
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

/// "batch=1,seq=128" as load options.
Result<LoadOptions> parse_dims(const std::optional<std::string>& text) {
    LoadOptions options;
    const std::string all = text.value_or("");
    std::string_view rest = all;
    while (!rest.empty()) {
        const std::size_t comma = rest.find(',');
        const std::string_view item = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view() : rest.substr(comma + 1);
        const std::size_t eq = item.find('=');
        std::int64_t value = 0;
        if (eq == std::string_view::npos ||
            std::from_chars(item.data() + eq + 1, item.data() + item.size(), value).ec != std::errc()) {
            return errors::invalid_argument("expected NAME=VALUE in --dims, got '{}'", item);
        }
        options.dims.emplace_back(std::string(item.substr(0, eq)), value);
    }
    return options;
}

}  // namespace

Result<std::vector<std::pair<std::string, std::uint64_t>>> parse_device_memory(std::string_view text) {
    std::vector<std::pair<std::string, std::uint64_t>> limits;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        const std::string_view item = text.substr(0, comma);
        text = comma == std::string_view::npos ? std::string_view() : text.substr(comma + 1);
        const std::size_t eq = item.find('=');
        std::string_view number = eq == std::string_view::npos ? std::string_view() : item.substr(eq + 1);
        std::uint64_t scale = 1;
        if (!number.empty()) {
            switch (number.back()) {
            case 'K':
                scale = std::uint64_t{1} << 10;
                break;
            case 'M':
                scale = std::uint64_t{1} << 20;
                break;
            case 'G':
                scale = std::uint64_t{1} << 30;
                break;
            default:
                break;
            }
            if (scale != 1) {
                number.remove_suffix(1);
            }
        }
        std::uint64_t value = 0;
        if (number.empty() || std::from_chars(number.data(), number.data() + number.size(), value).ec != std::errc() ||
            value > UINT64_MAX / scale) {
            return errors::invalid_argument("expected DEVICE=BYTES in --device-memory, got '{}'", item);
        }
        limits.emplace_back(std::string(item.substr(0, eq)), value * scale);
    }
    return limits;
}

int run_inspect(const std::vector<std::string>& args) {
    if (!args.empty() && is_llm_path(args.front()) &&
        std::find(args.begin(), args.end(), std::string("--plan")) != args.end()) {
        return inspect_llm_plan(args);
    }
    auto parsed = Args::parse(args, {"graph", "imported", "no-fold", "no-fuse", "decompose", "plan", "help"},
                              {"backends", "dims", "plan-dims", "device-memory"});
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
    if (is_llm_path(path) && path.size() > 5 && path.substr(path.size() - 5) == ".gguf") {
        return inspect_gguf(path, parsed->flag("graph"));
    }
    auto fail = [&](const Status& status) {
        std::fprintf(stderr, "ii inspect: %s: %s\n", path.c_str(), status.to_string().c_str());
        return 1;
    };
    const Result<LoadOptions> load_options = parse_dims(parsed->option("dims"));
    if (!load_options.ok()) {
        return fail(load_options.status());
    }
    const Result<Context> context = Context::create();
    if (!context.ok()) {
        return fail(context.status());
    }
    const Result<Model> model =
        Model::load(*context, std::filesystem::path(std::u8string(path.begin(), path.end())), *load_options);
    if (!model.ok()) {
        return fail(model.status());
    }
    if (parsed->flag("imported")) {
        std::printf("Imported graph:\n%s\n", model->to_text().c_str());
    }
    std::printf("Inputs:\n%sOutputs:\n%s", describe(model->inputs()).c_str(), describe(model->outputs()).c_str());
    if (!model->states().empty()) {
        std::printf("States:\n%s", describe(model->states()).c_str());
    }

    CompileOptions options;
    options.fold_constants = !parsed->flag("no-fold");
    options.fuse = !parsed->flag("no-fuse");
    options.decompose_composites = parsed->flag("decompose");
    if (const std::optional<std::string> list = parsed->option("backends")) {
        std::string_view rest = *list;
        while (!rest.empty()) {
            const std::size_t comma = rest.find(',');
            options.backends.emplace_back(rest.substr(0, comma));
            rest = comma == std::string_view::npos ? std::string_view() : rest.substr(comma + 1);
        }
    }
    const Result<LoadOptions> plan_dims = parse_dims(parsed->option("plan-dims"));
    if (!plan_dims.ok()) {
        return fail(plan_dims.status());
    }
    options.plan_dims = plan_dims->dims;
    const Result<std::vector<std::pair<std::string, std::uint64_t>>> memory =
        parse_device_memory(parsed->option("device-memory").value_or(""));
    if (!memory.ok()) {
        return fail(memory.status());
    }
    options.device_memory = *memory;
    const Result<Program> program = model->compile(options);
    if (!program.ok()) {
        return fail(program.status());
    }
    std::printf("\nExecution plan: %s", program->describe().c_str());
    if (parsed->flag("plan")) {
        std::printf("\n%s", program->describe_plan().c_str());
    }
    if (parsed->flag("graph")) {
        std::printf("\nOptimized graph:\n%s", program->to_text().c_str());
    }
    return 0;
}

}  // namespace ii::cli
