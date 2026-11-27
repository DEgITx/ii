// SPDX-License-Identifier: MIT
// `ii check-backends`: a quick differential self-test of the available backends against ii-ref.
#include <ii/ii.h>

#include "args.h"
#include "commands.h"
#include "json_writer.h"

#include <cstdio>
#include <format>
#include <string>
#include <vector>

namespace ii::cli {

namespace {

constexpr const char* kUsage = R"(Usage: ii check-backends [options]

Runs small models of common operations on every backend (except the reference ii-ref) and compares the results with
ii-ref: checks that backends and plugins work on this machine. Operations a backend does not support are skipped.
Plugins are loaded as usual (II_PLUGIN_PATH, next to the ii library) and configured by II_PLUGIN_OPTIONS; simulated
devices come from II_TEST_DEVICES.

Options:
  --backends LIST     Comma-separated backends to check (default: all but ii-ref)
  --plugin PATH       Also load this plugin file or directory
  --verbose           List every case, not only failures
  --json              Print JSON instead of text
  --help              Show this help

Exit code: 0 if no case failed, 1 otherwise.
)";

std::string_view state_name(BackendCheckState state) {
    switch (state) {
    case BackendCheckState::kPassed:
        return "passed";
    case BackendCheckState::kFailed:
        return "FAILED";
    case BackendCheckState::kSkipped:
        return "skipped";
    }
    return "?";
}

}  // namespace

int run_check_backends(const std::vector<std::string>& args) {
    const Result<Args> parsed = Args::parse(args, {"help", "verbose", "json"}, {"backends", "plugin"});
    if (!parsed.ok()) {
        std::fprintf(stderr, "ii check-backends: %s\n\n%s", parsed.status().to_string().c_str(), kUsage);
        return 2;
    }
    if (parsed->flag("help")) {
        std::printf("%s", kUsage);
        return 0;
    }
    ContextOptions options;
    if (const std::optional<std::string> plugin = parsed->option("plugin")) {
        options.plugin_paths.push_back(*plugin);
    }
    const Result<Context> context = Context::create(std::move(options));
    if (!context.ok()) {
        std::fprintf(stderr, "ii check-backends: %s\n", context.status().to_string().c_str());
        return 1;
    }
    std::vector<std::string> backends;
    if (const std::optional<std::string> list = parsed->option("backends")) {
        std::string_view rest = *list;
        while (!rest.empty()) {
            const std::size_t comma = rest.find(',');
            if (!rest.substr(0, comma).empty()) {
                backends.emplace_back(rest.substr(0, comma));
            }
            rest = comma == std::string_view::npos ? std::string_view() : rest.substr(comma + 1);
        }
    }
    const Result<std::vector<BackendCheck>> checks = check_backends(*context, backends);
    if (!checks.ok()) {
        std::fprintf(stderr, "ii check-backends: %s\n", checks.status().to_string().c_str());
        return 1;
    }
    bool failed = false;
    for (const BackendCheck& check : *checks) {
        failed = failed || check.failed > 0;
    }

    if (parsed->flag("json")) {
        JsonWriter json;
        json.begin_object();
        json.key("plugins");
        json.begin_array();
        for (const PluginDescription& plugin : context->plugins()) {
            json.begin_object();
            json.field("name", plugin.name);
            json.field("state", std::string(to_string(plugin.state)));
            json.field("message", plugin.message);
            json.end_object();
        }
        json.end_array();
        json.key("backends");
        json.begin_array();
        for (const BackendCheck& check : *checks) {
            json.begin_object();
            json.field("backend", check.backend);
            json.field("device", check.device);
            json.field("passed", check.passed);
            json.field("failed", check.failed);
            json.field("skipped", check.skipped);
            json.key("cases");
            json.begin_array();
            for (const BackendCheckCase& test : check.cases) {
                json.begin_object();
                json.field("name", test.name);
                json.field("state", std::string(state_name(test.state)));
                json.field("message", test.message);
                json.field("max_error", test.max_error);
                json.end_object();
            }
            json.end_array();
            json.end_object();
        }
        json.end_array();
        json.end_object();
        std::printf("%s\n", json.str().c_str());
        return failed ? 1 : 0;
    }

    for (const PluginDescription& plugin : context->plugins()) {
        if (plugin.state != PluginState::kLoaded) {
            std::printf("plugin %s: %s: %s\n", plugin.name.c_str(), std::string(to_string(plugin.state)).c_str(),
                        plugin.message.c_str());
        }
    }
    if (checks->empty()) {
        std::printf("no backends to check\n");
    }
    for (const BackendCheck& check : *checks) {
        std::printf("%-18s %-8s %2d passed, %d failed, %2d skipped\n", check.backend.c_str(), check.device.c_str(),
                    check.passed, check.failed, check.skipped);
        for (const BackendCheckCase& test : check.cases) {
            if (test.state == BackendCheckState::kFailed || parsed->flag("verbose")) {
                const std::string error =
                    test.state == BackendCheckState::kSkipped ? std::string() : std::format("{:.1e}", test.max_error);
                const std::string first_line = test.message.substr(0, test.message.find('\n'));
                std::printf("    %-20s %-8s %-8s %s\n", test.name.c_str(), std::string(state_name(test.state)).c_str(),
                            error.c_str(), first_line.c_str());
            }
        }
    }
    return failed ? 1 : 0;
}

}  // namespace ii::cli
