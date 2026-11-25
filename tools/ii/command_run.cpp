// SPDX-License-Identifier: MIT
// `ii run`: runs a model on random inputs, reports timings and optionally writes a profile.
#include <ii/ii.h>

#include "args.h"
#include "commands.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ii::cli {

namespace {

constexpr const char* kUsage = R"(Usage: ii run <model> [options]

Runs a model on random inputs and reports how long it takes. Supported formats: the ii IR text format.

Options:
  --dims LIST         Values of symbolic dimensions, e.g. N=8,S=128
  --shapes LIST       Input shapes, e.g. x=1x3x224x224,mask=1x128 (override --dims)
  --repeat N          Timed runs (default 10)
  --warmup N          Untimed runs before them (default 2)
  --threads N         Threads (default: all logical CPUs)
  --backends LIST     Comma-separated backends in order of preference (default: all)
  --isa NAME          Kernel variant of ii-cpu: generic, avx2, avx512, neon (default: the best supported)
  --no-fuse           Disable fusion of operations
  --profile FILE      Time every step; print a per-operation summary and write a Chrome trace to FILE
                      (open it in https://ui.perfetto.dev)
  --seed N            Seed of the random inputs (default 1)
  --help              Show this help
)";

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

std::vector<std::string> split(std::string_view text, char separator) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = std::min(text.find(separator, start), text.size());
        if (end > start) {
            parts.emplace_back(text.substr(start, end - start));
        }
        start = end + 1;
    }
    return parts;
}

Result<std::int64_t> parse_int(std::string_view text) {
    std::int64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size() || value < 0) {
        return errors::invalid_argument("expected a non-negative integer, got '{}'", text);
    }
    return value;
}

/// "N=8,S=128" or "x=1x3x224,y=4" as name/value pairs.
Result<std::map<std::string, std::string>> parse_assignments(const std::optional<std::string>& text) {
    std::map<std::string, std::string> result;
    if (!text) {
        return result;
    }
    for (const std::string& item : split(*text, ',')) {
        const std::size_t eq = item.find('=');
        if (eq == std::string::npos || eq == 0) {
            return errors::invalid_argument("expected NAME=VALUE, got '{}'", item);
        }
        result[item.substr(0, eq)] = item.substr(eq + 1);
    }
    return result;
}

/// A tensor of random values: floats uniform in [-1, 1), integers in [0, 10), bools 50/50.
Result<Tensor> random_tensor(DType dtype, const Shape& shape, std::mt19937& rng) {
    II_ASSIGN_OR_RETURN(Tensor tensor, Tensor::empty(dtype, shape));
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::uniform_int_distribution<int> digits(0, 9);
    const std::int64_t n = tensor.numel();
    void* data = tensor.data();
    for (std::int64_t i = 0; i < n; ++i) {
        switch (dtype) {
        case DType::kF32:
            static_cast<float*>(data)[i] = static_cast<float>(uniform(rng));
            break;
        case DType::kF64:
            static_cast<double*>(data)[i] = uniform(rng);
            break;
        case DType::kF16:
            static_cast<f16*>(data)[i] = f16(static_cast<float>(uniform(rng)));
            break;
        case DType::kBF16:
            static_cast<bf16*>(data)[i] = bf16(static_cast<float>(uniform(rng)));
            break;
        case DType::kBool:
        case DType::kU8:
        case DType::kI8:
            static_cast<std::uint8_t*>(data)[i] =
                static_cast<std::uint8_t>(dtype == DType::kBool ? digits(rng) % 2 : digits(rng));
            break;
        case DType::kI16:
        case DType::kU16:
            static_cast<std::uint16_t*>(data)[i] = static_cast<std::uint16_t>(digits(rng));
            break;
        case DType::kI32:
        case DType::kU32:
            static_cast<std::uint32_t*>(data)[i] = static_cast<std::uint32_t>(digits(rng));
            break;
        case DType::kI64:
        case DType::kU64:
            static_cast<std::uint64_t*>(data)[i] = static_cast<std::uint64_t>(digits(rng));
            break;
        default:
            return errors::unsupported("cannot generate random {} inputs", to_string(dtype));
        }
    }
    return tensor;
}

/// The concrete shape of an input from --shapes or from its symbolic shape and --dims.
Result<Shape> input_shape(const TensorInfo& info, const std::map<std::string, std::string>& shapes,
                          const std::map<std::string, std::int64_t>& dims) {
    if (const auto it = shapes.find(info.name); it != shapes.end()) {
        Shape shape;
        for (const std::string& part : split(it->second, 'x')) {
            II_ASSIGN_OR_RETURN(const std::int64_t value, parse_int(part));
            shape.push_back(value);
        }
        return shape;
    }
    Shape shape;
    for (const Dim& dim : info.shape) {
        const std::optional<std::int64_t> value =
            dim.evaluate([&](std::string_view name) -> std::optional<std::int64_t> {
                const auto it = dims.find(std::string(name));
                return it != dims.end() ? std::optional<std::int64_t>(it->second) : std::nullopt;
            });
        if (!value) {
            return errors::invalid_argument("input '{}' has the symbolic shape {}: give its dimensions with --dims or "
                                            "--shapes",
                                            info.name, to_string(info.shape));
        }
        shape.push_back(*value);
    }
    return shape;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void print_summary(const std::vector<Profile>& profiles) {
    struct Row {
        std::string op;
        std::string backend;
        int count = 0;
        double total_us = 0;
    };
    std::map<std::pair<std::string, std::string>, Row> rows;
    double total = 0;
    for (const Profile& profile : profiles) {
        for (const StepProfile& step : profile.steps) {
            Row& row = rows[{step.op, step.backend}];
            row.op = step.op;
            row.backend = step.backend;
            ++row.count;
            row.total_us += step.duration_us;
            total += step.duration_us;
        }
    }
    std::vector<Row> sorted;
    for (auto& [key, row] : rows) {
        sorted.push_back(row);
    }
    std::sort(sorted.begin(), sorted.end(), [](const Row& a, const Row& b) { return a.total_us > b.total_us; });
    const auto runs = static_cast<double>(std::max<std::size_t>(profiles.size(), 1));
    std::printf("\n%-20s %-8s %7s %12s %7s\n", "operation", "backend", "steps", "ms per run", "share");
    for (const Row& row : sorted) {
        std::printf("%-20s %-8s %7d %12.3f %6.1f%%\n", row.op.c_str(), row.backend.c_str(),
                    static_cast<int>(row.count / runs), row.total_us / runs / 1000.0,
                    total > 0 ? 100.0 * row.total_us / total : 0.0);
    }
}

}  // namespace

int run_run(const std::vector<std::string>& args) {
    auto parsed = Args::parse(args, {"no-fuse", "help"},
                              {"dims", "shapes", "repeat", "warmup", "threads", "backends", "isa", "profile", "seed"});
    if (!parsed.ok()) {
        std::fprintf(stderr, "ii run: %s\n\n%s", std::string(parsed.status().message()).c_str(), kUsage);
        return 2;
    }
    if (parsed->flag("help")) {
        std::printf("%s", kUsage);
        return 0;
    }
    if (parsed->positionals().size() != 1) {
        std::fprintf(stderr, "ii run: expected one model file\n\n%s", kUsage);
        return 2;
    }
    const std::string& path = parsed->positionals().front();
    auto fail = [&](const Status& status) {
        std::fprintf(stderr, "ii run: %s: %s\n", path.c_str(), status.to_string().c_str());
        return 1;
    };
    const Result<int> repeat = parsed->int_option("repeat", 10);
    const Result<int> warmup = parsed->int_option("warmup", 2);
    const Result<int> threads = parsed->int_option("threads", 0);
    const Result<int> seed = parsed->int_option("seed", 1);
    for (const Status& status : {repeat.status(), warmup.status(), threads.status(), seed.status()}) {
        if (!status.ok()) {
            return fail(status);
        }
    }
    const Result<std::map<std::string, std::string>> dim_text = parse_assignments(parsed->option("dims"));
    const Result<std::map<std::string, std::string>> shapes = parse_assignments(parsed->option("shapes"));
    if (!dim_text.ok() || !shapes.ok()) {
        return fail(!dim_text.ok() ? dim_text.status() : shapes.status());
    }
    std::map<std::string, std::int64_t> dims;
    for (const auto& [name, text] : *dim_text) {
        const Result<std::int64_t> value = parse_int(text);
        if (!value.ok()) {
            return fail(value.status());
        }
        dims[name] = *value;
    }

    const Result<std::string> text = read_file(path);
    if (!text.ok()) {
        return fail(text.status());
    }
    ContextOptions context_options;
    context_options.num_threads = *threads;
    context_options.cpu_isa = parsed->option("isa").value_or("");
    const Result<Context> context = Context::create(std::move(context_options));
    if (!context.ok()) {
        return fail(context.status());
    }
    const Result<Model> model = Model::from_text(*context, *text);
    if (!model.ok()) {
        return fail(model.status());
    }
    CompileOptions compile_options;
    if (const auto backends = parsed->option("backends")) {
        compile_options.backends = split(*backends, ',');
    }
    compile_options.fuse = !parsed->flag("no-fuse");
    using Clock = std::chrono::steady_clock;
    const Clock::time_point compile_start = Clock::now();
    const Result<Program> program = model->compile(compile_options);
    if (!program.ok()) {
        return fail(program.status());
    }
    const double compile_ms = std::chrono::duration<double, std::milli>(Clock::now() - compile_start).count();
    Result<Session> session = program->new_session();
    if (!session.ok()) {
        return fail(session.status());
    }

    std::mt19937 rng(static_cast<std::uint32_t>(*seed));
    for (const TensorInfo& info : program->inputs()) {
        const Result<Shape> shape = input_shape(info, *shapes, dims);
        if (!shape.ok()) {
            return fail(shape.status());
        }
        const Result<Tensor> tensor = random_tensor(info.dtype, *shape, rng);
        if (!tensor.ok()) {
            return fail(tensor.status());
        }
        if (Status status = session->set_input(info.name, *tensor); !status.ok()) {
            return fail(status);
        }
        std::printf("input   %-16s %s%s\n", info.name.c_str(), std::string(to_string(info.dtype)).c_str(),
                    shape->to_string().c_str());
    }

    for (int i = 0; i < *warmup; ++i) {
        if (Status status = session->run(); !status.ok()) {
            return fail(status);
        }
    }
    const std::optional<std::string> profile_path = parsed->option("profile");
    session->set_profiling(profile_path.has_value());
    std::vector<double> times;
    std::vector<Profile> profiles;
    for (int i = 0; i < std::max(*repeat, 1); ++i) {
        const Clock::time_point start = Clock::now();
        if (Status status = session->run(); !status.ok()) {
            return fail(status);
        }
        times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        if (profile_path) {
            profiles.push_back(session->profile().value());
        }
    }
    for (const TensorInfo& info : program->outputs()) {
        const Result<Tensor> output = session->output(info.name);
        if (output.ok()) {
            std::printf("output  %-16s %s%s\n", info.name.c_str(), std::string(to_string(output->dtype())).c_str(),
                        output->shape().to_string().c_str());
        }
    }
    const double min = *std::min_element(times.begin(), times.end());
    double mean = 0;
    for (const double t : times) {
        mean += t / static_cast<double>(times.size());
    }
    std::printf("\ncompile %.1f ms, %zu runs: min %.3f ms, median %.3f ms, mean %.3f ms\n", compile_ms, times.size(),
                min, median(times), mean);

    if (profile_path) {
        print_summary(profiles);
        const std::filesystem::path out(std::u8string(profile_path->begin(), profile_path->end()));
        std::ofstream file(out, std::ios::binary);
        file << to_chrome_trace(profiles);
        if (!file) {
            return fail(errors::io_error("cannot write '{}'", *profile_path));
        }
        std::printf("\ntrace written to %s\n", profile_path->c_str());
    }
    return 0;
}

}  // namespace ii::cli
