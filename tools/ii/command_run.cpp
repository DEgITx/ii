// SPDX-License-Identifier: MIT
// `ii run` and `ii bench`: run a model on given or random inputs; report outputs, timings and a profile.
#include <ii/ii.h>

#include "args.h"
#include "commands.h"
#include "image.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace ii::cli {

namespace {

constexpr const char* kRunUsage = R"(Usage: ii run <model> [options]

Runs a model (ONNX or the ii IR text format) and reports its outputs and how long it takes. Inputs come from
files or are random.

Inputs:
  --inputs FILE       An .npz archive with one array per model input (by name)
  --input LIST        NAME=FILE pairs, e.g. x=a.npy,image=cat.jpg; a FILE alone feeds the only (or first) input.
                      Images are decoded, resized to the input size and converted to its layout (NCHW / NHWC)
  --image-size HxW    Image size when the model input does not fix it
  --letterbox         Keep the image aspect ratio and pad (YOLO-style) instead of stretching
  --image-scale X     Pixel values are multiplied by X (default 1/255) ...
  --mean LIST         ... then (value - mean) / std per channel, e.g. 0.485,0.456,0.406
  --std LIST
  --bgr               Channel order BGR instead of RGB
  --dims LIST         Values of symbolic dimensions, e.g. N=8,S=128 (fixed when the model is loaded)
  --shapes LIST       Shapes of random inputs, e.g. x=1x3x224x224,mask=1x128 (override --dims)
  --seed N            Seed of random inputs (default 1)

Outputs:
  --save-outputs FILE Write all outputs to an .npz archive
  --top K             Print the K largest values (and their indices) of each output's last dimension

Execution:
  --repeat N          Timed runs (default 10)
  --warmup N          Untimed runs before them (default 2)
  --threads N         Threads (default: all logical CPUs)
  --backends LIST     Comma-separated backends in order of preference (default: all)
  --isa NAME          Kernel variant of ii-cpu: generic, avx2, avx512, neon (default: the best supported)
  --no-fuse           Disable fusion of operations
  --profile FILE      Time every step; print a per-operation summary and write a Chrome trace to FILE
                      (open it in https://ui.perfetto.dev)
  --help              Show this help
)";

constexpr const char* kBenchUsage = R"(Usage: ii bench <model> [options]

Benchmarks a model: latency percentiles and throughput over many runs (default 100 after 10 warm-up runs). Takes
the options of 'ii run'; inputs are random unless given.
)";

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

std::filesystem::path to_path(const std::string& utf8) {
    return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

Result<std::int64_t> parse_int(std::string_view text) {
    std::int64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size() || value < 0) {
        return errors::invalid_argument("expected a non-negative integer, got '{}'", text);
    }
    return value;
}

Result<std::vector<double>> parse_doubles(const std::optional<std::string>& text) {
    std::vector<double> values;
    if (!text) {
        return values;
    }
    for (const std::string& part : split(*text, ',')) {
        char* end = nullptr;
        const double value = std::strtod(part.c_str(), &end);
        if (end == part.c_str() || *end != '\0') {
            return errors::invalid_argument("expected a number, got '{}'", part);
        }
        values.push_back(value);
    }
    return values;
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

/// Converts a tensor to the dtype of a model input when that is lossless enough (e.g. f64 arrays from NumPy for an
/// f32 input, i32 for i64), by a tiny model.
Result<Tensor> convert_input(const Context& context, const Tensor& tensor, DType dtype) {
    if (tensor.dtype() == dtype) {
        return tensor;
    }
    GraphBuilder builder;
    Dims dims;
    for (int d = 0; d < tensor.rank(); ++d) {
        dims.emplace_back(tensor.shape()[d]);
    }
    const auto x = builder.input("x", tensor.dtype(), dims);
    builder.output("y", builder.op1("cast", {x}, {{"to", dtype}}));
    II_ASSIGN_OR_RETURN(Model model, Model::from_builder(context, std::move(builder)));
    II_ASSIGN_OR_RETURN(Program program, model.compile());
    II_ASSIGN_OR_RETURN(Session session, program.new_session());
    II_ASSIGN_OR_RETURN(std::vector<Tensor> outputs, session.run({{"x", tensor}}));
    return outputs.front().clone();
}

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    if (values.empty()) {
        return 0;
    }
    const double rank = p / 100.0 * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(rank));
    const auto high = std::min(low + 1, values.size() - 1);
    return values[low] + (values[high] - values[low]) * (rank - static_cast<double>(low));
}

}  // namespace

void print_profile_summary(const std::vector<Profile>& profiles) {
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

namespace {

/// Prints the K largest values of the last dimension (of the first row) of a float output.
void print_top(const std::string& name, const Tensor& output, int k) {
    if (!is_float(output.dtype()) || output.rank() == 0 || output.numel() == 0) {
        return;
    }
    const std::int64_t n = output.shape()[output.rank() - 1];
    const Result<Tensor> contiguous = output.contiguous();
    if (!contiguous.ok()) {
        return;
    }
    std::vector<double> row(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        switch (output.dtype()) {
        case DType::kF32:
            row[static_cast<std::size_t>(i)] = contiguous->data_as<float>()[i];
            break;
        case DType::kF16:
            row[static_cast<std::size_t>(i)] = contiguous->data_as<f16>()[i].to_float();
            break;
        case DType::kBF16:
            row[static_cast<std::size_t>(i)] = contiguous->data_as<bf16>()[i].to_float();
            break;
        default:
            row[static_cast<std::size_t>(i)] = contiguous->data_as<double>()[i];
            break;
        }
    }
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), std::int64_t{0});
    const auto count = static_cast<std::size_t>(std::min<std::int64_t>(k, n));
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count), order.end(),
                      [&](std::int64_t a, std::int64_t b) {
                          return row[static_cast<std::size_t>(a)] > row[static_cast<std::size_t>(b)];
                      });
    std::printf("top %zu of %s:", count, name.c_str());
    for (std::size_t i = 0; i < count; ++i) {
        std::printf(" %lld (%.4g)", static_cast<long long>(order[i]), row[static_cast<std::size_t>(order[i])]);
    }
    std::printf("\n");
}

int run_model(const std::vector<std::string>& args, bool bench) {
    const char* const usage = bench ? kBenchUsage : kRunUsage;
    const char* const command = bench ? "ii bench" : "ii run";
    auto parsed = Args::parse(args, {"no-fuse", "help", "letterbox", "bgr"},
                              {"dims", "shapes", "repeat", "warmup", "threads", "backends", "isa", "profile", "seed",
                               "inputs", "input", "save-outputs", "top", "image-size", "image-scale", "mean", "std"});
    if (!parsed.ok()) {
        std::fprintf(stderr, "%s: %s\n\n%s", command, std::string(parsed.status().message()).c_str(), usage);
        return 2;
    }
    if (parsed->flag("help")) {
        std::printf("%s", usage);
        if (bench) {
            std::printf("\n%s", kRunUsage);
        }
        return 0;
    }
    if (parsed->positionals().size() != 1) {
        std::fprintf(stderr, "%s: expected one model file\n\n%s", command, usage);
        return 2;
    }
    const std::string& path = parsed->positionals().front();
    auto fail = [&](const Status& status) {
        std::fprintf(stderr, "%s: %s: %s\n", command, path.c_str(), status.to_string().c_str());
        return 1;
    };
    const Result<int> repeat = parsed->int_option("repeat", bench ? 100 : 10);
    const Result<int> warmup = parsed->int_option("warmup", bench ? 10 : 2);
    const Result<int> threads = parsed->int_option("threads", 0);
    const Result<int> seed = parsed->int_option("seed", 1);
    const Result<int> top = parsed->int_option("top", 0);
    const Result<double> image_scale = parsed->double_option("image-scale", 1.0 / 255);
    for (const Status& status :
         {repeat.status(), warmup.status(), threads.status(), seed.status(), top.status(), image_scale.status()}) {
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
    LoadOptions load_options;
    for (const auto& [name, text] : *dim_text) {
        const Result<std::int64_t> value = parse_int(text);
        if (!value.ok()) {
            return fail(value.status());
        }
        dims[name] = *value;
        load_options.dims.emplace_back(name, *value);
    }
    ImageOptions image_options;
    image_options.letterbox = parsed->flag("letterbox");
    image_options.bgr = parsed->flag("bgr");
    image_options.scale = *image_scale;
    {
        const Result<std::vector<double>> mean = parse_doubles(parsed->option("mean"));
        const Result<std::vector<double>> std_dev = parse_doubles(parsed->option("std"));
        if (!mean.ok() || !std_dev.ok()) {
            return fail(!mean.ok() ? mean.status() : std_dev.status());
        }
        image_options.mean = *mean;
        image_options.std = *std_dev;
        if (const auto size = parsed->option("image-size")) {
            const std::vector<std::string> parts = split(*size, 'x');
            const Result<std::int64_t> h = parts.size() == 2 ? parse_int(parts[0]) : Result<std::int64_t>(0);
            const Result<std::int64_t> w = parts.size() == 2 ? parse_int(parts[1]) : Result<std::int64_t>(0);
            if (parts.size() != 2 || !h.ok() || !w.ok()) {
                return fail(errors::invalid_argument("--image-size expects HxW, got '{}'", *size));
            }
            image_options.height = static_cast<int>(*h);
            image_options.width = static_cast<int>(*w);
        }
    }

    ContextOptions context_options;
    context_options.num_threads = *threads;
    context_options.cpu_isa = parsed->option("isa").value_or("");
    const Result<Context> context = Context::create(std::move(context_options));
    if (!context.ok()) {
        return fail(context.status());
    }
    using Clock = std::chrono::steady_clock;
    const Clock::time_point load_start = Clock::now();
    const Result<Model> model = Model::load(*context, to_path(path), load_options);
    if (!model.ok()) {
        return fail(model.status());
    }
    const double load_ms = std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
    CompileOptions compile_options;
    if (const auto backends = parsed->option("backends")) {
        compile_options.backends = split(*backends, ',');
    }
    compile_options.fuse = !parsed->flag("no-fuse");
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

    // --- Inputs: files first, random values for the rest ---
    const std::vector<TensorInfo> inputs = program->inputs();
    std::map<std::string, Tensor> given;
    if (const auto archive = parsed->option("inputs")) {
        const Result<std::vector<NamedTensor>> arrays = load_npz(to_path(*archive));
        if (!arrays.ok()) {
            return fail(arrays.status());
        }
        for (const auto& [name, tensor] : *arrays) {
            given[name] = tensor;
        }
    }
    if (const auto list = parsed->option("input")) {
        for (const std::string& item : split(*list, ',')) {
            const std::size_t eq = item.find('=');
            std::string name = eq == std::string::npos ? std::string() : item.substr(0, eq);
            const std::string file = eq == std::string::npos ? item : item.substr(eq + 1);
            if (name.empty()) {
                if (inputs.empty()) {
                    return fail(errors::invalid_argument("the model has no inputs"));
                }
                name = inputs.front().name;
            }
            const auto info =
                std::find_if(inputs.begin(), inputs.end(), [&](const TensorInfo& i) { return i.name == name; });
            if (info == inputs.end()) {
                return fail(errors::not_found("the model has no input '{}'", name));
            }
            if (is_image_file(file)) {
                const Result<Image> image = load_image(to_path(file));
                if (!image.ok()) {
                    return fail(image.status());
                }
                const Result<Tensor> tensor = image_to_tensor(*image, *info, image_options);
                if (!tensor.ok()) {
                    return fail(tensor.status());
                }
                given[name] = *tensor;
            } else {
                const Result<Tensor> tensor = load_npy(to_path(file));
                if (!tensor.ok()) {
                    return fail(tensor.status());
                }
                given[name] = *tensor;
            }
        }
    }
    for (const auto& [name, tensor] : given) {
        if (std::none_of(inputs.begin(), inputs.end(), [&](const TensorInfo& i) { return i.name == name; })) {
            return fail(
                errors::not_found("the model has no input '{}' (inputs given in files must match by name)", name));
        }
    }
    std::mt19937 rng(static_cast<std::uint32_t>(*seed));
    std::vector<Tensor> bound;
    for (const TensorInfo& info : inputs) {
        Result<Tensor> tensor = Tensor();
        const char* origin = "random";
        if (const auto it = given.find(info.name); it != given.end()) {
            tensor = convert_input(*context, it->second, info.dtype);
            origin = "file";
        } else {
            const Result<Shape> shape = input_shape(info, *shapes, dims);
            if (!shape.ok()) {
                return fail(shape.status());
            }
            tensor = random_tensor(info.dtype, *shape, rng);
        }
        if (!tensor.ok()) {
            return fail(tensor.status());
        }
        if (Status status = session->set_input(info.name, *tensor); !status.ok()) {
            return fail(status);
        }
        bound.push_back(*tensor);
        std::printf("input   %-24s %s%s (%s)\n", info.name.c_str(), std::string(to_string(tensor->dtype())).c_str(),
                    tensor->shape().to_string().c_str(), origin);
    }

    // --- Runs ---
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

    // --- Outputs ---
    std::vector<NamedTensor> outputs;
    for (const TensorInfo& info : program->outputs()) {
        const Result<Tensor> output = session->output(info.name);
        if (!output.ok()) {
            return fail(output.status());
        }
        std::printf("output  %-24s %s%s\n", info.name.c_str(), std::string(to_string(output->dtype())).c_str(),
                    output->shape().to_string().c_str());
        outputs.emplace_back(info.name, *output);
    }
    if (*top > 0) {
        for (const auto& [name, tensor] : outputs) {
            print_top(name, tensor, *top);
        }
    }
    if (const auto save = parsed->option("save-outputs")) {
        if (Status status = save_npz(to_path(*save), outputs); !status.ok()) {
            return fail(status);
        }
        std::printf("outputs written to %s\n", save->c_str());
    }

    const double min = *std::min_element(times.begin(), times.end());
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    std::printf("\nload %.1f ms, compile %.1f ms\n", load_ms, compile_ms);
    if (bench) {
        std::printf("%zu runs: min %.3f ms, p50 %.3f ms, p90 %.3f ms, p99 %.3f ms, mean %.3f ms, %.1f runs/s\n",
                    times.size(), min, percentile(times, 50), percentile(times, 90), percentile(times, 99), mean,
                    1000.0 / mean);
    } else {
        std::printf("%zu runs: min %.3f ms, median %.3f ms, mean %.3f ms\n", times.size(), min, percentile(times, 50),
                    mean);
    }

    if (profile_path) {
        print_profile_summary(profiles);
        std::ofstream file(to_path(*profile_path), std::ios::binary);
        file << to_chrome_trace(profiles);
        if (!file) {
            return fail(errors::io_error("cannot write '{}'", *profile_path));
        }
        std::printf("\ntrace written to %s\n", profile_path->c_str());
    }
    return 0;
}

}  // namespace

int run_run(const std::vector<std::string>& args) {
    return run_model(args, false);
}

int run_bench(const std::vector<std::string>& args) {
    return run_model(args, true);
}

}  // namespace ii::cli
