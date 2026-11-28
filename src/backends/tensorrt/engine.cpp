// SPDX-License-Identifier: MIT
#include "backends/tensorrt/engine.h"

#include <ii/version.h>

#include "backends/tensorrt/translate.h"
#include "core/platform/platform.h"
#include "ops/fused.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>

namespace ii::tensorrt {

namespace nv = nvinfer1;

namespace {

thread_local std::string g_last_error;

/// Part of every engine's cache key: increase it whenever the translation of operations changes, so that engines
/// built by an earlier plugin are not reused.
constexpr int kTranslationRevision = 1;

template <class T>
struct TrtDelete {
    void operator()(T* object) const noexcept { delete object; }
};
template <class T>
using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

/// A 64-bit hash of bytes (multiply-rotate over 8-byte words): for cache keys, not for security.
std::uint64_t hash_bytes(const void* data, std::size_t size, std::uint64_t seed) {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = seed ^ (static_cast<std::uint64_t>(size) * 0x9E3779B97F4A7C15ULL);
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        std::uint64_t v = 0;
        std::memcpy(&v, p + i, 8);
        h ^= v * 0xFF51AFD7ED558CCDULL;
        h = ((h << 31) | (h >> 33)) * 0xC4CEB9FE1A85EC53ULL;
    }
    for (; i < size; ++i) {
        h ^= p[i];
        h *= 0x100000001B3ULL;
    }
    return h ^ (h >> 29);
}

std::string hash_text(std::string_view text) {
    return std::format("{:016x}{:016x}", hash_bytes(text.data(), text.size(), 1),
                       hash_bytes(text.data(), text.size(), 2));
}

Result<std::vector<char>> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return errors::not_found("cannot open {}", platform::path_to_utf8(path));
    }
    return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

/// Writes a file atomically enough for a cache: a temporary file renamed into place.
void write_file(const std::filesystem::path& path, const void* data, std::size_t size) {
    std::error_code ec;
    std::filesystem::path temporary = path;
    temporary += std::format(".{:x}.tmp", reinterpret_cast<std::uintptr_t>(&ec));
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            return;
        }
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!file) {
            file.close();
            std::filesystem::remove(temporary, ec);
            return;
        }
    }
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        std::filesystem::remove(temporary, ec);
    }
}

Status trt_error(std::string_view what) {
    const std::string message = TrtLogger::take_error();
    return errors::internal("TensorRT: {}{}{}", what, message.empty() ? "" : ": ", message);
}

/// Parses "N=1:8:64,S=1:128:512".
Result<std::map<std::string, SymbolRange>> parse_profile(std::string_view text) {
    std::map<std::string, SymbolRange> ranges;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        const std::string_view item = text.substr(0, comma);
        text = comma == std::string_view::npos ? std::string_view() : text.substr(comma + 1);
        const std::size_t eq = item.find('=');
        std::int64_t values[3] = {0, 0, 0};
        int count = 0;
        if (eq != std::string_view::npos) {
            std::string_view rest = item.substr(eq + 1);
            while (count < 3 && !rest.empty()) {
                const std::size_t colon = rest.find(':');
                values[count++] = std::strtoll(std::string(rest.substr(0, colon)).c_str(), nullptr, 10);
                rest = colon == std::string_view::npos ? std::string_view() : rest.substr(colon + 1);
            }
        }
        if (eq == std::string_view::npos || count != 3 || values[0] < 0 || values[0] > values[1] ||
            values[1] > values[2]) {
            return errors::invalid_argument("tensorrt.profile: expected symbol=min:opt:max, got '{}'", item);
        }
        ranges[std::string(item.substr(0, eq))] = SymbolRange{values[0], values[1], values[2]};
    }
    return ranges;
}

}  // namespace

// --- Logger ---------------------------------------------------------------------------------------------------------

void TrtLogger::log(Severity severity, const char* message) noexcept {
    try {
        if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
            g_last_error = message;
            logger_.log(LogLevel::kDebug, "TensorRT: {}", message);
        } else if (severity == Severity::kWARNING) {
            logger_.log(LogLevel::kTrace, "TensorRT: {}", message);
        }
    } catch (...) {  // NOLINT(bugprone-empty-catch): nothing may escape into TensorRT
    }
}

std::string TrtLogger::take_error() {
    std::string error = std::move(g_last_error);
    g_last_error.clear();
    return error;
}

// --- Compiler -------------------------------------------------------------------------------------------------------

Compiler::Compiler(const Library& library, const TrtDevice& device, const Options& options, Logger logger)
    : library_(library), device_(device), options_(options), logger_(std::move(logger)) {}

Compiler::~Compiler() {
    delete timing_;
    delete runtime_;
    delete builder_;
}

Result<std::unique_ptr<Compiler>> Compiler::create(const Library& library, const TrtDevice& device,
                                                   const Options& options, Logger logger) {
    std::unique_ptr<Compiler> compiler(new Compiler(library, device, options, std::move(logger)));
    II_ASSIGN_OR_RETURN(compiler->configured_, parse_profile(options.profile));
    II_TRY(device.activate());
    compiler->builder_ = library.builder(compiler->logger_);
    compiler->runtime_ = library.runtime(compiler->logger_);
    if (compiler->builder_ == nullptr || compiler->runtime_ == nullptr) {
        return errors::unavailable("TensorRT {} could not create a builder: {}", library.version_string(),
                                   TrtLogger::take_error());
    }
    if (options.cache != "none") {
        std::filesystem::path dir;
        if (options.cache.empty()) {
            if (Result<std::filesystem::path> root = platform::default_cache_dir(); root.ok()) {
                dir = *root / "tensorrt";
            }
        } else {
            dir = platform::path_from_utf8(options.cache);
        }
        std::error_code ec;
        if (!dir.empty() && (std::filesystem::create_directories(dir, ec), std::filesystem::is_directory(dir, ec))) {
            compiler->cache_dir_ = dir;
        }
    }
    // The timing cache (measured tactics) is shared by all builds of the device and kept on disk.
    std::vector<char> blob;
    if (compiler->cache_dir_) {
        if (Result<std::vector<char>> read =
                read_file(*compiler->cache_dir_ / std::format("timing-{}.cache", device.uuid()));
            read.ok()) {
            blob = std::move(*read);
        }
    }
    TrtPtr<nv::IBuilderConfig> config(compiler->builder_->createBuilderConfig());
    if (config) {
        compiler->timing_ = config->createTimingCache(blob.data(), blob.size());
        if (compiler->timing_ == nullptr && !blob.empty()) {
            compiler->timing_ = config->createTimingCache(nullptr, 0);  // an incompatible file: start afresh
        }
    }
    compiler->identity_ =
        std::format("TensorRT {} | {} cc {} {} | tf32 {} | level {} | ii {} translation {}", library.version_string(),
                    device.info().name, device.compute_capability(), device.uuid(), options.tf32 ? 1 : 0, options.level,
                    II_VERSION_STRING, kTranslationRevision);
    TrtLogger::take_error();
    return compiler;
}

std::optional<SymbolRange> Compiler::configured_range(const std::string& symbol) const {
    const auto it = configured_.find(symbol);
    return it == configured_.end() ? std::nullopt : std::optional<SymbolRange>(it->second);
}

void Compiler::save_timing_cache() {
    if (!cache_dir_ || timing_ == nullptr) {
        return;
    }
    TrtPtr<nv::IHostMemory> blob(timing_->serialize());
    if (blob) {
        write_file(*cache_dir_ / std::format("timing-{}.cache", device_.uuid()), blob->data(), blob->size());
    }
}

Result<std::shared_ptr<nv::ICudaEngine>>
Compiler::engine(const ir::Graph& program, const std::map<std::string, SymbolRange>& ranges, const std::string& key) {
    const std::lock_guard lock(mutex_);
    II_TRY(device_.activate());
    std::string profile;
    for (const auto& [symbol, range] : ranges) {
        profile += std::format("{}={}:{}:{};", symbol, range.min, range.opt, range.max);
    }
    const std::string name = hash_text(std::format("{}\n{}\n{}", identity_, key, profile));
    std::optional<std::filesystem::path> file;
    if (cache_dir_) {
        file = *cache_dir_ / std::format("{}.engine", name);
        if (Result<std::vector<char>> blob = read_file(*file); blob.ok()) {
            nv::ICudaEngine* cached = runtime_->deserializeCudaEngine(blob->data(), blob->size());
            if (cached != nullptr) {
                return std::shared_ptr<nv::ICudaEngine>(cached, TrtDelete<nv::ICudaEngine>());
            }
            TrtLogger::take_error();  // an engine of another TensorRT version: rebuilt below
        }
    }
    // Build: a strongly typed network keeps the model's types (TensorRT picks the kernels, not the precision).
    TrtPtr<nv::INetworkDefinition> network(builder_->createNetworkV2(
        1U << static_cast<std::uint32_t>(nv::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    TrtPtr<nv::IBuilderConfig> config(builder_->createBuilderConfig());
    if (!network || !config) {
        return trt_error("creating a network");
    }
    NetworkStorage storage;
    II_TRY(translate(program, *network, storage));
    if (!options_.tf32) {
        config->clearFlag(nv::BuilderFlag::kTF32);
    }
    config->setBuilderOptimizationLevel(options_.level);
    if (timing_ != nullptr) {
        config->setTimingCache(*timing_, false);
    }
    bool dynamic = false;
    nv::IOptimizationProfile* optimization = builder_->createOptimizationProfile();
    for (std::size_t j = 0; j < program.inputs().size(); ++j) {
        const ir::Value* input = program.inputs()[j];
        if (input->type().is_static()) {
            continue;
        }
        dynamic = true;
        nv::Dims dims[3]{};
        for (int s = 0; s < 3; ++s) {
            dims[s].nbDims = input->rank();
            for (int d = 0; d < input->rank(); ++d) {
                const std::optional<std::int64_t> value =
                    input->shape()[static_cast<std::size_t>(d)].evaluate([&](std::string_view symbol) {
                        const auto it = ranges.find(std::string(symbol));
                        if (it == ranges.end()) {
                            return std::optional<std::int64_t>();
                        }
                        return std::optional<std::int64_t>(s == 0   ? it->second.min
                                                           : s == 1 ? it->second.opt
                                                                    : it->second.max);
                    });
                if (!value) {
                    return errors::internal("no profile range for {} of input {}", input->type().to_string(), j);
                }
                dims[s].d[d] = *value;
            }
        }
        const std::string input_name = std::format("x{}", j);
        if (!optimization->setDimensions(input_name.c_str(), nv::OptProfileSelector::kMIN, dims[0]) ||
            !optimization->setDimensions(input_name.c_str(), nv::OptProfileSelector::kOPT, dims[1]) ||
            !optimization->setDimensions(input_name.c_str(), nv::OptProfileSelector::kMAX, dims[2])) {
            return trt_error("setting the optimization profile");
        }
    }
    if (dynamic && config->addOptimizationProfile(optimization) < 0) {
        return trt_error("adding the optimization profile");
    }
    TrtPtr<nv::IHostMemory> plan(builder_->buildSerializedNetwork(*network, *config));
    if (!plan) {
        return trt_error("building an engine");
    }
    nv::ICudaEngine* built = runtime_->deserializeCudaEngine(plan->data(), plan->size());
    if (built == nullptr) {
        return trt_error("loading the built engine");
    }
    if (file) {
        write_file(*file, plan->data(), plan->size());
        save_timing_cache();
    }
    return std::shared_ptr<nv::ICudaEngine>(built, TrtDelete<nv::ICudaEngine>());
}

// --- Kernel ---------------------------------------------------------------------------------------------------------

namespace {

/// A one-node program for a node that is not a `subgraph`: inputs that are constants stay constants.
Result<ops::SubgraphProgram> single_node_program(const ir::Node& node) {
    ops::SubgraphProgram out;
    out.graph = std::make_unique<ir::Graph>();
    ir::Graph& graph = *out.graph;
    std::vector<ir::Value*> inputs;
    for (int i = 0; i < node.num_inputs(); ++i) {
        const ir::Value* value = node.input(i);
        if (value == nullptr) {
            inputs.push_back(nullptr);
        } else if (value->kind() == ir::ValueKind::kConstant) {
            inputs.push_back(graph.add_constant(value->data()));
        } else {
            inputs.push_back(graph.add_input(std::format("x{}", out.inputs.size()), value->type()));
            out.inputs.push_back(i);
        }
    }
    std::vector<ir::TensorType> types;
    for (const ir::Value* output : node.outputs()) {
        types.push_back(output->type());
    }
    ir::Node* copy = graph.create_node(node.op(), std::move(inputs), node.attrs(), types);
    for (int o = 0; o < copy->num_outputs(); ++o) {
        graph.add_output(std::format("y{}", o), copy->output(o));
    }
    return out;
}

/// The execution context of an engine for one CUDA stream (one session).
struct StreamContext {
    void* stream = nullptr;
    TrtPtr<nv::IExecutionContext> context;
    std::vector<Shape> shapes;  // input shapes last set
};

/// An engine for a range of symbol values (or the error building it).
struct Engine {
    std::vector<SymbolRange> ranges;  // per symbol of the kernel
    std::shared_ptr<nv::ICudaEngine> engine;
    Status error;
    std::int64_t memory = 0;
    std::vector<bool> input_used;  // per program input
    std::mutex mutex;
    std::vector<std::unique_ptr<StreamContext>> contexts;
};

constexpr std::size_t kMemoryAlignment = 1024;  // TensorRT asks for 512 bytes for its device memory

class EngineKernel final : public backend::Kernel {
public:
    EngineKernel(Compiler& compiler, ops::SubgraphProgram program, std::vector<bool> constant_inputs)
        : compiler_(compiler),
          program_(std::move(program.graph)),
          inputs_(std::move(program.inputs)),
          constant_inputs_(std::move(constant_inputs)) {
        for (std::size_t j = 0; j < program_->inputs().size(); ++j) {
            input_names_.push_back(std::format("x{}", j));
            const ir::Value* input = program_->inputs()[j];
            for (int d = 0; d < input->rank(); ++d) {
                const std::string_view name = input->shape()[static_cast<std::size_t>(d)].symbol_name();
                if (!name.empty() &&
                    std::none_of(symbols_.begin(), symbols_.end(), [&](const Symbol& s) { return s.name == name; })) {
                    symbols_.push_back(Symbol{std::string(name), inputs_[j], d});
                }
            }
        }
        for (std::size_t k = 0; k < program_->outputs().size(); ++k) {
            output_names_.push_back(std::format("y{}", k));
        }
        // The key of the program: its text (constants elided) and the bytes of its constants.
        ir::PrintOptions print;
        print.max_constant_elements = 0;
        key_ = program_->to_text(print);
        std::uint64_t a = 3;
        std::uint64_t b = 4;
        for (const auto& value : program_->values()) {
            if (value->kind() == ir::ValueKind::kConstant) {
                const Result<Tensor> data = value->data().contiguous();
                if (data.ok()) {
                    a = hash_bytes(data->data(), data->nbytes(), a);
                    b = hash_bytes(data->data(), data->nbytes(), b ^ 0x5bd1e995);
                }
            }
        }
        key_ += std::format("\nweights {:016x}{:016x}", a, b);
    }

    bool captures_input(int input) const override {
        return static_cast<std::size_t>(input) < constant_inputs_.size() &&
               constant_inputs_[static_cast<std::size_t>(input)];
    }

    /// TensorRT can enqueue into a CUDA graph once an execution context has run with the current shapes, which the
    /// runtime guarantees (it records the second run with the same arguments).
    bool can_record() const override { return true; }

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        (void)outputs;
        std::vector<std::int64_t> values;
        for (const Symbol& symbol : symbols_) {
            values.push_back(inputs[static_cast<std::size_t>(symbol.input)][symbol.axis]);
        }
        const std::lock_guard lock(mutex_);
        if (const Engine* found = find(values); found != nullptr) {
            return found->engine ? static_cast<std::size_t>(found->memory) + kMemoryAlignment : 0;
        }
        engines_.push_back(build(values));
        const Engine& built = *engines_.back();
        return built.engine ? static_cast<std::size_t>(built.memory) + kMemoryAlignment : 0;
    }

    Status run(const backend::KernelContext& context) const override {
        void* stream = context.queue != nullptr ? context.queue->native_handle() : nullptr;
        if (stream == nullptr) {
            return errors::internal("ii-tensorrt: a kernel ran without a CUDA stream");
        }
        Engine* engine = nullptr;
        {
            const std::lock_guard lock(mutex_);
            for (const auto& candidate : engines_) {
                if (covers(*candidate, context)) {
                    engine = candidate.get();
                    break;
                }
            }
        }
        if (engine == nullptr) {
            return errors::internal("ii-tensorrt: the engine runs with shapes it was not prepared for");
        }
        if (!engine->engine) {
            return engine->error;
        }
        II_TRY(compiler_.device().activate());
        StreamContext* sc = nullptr;
        {
            const std::lock_guard lock(engine->mutex);
            for (const auto& candidate : engine->contexts) {
                if (candidate->stream == stream) {
                    sc = candidate.get();
                    break;
                }
            }
            if (sc == nullptr) {
                auto created = std::make_unique<StreamContext>();
                created->stream = stream;
                created->context.reset(
                    engine->engine->createExecutionContext(nv::ExecutionContextAllocationStrategy::kUSER_MANAGED));
                if (!created->context) {
                    return trt_error("creating an execution context");
                }
                created->shapes.resize(program_->inputs().size());
                sc = created.get();
                engine->contexts.push_back(std::move(created));
            }
        }
        nv::IExecutionContext& ctx = *sc->context;
        std::byte* memory = context.workspace.data();
        const auto misalignment = reinterpret_cast<std::uintptr_t>(memory) % kMemoryAlignment;
        if (misalignment != 0) {
            memory += kMemoryAlignment - misalignment;
        }
        if (context.workspace.size() < static_cast<std::size_t>(engine->memory) + kMemoryAlignment) {
            return errors::internal("ii-tensorrt: workspace of {} bytes, {} needed", context.workspace.size(),
                                    engine->memory + static_cast<std::int64_t>(kMemoryAlignment));
        }
        for (std::size_t j = 0; j < program_->inputs().size(); ++j) {
            if (!engine->input_used[j]) {
                continue;
            }
            const backend::TensorView& view = context.input(inputs_[j]);
            if (!(sc->shapes[j] == view.shape)) {
                nv::Dims dims{};
                dims.nbDims = view.rank();
                for (int d = 0; d < view.rank(); ++d) {
                    dims.d[d] = view.shape[d];
                }
                if (!ctx.setInputShape(input_names_[j].c_str(), dims)) {
                    return trt_error(std::format("setting the shape of input {} to {}", j, view.shape.to_string()));
                }
                sc->shapes[j] = view.shape;
            }
            if (!ctx.setTensorAddress(input_names_[j].c_str(), view.data != nullptr ? view.data : memory)) {
                return trt_error("binding an input");
            }
        }
        for (std::size_t k = 0; k < output_names_.size(); ++k) {
            void* data = context.output(static_cast<int>(k)).data;
            if (!ctx.setTensorAddress(output_names_[k].c_str(), data != nullptr ? data : memory)) {
                return trt_error("binding an output");
            }
        }
        TrtLogger::take_error();
        ctx.setDeviceMemoryV2(memory, engine->memory);
        if (const std::string error = TrtLogger::take_error(); !error.empty()) {
            return errors::internal("TensorRT: setting the device memory: {}", error);
        }
        if (!ctx.enqueueV3(static_cast<cudaStream_t>(stream))) {
            return trt_error(std::format("enqueueing the engine (memory {} at {}, workspace {})", engine->memory,
                                         static_cast<void*>(memory), context.workspace.size()));
        }
        return ok_status();
    }

private:
    struct Symbol {
        std::string name;
        int input;  // node input with the symbol as a whole dimension
        int axis;
    };

    const Engine* find(const std::vector<std::int64_t>& values) const {
        for (const auto& engine : engines_) {
            bool inside = true;
            for (std::size_t s = 0; s < values.size(); ++s) {
                inside = inside && values[s] >= engine->ranges[s].min && values[s] <= engine->ranges[s].max;
            }
            if (inside) {
                return engine.get();
            }
        }
        return nullptr;
    }

    bool covers(const Engine& engine, const backend::KernelContext& context) const {
        for (std::size_t s = 0; s < symbols_.size(); ++s) {
            const std::int64_t value = context.input(symbols_[s].input).shape[symbols_[s].axis];
            if (value < engine.ranges[s].min || value > engine.ranges[s].max) {
                return false;
            }
        }
        return true;
    }

    /// Builds an engine for symbol values around `values`: the configured range of a symbol, else 1 .. twice the
    /// value (at least 8); if TensorRT cannot build that (shapes invalid near 1), from the value up.
    std::unique_ptr<Engine> build(const std::vector<std::int64_t>& values) const {
        auto engine = std::make_unique<Engine>();
        std::map<std::string, SymbolRange> ranges;
        for (std::size_t s = 0; s < symbols_.size(); ++s) {
            const std::int64_t v = values[s];
            SymbolRange range{1, v, std::max<std::int64_t>(2 * v, 8)};
            if (const std::optional<SymbolRange> configured = compiler_.configured_range(symbols_[s].name)) {
                range = SymbolRange{std::min(configured->min, v), std::clamp(configured->opt, v, v),
                                    std::max(configured->max, v)};
                range.opt = std::clamp(configured->opt, range.min, range.max);
                if (v < configured->min || v > configured->max) {
                    range.opt = v;
                }
            }
            ranges[symbols_[s].name] = range;
        }
        Result<std::shared_ptr<nv::ICudaEngine>> built = compiler_.engine(*program_, ranges, key_);
        if (!built.ok() && !symbols_.empty()) {
            for (std::size_t s = 0; s < symbols_.size(); ++s) {
                ranges[symbols_[s].name].min = values[s];
            }
            built = compiler_.engine(*program_, ranges, key_);
        }
        for (const Symbol& symbol : symbols_) {
            engine->ranges.push_back(ranges[symbol.name]);
        }
        if (!built.ok()) {
            engine->error = built.status();
            for (std::size_t s = 0; s < symbols_.size(); ++s) {  // do not retry these values
                engine->ranges[s] = SymbolRange{values[s], values[s], values[s]};
            }
            return engine;
        }
        engine->engine = std::move(*built);
        engine->memory = engine->engine->getDeviceMemorySizeV2();
        for (const std::string& name : input_names_) {
            engine->input_used.push_back(engine->engine->getTensorIOMode(name.c_str()) == nv::TensorIOMode::kINPUT);
        }
        return engine;
    }

    Compiler& compiler_;
    std::unique_ptr<ir::Graph> program_;
    std::vector<int> inputs_;  // per program input: the node input it receives
    std::vector<bool> constant_inputs_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<Symbol> symbols_;
    std::string key_;
    mutable std::mutex mutex_;
    mutable std::vector<std::unique_ptr<Engine>> engines_;
};

}  // namespace

Result<std::unique_ptr<backend::Kernel>> create_engine_kernel(Compiler& compiler, const ir::Node& node) {
    ops::SubgraphProgram program;
    if (node.op_name() == ops::kSubgraph) {
        II_ASSIGN_OR_RETURN(program, ops::subgraph_program(node));
    } else {
        II_ASSIGN_OR_RETURN(program, single_node_program(node));
    }
    II_TRY(prepare_program(*program.graph));
    std::vector<bool> constant_inputs;
    for (int i = 0; i < node.num_inputs(); ++i) {
        constant_inputs.push_back(node.input(i) != nullptr && node.input(i)->kind() == ir::ValueKind::kConstant);
    }
    return std::unique_ptr<backend::Kernel>(
        std::make_unique<EngineKernel>(compiler, std::move(program), std::move(constant_inputs)));
}

}  // namespace ii::tensorrt
