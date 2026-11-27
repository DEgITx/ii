// SPDX-License-Identifier: MIT
#include "plugin/host.h"

#include <ii/version.h>

#include "core/platform/dynlib.h"
#include "core/platform/platform.h"
#include "ir/text.h"
#include "plugin/abi.h"
#include "runtime/runtime.h"

#include <algorithm>
#include <array>
#include <format>
#include <mutex>
#include <unordered_map>

namespace ii::plugin {

namespace {

/// An error struct ready for a call (only the code and the first byte are initialized: no 1 KB memset per call).
struct CallError {
    ii_plugin_error error;
    CallError() noexcept {  // NOLINT(cppcoreguidelines-pro-type-member-init)
        error.code = 0;
        error.message[0] = '\0';
    }
    ii_plugin_error* get() noexcept { return &error; }
};

Status check(int32_t code, const CallError& error) {
    return code == 0 ? ok_status() : abi::to_status(code, error.error);
}

// --- Nodes as text ---

/// A one-node graph in the text format with the data of its constants (see ii/plugin.h).
struct GraphText {
    std::string text;
    std::vector<std::string> names;
    std::vector<Tensor> data;  // keeps the constant data alive (contiguous copies where needed)
    std::vector<ii_plugin_constant> constants;

    ii_plugin_graph graph() const noexcept {
        return ii_plugin_graph{text.c_str(), text.size(), constants.data(), static_cast<int32_t>(constants.size())};
    }
};

Result<GraphText> describe_node(const ir::Node& node) {
    GraphText out;
    ir::Graph graph;
    graph.set_name("node");
    std::unordered_map<const ir::Value*, ir::Value*> mapped;
    std::vector<ir::Value*> inputs;
    int next_input = 0;
    for (const ir::Value* value : node.inputs()) {
        if (value == nullptr) {
            inputs.push_back(nullptr);
            continue;
        }
        if (const auto it = mapped.find(value); it != mapped.end()) {
            inputs.push_back(it->second);
            continue;
        }
        ir::Value* copy = nullptr;
        if (value->kind() == ir::ValueKind::kConstant) {
            Tensor data = value->data();
            if (data.device().is_host() && !data.is_contiguous()) {
                II_ASSIGN_OR_RETURN(data, data.contiguous());
            }
            std::string name = std::format("c{}", out.names.size());
            copy = graph.add_constant(data, name);
            out.names.push_back(std::move(name));
            out.data.push_back(std::move(data));
        } else {
            copy = graph.add_input(std::format("x{}", next_input++), value->type());
        }
        mapped.emplace(value, copy);
        inputs.push_back(copy);
    }
    std::vector<ir::TensorType> types;
    for (const ir::Value* output : node.outputs()) {
        types.push_back(output->type());
    }
    ir::Node* copy = graph.create_node(node.op(), std::move(inputs), node.attrs(), types);
    for (int i = 0; i < copy->num_outputs(); ++i) {
        graph.add_output(std::format("y{}", i), copy->output(i));
    }
    ir::PrintOptions print;
    print.max_constant_elements = 0;  // all constant data goes by pointer
    out.text = graph.to_text(print);
    out.constants.reserve(out.names.size());
    for (std::size_t i = 0; i < out.names.size(); ++i) {
        const Tensor& data = out.data[i];
        out.constants.push_back(ii_plugin_constant{out.names[i].c_str(), data.data(), data.nbytes()});
    }
    return out;
}

ii_plugin_shape to_plugin_shape(const Shape& shape, bool present) {
    ii_plugin_shape out{};
    out.rank = present ? shape.rank() : -1;
    std::copy(shape.begin(), shape.end(), out.dims);
    return out;
}

std::vector<ii_plugin_shape> to_plugin_shapes(std::span<const Shape> shapes, const ir::Node& node, bool inputs) {
    std::vector<ii_plugin_shape> out;
    out.reserve(shapes.size());
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        const bool present =
            !inputs || (i < static_cast<std::size_t>(node.num_inputs()) && node.input(static_cast<int>(i)) != nullptr);
        out.push_back(to_plugin_shape(shapes[i], present));
    }
    return out;
}

ii_plugin_tensor to_plugin_tensor(const backend::TensorView& view) noexcept {
    ii_plugin_tensor out;
    out.data = view.data;
    out.dtype = static_cast<int32_t>(view.dtype);
    out.rank = view.present() ? view.rank() : 0;
    std::copy(view.shape.begin(), view.shape.end(), out.shape);
    return out;
}

// --- Adapters: C interface -> ii::backend ---

class PluginQueue final : public backend::Queue {
public:
    PluginQueue(const ii_plugin_v1& table, ii_plugin_queue* handle) noexcept : table_(table), handle_(handle) {}
    ~PluginQueue() override { table_.destroy_queue(handle_); }
    PluginQueue(const PluginQueue&) = delete;
    PluginQueue& operator=(const PluginQueue&) = delete;

    Status copy_to_device(void* dst, const void* src, std::size_t size) override {
        CallError error;
        return check(table_.copy_to_device(handle_, dst, src, size, error.get()), error);
    }
    Status copy_to_host(void* dst, const void* src, std::size_t size) override {
        CallError error;
        return check(table_.copy_to_host(handle_, dst, src, size, error.get()), error);
    }
    Status copy(void* dst, const void* src, std::size_t size) override {
        CallError error;
        return check(table_.copy(handle_, dst, src, size, error.get()), error);
    }
    Status fill_zero(void* dst, std::size_t size) override {
        CallError error;
        return check(table_.fill_zero(handle_, dst, size, error.get()), error);
    }
    Status synchronize() override {
        CallError error;
        return check(table_.synchronize(handle_, error.get()), error);
    }

    const ii_plugin_v1& table() const noexcept { return table_; }
    ii_plugin_queue* handle() const noexcept { return handle_; }

private:
    const ii_plugin_v1& table_;
    ii_plugin_queue* handle_;
};

class PluginDeviceApi final : public backend::DeviceApi {
public:
    PluginDeviceApi(const ii_plugin_v1& table, ii_plugin_backend* backend, const backend::DeviceInfo& info) noexcept
        : table_(table), backend_(backend), info_(info) {}

    const backend::DeviceInfo& info() const noexcept override { return info_; }

    Result<std::shared_ptr<Buffer>> allocate(std::size_t size) override {
        CallError error;
        void* data = nullptr;
        II_TRY(check(table_.allocate(backend_, size, &data, error.get()), error));
        const ii_plugin_v1* table = &table_;
        ii_plugin_backend* backend = backend_;
        std::shared_ptr<void> owner(data, [table, backend, size](void* p) { table->deallocate(backend, p, size); });
        return Buffer::wrap(data, size, std::move(owner), info_.device);
    }

    Result<std::unique_ptr<backend::Queue>> create_queue() override {
        CallError error;
        ii_plugin_queue* queue = nullptr;
        II_TRY(check(table_.create_queue(backend_, &queue, error.get()), error));
        return std::unique_ptr<backend::Queue>(std::make_unique<PluginQueue>(table_, queue));
    }

private:
    const ii_plugin_v1& table_;
    ii_plugin_backend* backend_;
    const backend::DeviceInfo& info_;
};

void parallel_for_trampoline(void* context, int64_t n, int64_t grain, void (*fn)(void*, int64_t, int64_t), void* user) {
    static_cast<Executor*>(context)->parallel_for(
        n, grain, [fn, user](std::int64_t begin, std::int64_t end) { fn(user, begin, end); });
}

class PluginKernel final : public backend::Kernel {
public:
    PluginKernel(const ii_plugin_v1& table, ii_plugin_kernel* handle, GraphText graph, int inputs) noexcept
        : table_(table), handle_(handle), graph_(std::move(graph)), inputs_(inputs) {}
    ~PluginKernel() override { table_.destroy_kernel(handle_); }
    PluginKernel(const PluginKernel&) = delete;
    PluginKernel& operator=(const PluginKernel&) = delete;

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        std::vector<ii_plugin_shape> in;
        std::vector<ii_plugin_shape> out;
        in.reserve(inputs.size());
        out.reserve(outputs.size());
        for (const Shape& shape : inputs) {
            in.push_back(to_plugin_shape(shape, true));
        }
        for (const Shape& shape : outputs) {
            out.push_back(to_plugin_shape(shape, true));
        }
        CallError error;
        std::size_t size = 0;
        if (table_.workspace_size(handle_, in.data(), static_cast<int32_t>(in.size()), out.data(),
                                  static_cast<int32_t>(out.size()), &size, error.get()) != 0) {
            return 0;  // the error shows up when the kernel runs
        }
        return size;
    }

    Status run(const backend::KernelContext& context) const override {
        constexpr std::size_t kInline = 16;
        std::array<ii_plugin_tensor, kInline> inline_tensors;  // NOLINT(cppcoreguidelines-pro-type-member-init)
        thread_local std::vector<ii_plugin_tensor> heap_tensors;
        const std::size_t count = context.inputs.size() + context.outputs.size();
        ii_plugin_tensor* tensors = inline_tensors.data();
        if (count > kInline) {
            heap_tensors.resize(count);
            tensors = heap_tensors.data();
        }
        std::size_t n = 0;
        for (const backend::TensorView& view : context.inputs) {
            tensors[n++] = to_plugin_tensor(view);
        }
        for (const backend::TensorView& view : context.outputs) {
            tensors[n++] = to_plugin_tensor(view);
        }
        ii_plugin_executor executor{context.executor, context.executor->concurrency(), &parallel_for_trampoline};
        ii_plugin_run_context run{};
        run.inputs = tensors;
        run.input_count = static_cast<int32_t>(context.inputs.size());
        run.outputs = tensors + context.inputs.size();
        run.output_count = static_cast<int32_t>(context.outputs.size());
        run.workspace = context.workspace.data();
        run.workspace_size = context.workspace.size();
        run.executor = &executor;
        if (context.queue != nullptr) {
            const auto* queue = dynamic_cast<const PluginQueue*>(context.queue);
            if (queue == nullptr || &queue->table() != &table_) {
                return errors::internal("a kernel of plugin '{}' got a queue of another backend", table_.name);
            }
            run.queue = queue->handle();
        }
        CallError error;
        return check(table_.run(handle_, &run, error.get()), error);
    }

    bool can_run_in_place(int input) const override {
        return input < inputs_ && (table_.kernel_flags(handle_, input) & II_PLUGIN_KERNEL_IN_PLACE) != 0;
    }

    bool captures_input(int input) const override {
        return input < inputs_ && (table_.kernel_flags(handle_, input) & II_PLUGIN_KERNEL_CAPTURES_INPUT) != 0;
    }

private:
    const ii_plugin_v1& table_;
    ii_plugin_kernel* handle_;
    GraphText graph_;  // constant data must stay valid while the kernel lives
    int inputs_;
};

class PluginBackend final : public backend::Backend {
public:
    PluginBackend(const ii_plugin_v1& table, ii_plugin_backend* handle) : table_(table), handle_(handle) {
        name_ = table_.backend_name(handle_);
        ii_plugin_device_info info{};
        table_.device_info(handle_, &info);
        info_.device = Device{static_cast<DeviceKind>(info.kind), info.index};
        info_.name = info.name != nullptr ? info.name : "";
        info_.memory_bytes = info.memory_bytes;
        info_.gflops = info.gflops;
        info_.bandwidth_gbs = info.bandwidth_gbs;
        info_.launch_us = info.launch_us;
        info_.link_gbs = info.link_gbs;
        info_.link_latency_us = info.link_latency_us;
        if (!info_.device.is_host()) {
            api_ = std::make_unique<PluginDeviceApi>(table_, handle_, info_);
        }
    }
    ~PluginBackend() override {
        api_.reset();
        table_.destroy_backend(handle_);
    }
    PluginBackend(const PluginBackend&) = delete;
    PluginBackend& operator=(const PluginBackend&) = delete;

    std::string_view name() const noexcept override { return name_; }
    const backend::DeviceInfo& device_info() const noexcept override { return info_; }
    backend::DeviceApi* device_api() const noexcept override { return api_.get(); }

    double estimate_us(const ir::CostContext& context) const override {
        if (table_.estimate_us == nullptr) {
            return Backend::estimate_us(context);
        }
        Result<GraphText> text = describe_node(context.node);
        if (!text.ok()) {
            return Backend::estimate_us(context);
        }
        const std::vector<ii_plugin_shape> inputs = to_plugin_shapes(context.inputs, context.node, true);
        const std::vector<ii_plugin_shape> outputs = to_plugin_shapes(context.outputs, context.node, false);
        const ii_plugin_graph graph = text->graph();
        CallError error;
        double us = 0;
        if (table_.estimate_us(handle_, &graph, inputs.data(), static_cast<int32_t>(inputs.size()), outputs.data(),
                               static_cast<int32_t>(outputs.size()), &us, error.get()) != 0) {
            return Backend::estimate_us(context);
        }
        return us;
    }

    Status supports(const ir::Node& node) const override {
        Result<GraphText> text = describe_node(node);
        if (!text.ok()) {
            return errors::unsupported("{}: {}", name_, text.status().message());
        }
        const ii_plugin_graph graph = text->graph();
        CallError error;
        const int32_t code = table_.supports(handle_, &graph, error.get());
        if (code == 0) {
            return ok_status();
        }
        Status status = abi::to_status(code, error.error);
        if (status.code() != StatusCode::kUnsupported) {
            return errors::unsupported("{}: {}", name_, status.message());
        }
        return status;
    }

    Result<std::unique_ptr<backend::Kernel>> create_kernel(const ir::Node& node) const override {
        II_ASSIGN_OR_RETURN(GraphText text, describe_node(node));
        const ii_plugin_graph graph = text.graph();
        CallError error;
        ii_plugin_kernel* kernel = nullptr;
        II_TRY(check(table_.create_kernel(handle_, &graph, &kernel, error.get()), error));
        return std::unique_ptr<backend::Kernel>(
            std::make_unique<PluginKernel>(table_, kernel, std::move(text), node.num_inputs()));
    }

private:
    const ii_plugin_v1& table_;
    ii_plugin_backend* handle_;
    std::string name_;
    backend::DeviceInfo info_;
    std::unique_ptr<PluginDeviceApi> api_;
};

// --- Host services ---

void log_callback(void* context, int32_t level, const char* message) {
    const auto* logger = static_cast<const Logger*>(context);
    if (logger != nullptr && message != nullptr) {
        logger->log(static_cast<LogLevel>(std::clamp<int32_t>(level, II_PLUGIN_LOG_TRACE, II_PLUGIN_LOG_ERROR)),
                    message);
    }
}

const char* option_callback(void* context, const char* key) {
    const auto* config = static_cast<const HostConfig*>(context);
    if (config == nullptr || key == nullptr) {
        return nullptr;
    }
    for (const auto& [name, value] : config->options) {
        if (name == key) {
            return value.c_str();
        }
    }
    return nullptr;
}

// --- Loaded libraries ---

struct Library {
    std::string key;
    platform::DynLib library;
    const ii_plugin_v1* table = nullptr;
};

std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

/// Never destroyed: plugin code may run until the process exits.
std::vector<Library*>& registry() {
    static auto* libraries = new std::vector<Library*>();  // NOLINT(cppcoreguidelines-owning-memory)
    return *libraries;
}

std::string_view platform_extension() {
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

void add_directory(const std::filesystem::path& directory, bool explicit_path,
                   std::vector<std::pair<std::filesystem::path, bool>>& out) {
    std::error_code error;
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        std::error_code file_error;
        if (entry.is_regular_file(file_error) && is_plugin_file_name(entry.path())) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    for (auto& file : files) {
        out.emplace_back(std::move(file), explicit_path);
    }
}

/// The absolute, normalized form of a path: the identity of a plugin file.
std::filesystem::path canonical_path(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::path canonical = std::filesystem::weakly_canonical(path, error);
    if (error) {
        canonical = path;
    }
    canonical.make_preferred();
    return canonical;
}

std::string describe_path(const std::filesystem::path& path) {
    return platform::path_to_utf8(canonical_path(path));
}

}  // namespace

Status validate_table(const ii_plugin_v1* table) {
    if (table == nullptr) {
        return errors::unsupported("the plugin does not support plugin ABI version {}", II_PLUGIN_ABI_VERSION);
    }
    if (table->abi_version != II_PLUGIN_ABI_VERSION) {
        return errors::unsupported("the plugin uses plugin ABI version {}; this ii supports version {}",
                                   table->abi_version, II_PLUGIN_ABI_VERSION);
    }
    if (table->struct_size < sizeof(ii_plugin_v1)) {
        return errors::unsupported("the plugin's function table is incomplete ({} of {} bytes)", table->struct_size,
                                   sizeof(ii_plugin_v1));
    }
    if (table->name == nullptr || table->create_backends == nullptr || table->destroy_backend == nullptr ||
        table->backend_name == nullptr || table->device_info == nullptr || table->supports == nullptr ||
        table->create_kernel == nullptr || table->destroy_kernel == nullptr || table->kernel_flags == nullptr ||
        table->workspace_size == nullptr || table->run == nullptr) {
        return errors::unsupported("the plugin's function table lacks required functions");
    }
    return ok_status();
}

Result<const ii_plugin_v1*> load_library(const std::filesystem::path& path) {
    const std::filesystem::path canonical = canonical_path(path);
    const std::string key = platform::path_to_utf8(canonical);
    const std::lock_guard lock(registry_mutex());
    for (const Library* library : registry()) {
        if (library->key == key) {
            return library->table;
        }
    }
    II_ASSIGN_OR_RETURN(platform::DynLib library, platform::DynLib::open(canonical));
    Result<ii_backend_plugin_v1_fn> entry = library.function<ii_backend_plugin_v1_fn>(II_PLUGIN_ENTRY_POINT);
    if (!entry.ok()) {
        return errors::unsupported("not an ii backend plugin: it does not export {}", II_PLUGIN_ENTRY_POINT);
    }
    const ii_plugin_v1* table = (*entry)(II_PLUGIN_ABI_VERSION);
    II_TRY(validate_table(table));
    auto* loaded = new Library{key, std::move(library), table};  // NOLINT(cppcoreguidelines-owning-memory)
    registry().push_back(loaded);
    return table;
}

Result<std::vector<std::unique_ptr<backend::Backend>>> create_backends(const ii_plugin_v1& table,
                                                                       const HostConfig& config) {
    ii_plugin_host host{};
    host.struct_size = sizeof(ii_plugin_host);
    host.abi_version = II_PLUGIN_ABI_VERSION;
    host.ii_version = II_VERSION_STRING;
    host.log_context = const_cast<Logger*>(config.logger);
    host.log = &log_callback;
    host.options_context = const_cast<HostConfig*>(&config);
    host.get_option = &option_callback;

    constexpr int32_t kCapacity = 64;
    std::array<ii_plugin_backend*, kCapacity> handles{};
    int32_t count = 0;
    CallError error;
    const int32_t code = table.create_backends(&host, handles.data(), kCapacity, &count, error.get());
    if (code != 0) {
        return abi::to_status(code, error.error);
    }
    count = std::clamp<int32_t>(count, 0, kCapacity);
    std::vector<std::unique_ptr<backend::Backend>> backends;
    Status status = ok_status();
    for (int32_t i = 0; i < count; ++i) {
        ii_plugin_device_info info{};
        table.device_info(handles[static_cast<std::size_t>(i)], &info);
        const bool device = info.kind != static_cast<int32_t>(DeviceKind::kCpu);
        if (device &&
            (table.allocate == nullptr || table.deallocate == nullptr || table.create_queue == nullptr ||
             table.destroy_queue == nullptr || table.copy_to_device == nullptr || table.copy_to_host == nullptr ||
             table.copy == nullptr || table.fill_zero == nullptr || table.synchronize == nullptr)) {
            status = errors::unsupported("backend '{}' runs on a device but the plugin provides no device memory",
                                         table.backend_name(handles[static_cast<std::size_t>(i)]));
        }
        if (info.kind < 0 || info.kind > static_cast<int32_t>(DeviceKind::kTest)) {
            status = errors::unsupported("backend '{}' runs on an unknown kind of device ({})",
                                         table.backend_name(handles[static_cast<std::size_t>(i)]), info.kind);
        }
        backends.push_back(std::make_unique<PluginBackend>(table, handles[static_cast<std::size_t>(i)]));
    }
    if (!status.ok()) {
        return status;  // destroys every backend
    }
    return backends;
}

bool is_plugin_file_name(const std::filesystem::path& path) {
    const std::string name = platform::path_to_utf8(path.filename());
    const std::string_view extension = platform_extension();
    const std::string_view prefix = name.starts_with("lib") ? "libii-backend-" : "ii-backend-";
    if (!name.starts_with(prefix)) {
        return false;
    }
    // CMake names MODULE libraries "*.so" on macOS too.
    for (const std::string_view suffix : {extension, std::string_view(".so")}) {
        if (name.size() > prefix.size() + suffix.size() && name.ends_with(suffix) &&
            (suffix == extension || extension == ".dylib")) {
            return true;
        }
    }
    return false;
}

std::vector<std::pair<std::filesystem::path, bool>> find_plugins(const ContextOptions& options) {
    std::vector<std::pair<std::filesystem::path, bool>> out;
    for (const std::string& item : options.plugin_paths) {
        const std::filesystem::path path = platform::path_from_utf8(item);
        std::error_code error;
        if (std::filesystem::is_directory(path, error)) {
            add_directory(path, true, out);
        } else {
            out.emplace_back(path, true);
        }
    }
    if (!options.search_plugins) {
        return out;
    }
#if defined(_WIN32)
    constexpr char kSeparator = ';';
#else
    constexpr char kSeparator = ':';
#endif
    if (const std::optional<std::string> env = platform::get_env("II_PLUGIN_PATH")) {
        std::string_view list = *env;
        while (!list.empty()) {
            const std::size_t end = list.find(kSeparator);
            const std::string_view item = list.substr(0, end);
            list = end == std::string_view::npos ? std::string_view() : list.substr(end + 1);
            if (item.empty()) {
                continue;
            }
            const std::filesystem::path path = platform::path_from_utf8(item);
            std::error_code error;
            if (std::filesystem::is_directory(path, error)) {
                add_directory(path, false, out);
            } else if (std::filesystem::is_regular_file(path, error)) {
                out.emplace_back(path, false);
            }
        }
    }
    if (const Result<std::filesystem::path> library = platform::this_library_path(); library.ok()) {
        add_directory(library->parent_path(), false, out);
    }
    return out;
}

Status add_plugin(detail::ContextImpl& context, const ii_plugin_v1& table, std::string path,
                  std::span<const std::pair<std::string, std::string>> options) {
    PluginDescription description;
    description.name = table.name;
    description.path = std::move(path);
    description.version = table.version != nullptr ? table.version : "";
    for (const PluginDescription& other : context.plugins) {
        if (other.name == description.name && other.state == PluginState::kLoaded) {
            if (other.path == description.path) {
                return ok_status();  // found twice
            }
            description.state = PluginState::kFailed;
            description.message = std::format("skipped: '{}' is already loaded from {}", other.name, other.path);
            context.logger.log(LogLevel::kWarning, "plugin {}: {}", description.path, description.message);
            context.plugins.push_back(std::move(description));
            return ok_status();
        }
    }
    HostConfig config;
    config.logger = &context.logger;
    config.options.assign(options.begin(), options.end());
    Result<std::vector<std::unique_ptr<backend::Backend>>> backends = create_backends(table, config);
    if (!backends.ok()) {
        const bool unavailable = backends.status().code() == StatusCode::kUnavailable;
        description.state = unavailable ? PluginState::kUnavailable : PluginState::kFailed;
        description.message = std::string(backends.status().message());
        context.logger.log(unavailable ? LogLevel::kInfo : LogLevel::kWarning, "plugin {}: {}", description.name,
                           description.message);
        context.plugins.push_back(std::move(description));
        return ok_status();
    }
    description.state = PluginState::kLoaded;
    for (auto& backend : *backends) {
        if (context.find_backend(backend->name()) != nullptr) {
            const std::string note =
                std::format("backend '{}' skipped: a backend with that name exists", backend->name());
            context.logger.log(LogLevel::kWarning, "plugin {}: {}", description.name, note);
            description.message += description.message.empty() ? note : "; " + note;
            continue;
        }
        description.backends.emplace_back(backend->name());
        context.backends.push_back(std::move(backend));
    }
    context.plugins.push_back(std::move(description));
    return ok_status();
}

Status load_plugins(detail::ContextImpl& context, const ContextOptions& options, std::string_view test_devices) {
    // ContextOptions first: lookups return the first match, so they override II_PLUGIN_OPTIONS.
    std::vector<std::pair<std::string, std::string>> plugin_options = options.plugin_options;
    plugin_options.emplace_back("test_devices", std::string(test_devices));
    const std::string environment = platform::get_env("II_PLUGIN_OPTIONS").value_or("");
    for (std::string_view rest = environment; !rest.empty();) {
        const std::size_t end = rest.find(';');
        const std::string_view item = rest.substr(0, end);
        rest = end == std::string_view::npos ? std::string_view() : rest.substr(end + 1);
        const std::size_t eq = item.find('=');
        if (eq != std::string_view::npos && eq > 0) {
            plugin_options.emplace_back(std::string(item.substr(0, eq)), std::string(item.substr(eq + 1)));
        }
    }
    for (const auto& [path, explicit_path] : find_plugins(options)) {
        Result<const ii_plugin_v1*> table = load_library(path);
        if (!table.ok()) {
            if (explicit_path) {
                return std::move(table).status().add_context(std::format("plugin {}", describe_path(path)));
            }
            PluginDescription description;
            description.name = platform::path_to_utf8(path.stem());
            description.path = describe_path(path);
            description.state = PluginState::kFailed;
            description.message = std::string(table.status().message());
            context.logger.log(LogLevel::kWarning, "plugin {}: {}", description.path, description.message);
            context.plugins.push_back(std::move(description));
            continue;
        }
        II_TRY(add_plugin(context, **table, describe_path(path), plugin_options));
    }
    return ok_status();
}

}  // namespace ii::plugin
