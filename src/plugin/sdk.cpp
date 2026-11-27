// SPDX-License-Identifier: MIT
#include "plugin/sdk.h"

#include <ii/version.h>

#include "ir/text.h"
#include "plugin/abi.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <new>

// Objects behind the opaque handles of the interface (global namespace: they complete the C declarations).
struct ii_plugin_backend {
    std::unique_ptr<ii::backend::Backend> backend;
    std::string name;  // NUL-terminated copy of the backend's name
    std::mutex mutex;
    std::map<void*, std::shared_ptr<ii::Buffer>> allocations;  // device memory handed out to ii
};

struct ii_plugin_kernel {
    std::unique_ptr<ii::ir::Graph> graph;  // the node the kernel was created from (kernels may refer to it)
    std::unique_ptr<ii::backend::Kernel> kernel;
};

namespace ii::plugin::detail {

/// Forwards `parallel_for` to the executor of the loading ii.
class HostExecutor final : public Executor {
public:
    HostExecutor() = default;
    explicit HostExecutor(const ii_plugin_executor* host) noexcept { set(host); }

    void set(const ii_plugin_executor* host) noexcept {
        if (host != nullptr) {
            host_ = *host;
        }
    }

    int concurrency() const noexcept override { return std::max(1, host_.concurrency); }

    void parallel_for(std::int64_t n, std::int64_t grain,
                      FunctionRef<void(std::int64_t begin, std::int64_t end)> fn) override {
        if (n <= 0) {
            return;
        }
        if (host_.parallel_for == nullptr) {
            fn(0, n);
            return;
        }
        host_.parallel_for(
            host_.context, n, grain,
            [](void* user, std::int64_t begin, std::int64_t end) {
                (*static_cast<FunctionRef<void(std::int64_t, std::int64_t)>*>(user))(begin, end);
            },
            &fn);
    }

private:
    ii_plugin_executor host_{};
};

}  // namespace ii::plugin::detail

struct ii_plugin_queue {
    std::unique_ptr<ii::backend::Queue> queue;
    ii::plugin::detail::HostExecutor executor;  // kernels enqueued on the queue may run after `run` returns
};

struct ii_plugin_recording {
    std::unique_ptr<ii::backend::Recording> recording;
};

namespace ii::plugin {

namespace {

using detail::HostExecutor;

const PluginDefinition* g_definition = nullptr;

int32_t fail(ii_plugin_error* error, const Status& status) noexcept {
    return abi::set_error(error, static_cast<int32_t>(status.code()), status.message());
}

int32_t fail_exception(ii_plugin_error* error) noexcept {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        return abi::set_error(error, static_cast<int32_t>(StatusCode::kOutOfMemory), "out of memory");
    } catch (const std::exception& e) {
        return abi::set_error(error, static_cast<int32_t>(StatusCode::kInternal), e.what());
    } catch (...) {
        return abi::set_error(error, static_cast<int32_t>(StatusCode::kInternal), "unknown exception");
    }
}

/// Runs `fn` (returning `Status`) and converts the outcome.
template <class Fn>
int32_t guarded(ii_plugin_error* error, Fn&& fn) noexcept {
    try {
        const Status status = fn();
        return status.ok() ? 0 : fail(error, status);
    } catch (...) {
        return fail_exception(error);
    }
}

/// Parses the graph ii passed; constant data is wrapped, not copied.
Result<std::unique_ptr<ir::Graph>> parse(const ii_plugin_graph* graph) {
    if (graph == nullptr || graph->text == nullptr) {
        return errors::invalid_argument("no graph");
    }
    ir::ParseOptions options;
    options.resolve_constant = [graph](std::string_view name, const ir::TensorType& type) -> Result<Tensor> {
        for (int32_t i = 0; i < graph->constant_count; ++i) {
            const ii_plugin_constant& constant = graph->constants[i];
            if (constant.name == nullptr || name != constant.name) {
                continue;
            }
            const std::optional<Shape> shape = type.static_shape();
            if (!shape) {
                return errors::invalid_argument("constant %{} has a symbolic shape", name);
            }
            if (constant.size != storage_size(type.dtype, shape->numel())) {
                return errors::invalid_argument("constant %{}: {} bytes for {}", name, constant.size, type.to_string());
            }
            return Tensor::wrap(const_cast<void*>(constant.data), type.dtype, *shape);
        }
        return errors::invalid_argument("no data for constant %{}", name);
    };
    return ir::parse_graph(std::string_view(graph->text, graph->text_size), options);
}

/// The node of a graph passed by ii (graphs hold exactly one node in this ABI version).
Result<const ir::Node*> single_node(const ir::Graph& graph) {
    if (graph.num_nodes() != 1) {
        return errors::unsupported("expected a graph with one node, got {}", graph.num_nodes());
    }
    return graph.nodes().front().get();
}

Shape to_shape(const ii_plugin_shape& shape) {
    if (shape.rank < 0) {
        return Shape();
    }
    return Shape(std::span<const std::int64_t>(shape.dims, static_cast<std::size_t>(shape.rank)));
}

// --- Backends ---

int32_t create_backends(const ii_plugin_host* host, ii_plugin_backend** backends, int32_t capacity, int32_t* count,
                        ii_plugin_error* error) {
    *count = 0;
    return guarded(error, [&]() -> Status {
        if (host == nullptr || host->abi_version != II_PLUGIN_ABI_VERSION) {
            return errors::unsupported("unsupported plugin ABI version");
        }
        const PluginHost plugin_host(*host);
        II_ASSIGN_OR_RETURN(BackendList list, g_definition->create_backends(plugin_host));
        if (list.size() > static_cast<std::size_t>(capacity)) {
            return errors::out_of_range("{} backends, ii accepts at most {}", list.size(), capacity);
        }
        for (auto& backend : list) {
            auto handle = std::make_unique<ii_plugin_backend>();
            handle->name = std::string(backend->name());
            handle->backend = std::move(backend);
            backends[(*count)++] = handle.release();
        }
        return ok_status();
    });
}

void destroy_backend(ii_plugin_backend* backend) {
    delete backend;
}

const char* backend_name(const ii_plugin_backend* backend) {
    return backend->name.c_str();
}

void device_info(const ii_plugin_backend* backend, ii_plugin_device_info* info) {
    const backend::DeviceInfo& device = backend->backend->device_info();
    info->kind = static_cast<int32_t>(device.device.kind);
    info->index = device.device.index;
    info->name = device.name.c_str();
    info->memory_bytes = device.memory_bytes;
    info->gflops = device.gflops;
    info->bandwidth_gbs = device.bandwidth_gbs;
    info->launch_us = device.launch_us;
    info->link_gbs = device.link_gbs;
    info->link_latency_us = device.link_latency_us;
}

int32_t supports(const ii_plugin_backend* backend, const ii_plugin_graph* graph, ii_plugin_error* error) {
    return guarded(error, [&]() -> Status {
        Result<std::unique_ptr<ir::Graph>> parsed = parse(graph);
        if (!parsed.ok()) {
            return errors::unsupported("{}: {}", backend->backend->name(), parsed.status().message());
        }
        II_ASSIGN_OR_RETURN(const ir::Node* node, single_node(**parsed));
        return backend->backend->supports(*node);
    });
}

int32_t estimate_us(const ii_plugin_backend* backend, const ii_plugin_graph* graph, const ii_plugin_shape* inputs,
                    int32_t input_count, const ii_plugin_shape* outputs, int32_t output_count, double* us,
                    ii_plugin_error* error) {
    return guarded(error, [&]() -> Status {
        II_ASSIGN_OR_RETURN(std::unique_ptr<ir::Graph> parsed, parse(graph));
        II_ASSIGN_OR_RETURN(const ir::Node* node, single_node(*parsed));
        std::vector<Shape> input_shapes;
        std::vector<Shape> output_shapes;
        for (int32_t i = 0; i < input_count; ++i) {
            input_shapes.push_back(to_shape(inputs[i]));
        }
        for (int32_t i = 0; i < output_count; ++i) {
            output_shapes.push_back(to_shape(outputs[i]));
        }
        *us = backend->backend->estimate_us(ir::CostContext{*node, input_shapes, output_shapes});
        return ok_status();
    });
}

int32_t create_kernel(const ii_plugin_backend* backend, const ii_plugin_graph* graph, ii_plugin_kernel** kernel,
                      ii_plugin_error* error) {
    *kernel = nullptr;
    return guarded(error, [&]() -> Status {
        auto handle = std::make_unique<ii_plugin_kernel>();
        II_ASSIGN_OR_RETURN(handle->graph, parse(graph));
        II_ASSIGN_OR_RETURN(const ir::Node* node, single_node(*handle->graph));
        II_ASSIGN_OR_RETURN(handle->kernel, backend->backend->create_kernel(*node));
        *kernel = handle.release();
        return ok_status();
    });
}

void destroy_kernel(ii_plugin_kernel* kernel) {
    delete kernel;
}

uint32_t kernel_flags(const ii_plugin_kernel* kernel, int32_t input) {
    if (input < 0) {
        return kernel->kernel->can_record() ? II_PLUGIN_KERNEL_RECORDABLE : 0;
    }
    uint32_t flags = 0;
    if (kernel->kernel->can_run_in_place(input)) {
        flags |= II_PLUGIN_KERNEL_IN_PLACE;
    }
    if (kernel->kernel->captures_input(input)) {
        flags |= II_PLUGIN_KERNEL_CAPTURES_INPUT;
    }
    return flags;
}

int32_t workspace_size(const ii_plugin_kernel* kernel, const ii_plugin_shape* inputs, int32_t input_count,
                       const ii_plugin_shape* outputs, int32_t output_count, size_t* size, ii_plugin_error* error) {
    return guarded(error, [&]() -> Status {
        std::vector<Shape> input_shapes;
        std::vector<Shape> output_shapes;
        for (int32_t i = 0; i < input_count; ++i) {
            input_shapes.push_back(to_shape(inputs[i]));
        }
        for (int32_t i = 0; i < output_count; ++i) {
            output_shapes.push_back(to_shape(outputs[i]));
        }
        *size = kernel->kernel->workspace_size(input_shapes, output_shapes);
        return ok_status();
    });
}

backend::TensorView to_view(const ii_plugin_tensor& tensor) {
    backend::TensorView view;
    view.data = tensor.data;
    view.dtype = static_cast<DType>(tensor.dtype);
    view.shape = Shape(std::span<const std::int64_t>(tensor.shape, static_cast<std::size_t>(tensor.rank)));
    return view;
}

int32_t run(const ii_plugin_kernel* kernel, const ii_plugin_run_context* context, ii_plugin_error* error) {
    try {
        // Reused between runs: no allocation once the vector has grown (kernels must not allocate).
        thread_local std::vector<backend::TensorView> views;
        views.clear();
        for (int32_t i = 0; i < context->input_count; ++i) {
            views.push_back(to_view(context->inputs[i]));
        }
        for (int32_t i = 0; i < context->output_count; ++i) {
            views.push_back(to_view(context->outputs[i]));
        }
        const auto inputs = static_cast<std::size_t>(context->input_count);
        backend::KernelContext kernel_context;
        kernel_context.inputs = std::span<const backend::TensorView>(views.data(), inputs);
        kernel_context.outputs = std::span<const backend::TensorView>(views.data() + inputs,
                                                                      static_cast<std::size_t>(context->output_count));
        kernel_context.workspace = std::span<std::byte>(static_cast<std::byte*>(context->workspace),
                                                        context->workspace == nullptr ? 0 : context->workspace_size);
        HostExecutor executor(context->executor);
        if (context->queue != nullptr) {
            context->queue->executor.set(context->executor);
            kernel_context.executor = &context->queue->executor;
            kernel_context.queue = context->queue->queue.get();
        } else {
            kernel_context.executor = &executor;
        }
        const Status status = kernel->kernel->run(kernel_context);
        return status.ok() ? 0 : fail(error, status);
    } catch (...) {
        return fail_exception(error);
    }
}

// --- Device memory and queues ---

int32_t allocate(ii_plugin_backend* backend, size_t size, void** data, ii_plugin_error* error) {
    *data = nullptr;
    return guarded(error, [&]() -> Status {
        backend::DeviceApi* api = backend->backend->device_api();
        if (api == nullptr) {
            return errors::unsupported("{} has no device memory", backend->backend->name());
        }
        II_ASSIGN_OR_RETURN(std::shared_ptr<Buffer> buffer, api->allocate(size));
        const std::lock_guard lock(backend->mutex);
        *data = buffer->data();
        backend->allocations.emplace(*data, std::move(buffer));
        return ok_status();
    });
}

void deallocate(ii_plugin_backend* backend, void* data, size_t /*size*/) {
    std::shared_ptr<Buffer> buffer;  // released outside the lock
    const std::lock_guard lock(backend->mutex);
    const auto it = backend->allocations.find(data);
    if (it != backend->allocations.end()) {
        buffer = std::move(it->second);
        backend->allocations.erase(it);
    }
}

int32_t create_queue(ii_plugin_backend* backend, ii_plugin_queue** queue, ii_plugin_error* error) {
    *queue = nullptr;
    return guarded(error, [&]() -> Status {
        backend::DeviceApi* api = backend->backend->device_api();
        if (api == nullptr) {
            return errors::unsupported("{} has no device queues", backend->backend->name());
        }
        auto handle = std::make_unique<ii_plugin_queue>();
        II_ASSIGN_OR_RETURN(handle->queue, api->create_queue());
        *queue = handle.release();
        return ok_status();
    });
}

void destroy_queue(ii_plugin_queue* queue) {
    delete queue;
}

int32_t copy_to_device(ii_plugin_queue* queue, void* dst, const void* src, size_t size, ii_plugin_error* error) {
    return guarded(error, [&] { return queue->queue->copy_to_device(dst, src, size); });
}

int32_t copy_to_host(ii_plugin_queue* queue, void* dst, const void* src, size_t size, ii_plugin_error* error) {
    return guarded(error, [&] { return queue->queue->copy_to_host(dst, src, size); });
}

int32_t copy(ii_plugin_queue* queue, void* dst, const void* src, size_t size, ii_plugin_error* error) {
    return guarded(error, [&] { return queue->queue->copy(dst, src, size); });
}

int32_t fill_zero(ii_plugin_queue* queue, void* dst, size_t size, ii_plugin_error* error) {
    return guarded(error, [&] { return queue->queue->fill_zero(dst, size); });
}

int32_t synchronize(ii_plugin_queue* queue, ii_plugin_error* error) {
    return guarded(error, [&] { return queue->queue->synchronize(); });
}

int32_t begin_recording(ii_plugin_queue* queue, ii_plugin_error* error) {
    return guarded(error, [&] { return queue->queue->begin_recording(); });
}

int32_t end_recording(ii_plugin_queue* queue, ii_plugin_recording** recording, ii_plugin_error* error) {
    *recording = nullptr;
    return guarded(error, [&]() -> Status {
        auto handle = std::make_unique<ii_plugin_recording>();
        II_ASSIGN_OR_RETURN(handle->recording, queue->queue->end_recording());
        *recording = handle.release();
        return ok_status();
    });
}

int32_t replay(ii_plugin_queue* queue, const ii_plugin_recording* recording, ii_plugin_error* error) {
    return guarded(error, [&] { return queue->queue->replay(*recording->recording); });
}

void destroy_recording(ii_plugin_recording* recording) {
    delete recording;
}

}  // namespace

std::string PluginHost::option(std::string_view key) const {
    if (host_.get_option == nullptr) {
        return {};
    }
    const std::string name(key);
    const char* value = host_.get_option(host_.options_context, name.c_str());
    return value != nullptr ? std::string(value) : std::string();
}

Logger PluginHost::logger() const {
    if (host_.log == nullptr) {
        return Logger(LogLevel::kOff, nullptr);
    }
    auto* log = host_.log;
    void* context = host_.log_context;
    return Logger(LogLevel::kTrace, [log, context](LogLevel level, std::string_view message) {
        const std::string text(message);
        log(context, static_cast<int32_t>(level), text.c_str());
    });
}

const ii_plugin_v1* make_plugin_table(const PluginDefinition& definition) {
    g_definition = &definition;
    static const ii_plugin_v1 table = [] {
        ii_plugin_v1 t{};
        t.struct_size = sizeof(ii_plugin_v1);
        t.abi_version = II_PLUGIN_ABI_VERSION;
        t.name = g_definition->name;
        t.version = g_definition->version;
        t.ii_version = II_VERSION_STRING;
        t.create_backends = &create_backends;
        t.destroy_backend = &destroy_backend;
        t.backend_name = &backend_name;
        t.device_info = &device_info;
        t.supports = &supports;
        t.estimate_us = g_definition->custom_estimates ? &estimate_us : nullptr;
        t.create_kernel = &create_kernel;
        t.destroy_kernel = &destroy_kernel;
        t.kernel_flags = &kernel_flags;
        t.workspace_size = &workspace_size;
        t.run = &run;
        t.allocate = &allocate;
        t.deallocate = &deallocate;
        t.create_queue = &create_queue;
        t.destroy_queue = &destroy_queue;
        t.copy_to_device = &copy_to_device;
        t.copy_to_host = &copy_to_host;
        t.copy = &copy;
        t.fill_zero = &fill_zero;
        t.synchronize = &synchronize;
        t.begin_recording = &begin_recording;
        t.end_recording = &end_recording;
        t.replay = &replay;
        t.destroy_recording = &destroy_recording;
        return t;
    }();
    return &table;
}

}  // namespace ii::plugin
