// SPDX-License-Identifier: MIT
#include "backends/test_remote/test_remote.h"

#include "backends/ref/ref_backend.h"
#include "ir/op_schema.h"
#include "ops/fused.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <map>
#include <mutex>
#include <unordered_map>

namespace ii::test_remote {

namespace {

constexpr std::size_t kAlignment = 256;
constexpr unsigned char kPoison = 0xFF;  // all-ones bytes: NaN in every float format

/// The memory of one device: tracked allocations.
class DeviceMemory {
public:
    explicit DeviceMemory(std::uint64_t capacity) : capacity_(capacity) {}

    Result<void*> allocate(std::size_t size) {
        const std::lock_guard lock(mutex_);
        if (capacity_ != 0 && used_ + size > capacity_) {
            return errors::out_of_memory("test device: allocating {} bytes with {} of {} in use", size, used_,
                                         capacity_);
        }
        void* data = default_allocator().allocate(size, kAlignment);
        if (data == nullptr) {
            return errors::out_of_memory("test device: allocating {} bytes", size);
        }
        std::memset(data, kPoison, size);
        blocks_.emplace(reinterpret_cast<std::uintptr_t>(data), size);
        used_ += size;
        return data;
    }

    void free(void* data, std::size_t size) noexcept {
        {
            const std::lock_guard lock(mutex_);
            blocks_.erase(reinterpret_cast<std::uintptr_t>(data));
            used_ -= size;
        }
        default_allocator().deallocate(data, size, kAlignment);
    }

    /// True if [data, data + size) lies inside one allocation.
    bool contains(const void* data, std::size_t size) const {
        const auto address = reinterpret_cast<std::uintptr_t>(data);
        const std::lock_guard lock(mutex_);
        auto it = blocks_.upper_bound(address);
        if (it == blocks_.begin()) {
            return false;
        }
        --it;
        return address >= it->first && address - it->first + size <= it->second;
    }

    std::uint64_t used() const {
        const std::lock_guard lock(mutex_);
        return used_;
    }

private:
    const std::uint64_t capacity_;
    mutable std::mutex mutex_;
    std::map<std::uintptr_t, std::size_t> blocks_;
    std::uint64_t used_ = 0;
};

/// Deferred work, executed by `synchronize()` in order.
class TestQueue final : public backend::Queue {
public:
    TestQueue(const DeviceMemory& memory, std::string device) : memory_(memory), device_(std::move(device)) {
        tasks_.reserve(1024);
        views_.reserve(4096);
    }

    Status copy_to_device(void* dst, const void* src, std::size_t size) override {
        II_TRY(not_recording("copy_to_device"));
        II_TRY(check(dst, size, "copy destination"));
        // Earlier work may still read the destination; the source is consumed now.
        II_TRY(synchronize());
        std::memcpy(dst, src, size);
        return ok_status();
    }

    Status copy_to_host(void* dst, const void* src, std::size_t size) override {
        II_TRY(check(src, size, "copy source"));
        tasks_.push_back(Task{Kind::kCopy, nullptr, {}, dst, src, size});
        return ok_status();
    }

    Status copy(void* dst, const void* src, std::size_t size) override {
        II_TRY(check(src, size, "copy source"));
        II_TRY(check(dst, size, "copy destination"));
        tasks_.push_back(Task{Kind::kCopy, nullptr, {}, dst, src, size});
        return ok_status();
    }

    Status fill_zero(void* dst, std::size_t size) override {
        II_TRY(check(dst, size, "fill destination"));
        tasks_.push_back(Task{Kind::kFill, nullptr, {}, dst, nullptr, size});
        return ok_status();
    }

    Status synchronize() override {
        II_TRY(not_recording("synchronize"));
        Status first = ok_status();
        for (const Task& task : tasks_) {
            if (!first.ok()) {
                break;  // like a failed GPU stream: later work is not executed
            }
            switch (task.kind) {
            case Kind::kKernel: {
                backend::KernelContext context = task.context;
                context.inputs =
                    std::span<const backend::TensorView>(views_.data() + task.first_view, context.inputs.size());
                context.outputs = std::span<const backend::TensorView>(
                    views_.data() + task.first_view + context.inputs.size(), context.outputs.size());
                first = task.kernel->run(context);
                break;
            }
            case Kind::kCopy:
                std::memmove(task.dst, task.src, task.size);
                break;
            case Kind::kFill:
                std::memset(task.dst, 0, task.size);
                break;
            }
        }
        tasks_.clear();
        views_.clear();
        return first;
    }

    // Recording keeps the tasks enqueued meanwhile (with their arguments) instead of running them, like CUDA stream
    // capture; replay enqueues copies of them.
    Status begin_recording() override {
        II_TRY(not_recording("begin_recording"));
        recording_ = true;
        record_tasks_ = tasks_.size();
        record_views_ = views_.size();
        return ok_status();
    }

    Result<std::unique_ptr<backend::Recording>> end_recording() override {
        if (!recording_) {
            return errors::internal("{}: end_recording without begin_recording", device_);
        }
        recording_ = false;
        auto recording = std::make_unique<TestRecording>();
        recording->queue = this;
        recording->tasks.assign(tasks_.begin() + static_cast<std::ptrdiff_t>(record_tasks_), tasks_.end());
        recording->views.assign(views_.begin() + static_cast<std::ptrdiff_t>(record_views_), views_.end());
        for (Task& task : recording->tasks) {
            task.first_view -= record_views_;
        }
        tasks_.resize(record_tasks_);
        views_.resize(record_views_);
        return std::unique_ptr<backend::Recording>(std::move(recording));
    }

    Status replay(const backend::Recording& recording) override {
        II_TRY(not_recording("replay"));
        const auto* own = dynamic_cast<const TestRecording*>(&recording);
        if (own == nullptr || own->queue != this) {
            return errors::internal("{}: a recording of another queue", device_);
        }
        const std::size_t offset = views_.size();
        views_.insert(views_.end(), own->views.begin(), own->views.end());
        for (Task task : own->tasks) {
            task.first_view += offset;
            tasks_.push_back(task);
        }
        return ok_status();
    }

    /// Defers a kernel. The views are copied (like the arguments of a GPU kernel launch): the caller's arrays need not
    /// outlive the call. The executor must stay valid until the queue is synchronized.
    void enqueue(const backend::Kernel& kernel, const backend::KernelContext& context) {
        const std::size_t first_view = views_.size();
        views_.insert(views_.end(), context.inputs.begin(), context.inputs.end());
        views_.insert(views_.end(), context.outputs.begin(), context.outputs.end());
        tasks_.push_back(Task{Kind::kKernel, &kernel, context, nullptr, nullptr, 0, first_view});
    }

    const DeviceMemory& memory() const noexcept { return memory_; }
    const std::string& device() const noexcept { return device_; }

    /// Kernels reach the queue through its native handle, like TensorRT reaches a CUDA stream of ii-cuda.
    void* native_handle() const noexcept override { return const_cast<TestQueue*>(this); }

private:
    enum class Kind : std::uint8_t { kKernel, kCopy, kFill };
    struct Task {
        Kind kind;
        const backend::Kernel* kernel;
        backend::KernelContext context;
        void* dst;
        const void* src;
        std::size_t size;
        std::size_t first_view = 0;  // kernels: inputs, then outputs, in views_
    };

    struct TestRecording final : backend::Recording {
        const TestQueue* queue = nullptr;
        std::vector<Task> tasks;
        std::vector<backend::TensorView> views;
    };

    Status check(const void* data, std::size_t size, std::string_view what) const {
        if (size != 0 && !memory_.contains(data, size)) {
            return errors::invalid_argument("{} of {} bytes is not in the memory of {}", what, size, device_);
        }
        return ok_status();
    }

    /// Operations that cannot be recorded (they run or wait on the host now) fail while recording, as with CUDA.
    Status not_recording(std::string_view what) const {
        if (recording_) {
            return errors::internal("{}: {} while recording", device_, what);
        }
        return ok_status();
    }

    const DeviceMemory& memory_;
    std::string device_;
    std::vector<Task> tasks_;
    std::vector<backend::TensorView> views_;
    bool recording_ = false;
    std::size_t record_tasks_ = 0;
    std::size_t record_views_ = 0;
};

class TestDeviceApi final : public backend::DeviceApi {
public:
    explicit TestDeviceApi(backend::DeviceInfo info) : info_(std::move(info)), memory_(info_.memory_bytes) {}

    const backend::DeviceInfo& info() const noexcept override { return info_; }

    Result<std::shared_ptr<Buffer>> allocate(std::size_t size) override {
        II_ASSIGN_OR_RETURN(void* data, memory_.allocate(size));
        DeviceMemory* memory = &memory_;
        std::shared_ptr<void> owner(data, [memory, size](void* p) { memory->free(p, size); });
        return Buffer::wrap(data, size, std::move(owner), info_.device);
    }

    Result<std::unique_ptr<backend::Queue>> create_queue() override {
        return std::unique_ptr<backend::Queue>(std::make_unique<TestQueue>(memory_, to_string(info_.device)));
    }

private:
    backend::DeviceInfo info_;
    DeviceMemory memory_;
};

/// An ii-ref kernel executed on the device's queue.
class RemoteKernel final : public backend::Kernel {
public:
    explicit RemoteKernel(std::unique_ptr<backend::Kernel> inner) : inner_(std::move(inner)) {}

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        return inner_->workspace_size(inputs, outputs);
    }

    bool can_run_in_place(int input) const override { return inner_->can_run_in_place(input); }
    bool captures_input(int input) const override { return inner_->captures_input(input); }

    /// The kernel only enqueues itself; the queue runs it with the recorded arguments.
    bool can_record() const override { return true; }

    Status run(const backend::KernelContext& context) const override {
        if (context.queue == nullptr || context.queue->native_handle() == nullptr) {
            return errors::internal("ii-test-remote kernel run without a device queue");
        }
        auto& queue = *static_cast<TestQueue*>(context.queue->native_handle());
        for (std::size_t i = 0; i < context.inputs.size(); ++i) {
            if (!inner_->captures_input(static_cast<int>(i))) {  // captured constants stay in host memory
                II_TRY(check(queue, context.inputs[i], "input", i));
            }
        }
        for (std::size_t o = 0; o < context.outputs.size(); ++o) {
            II_TRY(check(queue, context.outputs[o], "output", o));
        }
        if (!context.workspace.empty() &&
            !queue.memory().contains(context.workspace.data(), context.workspace.size())) {
            return errors::internal("the workspace is not in the memory of {}", queue.device());
        }
        queue.enqueue(*inner_, context);
        return ok_status();
    }

private:
    /// Allocation-free unless the check fails (the runtime's steady state allocates nothing).
    static Status check(const TestQueue& queue, const backend::TensorView& view, const char* what, std::size_t index) {
        if (!view.present() || view.nbytes() == 0 || queue.memory().contains(view.data, view.nbytes())) {
            return ok_status();
        }
        return errors::internal("{} {} ({}{}) is not in the memory of {}: a transfer is missing", what, index,
                                to_string(view.dtype), view.shape.to_string(), queue.device());
    }

    std::unique_ptr<backend::Kernel> inner_;
};

class TestRemoteBackend final : public backend::Backend {
public:
    explicit TestRemoteBackend(const TestDeviceOptions& options)
        : name_(options.index == 0 ? std::string(kBackendName) : std::format("{}.{}", kBackendName, options.index)),
          ops_(options.ops),
          reference_(ref::create_ref_backend()) {
        backend::DeviceInfo info;
        info.device = Device{DeviceKind::kTest, options.index};
        info.name = std::format("ii test device {}", options.index);
        info.memory_bytes = options.memory_bytes;
        info.gflops = options.gflops;
        info.bandwidth_gbs = options.bandwidth_gbs;
        info.launch_us = options.launch_us;
        info.link_gbs = options.link_gbs;
        info.link_latency_us = options.link_latency_us;
        api_ = std::make_unique<TestDeviceApi>(std::move(info));
    }

    std::string_view name() const noexcept override { return name_; }
    const backend::DeviceInfo& device_info() const noexcept override { return api_->info(); }
    backend::DeviceApi* device_api() const noexcept override { return api_.get(); }

    Status supports(const ir::Node& node) const override {
        if (!ops_.empty() && std::find(ops_.begin(), ops_.end(), node.op_name()) == ops_.end()) {
            return errors::unsupported("{}: no kernel for '{}'", name_, node.op_name());
        }
        if (Status status = reference_->supports(node); !status.ok()) {
            std::string_view message = status.message();
            const std::string_view prefix = "ii-ref: ";
            if (message.starts_with(prefix)) {
                message.remove_prefix(prefix.size());
            }
            return errors::unsupported("{}: {}", name_, message);
        }
        return ok_status();
    }

    Result<std::unique_ptr<backend::Kernel>> create_kernel(const ir::Node& node) const override {
        II_TRY(supports(node));
        II_ASSIGN_OR_RETURN(std::unique_ptr<backend::Kernel> inner, reference_->create_kernel(node));
        return std::unique_ptr<backend::Kernel>(std::make_unique<RemoteKernel>(std::move(inner)));
    }

private:
    std::string name_;
    std::vector<std::string> ops_;
    std::unique_ptr<backend::Backend> reference_;
    std::unique_ptr<TestDeviceApi> api_;
};

/// Runs the program of a `subgraph` node with ii-ref kernels, one after the other, keeping intermediate values in the
/// workspace: a stand-in for a graph backend's compiled kernel. Constants of the program are captured (read where
/// they are), like weights built into a TensorRT engine.
class SubgraphKernel final : public backend::Kernel {
public:
    static Result<std::unique_ptr<SubgraphKernel>> create(const ir::Node& node, const backend::Backend& reference) {
        auto kernel = std::unique_ptr<SubgraphKernel>(new SubgraphKernel());
        II_ASSIGN_OR_RETURN(ops::SubgraphProgram program, ops::subgraph_program(node));
        kernel->graph_ = std::move(program.graph);
        kernel->inputs_ = std::move(program.inputs);
        for (int i = 0; i < node.num_inputs(); ++i) {
            kernel->constant_inputs_.push_back(node.input(i)->kind() == ir::ValueKind::kConstant);
        }
        const ir::Graph& graph = *kernel->graph_;
        for (const auto& value : graph.values()) {
            kernel->value_index_.emplace(value.get(), static_cast<int>(kernel->value_index_.size()));
        }
        kernel->output_of_value_.assign(graph.values().size(), -1);
        for (std::size_t o = 0; o < graph.outputs().size(); ++o) {
            kernel->output_of_value_[static_cast<std::size_t>(kernel->index(graph.outputs()[o].value))] =
                static_cast<int>(o);
        }
        for (const auto& inner : graph.nodes()) {
            II_ASSIGN_OR_RETURN(std::unique_ptr<backend::Kernel> created, reference.create_kernel(*inner));
            kernel->kernels_.push_back(std::move(created));
        }
        return kernel;
    }

    bool captures_input(int input) const override {
        return static_cast<std::size_t>(input) < constant_inputs_.size() &&
               constant_inputs_[static_cast<std::size_t>(input)];
    }

    std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const override {
        (void)outputs;
        const Result<std::shared_ptr<const Layout>> layout = layout_for(inputs);
        return layout.ok() ? (*layout)->size : 0;
    }

    Status run(const backend::KernelContext& context) const override {
        std::shared_ptr<const Layout> layout;
        {
            const std::lock_guard lock(mutex_);
            for (const auto& known : layouts_) {
                if (matches(*known, context)) {
                    layout = known;
                    break;
                }
            }
        }
        if (!layout) {
            return errors::internal("{}: a subgraph runs with shapes it was not prepared for", kGraphBackendName);
        }
        if (context.workspace.size() < layout->size) {
            return errors::internal("{}: workspace of {} bytes, {} needed", kGraphBackendName, context.workspace.size(),
                                    layout->size);
        }
        const ir::Graph& graph = *graph_;
        // Reused: no allocation once grown (kernels must not allocate).
        thread_local std::vector<backend::TensorView> views;
        thread_local std::vector<backend::TensorView> in;
        thread_local std::vector<backend::TensorView> out;
        views.resize(graph.values().size());
        for (std::size_t v = 0; v < graph.values().size(); ++v) {
            const ir::Value& value = *graph.values()[v];
            backend::TensorView& view = views[v];
            view.dtype = value.dtype();
            view.shape = layout->shapes[v];
            view.data = nullptr;
            if (value.kind() == ir::ValueKind::kConstant) {
                view.data = const_cast<void*>(value.data().data());
            } else if (output_of_value_[v] >= 0) {
                view.data = context.output(output_of_value_[v]).data;
            } else if (layout->offsets[v] != kExternal) {
                view.data = context.workspace.data() + layout->offsets[v];
            }
        }
        for (std::size_t j = 0; j < graph.inputs().size(); ++j) {
            views[static_cast<std::size_t>(index(graph.inputs()[j]))].data = context.input(inputs_[j]).data;
        }
        const std::span<std::byte> scratch = context.workspace.subspan(layout->scratch);
        for (std::size_t n = 0; n < graph.nodes().size(); ++n) {
            const ir::Node& node = *graph.nodes()[n];
            in.clear();
            out.clear();
            for (const ir::Value* value : node.inputs()) {
                in.push_back(value == nullptr ? backend::TensorView{} : views[static_cast<std::size_t>(index(value))]);
            }
            for (const ir::Value* value : node.outputs()) {
                out.push_back(views[static_cast<std::size_t>(index(value))]);
            }
            backend::KernelContext inner;
            inner.inputs = in;
            inner.outputs = out;
            inner.workspace = scratch;
            inner.executor = context.executor;
            II_TRY(kernels_[n]->run(inner));
        }
        return ok_status();
    }

private:
    static constexpr std::size_t kExternal = ~std::size_t{0};

    /// Shapes and workspace offsets of the program's values for one set of input shapes.
    struct Layout {
        std::vector<Shape> inputs;         // the node's input shapes this layout is for
        std::vector<Shape> shapes;         // per value of the program
        std::vector<std::size_t> offsets;  // per value: offset in the workspace, or kExternal
        std::size_t scratch = 0;           // offset of the inner kernels' workspace
        std::size_t size = 0;
    };

    SubgraphKernel() = default;

    int index(const ir::Value* value) const { return value_index_.at(value); }

    static bool matches(const Layout& layout, const backend::KernelContext& context) {
        if (layout.inputs.size() != context.inputs.size()) {
            return false;
        }
        for (std::size_t i = 0; i < layout.inputs.size(); ++i) {
            if (!(layout.inputs[i] == context.inputs[i].shape)) {
                return false;
            }
        }
        return true;
    }

    /// Shape inference of the program for concrete input shapes (cached).
    Result<std::shared_ptr<const Layout>> layout_for(std::span<const Shape> inputs) const {
        {
            const std::lock_guard lock(mutex_);
            for (const auto& known : layouts_) {
                if (std::equal(known->inputs.begin(), known->inputs.end(), inputs.begin(), inputs.end())) {
                    return known;
                }
            }
        }
        const ir::Graph& graph = *graph_;
        auto layout = std::make_shared<Layout>();
        layout->inputs.assign(inputs.begin(), inputs.end());
        layout->shapes.assign(graph.values().size(), Shape());
        layout->offsets.assign(graph.values().size(), kExternal);
        std::vector<Shape> program_inputs;
        for (std::size_t j = 0; j < graph.inputs().size(); ++j) {
            const Shape& shape = inputs[static_cast<std::size_t>(inputs_[j])];
            program_inputs.push_back(shape);
            layout->shapes[static_cast<std::size_t>(index(graph.inputs()[j]))] = shape;
        }
        II_ASSIGN_OR_RETURN(const auto symbols, ops::subgraph_symbols(graph, program_inputs));
        auto lookup = [&](std::string_view name) -> std::optional<std::int64_t> {
            for (const auto& [symbol, value] : symbols) {
                if (symbol == name) {
                    return value;
                }
            }
            return std::nullopt;
        };
        for (const auto& value : graph.values()) {
            if (value->kind() == ir::ValueKind::kConstant) {
                layout->shapes[static_cast<std::size_t>(index(value.get()))] = value->data().shape();
            }
        }
        std::size_t offset = 0;
        std::size_t scratch = 0;
        std::vector<ir::TensorType> in_types;
        std::vector<const ir::TensorType*> in_ptrs;
        std::vector<ir::TensorType> out_types;
        std::vector<Shape> in_shapes;
        std::vector<Shape> out_shapes;
        for (std::size_t n = 0; n < graph.nodes().size(); ++n) {
            const ir::Node& node = *graph.nodes()[n];
            in_types.assign(static_cast<std::size_t>(node.num_inputs()), ir::TensorType{});
            in_ptrs.assign(static_cast<std::size_t>(node.num_inputs()), nullptr);
            in_shapes.clear();
            for (int i = 0; i < node.num_inputs(); ++i) {
                const ir::Value* value = node.input(i);
                if (value == nullptr) {
                    in_shapes.emplace_back();
                    continue;
                }
                const Shape& shape = layout->shapes[static_cast<std::size_t>(index(value))];
                in_types[static_cast<std::size_t>(i)].dtype = value->dtype();
                in_types[static_cast<std::size_t>(i)].shape.assign(shape.begin(), shape.end());
                in_ptrs[static_cast<std::size_t>(i)] = &in_types[static_cast<std::size_t>(i)];
                in_shapes.push_back(shape);
            }
            Attributes attrs = node.attrs();
            for (const auto& [name, attr] : node.attrs()) {
                if (const Dims* dims = std::get_if<Dims>(&attr)) {
                    Dims concrete;
                    for (const Dim& dim : *dims) {
                        const std::optional<std::int64_t> value = dim.evaluate(lookup);
                        if (!value) {
                            return errors::invalid_argument("{}: unbound symbols in {}", kGraphBackendName,
                                                            to_string(*dims));
                        }
                        concrete.emplace_back(*value);
                    }
                    attrs.set(name, std::move(concrete));
                }
            }
            out_types.clear();
            II_TRY(ir::infer_types(node.op(), in_ptrs, attrs, out_types));
            out_shapes.clear();
            for (int o = 0; o < node.num_outputs(); ++o) {
                const std::optional<Shape> shape = out_types[static_cast<std::size_t>(o)].static_shape();
                if (!shape) {
                    return errors::internal("{}: no concrete shape for output {} of '{}'", kGraphBackendName, o,
                                            node.op_name());
                }
                const auto v = static_cast<std::size_t>(index(node.output(o)));
                layout->shapes[v] = *shape;
                out_shapes.push_back(*shape);
                if (output_of_value_[v] < 0) {
                    layout->offsets[v] = offset;
                    offset += (storage_size(node.output(o)->dtype(), shape->numel()) + 63) / 64 * 64;
                }
            }
            scratch = std::max(scratch, kernels_[n]->workspace_size(in_shapes, out_shapes));
        }
        layout->scratch = offset;
        layout->size = offset + scratch;
        const std::lock_guard lock(mutex_);
        layouts_.push_back(layout);
        return std::shared_ptr<const Layout>(std::move(layout));
    }

    std::unique_ptr<ir::Graph> graph_;
    std::vector<int> inputs_;  // per program input: the node input it receives
    std::vector<bool> constant_inputs_;
    std::vector<std::unique_ptr<backend::Kernel>> kernels_;  // per node of the program
    std::unordered_map<const ir::Value*, int> value_index_;
    std::vector<int> output_of_value_;  // per value: the program output it is, or -1
    mutable std::mutex mutex_;
    mutable std::vector<std::shared_ptr<const Layout>> layouts_;
};

/// ii-test-graph: a graph backend on a simulated device. It compiles `subgraph` nodes into a `SubgraphKernel` and
/// shares the device (memory and queues) with ii-test-remote, so that partitioning for graph backends and plans that
/// mix a graph backend and a kernel backend on one device are tested without TensorRT.
class TestGraphBackend final : public backend::Backend {
public:
    TestGraphBackend(const TestDeviceOptions& options, const backend::Backend& device_owner)
        : name_(options.index == 0 ? std::string(kGraphBackendName)
                                   : std::format("{}.{}", kGraphBackendName, options.index)),
          ops_(options.graph_ops),
          owner_(device_owner),
          reference_(ref::create_ref_backend()) {}

    std::string_view name() const noexcept override { return name_; }
    const backend::DeviceInfo& device_info() const noexcept override { return owner_.device_info(); }
    backend::DeviceApi* device_api() const noexcept override { return owner_.device_api(); }
    bool compiles_subgraphs() const override { return true; }

    /// A part runs 10% faster than its operations one by one (a compiled engine), so that the graph backend takes
    /// what the device runs also when kernels launch for free (the fast test devices).
    double estimate_us(const ir::CostContext& context) const override {
        const double us = Backend::estimate_us(context);
        return context.node.op_name() == ops::kSubgraph ? 0.9 * us : us;
    }

    Status supports(const ir::Node& node) const override {
        if (node.op_name() != ops::kSubgraph) {
            return supports_node(node);
        }
        Result<std::shared_ptr<const ir::Graph>> program = ops::parse_subgraph_program(node);
        if (!program.ok()) {
            return errors::unsupported("{}: {}", name_, program.status().message());
        }
        for (const auto& inner : (*program)->nodes()) {
            II_TRY(supports_node(*inner));
        }
        return ok_status();
    }

    Result<std::unique_ptr<backend::Kernel>> create_kernel(const ir::Node& node) const override {
        II_TRY(supports(node));
        if (node.op_name() == ops::kSubgraph) {
            II_ASSIGN_OR_RETURN(std::unique_ptr<SubgraphKernel> inner, SubgraphKernel::create(node, *reference_));
            return std::unique_ptr<backend::Kernel>(std::make_unique<RemoteKernel>(std::move(inner)));
        }
        II_ASSIGN_OR_RETURN(std::unique_ptr<backend::Kernel> inner, reference_->create_kernel(node));
        return std::unique_ptr<backend::Kernel>(std::make_unique<RemoteKernel>(std::move(inner)));
    }

private:
    Status supports_node(const ir::Node& node) const {
        if ((!ops_.empty() && std::find(ops_.begin(), ops_.end(), node.op_name()) == ops_.end()) ||
            node.op_name() == ops::kTransfer || node.op_name() == ops::kSubgraph) {
            return errors::unsupported("{}: does not compile '{}'", name_, node.op_name());
        }
        if (Status status = reference_->supports(node); !status.ok()) {
            return errors::unsupported("{}: {}", name_, status.message());
        }
        return ok_status();
    }

    std::string name_;
    std::vector<std::string> ops_;
    const backend::Backend& owner_;
    std::unique_ptr<backend::Backend> reference_;
};

Result<double> parse_number(std::string_view key, std::string_view text) {
    const std::string copy(text);
    char* end = nullptr;
    const double value = std::strtod(copy.c_str(), &end);
    if (copy.empty() || end != copy.c_str() + copy.size() || !(value >= 0)) {
        return errors::invalid_argument("test device: invalid {} '{}'", key, text);
    }
    return value;
}

}  // namespace

Result<std::vector<TestDeviceOptions>> parse_test_devices(std::string_view spec) {
    std::vector<TestDeviceOptions> devices;
    while (true) {
        const std::size_t semicolon = spec.find(';');
        std::string_view item = spec.substr(0, semicolon);
        TestDeviceOptions options;
        options.index = static_cast<int>(devices.size());
        if (item == "default") {
            item = {};
        }
        while (!item.empty()) {
            const std::size_t comma = item.find(',');
            const std::string_view pair = item.substr(0, comma);
            item = comma == std::string_view::npos ? std::string_view() : item.substr(comma + 1);
            const std::size_t eq = pair.find('=');
            if (eq == std::string_view::npos) {
                return errors::invalid_argument("test device: expected key=value, got '{}'", pair);
            }
            const std::string_view key = pair.substr(0, eq);
            std::string_view value = pair.substr(eq + 1);
            if (key == "ops" || key == "graph_ops") {
                std::vector<std::string>& list = key == "ops" ? options.ops : options.graph_ops;
                while (!value.empty()) {
                    const std::size_t plus = value.find('+');
                    list.emplace_back(value.substr(0, plus));
                    value = plus == std::string_view::npos ? std::string_view() : value.substr(plus + 1);
                }
                continue;
            }
            double scale = 1;
            if (key == "memory" && !value.empty()) {
                switch (value.back()) {
                case 'K':
                    scale = 1 << 10;
                    break;
                case 'M':
                    scale = 1 << 20;
                    break;
                case 'G':
                    scale = 1 << 30;
                    break;
                default:
                    break;
                }
                if (scale != 1) {
                    value.remove_suffix(1);
                }
            }
            II_ASSIGN_OR_RETURN(const double number, parse_number(key, value));
            if (key == "memory") {
                options.memory_bytes = static_cast<std::uint64_t>(number * scale);
            } else if (key == "gflops") {
                options.gflops = number;
            } else if (key == "bandwidth") {
                options.bandwidth_gbs = number;
            } else if (key == "launch") {
                options.launch_us = number;
            } else if (key == "link") {
                options.link_gbs = number;
            } else if (key == "latency") {
                options.link_latency_us = number;
            } else if (key == "graph") {
                options.graph = number != 0;
            } else {
                return errors::invalid_argument("test device: unknown key '{}'", key);
            }
        }
        devices.push_back(std::move(options));
        if (semicolon == std::string_view::npos) {
            break;
        }
        spec.remove_prefix(semicolon + 1);
    }
    return devices;
}

std::vector<std::unique_ptr<backend::Backend>> create_test_remote_backends(const TestDeviceOptions& options) {
    std::vector<std::unique_ptr<backend::Backend>> backends;
    backends.push_back(std::make_unique<TestRemoteBackend>(options));
    if (options.graph) {
        backends.push_back(std::make_unique<TestGraphBackend>(options, *backends.front()));
    }
    return backends;
}

}  // namespace ii::test_remote
