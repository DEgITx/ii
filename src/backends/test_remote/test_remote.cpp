// SPDX-License-Identifier: MIT
#include "backends/test_remote/test_remote.h"

#include "backends/ref/ref_backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <map>
#include <mutex>

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
    }

    Status copy_to_device(void* dst, const void* src, std::size_t size) override {
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
        Status first = ok_status();
        for (const Task& task : tasks_) {
            if (!first.ok()) {
                break;  // like a failed GPU stream: later work is not executed
            }
            switch (task.kind) {
            case Kind::kKernel:
                first = task.kernel->run(task.context);
                break;
            case Kind::kCopy:
                std::memmove(task.dst, task.src, task.size);
                break;
            case Kind::kFill:
                std::memset(task.dst, 0, task.size);
                break;
            }
        }
        tasks_.clear();
        return first;
    }

    void enqueue(const backend::Kernel& kernel, const backend::KernelContext& context) {
        tasks_.push_back(Task{Kind::kKernel, &kernel, context, nullptr, nullptr, 0});
    }

    const DeviceMemory& memory() const noexcept { return memory_; }
    const std::string& device() const noexcept { return device_; }

private:
    enum class Kind : std::uint8_t { kKernel, kCopy, kFill };
    struct Task {
        Kind kind;
        const backend::Kernel* kernel;
        backend::KernelContext context;
        void* dst;
        const void* src;
        std::size_t size;
    };

    Status check(const void* data, std::size_t size, std::string_view what) const {
        if (size != 0 && !memory_.contains(data, size)) {
            return errors::invalid_argument("{} of {} bytes is not in the memory of {}", what, size, device_);
        }
        return ok_status();
    }

    const DeviceMemory& memory_;
    std::string device_;
    std::vector<Task> tasks_;
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

    Status run(const backend::KernelContext& context) const override {
        if (context.queue == nullptr) {
            return errors::internal("ii-test-remote kernel run without a device queue");
        }
        auto& queue = static_cast<TestQueue&>(*context.queue);
        for (std::size_t i = 0; i < context.inputs.size(); ++i) {
            II_TRY(check(queue, context.inputs[i], "input", i));
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
            if (key == "ops") {
                while (!value.empty()) {
                    const std::size_t plus = value.find('+');
                    options.ops.emplace_back(value.substr(0, plus));
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

std::unique_ptr<backend::Backend> create_test_remote_backend(const TestDeviceOptions& options) {
    return std::make_unique<TestRemoteBackend>(options);
}

}  // namespace ii::test_remote
