// SPDX-License-Identifier: MIT
#include "backends/cpu/cpu_backend.h"
#include "backends/ref/ref_backend.h"
#include "backends/test_remote/test_remote.h"
#include "core/assert.h"
#include "core/platform/platform.h"
#include "plugin/host.h"
#include "runtime/runtime.h"

#include <algorithm>
#include <optional>
#include <string_view>

namespace ii {

namespace detail {

const backend::Backend* ContextImpl::find_backend(std::string_view name) const noexcept {
    for (const auto& backend : backends) {
        if (backend->name() == name) {
            return backend.get();
        }
    }
    return nullptr;
}

const backend::Backend& ContextImpl::reference() const {
    const backend::Backend* backend = find_backend(ref::kBackendName);
    II_CHECK(backend != nullptr, "the reference backend is missing");
    return *backend;
}

backend::DeviceApi* ContextImpl::device_api(Device device) const noexcept {
    backend::DeviceApi* graph_backend_api = nullptr;
    for (const auto& backend : backends) {
        if (backend->device() != device || backend->device_api() == nullptr) {
            continue;
        }
        if (!backend->compiles_subgraphs()) {
            return backend->device_api();
        }
        if (graph_backend_api == nullptr) {
            graph_backend_api = backend->device_api();
        }
    }
    return graph_backend_api;
}

void* PinnedAllocator::allocate(std::size_t size, std::size_t alignment) noexcept {
    if (alignment <= 256) {  // what devices guarantee
        for (backend::DeviceApi* device : devices_) {
            try {
                Result<std::shared_ptr<Buffer>> buffer = device->allocate_host(std::max<std::size_t>(size, 1));
                if (!buffer.ok()) {
                    continue;
                }
                void* data = (*buffer)->data();
                const std::lock_guard lock(mutex_);
                blocks_.emplace(data, std::move(*buffer));
                return data;
            } catch (...) {
                // out of memory: the next device, then pageable memory
            }
        }
    }
    return fallback_.allocate(size, alignment);
}

bool PinnedAllocator::is_pinned(const void* ptr) noexcept {
    const std::lock_guard lock(mutex_);
    return blocks_.contains(const_cast<void*>(ptr));
}

void PinnedAllocator::deallocate(void* ptr, std::size_t size, std::size_t alignment) noexcept {
    if (ptr == nullptr) {
        return;
    }
    std::shared_ptr<Buffer> block;  // released outside the lock
    {
        const std::lock_guard lock(mutex_);
        if (const auto it = blocks_.find(ptr); it != blocks_.end()) {
            block = std::move(it->second);
            blocks_.erase(it);
        }
    }
    if (!block) {
        fallback_.deallocate(ptr, size, alignment);
    }
}

}  // namespace detail

Result<Context> Context::create(ContextOptions options) {
    auto impl = std::make_shared<detail::ContextImpl>();
    impl->allocator = options.allocator != nullptr ? options.allocator : &default_allocator();
    if (options.executor != nullptr) {
        impl->executor = options.executor;
    } else {
        if (options.num_threads < 0) {
            return errors::invalid_argument("num_threads must not be negative");
        }
        ThreadPoolOptions pool_options;
        pool_options.num_threads = options.num_threads;
        impl->own_pool = std::make_unique<ThreadPool>(pool_options);
        impl->executor = impl->own_pool.get();
    }
    impl->logger = std::move(options.logger);
    // Built-in backends are registered explicitly, in priority order (no static registration): ii-cpu first,
    // ii-ref as the fallback for everything ii-cpu does not accelerate.
    std::optional<std::string_view> cpu_isa;
    if (!options.cpu_isa.empty()) {
        cpu_isa = options.cpu_isa;
    }
    II_ASSIGN_OR_RETURN(std::unique_ptr<backend::Backend> cpu,
                        cpu::create_cpu_backend(cpu_isa, impl->executor->concurrency()));
    impl->backends.push_back(std::move(cpu));
    impl->backends.push_back(ref::create_ref_backend());
    std::string test_devices = options.test_devices;
    if (test_devices.empty()) {
        test_devices = platform::get_env("II_TEST_DEVICES").value_or("");
    }
    if (test_devices == "none") {
        test_devices.clear();
    }
    II_TRY(plugin::load_plugins(*impl, options, test_devices));
    // The ii-backend-test-remote plugin, when loaded, provides the simulated devices instead of the built-in backend.
    if (!test_devices.empty() && impl->find_backend(test_remote::kBackendName) == nullptr) {
        Result<std::vector<test_remote::TestDeviceOptions>> devices = test_remote::parse_test_devices(test_devices);
        if (!devices.ok()) {
            Status status = std::move(devices).status();
            return status.add_context("test devices");
        }
        for (const test_remote::TestDeviceOptions& device : *devices) {
            for (auto& backend : test_remote::create_test_remote_backends(device)) {
                impl->backends.push_back(std::move(backend));
            }
        }
    }
    // Page-locked memory from the devices (once per device: the backends of a GPU share its memory).
    std::vector<backend::DeviceApi*> pinned_devices;
    for (const auto& backend : impl->backends) {
        backend::DeviceApi* api = backend->device().is_host() ? nullptr : impl->device_api(backend->device());
        if (api != nullptr && std::find(pinned_devices.begin(), pinned_devices.end(), api) == pinned_devices.end()) {
            pinned_devices.push_back(api);
        }
    }
    impl->pinned = std::make_unique<detail::PinnedAllocator>(*impl->allocator, std::move(pinned_devices));
    return Context(std::move(impl));
}

Allocator& Context::allocator() const noexcept {
    return *impl_->allocator;
}

Allocator& Context::pinned_allocator() const noexcept {
    return *impl_->pinned;
}

Result<Tensor> Context::allocate_tensor(DType dtype, const Shape& shape, Device device) const {
    if (device.is_host()) {
        return Tensor::empty(dtype, shape, *impl_->allocator);
    }
    backend::DeviceApi* api = impl_->device_api(device);
    if (api == nullptr) {
        return errors::not_found("the context has no device '{}'", to_string(device));
    }
    const std::size_t nbytes = storage_size(dtype, shape.numel());
    II_ASSIGN_OR_RETURN(std::shared_ptr<Buffer> buffer, api->allocate(std::max<std::size_t>(nbytes, 1)));
    return Tensor::view(std::move(buffer), dtype, shape, contiguous_strides(shape));
}

Status Context::copy(const Tensor& destination, const Tensor& source) const {
    if (!destination.defined() || !source.defined()) {
        return errors::invalid_argument("copy: empty tensor");
    }
    if (destination.dtype() != source.dtype() || !(destination.shape() == source.shape())) {
        return errors::invalid_argument("copy: {}{} into {}{}", to_string(source.dtype()), source.shape().to_string(),
                                        to_string(destination.dtype()), destination.shape().to_string());
    }
    const Device to = destination.device();
    const Device from = source.device();
    if (to.is_host() && from.is_host()) {
        return destination.copy_from(source);
    }
    if ((!to.is_host() && !destination.is_contiguous()) || (!from.is_host() && !source.is_contiguous())) {
        return errors::invalid_argument("copy: tensors in device memory must be contiguous");
    }
    if (!to.is_host() && !from.is_host() && to != from) {
        // Between two devices: through the host.
        II_ASSIGN_OR_RETURN(Tensor staging, Tensor::empty(source.dtype(), source.shape(), *impl_->pinned));
        II_TRY(copy(staging, source));
        return copy(destination, staging);
    }
    const Device device = to.is_host() ? from : to;
    backend::DeviceApi* api = impl_->device_api(device);
    if (api == nullptr) {
        return errors::not_found("the context has no device '{}'", to_string(device));
    }
    Tensor host_side;  // the contiguous host end of the copy
    if (from.is_host()) {
        II_ASSIGN_OR_RETURN(host_side, source.contiguous(*impl_->allocator));
    } else if (to.is_host() && !destination.is_contiguous()) {
        II_ASSIGN_OR_RETURN(host_side, Tensor::empty(destination.dtype(), destination.shape(), *impl_->allocator));
    } else {
        host_side = destination;
    }
    detail::ContextImpl& impl = *impl_;
    const std::lock_guard lock(impl.copy_mutex);
    backend::Queue* queue = nullptr;
    for (const auto& [owner, existing] : impl.copy_queues) {
        if (owner == api) {
            queue = existing.get();
        }
    }
    if (queue == nullptr) {
        II_ASSIGN_OR_RETURN(std::unique_ptr<backend::Queue> created, api->create_queue());
        queue = created.get();
        impl.copy_queues.emplace_back(api, std::move(created));
    }
    const std::size_t nbytes = source.nbytes();
    if (!to.is_host() && !from.is_host()) {
        II_TRY(queue->copy(destination.data(), source.data(), nbytes));
    } else if (from.is_host()) {
        II_TRY(queue->copy_to_device(destination.data(), host_side.data(), nbytes));
    } else {
        II_TRY(queue->copy_to_host(host_side.data(), source.data(), nbytes));
    }
    II_TRY(queue->synchronize());
    if (to.is_host() && host_side.data() != destination.data()) {
        return destination.copy_from(host_side);
    }
    return ok_status();
}

Executor& Context::executor() const noexcept {
    return *impl_->executor;
}

const Logger& Context::logger() const noexcept {
    return impl_->logger;
}

std::vector<DeviceDescription> Context::devices() const {
    std::vector<DeviceDescription> result;
    auto add = [&](const backend::Backend& backend) {
        const backend::DeviceInfo& info = backend.device_info();
        const std::string id = to_string(info.device);
        for (DeviceDescription& device : result) {
            if (device.id == id) {
                device.backends.emplace_back(backend.name());
                return;
            }
        }
        result.push_back(DeviceDescription{id, info.name, info.memory_bytes, {std::string(backend.name())}});
    };
    for (const auto& backend : impl_->backends) {
        if (backend->device().is_host()) {
            add(*backend);
        }
    }
    for (const auto& backend : impl_->backends) {
        if (!backend->device().is_host()) {
            add(*backend);
        }
    }
    return result;
}

std::vector<PluginDescription> Context::plugins() const {
    return impl_->plugins;
}

std::string_view to_string(PluginState state) noexcept {
    switch (state) {
    case PluginState::kLoaded:
        return "loaded";
    case PluginState::kUnavailable:
        return "unavailable";
    case PluginState::kFailed:
        return "failed";
    }
    return "unknown";
}

std::vector<std::string> Context::backend_names() const {
    std::vector<std::string> names;
    for (const auto& backend : impl_->backends) {
        names.emplace_back(backend->name());
    }
    return names;
}

}  // namespace ii
