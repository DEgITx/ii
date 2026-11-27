// SPDX-License-Identifier: MIT
#include "backends/cpu/cpu_backend.h"
#include "backends/ref/ref_backend.h"
#include "backends/test_remote/test_remote.h"
#include "core/assert.h"
#include "core/platform/platform.h"
#include "plugin/host.h"
#include "runtime/runtime.h"

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
            impl->backends.push_back(test_remote::create_test_remote_backend(device));
        }
    }
    return Context(std::move(impl));
}

Allocator& Context::allocator() const noexcept {
    return *impl_->allocator;
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
