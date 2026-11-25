// SPDX-License-Identifier: MIT
#include "backends/ref/ref_backend.h"
#include "core/assert.h"
#include "runtime/runtime.h"

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
    // Built-in backends are registered explicitly, in priority order (no static registration).
    impl->backends.push_back(ref::create_ref_backend());
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

std::vector<std::string> Context::backend_names() const {
    std::vector<std::string> names;
    for (const auto& backend : impl_->backends) {
        names.emplace_back(backend->name());
    }
    return names;
}

}  // namespace ii
