// SPDX-License-Identifier: MIT
/// @file
/// `ii::Context` — the root object of the library: backends, devices, thread pool, allocator and logger.
///
/// An application usually creates one context and shares it. Contexts, models, programs and sessions are cheap
/// handles to reference-counted objects; a session keeps its program, model and context alive.
#pragma once

#include <ii/allocator.h>
#include <ii/executor.h>
#include <ii/export.h>
#include <ii/log.h>
#include <ii/status.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

II_BEGIN_DLL_EXPORTS

namespace ii {

namespace detail {
struct ContextImpl;
}  // namespace detail

/// Options of `Context::create`.
struct ContextOptions {
    /// Allocator for weights, activation arenas and tensors created by ii. nullptr: `default_allocator()`.
    /// Must outlive the context and everything created from it.
    Allocator* allocator = nullptr;

    /// Executor for parallel work. nullptr: the context creates its own thread pool with `num_threads` threads.
    /// Must outlive the context and everything created from it.
    Executor* executor = nullptr;

    /// Threads of the context's own pool (including the calling thread). 0: all logical CPUs.
    int num_threads = 0;

    /// Kernel instruction-set variant of the ii-cpu backend, e.g. "generic", "avx2", "avx512", "neon". Empty: the
    /// `II_CPU_ISA` environment variable if set, otherwise the most advanced variant the CPU supports.
    std::string cpu_isa;

    /// Receives diagnostics. The default logs warnings and errors to stderr.
    Logger logger{LogLevel::kWarning, stderr_log_sink()};

    /// Simulated devices for testing multi-device execution without a GPU (the ii-test-remote backend), e.g.
    /// "memory=64M,ops=matmul+linear" or "default"; several devices separated by ';'. Empty: the `II_TEST_DEVICES`
    /// environment variable if set, otherwise none; "none": none. Keys are documented in
    /// docs/testing-and-benchmarks.md.
    std::string test_devices;
};

/// A device of a context and the backends running on it.
struct DeviceDescription {
    std::string id;                  ///< "cpu", "cuda:0", "test:0"
    std::string name;                ///< e.g. the CPU or GPU model
    std::uint64_t memory_bytes = 0;  ///< memory available to ii; 0: unlimited (host memory)
    std::vector<std::string> backends;
};

/// The root object: backends, thread pool, allocator, logger. Thread-safe.
class II_API Context {
public:
    /// Creates a context and registers the built-in backends.
    static Result<Context> create(ContextOptions options = {});

    Allocator& allocator() const noexcept;
    Executor& executor() const noexcept;
    const Logger& logger() const noexcept;

    /// Names of the available backends in priority order, e.g. {"ii-cpu", "ii-ref"}.
    std::vector<std::string> backend_names() const;

    /// The devices of the available backends; the host comes first.
    std::vector<DeviceDescription> devices() const;

    /// Internal: the implementation object.
    const std::shared_ptr<detail::ContextImpl>& impl() const noexcept { return impl_; }

private:
    explicit Context(std::shared_ptr<detail::ContextImpl> impl) noexcept : impl_(std::move(impl)) {}
    std::shared_ptr<detail::ContextImpl> impl_;
};

}  // namespace ii

II_END_DLL_EXPORTS
