// SPDX-License-Identifier: MIT
/// @file
/// The parallel execution interface. ii runs its parallel work through an `Executor`; applications with their own
/// job system can implement it so that ii does not create threads of its own.
#pragma once

#include <ii/function_ref.h>

#include <cstdint>

namespace ii {

/// Executes data-parallel loops.
class Executor {
public:
    virtual ~Executor() = default;

    /// Maximum number of threads (including the calling thread) that may run work of one `parallel_for` at once.
    virtual int concurrency() const noexcept = 0;

    /// Calls `fn(begin, end)` for disjoint subranges that together cover exactly `[0, n)`, possibly concurrently,
    /// and returns when all calls have finished. Subranges are at least `grain` elements long (except the last).
    /// `fn` must be safe to call concurrently and must not throw.
    virtual void parallel_for(std::int64_t n, std::int64_t grain,
                              FunctionRef<void(std::int64_t begin, std::int64_t end)> fn) = 0;

protected:
    Executor() = default;
    Executor(const Executor&) = default;
    Executor& operator=(const Executor&) = default;
};

}  // namespace ii
