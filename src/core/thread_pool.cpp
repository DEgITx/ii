// SPDX-License-Identifier: MIT
#include "core/thread_pool.h"

#include <ii/system_info.h>

#include "core/assert.h"
#include "core/cpu_relax.h"
#include "core/platform/platform.h"

#include <algorithm>
#include <format>

namespace ii {

namespace {

thread_local const ThreadPool* t_current_pool = nullptr;
thread_local int t_thread_index = -1;

/// Chunks per thread: enough for dynamic load balancing, few enough to keep atomic traffic low.
constexpr std::int64_t kChunksPerThread = 4;

std::uint32_t epoch_of(std::uint64_t control) noexcept {
    return static_cast<std::uint32_t>(control >> 32);
}

std::int64_t ceil_div(std::int64_t a, std::int64_t b) noexcept {
    return (a + b - 1) / b;
}

/// Marks the current thread as executing a job of `pool` and restores the previous state on destruction.
class ScopedJobContext {
public:
    ScopedJobContext(const ThreadPool* pool, int index) noexcept
        : previous_pool_(t_current_pool), previous_index_(t_thread_index) {
        t_current_pool = pool;
        t_thread_index = index;
    }
    ~ScopedJobContext() {
        t_current_pool = previous_pool_;
        t_thread_index = previous_index_;
    }
    ScopedJobContext(const ScopedJobContext&) = delete;
    ScopedJobContext& operator=(const ScopedJobContext&) = delete;

private:
    const ThreadPool* previous_pool_;
    int previous_index_;
};

}  // namespace

ThreadPool::ThreadPool(ThreadPoolOptions options)
    : spin_duration_(std::chrono::duration_cast<std::chrono::steady_clock::duration>(options.spin_duration)) {
    const int threads = options.num_threads > 0 ? options.num_threads : std::max(1, cpu_info().logical_cpus);
    workers_.reserve(static_cast<std::size_t>(threads - 1));
    for (int index = 1; index < threads; ++index) {
        workers_.emplace_back([this, index, name = std::format("{}-{}", options.name, index)] {
            platform::set_current_thread_name(name);
            worker_main(index);
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        const std::lock_guard lock(submit_mutex_);
        stop_.store(true, std::memory_order_seq_cst);
        // Change the control word (new epoch, still closed) so that sleeping workers wake up and observe stop_.
        control_.fetch_add(kEpochUnit, std::memory_order_seq_cst);
        const std::lock_guard sleep_lock(sleep_mutex_);
        sleep_cv_.notify_all();
    }
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

int ThreadPool::current_thread_index() noexcept {
    return t_thread_index;
}

void ThreadPool::parallel_for(std::int64_t n, std::int64_t grain, FunctionRef<void(std::int64_t, std::int64_t)> fn) {
    if (n <= 0) {
        return;
    }
    grain = std::max<std::int64_t>(grain, 1);
    if (workers_.empty() || n <= grain || t_current_pool == this) {
        fn(0, n);  // too small to split, single-threaded pool, or nested call: run inline
        return;
    }

    const std::lock_guard lock(submit_mutex_);
    const std::int64_t chunk = std::max(grain, ceil_div(n, concurrency() * kChunksPerThread));
    job_.kind = JobKind::kRange;
    job_.range_fn = &fn;
    job_.broadcast_fn = nullptr;
    job_.n = n;
    job_.chunk = chunk;
    job_.num_chunks = ceil_div(n, chunk);
    next_chunk_.store(0, std::memory_order_relaxed);

    open_job();
    {
        const ScopedJobContext context(this, 0);
        run_chunks();
    }
    close_job_and_wait();
}

void ThreadPool::run_on_all(FunctionRef<void(int)> fn) {
    II_CHECK(t_current_pool != this, "run_on_all() called from inside a job of the same pool");
    if (workers_.empty()) {
        const ScopedJobContext context(this, 0);
        fn(0);
        return;
    }

    const std::lock_guard lock(submit_mutex_);
    job_.kind = JobKind::kBroadcast;
    job_.range_fn = nullptr;
    job_.broadcast_fn = &fn;
    broadcast_pending_.store(static_cast<std::int32_t>(workers_.size()), std::memory_order_relaxed);

    open_job();
    {
        const ScopedJobContext context(this, 0);
        fn(0);
    }
    // Every worker must run fn once, so the job may only be closed after all of them have joined and finished.
    wait_for_done_signal(&broadcast_pending_);
    close_job_and_wait();
}

void ThreadPool::open_job() {
    epoch_ = epoch_ + 1 == 0 ? 1 : epoch_ + 1;  // epoch 0 is never published: workers start with "seen = 0"
    control_.store(static_cast<std::uint64_t>(epoch_) << 32, std::memory_order_seq_cst);
    // Pairs with the sleepers_ increment in wait_for_change(): either we see the sleeper and notify it, or the
    // sleeper sees the new control value (its predicate is checked under sleep_mutex_) and does not sleep.
    if (sleepers_.load(std::memory_order_seq_cst) > 0) {
        const std::lock_guard lock(sleep_mutex_);
        sleep_cv_.notify_all();
    }
}

void ThreadPool::close_job_and_wait() {
    const std::uint64_t value = control_.fetch_or(kClosedBit, std::memory_order_acq_rel);
    if ((value & kActiveMask) != 0) {
        wait_for_done_signal(nullptr);
    }
}

void ThreadPool::wait_for_done_signal(const std::atomic<std::int32_t>* counter) {
    // Waits until `*counter` reaches zero, or (counter == nullptr) until no worker is active in the closed job.
    const auto finished = [&] {
        return counter != nullptr ? counter->load(std::memory_order_acquire) == 0
                                  : (control_.load(std::memory_order_acquire) & kActiveMask) == 0;
    };
    const auto deadline = std::chrono::steady_clock::now() + spin_duration_;
    for (int spins = 0; !finished(); ++spins) {
        if ((spins & 63) == 63 && std::chrono::steady_clock::now() >= deadline) {
            // signal_done() notifies under done_mutex_ after the state change, so the predicate cannot miss it.
            std::unique_lock lock(done_mutex_);
            done_cv_.wait(lock, finished);
            return;
        }
        cpu_relax();
    }
}

void ThreadPool::signal_done() {
    const std::lock_guard lock(done_mutex_);
    done_cv_.notify_all();
}

void ThreadPool::worker_main(int index) {
    const ScopedJobContext context(this, index);
    std::uint32_t seen_epoch = 0;
    for (;;) {
        std::uint64_t value = control_.load(std::memory_order_acquire);
        if (stop_.load(std::memory_order_acquire)) {
            return;
        }
        if ((value & kClosedBit) == 0 && epoch_of(value) != seen_epoch) {
            if (control_.compare_exchange_weak(value, value + 1, std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
                seen_epoch = epoch_of(value);
                participate(index);
                const std::uint64_t after = control_.fetch_sub(1, std::memory_order_acq_rel) - 1;
                if ((after & kClosedBit) != 0 && (after & kActiveMask) == 0) {
                    signal_done();
                }
            }
            continue;
        }
        wait_for_change(value);
    }
}

void ThreadPool::wait_for_change(std::uint64_t observed) {
    const auto deadline = std::chrono::steady_clock::now() + spin_duration_;
    for (int spins = 0; control_.load(std::memory_order_relaxed) == observed; ++spins) {
        if (stop_.load(std::memory_order_relaxed)) {
            return;
        }
        if ((spins & 63) == 63 && std::chrono::steady_clock::now() >= deadline) {
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            {
                std::unique_lock lock(sleep_mutex_);
                sleep_cv_.wait(lock, [&] {
                    return control_.load(std::memory_order_seq_cst) != observed ||
                           stop_.load(std::memory_order_seq_cst);
                });
            }
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            return;
        }
        cpu_relax();
    }
}

void ThreadPool::participate(int thread_index) {
    if (job_.kind == JobKind::kBroadcast) {
        (*job_.broadcast_fn)(thread_index);
        if (broadcast_pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            signal_done();
        }
        return;
    }
    run_chunks();
}

void ThreadPool::run_chunks() {
    const auto& fn = *job_.range_fn;
    for (;;) {
        const std::int64_t chunk = next_chunk_.fetch_add(1, std::memory_order_relaxed);
        if (chunk >= job_.num_chunks) {
            return;
        }
        const std::int64_t begin = chunk * job_.chunk;
        fn(begin, std::min(job_.n, begin + job_.chunk));
    }
}

}  // namespace ii
