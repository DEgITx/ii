// SPDX-License-Identifier: MIT
#pragma once

#include <ii/executor.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ii {

struct ThreadPoolOptions {
    /// Total number of threads that execute work, including the thread that submits it. 0 = all logical CPUs.
    int num_threads = 0;
    /// How long idle workers spin before going to sleep. Spinning makes back-to-back parallel loops (one per graph
    /// node) cheap; sleeping keeps an idle pool from burning CPU.
    std::chrono::microseconds spin_duration{50};
    /// Thread name prefix shown in debuggers and profilers.
    std::string name = "ii-worker";
};

/// A fixed-size pool of worker threads executing data-parallel loops.
///
/// * The submitting thread participates in the work, so a pool of N threads owns N-1 workers.
/// * Work is handed out dynamically in chunks, which balances load on hybrid CPUs (performance + efficiency cores).
/// * Calls from several external threads are serialized. A nested call from inside a running loop executes inline.
///
/// Synchronization protocol. `control_` packs [epoch:32 | closed:1 | active:31]:
///   1. A submitter fills `job_`, then publishes a new epoch with closed=0 (seq_cst store) and wakes sleeping workers.
///   2. A worker joins by CAS-incrementing `active` while the job is open and its epoch is new to the worker.
///   3. When the submitter has finished its own share, it sets closed=1. No worker can join afterwards.
///   4. The submitter waits until active == 0; the last worker to leave a closed job signals `done_signal_`.
/// A worker therefore only touches `job_` between a successful join and its leave, while the submitter waits.
class ThreadPool final : public Executor {
public:
    explicit ThreadPool(ThreadPoolOptions options = {});
    ~ThreadPool() override;

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int concurrency() const noexcept override { return static_cast<int>(workers_.size()) + 1; }

    void parallel_for(std::int64_t n, std::int64_t grain,
                      FunctionRef<void(std::int64_t begin, std::int64_t end)> fn) override;

    /// Calls `fn(thread_index)` exactly once on every thread of the pool, concurrently; index 0 is the calling
    /// thread, workers are 1..concurrency()-1. Used for per-thread setup and for measurements that need every thread.
    /// Must not be called from inside a job of the same pool.
    void run_on_all(FunctionRef<void(int thread_index)> fn);

    /// Index of the calling thread within the pool whose job it is executing, or -1 outside of any pool job.
    static int current_thread_index() noexcept;

private:
    enum class JobKind : std::uint8_t { kRange, kBroadcast };

    struct Job {
        JobKind kind = JobKind::kRange;
        const FunctionRef<void(std::int64_t, std::int64_t)>* range_fn = nullptr;
        const FunctionRef<void(int)>* broadcast_fn = nullptr;
        std::int64_t n = 0;
        std::int64_t chunk = 1;
        std::int64_t num_chunks = 0;
    };

    static constexpr std::uint64_t kClosedBit = std::uint64_t{1} << 31;
    static constexpr std::uint64_t kActiveMask = kClosedBit - 1;
    static constexpr std::uint64_t kEpochUnit = std::uint64_t{1} << 32;

    void worker_main(int index);
    void wait_for_change(std::uint64_t observed);
    void participate(int thread_index);
    void run_chunks();
    void open_job();
    void close_job_and_wait();
    void wait_for_done_signal(const std::atomic<std::int32_t>* counter);
    void signal_done();

    std::vector<std::thread> workers_;
    std::chrono::steady_clock::duration spin_duration_;
    std::mutex submit_mutex_;
    std::uint32_t epoch_ = 0;  // guarded by submit_mutex_
    Job job_;                  // written by the submitter while no worker is active

    alignas(64) std::atomic<std::uint64_t> control_{kClosedBit};
    alignas(64) std::atomic<std::int64_t> next_chunk_{0};
    alignas(64) std::atomic<std::int32_t> broadcast_pending_{0};
    alignas(64) std::atomic<std::uint32_t> done_signal_{0};
    alignas(64) std::atomic<std::int32_t> sleepers_{0};
    std::atomic<bool> stop_{false};
};

}  // namespace ii
