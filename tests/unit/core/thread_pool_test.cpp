// SPDX-License-Identifier: MIT
#include "core/thread_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace ii {
namespace {

using namespace std::chrono_literals;

ThreadPool make_pool(int threads, std::chrono::microseconds spin = std::chrono::microseconds(50)) {
    return ThreadPool({.num_threads = threads, .spin_duration = spin});
}

TEST(ThreadPool, ConcurrencyMatchesOptions) {
    for (const int threads : {1, 2, 5}) {
        ThreadPool pool({.num_threads = threads});
        EXPECT_EQ(pool.concurrency(), threads);
    }
    ThreadPool default_pool;
    EXPECT_GE(default_pool.concurrency(), 1);
}

TEST(ThreadPool, CoversRangeExactlyOnceWithGrain) {
    for (const int threads : {1, 2, 3, 8}) {
        ThreadPool pool({.num_threads = threads});
        for (const std::int64_t n : {1, 2, 7, 64, 1000, 12345}) {
            for (const std::int64_t grain : {1, 3, 16, 5000}) {
                std::vector<std::atomic<int>> hits(static_cast<std::size_t>(n));
                std::mutex mutex;
                std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
                pool.parallel_for(n, grain, [&](std::int64_t begin, std::int64_t end) {
                    for (std::int64_t i = begin; i < end; ++i) {
                        hits[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
                    }
                    const std::lock_guard lock(mutex);
                    ranges.emplace_back(begin, end);
                });
                for (std::int64_t i = 0; i < n; ++i) {
                    ASSERT_EQ(hits[static_cast<std::size_t>(i)].load(), 1)
                        << "threads=" << threads << " n=" << n << " grain=" << grain << " i=" << i;
                }
                for (const auto& [begin, end] : ranges) {
                    ASSERT_LT(begin, end);
                    if (end != n) {
                        ASSERT_GE(end - begin, grain) << "only the last range may be shorter than the grain";
                    }
                }
            }
        }
    }
}

TEST(ThreadPool, EmptyRangeDoesNothing) {
    ThreadPool pool = make_pool(4);
    int calls = 0;
    pool.parallel_for(0, 1, [&](std::int64_t, std::int64_t) { ++calls; });
    pool.parallel_for(-5, 1, [&](std::int64_t, std::int64_t) { ++calls; });
    EXPECT_EQ(calls, 0);
}

TEST(ThreadPool, SmallRangeRunsInlineOnCaller) {
    ThreadPool pool = make_pool(4);
    std::thread::id runner;
    int calls = 0;
    pool.parallel_for(10, 100, [&](std::int64_t begin, std::int64_t end) {
        runner = std::this_thread::get_id();
        ++calls;
        EXPECT_EQ(begin, 0);
        EXPECT_EQ(end, 10);
    });
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(runner, std::this_thread::get_id());
}

TEST(ThreadPool, WorkIsSpreadAcrossThreads) {
    ThreadPool pool = make_pool(4);
    std::mutex mutex;
    std::set<std::thread::id> threads;
    pool.parallel_for(64, 1, [&](std::int64_t, std::int64_t) {
        std::this_thread::sleep_for(2ms);
        const std::lock_guard lock(mutex);
        threads.insert(std::this_thread::get_id());
    });
    EXPECT_GE(threads.size(), 2u);
}

TEST(ThreadPool, ThreadIndexIsInRange) {
    ThreadPool pool = make_pool(4);
    EXPECT_EQ(ThreadPool::current_thread_index(), -1);
    std::atomic<bool> in_range{true};
    pool.parallel_for(1000, 1, [&](std::int64_t, std::int64_t) {
        const int index = ThreadPool::current_thread_index();
        if (index < 0 || index >= pool.concurrency()) {
            in_range = false;
        }
    });
    EXPECT_TRUE(in_range);
    EXPECT_EQ(ThreadPool::current_thread_index(), -1);
}

TEST(ThreadPool, NestedCallRunsInline) {
    ThreadPool pool = make_pool(4);
    std::atomic<std::int64_t> total{0};
    pool.parallel_for(16, 1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            pool.parallel_for(100, 1, [&](std::int64_t b, std::int64_t e) { total += e - b; });
        }
    });
    EXPECT_EQ(total.load(), 1600);
}

TEST(ThreadPool, RunOnAllRunsEveryThreadOnce) {
    for (const int threads : {1, 2, 6}) {
        ThreadPool pool = make_pool(threads);
        for (int repeat = 0; repeat < 50; ++repeat) {
            std::mutex mutex;
            std::vector<int> indices;
            std::set<std::thread::id> ids;
            pool.run_on_all([&](int index) {
                const std::lock_guard lock(mutex);
                indices.push_back(index);
                ids.insert(std::this_thread::get_id());
            });
            std::sort(indices.begin(), indices.end());
            ASSERT_EQ(static_cast<int>(indices.size()), threads);
            for (int i = 0; i < threads; ++i) {
                ASSERT_EQ(indices[static_cast<std::size_t>(i)], i);
            }
            ASSERT_EQ(static_cast<int>(ids.size()), threads) << "every index must run on a different thread";
        }
    }
}

TEST(ThreadPool, ConcurrentSubmittersAreSerialized) {
    ThreadPool pool = make_pool(4);
    constexpr int kSubmitters = 4;
    constexpr int kJobs = 200;
    std::vector<std::int64_t> sums(kSubmitters, 0);
    std::vector<std::thread> submitters;
    for (int s = 0; s < kSubmitters; ++s) {
        submitters.emplace_back([&, s] {
            for (int job = 0; job < kJobs; ++job) {
                std::atomic<std::int64_t> sum{0};
                pool.parallel_for(1000, 10, [&](std::int64_t begin, std::int64_t end) {
                    std::int64_t local = 0;
                    for (std::int64_t i = begin; i < end; ++i) {
                        local += i;
                    }
                    sum += local;
                });
                sums[static_cast<std::size_t>(s)] += sum.load();
            }
        });
    }
    for (std::thread& t : submitters) {
        t.join();
    }
    for (const std::int64_t sum : sums) {
        EXPECT_EQ(sum, std::int64_t{kJobs} * 999 * 1000 / 2);
    }
}

TEST(ThreadPool, ManySmallJobs) {
    ThreadPool pool = make_pool(4);
    std::int64_t total = 0;
    for (int job = 0; job < 20000; ++job) {
        std::atomic<std::int64_t> count{0};
        pool.parallel_for(64, 1, [&](std::int64_t begin, std::int64_t end) { count += end - begin; });
        total += count.load();
    }
    EXPECT_EQ(total, 20000 * 64);
}

TEST(ThreadPool, WorkersWakeUpAfterSleeping) {
    ThreadPool pool = make_pool(4, std::chrono::microseconds(0));  // workers sleep immediately
    for (int job = 0; job < 50; ++job) {
        std::this_thread::sleep_for(1ms);
        std::atomic<std::int64_t> count{0};
        pool.parallel_for(256, 1, [&](std::int64_t begin, std::int64_t end) { count += end - begin; });
        ASSERT_EQ(count.load(), 256);
        int ran = 0;
        std::mutex mutex;
        pool.run_on_all([&](int) {
            const std::lock_guard lock(mutex);
            ++ran;
        });
        ASSERT_EQ(ran, 4);
    }
}

TEST(ThreadPool, CreateAndDestroyRepeatedly) {
    for (int i = 0; i < 100; ++i) {
        ThreadPool pool = make_pool(3, std::chrono::microseconds(i % 2 == 0 ? 0 : 50));
        if (i % 3 == 0) {
            std::atomic<int> count{0};
            pool.parallel_for(100, 1,
                              [&](std::int64_t begin, std::int64_t end) { count += static_cast<int>(end - begin); });
            ASSERT_EQ(count.load(), 100);
        }
    }
}

TEST(ThreadPool, UsableThroughExecutorInterface) {
    ThreadPool pool = make_pool(3);
    Executor& executor = pool;
    std::atomic<std::int64_t> count{0};
    executor.parallel_for(500, 7, [&](std::int64_t begin, std::int64_t end) { count += end - begin; });
    EXPECT_EQ(count.load(), 500);
    EXPECT_EQ(executor.concurrency(), 3);
}

}  // namespace
}  // namespace ii
