// Реализация parallel.h: пул потоков и parallel_for.

#include "parallel.h"

namespace parallel {

ThreadPool::ThreadPool(unsigned num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;  // fallback, если ОС не знает
    }
    workers_.reserve(num_threads);
    for (unsigned i = 0; i < num_threads; ++i)
        workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_)
        if (t.joinable()) t.join();
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();  // вне мьютекса — иначе очередь стоит, пока чанк считается
    }
}

void ThreadPool::parallel_for(
    std::size_t begin, std::size_t end,
    const std::function<void(std::size_t, std::size_t)>& body,
    std::size_t grain) {
    if (begin >= end) return;
    const std::size_t n = end - begin;

    // Сколько чанков нарезать. По умолчанию — по одному на «исполнителя»
    // (рабочие потоки + сам вызывающий). При заданном grain — столько,
    // чтобы каждый чанк был не меньше grain.
    const std::size_t executors = size() + 1;
    std::size_t chunks = (grain == 0)
                             ? executors
                             : (n + grain - 1) / grain;
    if (chunks < 1) chunks = 1;
    if (chunks > n) chunks = n;  // не дробим мельче одного элемента

    // Один чанк (или пустой пул) — просто выполняем на месте, без
    // futures и блокировок.
    if (chunks == 1 || workers_.empty()) {
        body(begin, end);
        return;
    }

    // Раскладка границ чанков: первые (n % chunks) чанков на 1 элемент
    // длиннее — равномернее некуда.
    const std::size_t base = n / chunks;
    const std::size_t rem  = n % chunks;

    std::vector<std::pair<std::size_t, std::size_t>> ranges(chunks);
    {
        std::size_t lo = begin;
        for (std::size_t i = 0; i < chunks; ++i) {
            std::size_t len = base + (i < rem ? 1 : 0);
            ranges[i] = {lo, lo + len};
            lo += len;
        }
    }

    // Чанки [1..chunks) уходят в пул; чанк 0 выполняет вызывающий поток
    // (пока рабочие разбирают остальное). Затем дожидаемся всех и
    // прокидываем первое исключение, если было.
    std::vector<std::future<void>> futs;
    futs.reserve(chunks - 1);
    for (std::size_t i = 1; i < chunks; ++i) {
        const auto r = ranges[i];
        futs.push_back(submit([&body, r] { body(r.first, r.second); }));
    }

    std::exception_ptr err;
    try {
        body(ranges[0].first, ranges[0].second);
    } catch (...) {
        err = std::current_exception();
    }
    for (auto& f : futs) {
        try {
            f.get();
        } catch (...) {
            if (!err) err = std::current_exception();
        }
    }
    if (err) std::rethrow_exception(err);
}

ThreadPool& default_pool() {
    static ThreadPool pool;  // hardware_concurrency; thread-safe init (C++11)
    return pool;
}

}  // namespace parallel
