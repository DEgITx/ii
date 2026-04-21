// Переиспользуемый пул потоков + parallel_for.
//
// Модуль намеренно не знает ничего про инференс/изображения — это
// generic-инфраструктура распараллеливания, пригодная и в этом проекте
// (tile-сборка, декод, препроцесс), и в любом другом C++17-коде.
//
// Дизайн:
//   * ThreadPool — фиксированный набор рабочих потоков + FIFO-очередь
//     задач, защищённая mutex + condition_variable. Потоки живут от
//     конструктора до деструктора;
//   * submit(f, args...) — поставить задачу, получить std::future с
//     результатом (исключения прокидываются через future);
//   * parallel_for(begin, end, body, grain) — разбить диапазон на чанки
//     и выполнить body(chunk_begin, chunk_end) для каждого; блокирует до
//     конца. Вызывающий поток сам выполняет один чанк (не простаивает и
//     не даёт дедлок на занятом пуле для НЕвложенных вызовов);
//   * default_pool() — ленивый глобальный пул на hardware_concurrency.
//
// Требования: C++17 (std::invoke_result_t, structured init статиков).
// Никаких внешних зависимостей кроме STL и потоков.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace parallel {

class ThreadPool {
public:
    // num_threads == 0 → std::thread::hardware_concurrency() (минимум 1).
    explicit ThreadPool(unsigned num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Поставить задачу в очередь. Возвращает future с результатом
    // f(args...); исключение из задачи всплывёт при future::get().
    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>;

    // Разбить [begin, end) на чанки и выполнить body(chunk_begin,
    // chunk_end) для каждого. Блокирует до завершения всех чанков.
    // Вызывающий поток выполняет один из чанков сам.
    //   grain == 0 → делим ровно на size()+1 частей (вызывающий +
    //                рабочие потоки), по чанку на поток;
    //   grain  > 0 → размер чанка не меньше grain (число чанков =
    //                ceil(n / grain), может превышать число потоков —
    //                тогда они разбираются из очереди по мере готовности,
    //                что даёт балансировку при неравномерной нагрузке).
    // Исключения из body прокидываются вызывающему (первое пойманное).
    void parallel_for(std::size_t begin, std::size_t end,
                      const std::function<void(std::size_t, std::size_t)>& body,
                      std::size_t grain = 0);

    std::size_t size() const noexcept { return workers_.size(); }

private:
    void worker_loop();

    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        mtx_;
    std::condition_variable           cv_;
    bool                              stop_ = false;
};

// Глобальный пул процесса (ленивая инициализация, размер =
// hardware_concurrency). Подходит для большинства вызовов parallel_for.
ThreadPool& default_pool();

// Удобная обёртка над default_pool().parallel_for(...).
inline void parallel_for(
    std::size_t begin, std::size_t end,
    const std::function<void(std::size_t, std::size_t)>& body,
    std::size_t grain = 0) {
    default_pool().parallel_for(begin, end, body, grain);
}

// ------------------------- template impl -------------------------------

template <class F, class... Args>
auto ThreadPool::submit(F&& f, Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>> {
    using R = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<R()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<R> fut = task->get_future();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (stop_)
            throw std::runtime_error("ThreadPool::submit после остановки пула");
        tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return fut;
}

}  // namespace parallel
