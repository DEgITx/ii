// Реализация engine/parallel.h: пул рабочих потоков + планирование кусков.
//
// Пул заточен под parallel_for: за раз исполняется одна «волна» одинаковых
// задач (rank 0..chunks-1), без очереди и без packaged_task — вызывающий
// поток сам считает кусок 0 и ждёт, пока рабочие добьют остальные. Это
// дешевле, чем future на каждый кусок, и точно соответствует тому, как
// движок вызывает parallel_for (одна волна на ядро графа).

#include "engine/parallel.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ii {
namespace {

// Помечаем поток как «уже внутри параллельного региона». Рабочие потоки
// помечены навсегда (они не должны запускать вложенные волны — это привело
// бы к дедлоку на занятом пуле), вызывающий — только на время своей волны.
thread_local bool t_in_parallel = false;

// Пул фиксированного размера. dispatch() раздаёт rank'и: rank 0 — вызывающий
// поток, rank 1..workers — рабочие. За одну волну работают ранги [0, n_ranks).
class WorkerPool {
public:
    explicit WorkerPool(int workers) {
        workers_.reserve(workers);
        for (int i = 0; i < workers; ++i)
            workers_.emplace_back([this, rank = i + 1] { worker_loop(rank); });
    }

    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
            ++epoch_;
        }
        wake_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    // Число исполнителей = рабочие + вызывающий.
    int size() const { return static_cast<int>(workers_.size()) + 1; }

    // Запустить job(rank) для rank в [0, n_ranks) и дождаться завершения.
    // n_ranks гарантированно в диапазоне [2, size()].
    void dispatch(int n_ranks, const std::function<void(int)>& job) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            job_     = &job;
            n_ranks_ = n_ranks;
            running_ = n_ranks - 1;   // рабочие, занятые этой волной (ранги 1..)
            ++epoch_;
        }
        wake_.notify_all();

        job(0);                       // вызывающий — rank 0

        std::unique_lock<std::mutex> lk(mtx_);
        done_.wait(lk, [this] { return running_ == 0; });
    }

private:
    void worker_loop(int rank) {
        t_in_parallel = true;         // рабочие никогда не дробят вложенно
        unsigned seen = 0;
        for (;;) {
            const std::function<void(int)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                wake_.wait(lk, [this, &seen] { return epoch_ != seen; });
                seen = epoch_;
                if (stop_) return;
                if (rank >= n_ranks_) continue;   // эта волна меньше — мы не нужны
                job = job_;
            }
            (*job)(rank);
            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (--running_ == 0) done_.notify_one();
            }
        }
    }

    std::vector<std::thread>          workers_;
    std::mutex                        mtx_;
    std::condition_variable           wake_;    // будит рабочих на новую волну
    std::condition_variable           done_;    // будит вызывающего по завершении
    const std::function<void(int)>*   job_     = nullptr;
    int                               n_ranks_ = 0;
    int                               running_ = 0;     // рабочих осталось в волне
    unsigned                          epoch_   = 0;     // номер волны
    bool                              stop_    = false;
};

std::mutex                  g_mtx;          // защищает конфиг и ленивый пул
int                         g_threads = 0;  // 0 = авто (число ядер)
std::unique_ptr<WorkerPool> g_pool;         // null => последовательный режим

// Разрешить желаемое число потоков в фактическое (>= 1).
int resolve_threads(int requested) {
    if (requested == 1) return 1;
    if (requested > 1)  return requested;
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

// Текущий пул под размер g_threads; null, если потоков <= 1 (тогда всё
// считается последовательно). Создаётся лениво и переиспользуется.
WorkerPool* active_pool() {
    std::lock_guard<std::mutex> lk(g_mtx);
    int threads = resolve_threads(g_threads);
    if (threads <= 1) {
        g_pool.reset();
        return nullptr;
    }
    if (!g_pool || g_pool->size() != threads)
        g_pool = std::make_unique<WorkerPool>(threads - 1);  // +1 вызывающий
    return g_pool.get();
}

}  // namespace

void set_num_threads(int threads) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_threads = threads;
    g_pool.reset();   // пересоздастся нужного размера при следующем active_pool()
}

int num_threads() {
    std::lock_guard<std::mutex> lk(g_mtx);
    return resolve_threads(g_threads);
}

namespace detail {

int plan_chunks(std::int64_t count, std::int64_t min_grain) {
    if (count <= 0 || t_in_parallel) return 1;
    if (min_grain < 1) min_grain = 1;

    WorkerPool* pool = active_pool();
    if (!pool) return 1;                       // последовательный режим

    std::int64_t by_grain = count / min_grain; // сколько кусков допускает грейн
    if (by_grain < 2) return 1;                // работы меньше двух кусков
    std::int64_t chunks = std::min<std::int64_t>(by_grain, pool->size());
    return static_cast<int>(chunks);
}

void run_on_pool(int chunks, const std::function<void(int)>& job) {
    WorkerPool* pool = active_pool();
    if (!pool) {                  // подстраховка: пул исчез между plan и run
        for (int r = 0; r < chunks; ++r) job(r);
        return;
    }

    t_in_parallel = true;         // запрет вложенного дробления на этом потоке
    std::exception_ptr err;
    std::mutex          err_mtx;
    auto guarded = [&](int rank) {
        try {
            job(rank);
        } catch (...) {
            std::lock_guard<std::mutex> lk(err_mtx);
            if (!err) err = std::current_exception();
        }
    };
    pool->dispatch(chunks, guarded);
    t_in_parallel = false;

    if (err) std::rethrow_exception(err);
}

}  // namespace detail

}  // namespace ii
