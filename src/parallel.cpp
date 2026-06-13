// Реализация parallel.h: «волновой» пул потоков под parallel_for.
//
// ── Почему «волна», а не очередь задач ─────────────────────────────────────
//
// Привычный пул задач кладёт каждую под-задачу в очередь как std::function,
// оборачивает в packaged_task + future и будит рабочих, которые тянут задачи
// динамически. Это удобно для разнородной работы, но дорого: куча-аллокация на
// КАЖДУЮ под-задачу. Движок зовёт parallel_for тысячи раз за инференс, поэтому
// такая модель тут неприемлема.
//
// Волновой пул устроен как заранее собранная бригада фиксированного размера:
//
//   1. Рабочие потоки создаются ОДИН раз и спят на условной переменной.
//   2. parallel_for, решив дробить на N кусков, заполняет общее поле «текущая
//      работа», увеличивает номер волны (wave_seq_) и будит бригаду.
//   3. Каждый рабочий и сам вызывающий берут СВОЙ номер куска (chunk_index) и
//      считают свой кусок. Вызывающий поток — это chunk 0 (он не простаивает).
//   4. Вызывающий ждёт, пока рабочие добьют свои куски, и возвращается.
//
// Куски однородны и нарезаны заранее (см. parallel.h), очередь и future не
// нужны — за всю волну ноль куча-аллокаций. Один parallel_for = одна волна.

#include "parallel.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ii {

namespace {

// Поток помечается «уже внутри параллельной волны», чтобы вложенный
// parallel_for не дробил повторно. Рабочие потоки помечены навсегда (запуск
// вложенной волны на занятом пуле привёл бы к дедлоку), вызывающий — только на
// время своей волны (снимается в run_chunks по выходе).
thread_local bool t_inside_wave = false;

// Пул фиксированного размера. Исполнители делятся по «номеру куска»
// (chunk_index): 0 — вызывающий поток, 1..workers — рабочие потоки. За одну
// волну работают исполнители с индексами [0, chunk_count).
class WavePool {
public:
    // worker_count — число фоновых рабочих (вызывающий поток считается отдельно
    // и получает chunk_index 0).
    explicit WavePool(int worker_count) {
        workers_.reserve(worker_count);
        for (int i = 0; i < worker_count; ++i)
            workers_.emplace_back([this, chunk_index = i + 1] {
                worker_loop(chunk_index);
            });
    }

    ~WavePool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stopping_ = true;
            ++wave_seq_;          // разбудить рабочих «пустой» волной на выход
        }
        wake_workers_.notify_all();
        for (std::thread& t : workers_)
            if (t.joinable()) t.join();
    }

    // Число исполнителей = рабочие + вызывающий. Это потолок числа кусков.
    int executor_count() const {
        return static_cast<int>(workers_.size()) + 1;
    }

    // Запустить run_chunk(chunk_index) для индексов [0, chunk_count) и дождаться
    // завершения. chunk_count гарантированно в диапазоне [2, executor_count()].
    void run_wave(int chunk_count,
                  const std::function<void(int chunk_index)>& run_chunk) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            current_work_     = &run_chunk;
            chunk_count_      = chunk_count;
            workers_pending_  = chunk_count - 1;  // рабочие с индексами 1..
            ++wave_seq_;
        }
        wake_workers_.notify_all();

        run_chunk(0);             // вызывающий поток — chunk 0

        std::unique_lock<std::mutex> lk(mtx_);
        wave_done_.wait(lk, [this] { return workers_pending_ == 0; });
    }

private:
    void worker_loop(int chunk_index) {
        t_inside_wave = true;     // рабочие никогда не дробят вложенно
        unsigned last_seen_wave = 0;
        for (;;) {
            const std::function<void(int)>* work = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                wake_workers_.wait(lk, [this, last_seen_wave] {
                    return wave_seq_ != last_seen_wave;
                });
                last_seen_wave = wave_seq_;
                if (stopping_) return;
                if (chunk_index >= chunk_count_)
                    continue;     // эта волна мельче — наш кусок не нужен
                work = current_work_;
            }

            (*work)(chunk_index);

            std::lock_guard<std::mutex> lk(mtx_);
            if (--workers_pending_ == 0)
                wave_done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex               mtx_;
    std::condition_variable  wake_workers_;  // будит бригаду на новую волну
    std::condition_variable  wave_done_;     // будит вызывающего по завершении

    // Состояние текущей волны (под mtx_):
    const std::function<void(int)>* current_work_ = nullptr;  // тело куска
    int      chunk_count_     = 0;   // сколько кусков в этой волне (вкл. chunk 0)
    int      workers_pending_ = 0;   // сколько рабочих ещё не добили свой кусок
    unsigned wave_seq_        = 0;   // монотонный номер волны (детектор пробуждения)
    bool     stopping_        = false;
};

// ── Глобальная конфигурация и ленивый пул ──────────────────────────────────

std::mutex                g_config_mtx;        // защищает желаемый размер и пул
int                       g_requested_threads = 0;  // 0 = авто (число ядер)
std::unique_ptr<WavePool> g_pool;              // null => последовательный режим

// Разрешить желаемое число потоков в фактическое (>= 1).
int resolve_threads(int requested) {
    if (requested == 1) return 1;
    if (requested > 1)  return requested;
    unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

// Пул под текущий g_requested_threads; null, если потоков <= 1 (тогда всё
// считается последовательно). Создаётся лениво и переиспользуется между волнами.
WavePool* active_pool() {
    std::lock_guard<std::mutex> lk(g_config_mtx);
    const int threads = resolve_threads(g_requested_threads);
    if (threads <= 1) {
        g_pool.reset();
        return nullptr;
    }
    if (!g_pool || g_pool->executor_count() != threads)
        g_pool = std::make_unique<WavePool>(threads - 1);  // -1 на вызывающего
    return g_pool.get();
}

}  // namespace

void set_num_threads(int threads) {
    std::lock_guard<std::mutex> lk(g_config_mtx);
    g_requested_threads = threads;
    g_pool.reset();   // пересоздастся нужного размера при следующем active_pool()
}

int num_threads() {
    std::lock_guard<std::mutex> lk(g_config_mtx);
    return resolve_threads(g_requested_threads);
}

namespace detail {

int plan_chunks(std::int64_t count, std::int64_t min_grain) {
    if (count <= 0 || t_inside_wave) return 1;
    if (min_grain < 1) min_grain = 1;

    WavePool* pool = active_pool();
    if (!pool) return 1;                       // последовательный режим

    const std::int64_t by_grain = count / min_grain;  // макс. кусков по грейну
    if (by_grain < 2) return 1;                // работы меньше двух кусков
    const std::int64_t chunks =
        std::min<std::int64_t>(by_grain, pool->executor_count());
    return static_cast<int>(chunks);
}

void run_chunks(int chunk_count,
                const std::function<void(int chunk_index)>& run_chunk) {
    WavePool* pool = active_pool();
    if (!pool) {                  // подстраховка: пул исчез между plan и run
        for (int c = 0; c < chunk_count; ++c) run_chunk(c);
        return;
    }

    t_inside_wave = true;         // запрет вложенного дробления на этом потоке
    std::exception_ptr first_error;
    std::mutex          error_mtx;
    auto guarded = [&](int chunk_index) {
        try {
            run_chunk(chunk_index);
        } catch (...) {
            std::lock_guard<std::mutex> lk(error_mtx);
            if (!first_error) first_error = std::current_exception();
        }
    };
    pool->run_wave(chunk_count, guarded);
    t_inside_wave = false;

    if (first_error) std::rethrow_exception(first_error);
}

}  // namespace detail

}  // namespace ii
