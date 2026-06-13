// Единый модуль распараллеливания всего проекта `ii`.
//
// Раньше параллелизм был раздроблён: движок (engine/) имел свой intra-op
// `parallel_for`, а раннер — отдельный пул потоков, который вдобавок никто не
// использовал. Теперь оба инструмента живут здесь, в одном пространстве имён
// `ii`, и дополняют друг друга:
//
//   1. parallel_for(count, min_grain, body) — intra-op параллелизм ПО ДАННЫМ.
//      Граф исполняется последовательно (нода за нодой), но внутри каждого
//      тяжёлого ядра независимые выходные элементы раскладываются на ядра CPU.
//      Разные потоки пишут в НЕпересекающиеся диапазоны выхода и в исходном
//      порядке индексов внутри своего куска — поэтому результат побитово
//      совпадает с однопоточным. Это важно: движок `ii` служит эталоном
//      корректности для TFLite/TensorRT/DirectML, и распараллеливание не
//      должно менять числа. Это основной примитив для горячих ядер
//      (engine/ops.cpp), а также для tile-сборки / декода / препроцесса.
//      Одноядерный режим бесплатен: set_num_threads(1) (или машина с одним
//      ядром) вообще не создаёт потоков, а parallel_for сводится к прямому
//      инлайн-вызову тела — без мьютексов, futures и атомиков на горячем пути.
//
//   2. ThreadPool + submit() + default_pool() — обобщённый пул задач c
//      std::future для КРУПНОЗЕРНИСТЫХ / РАЗНОРОДНЫХ задач, где удобнее
//      раздать функции и собрать результаты, а не нарезать однородный
//      диапазон. Это отдельный, независимый от (1) пул.
//
// Самодостаточность: модуль не зависит ни от чего, кроме STL и std::thread,
// поэтому и движок, и раннер остаются standalone (см. engine/ii.h).
//
// Требования: C++17 (std::invoke_result_t, thread-safe инициализация
// локальных статиков).

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
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

namespace ii {

// ---------------------------------------------------------------------------
// 1. intra-op параллелизм по данным (бит-в-бит детерминированный)
// ---------------------------------------------------------------------------

// Сколько потоков задействует parallel_for, ВКЛЮЧАЯ вызывающий.
//   threads == 1  -> строго последовательно (потоки не создаются);
//   threads == 0  -> по числу аппаратных ядер (hardware_concurrency);
//   threads  > 1  -> ровно столько (1 вызывающий + threads-1 рабочих).
// Вызывать до инференса (бэкенд делает это в load() из Options::num_threads).
// Смена размера пересоздаёт пул при следующем параллельном вызове.
void set_num_threads(int threads);

// Фактическое число потоков (с учётом разрешения 0 -> число ядер).
int num_threads();

namespace detail {

// Сколько кусков нарезать для диапазона [0, count) при минимуме min_grain
// элементов на кусок. Возвращает 1, если дробить не нужно или нельзя:
// мало работы (count < 2*min_grain), включён 1-поточный режим, либо вызов
// вложен в другой parallel_for (рабочие потоки всегда «внутри», поэтому
// вложенного распараллеливания и связанного с ним дедлока не возникает).
int plan_chunks(std::int64_t count, std::int64_t min_grain);

// Выполнить job(rank) для rank в [0, chunks) — rank 0 на вызывающем потоке,
// остальные на рабочих — и дождаться завершения. Первое исключение из job
// прокидывается вызывающему. Вызывается только при chunks >= 2.
void run_on_pool(int chunks, const std::function<void(int rank)>& job);

}  // namespace detail

// Разрезать [0, count) на куски и выполнить body(begin, end) над каждым,
// по возможности параллельно. min_grain — минимальный размер куска, при
// котором распараллеливание оправдано (см. plan_chunks). Тело должно писать
// только в «свой» диапазон выхода, не зависящий от других кусков.
//
// Шаблон, а не std::function: на последовательном пути (1 поток / мелкий
// тензор / вложенный вызов) тело вызывается напрямую и инлайнится, без
// единой аллокации — поэтому одноядерная производительность не страдает.
template <class Body>
void parallel_for(std::int64_t count, std::int64_t min_grain, Body&& body) {
    if (count <= 0) return;

    int chunks = detail::plan_chunks(count, min_grain);
    if (chunks <= 1) {                       // последовательный быстрый путь
        body(static_cast<std::int64_t>(0), count);
        return;
    }

    // Равномерная раскладка: первые (count % chunks) кусков на 1 элемент длиннее.
    const std::int64_t base = count / chunks;
    const std::int64_t rem  = count % chunks;
    detail::run_on_pool(chunks, [&](int rank) {
        std::int64_t begin = static_cast<std::int64_t>(rank) * base +
                             (rank < rem ? rank : rem);
        std::int64_t len = base + (rank < rem ? 1 : 0);
        if (len > 0) body(begin, begin + len);
    });
}

// ---------------------------------------------------------------------------
// 2. обобщённый пул задач с futures (крупнозернистые / разнородные задачи)
// ---------------------------------------------------------------------------

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
// hardware_concurrency). Подходит для большинства задач submit/parallel_for.
ThreadPool& default_pool();

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

}  // namespace ii
