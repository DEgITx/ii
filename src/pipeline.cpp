// Реализация pipeline.h: bounded single-producer / single-consumer конвейер
// на одном персистентном фоновом потоке.
//
// Синхронизация — классический bounded buffer на двух счётчиках и двух
// condvar, структурно как WavePool::run_wave в parallel.cpp:
//   * produced_ — сколько слотов произведено (готово к consume);
//   * consumed_ — сколько уже потреблено;
//   * in-flight = produced_ - consumed_; producer ждёт, пока он < depth_
//     (значит слот i % depth_ уже освобождён consume'ом item'а i-depth_);
//   * consumer ждёт появления готового слота (consumed_ < produced_).
// Оба счётчика монотонны ВНУТРИ одного run() и сбрасываются в его начале.

#include "pipeline.h"

namespace ii {

Pipeline::Pipeline(int depth) : depth_(depth) {}

Pipeline::~Pipeline() { stop_worker(); }

void Pipeline::ensure_worker() {
    // run() вызывается из одного потока, гонки на создании нет.
    if (!worker_.joinable())
        worker_ = std::thread([this] { worker_loop(); });
}

void Pipeline::stop_worker() {
    if (!worker_.joinable()) return;   // depth<=1: фонового потока и не было
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stopping_ = true;
    }
    cv_consumer_.notify_all();
    worker_.join();
}

void Pipeline::worker_loop() {
    std::unique_lock<std::mutex> lk(mtx_);
    for (;;) {
        // Ждём готовый слот текущего прогона либо команду на выход. При abort_
        // новых слотов не берём (прогон фатально остановлен) — только stopping_
        // выведет поток из ожидания.
        cv_consumer_.wait(lk, [this] {
            return stopping_ || (!abort_ && consumed_ < produced_);
        });
        if (stopping_) return;

        const int i = consumed_;
        busy_ = true;
        lk.unlock();

        std::exception_ptr caught;
        try {
            (*consume_)(i);
        } catch (...) {
            caught = std::current_exception();
        }

        lk.lock();
        consumed_ = i + 1;
        busy_ = false;
        if (caught) {
            if (!error_) error_ = caught;
            abort_ = true;          // останавливаем прогон, run() пробросит
        }
        cv_producer_.notify_one();  // освободили слот / сообщили об ошибке
    }
}

bool Pipeline::run(int count,
                   const std::function<bool(int)>& produce,
                   const std::function<void(int)>& consume) {
    if (count <= 0) return true;

    // Вырожденный последовательный режим: без фонового потока и пробуждений.
    // Полезно для A/B-замеров (--tile-pipeline-depth 1) и одиночного тайла.
    if (depth_ <= 1) {
        for (int i = 0; i < count; ++i) {
            if (!produce(i)) return false;
            consume(i);
        }
        return true;
    }

    ensure_worker();

    {
        std::lock_guard<std::mutex> lk(mtx_);
        consume_  = &consume;
        count_    = count;
        produced_ = 0;
        consumed_ = 0;
        busy_     = false;
        abort_    = false;
        error_    = nullptr;
    }

    for (int i = 0; i < count; ++i) {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            // Backpressure: не более depth_ слотов в работе одновременно.
            cv_producer_.wait(lk, [this] {
                return abort_ || (produced_ - consumed_) < depth_;
            });
            if (abort_) break;      // consume() кинул исключение — прекращаем
        }

        if (!produce(i)) {          // фатальная ошибка продюсера
            std::lock_guard<std::mutex> lk(mtx_);
            abort_ = true;
            cv_consumer_.notify_one();
            break;
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            produced_ = i + 1;
        }
        cv_consumer_.notify_one();  // появился готовый слот
    }

    // Дождаться, пока фоновый поток освободится: при штатном завершении —
    // когда потреблено всё произведённое; при abort — когда воркер дочитал
    // свой текущий слот (новых он уже не берёт).
    std::exception_ptr err;
    bool aborted;
    {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_producer_.wait(lk, [this] {
            return !busy_ && (abort_ || consumed_ == produced_);
        });
        err      = error_;
        error_   = nullptr;
        aborted  = abort_;
        consume_ = nullptr;
    }

    if (err) std::rethrow_exception(err);
    return !aborted;
}

}  // namespace ii
