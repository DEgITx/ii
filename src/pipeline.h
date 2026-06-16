// Двухстадийный конвейер «producer → consumer» на одном фоновом потоке.
//
// ── Зачем оно надо ─────────────────────────────────────────────────────────
//
// Когда работа естественно делится на стадию, занимающую УСКОРИТЕЛЬ (NPU/GPU:
// invoke), и стадию, грузящую CPU (decode + сборка пикселей), последовательный
// прогон «invoke; decode; invoke; decode; …» простаивает: пока считает NPU —
// CPU отдыхает, пока считает CPU — NPU отдыхает. Этот модуль перекрывает их:
// producer кадра i+1 идёт параллельно consumer'у кадра i. В tile-режиме это
// invoke тайла vs decode+composite предыдущего тайла (см. ii.cpp:run_tile_pass).
//
// ── Чем это НЕ является ─────────────────────────────────────────────────────
//
// Это НЕ data-parallelism (для него есть parallel.h:parallel_for, который рубит
// ОДНУ задачу на куски по ядрам). Здесь две РАЗНЫЕ по природе стадии на двух
// потоках. parallel.h осознанно держит только параллелизм по данным и в своей
// шапке отсылает «долгоживущий producer/consumer — это обычный std::thread»;
// Pipeline — ровно этот std::thread, аккуратно упакованный и переиспользуемый.
//
// ── Контракт ───────────────────────────────────────────────────────────────
//
// Persistent: фоновый поток создаётся лениво при первом run() и живёт до
// разрушения объекта — нулевой per-frame thread-churn (как WavePool в
// parallel.cpp). Объект НЕ потокобезопасен: run() зовётся из одного потока.

#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace ii {

class Pipeline {
public:
    // depth — глубина кольца слотов (двойная буферизация = 2). produce(i+depth)
    // не начнётся, пока не завершится consume(i): backpressure + ограниченная
    // память (вызывающий держит ровно depth слотов). depth <= 1 → строго
    // последовательный режим (produce(i); consume(i) на вызывающем потоке,
    // фоновый поток не создаётся) — удобно для A/B-замеров.
    explicit Pipeline(int depth = 2);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Прогнать count единиц работы через две стадии:
    //   * produce(i) — НА ВЫЗЫВАЮЩЕМ потоке, строго i = 0..count-1; готовит
    //     слот i (например, invoke + копия выхода в буфер i % depth). Возврат
    //     false = фатальная ошибка: конвейер останавливается, run() вернёт
    //     false (поведение старого `return false` в горячем цикле);
    //   * consume(i) — НА ФОНОВОМ потоке, строго в том же порядке i; забирает
    //     слот i (decode + сборка). Нефатальная постобработка — сама решает,
    //     что делать с битым слотом (просто не использовать; аналог `continue`).
    //
    // Возвращает true, если все count единиц обработаны; false — если produce
    // вернул false (тогда оставшиеся единицы не запускаются). Исключение из
    // любой стадии пробрасывается из run() после остановки обоих потоков (как
    // parallel_for). Бит-в-бит эквивалентно последовательному «produce(i);
    // consume(i)» по наблюдаемому результату — порядок consume сохранён.
    bool run(int count,
             const std::function<bool(int)>& produce,
             const std::function<void(int)>& consume);

private:
    void ensure_worker();             // лениво поднять фоновый поток
    void stop_worker();               // корректно погасить (из деструктора)
    void worker_loop();

    const int depth_;

    std::thread             worker_;
    std::mutex              mtx_;
    std::condition_variable cv_producer_;  // ждёт освобождения слота / завершения
    std::condition_variable cv_consumer_;  // ждёт появления готового слота / стоп

    // Состояние текущего прогона (под mtx_). produced_/consumed_ — монотонные
    // счётчики ВНУТРИ run(); слот i живёт в кольце как i % depth_.
    const std::function<void(int)>* consume_ = nullptr;  // тело consumer'а на run()
    int  count_     = 0;       // сколько единиц в этом прогоне
    int  produced_  = 0;       // сколько произведено (готово к consume)
    int  consumed_  = 0;       // сколько уже потреблено
    bool busy_      = false;   // фоновый поток сейчас внутри consume()
    bool abort_     = false;   // фатальная остановка прогона
    bool stopping_  = false;   // гасим фоновый поток (деструктор)
    std::exception_ptr error_; // первое исключение из consume()
};

}  // namespace ii
