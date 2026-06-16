// Тесты для ii::Pipeline — двухстадийного producer/consumer конвейера.
//
// Проверяем то, что нельзя «увидеть глазами» в горячем коде tile-режима:
//   * consume вызывается строго по порядку 0..count-1;
//   * backpressure: одновременно «в работе» не больше depth слотов
//     (значит слот t % depth можно безопасно переиспользовать);
//   * produce тоже идёт по порядку и на вызывающем потоке;
//   * фатальный возврат produce останавливает конвейер;
//   * исключение из consume пробрасывается из run();
//   * depth=1 — строго последовательный режим;
//   * один объект переиспользуется между run() (persistent worker).

#include <atomic>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "pipeline.h"

namespace {

// consume получает каждый индекс ровно один раз и строго по возрастанию.
TEST(Pipeline, ConsumesInOrder) {
    ii::Pipeline pipe(2);
    std::vector<int> consumed;
    bool ok = pipe.run(
        100,
        [](int) { return true; },
        [&](int i) { consumed.push_back(i); });
    EXPECT_TRUE(ok);
    ASSERT_EQ(consumed.size(), 100u);
    for (int i = 0; i < 100; ++i) EXPECT_EQ(consumed[i], i);
}

// produce вызывается по порядку и ровно count раз.
TEST(Pipeline, ProducesInOrder) {
    ii::Pipeline pipe(3);
    int next = 0;
    bool ok = pipe.run(
        50,
        [&](int i) { EXPECT_EQ(i, next++); return true; },
        [](int) {});
    EXPECT_TRUE(ok);
    EXPECT_EQ(next, 50);
}

// Backpressure: число «в работе» (произведено, но ещё не потреблено) никогда
// не превышает depth. Consumer держим медленным, чтобы продюсер уперся в лимит.
TEST(Pipeline, RespectsDepthBackpressure) {
    const int depth = 2;
    ii::Pipeline pipe(depth);
    std::atomic<int> in_flight{0};
    std::atomic<int> max_in_flight{0};
    bool ok = pipe.run(
        200,
        [&](int) {
            int cur = ++in_flight;
            int prev = max_in_flight.load();
            while (cur > prev && !max_in_flight.compare_exchange_weak(prev, cur)) {}
            return true;
        },
        [&](int) {
            // Небольшая «работа», чтобы продюсер успел упереться в backpressure.
            for (volatile int k = 0; k < 1000; ++k) {}
            --in_flight;
        });
    EXPECT_TRUE(ok);
    EXPECT_LE(max_in_flight.load(), depth);
    EXPECT_GE(max_in_flight.load(), 1);
}

// Слот t % depth не переиспользуется, пока consume(t) не завершён: продюсер,
// записав «свой» индекс в слот, после backpressure-ожидания должен видеть, что
// consume уже забрал предыдущего жильца слота.
TEST(Pipeline, SlotReuseIsSafe) {
    const int depth = 2;
    ii::Pipeline pipe(depth);
    std::vector<int> slot_owner(depth, -1);     // кто сейчас «владелец» слота
    std::atomic<bool> race{false};
    bool ok = pipe.run(
        500,
        [&](int t) {
            // Продюсер занимает слот. Если consumer ещё не освободил прошлого
            // владельца этого слота — это нарушение backpressure.
            slot_owner[t % depth] = t;
            return true;
        },
        [&](int t) {
            if (slot_owner[t % depth] != t) race = true;  // слот уже перезаписан
        });
    EXPECT_TRUE(ok);
    EXPECT_FALSE(race.load());
}

// produce вернул false → run() возвращает false и больше единиц не запускает.
TEST(Pipeline, ProducerAbortStops) {
    ii::Pipeline pipe(2);
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    bool ok = pipe.run(
        100,
        [&](int i) {
            ++produced;
            return i < 10;          // на 10-м продюсер «падает»
        },
        [&](int) { ++consumed; });
    EXPECT_FALSE(ok);
    // Запущено produce примерно до точки отказа (плюс возможные in-flight),
    // но никак не все 100.
    EXPECT_LE(produced.load(), 13);
    EXPECT_LE(consumed.load(), produced.load());
}

// Исключение из consume пробрасывается из run().
TEST(Pipeline, ConsumerExceptionPropagates) {
    ii::Pipeline pipe(2);
    EXPECT_THROW(
        pipe.run(
            100,
            [](int) { return true; },
            [](int i) { if (i == 5) throw std::runtime_error("boom"); }),
        std::runtime_error);
}

// depth=1 — строго последовательный режим: produce(i) полностью предшествует
// consume(i), порядок сохранён, фоновый поток не нужен.
TEST(Pipeline, DepthOneIsSerial) {
    ii::Pipeline pipe(1);
    std::vector<int> trace;
    bool ok = pipe.run(
        5,
        [&](int i) { trace.push_back(100 + i); return true; },
        [&](int i) { trace.push_back(200 + i); });
    EXPECT_TRUE(ok);
    // Ожидаем строгое чередование produce(i), consume(i): 100,200,101,201,...
    const std::vector<int> expected = {100, 200, 101, 201, 102, 202,
                                       103, 203, 104, 204};
    EXPECT_EQ(trace, expected);
}

// Один объект переиспользуется между прогонами (persistent worker), счётчики
// корректно сбрасываются на каждый run().
TEST(Pipeline, ReusableAcrossRuns) {
    ii::Pipeline pipe(2);
    for (int r = 0; r < 5; ++r) {
        std::vector<int> consumed;
        bool ok = pipe.run(
            30,
            [](int) { return true; },
            [&](int i) { consumed.push_back(i); });
        EXPECT_TRUE(ok);
        ASSERT_EQ(consumed.size(), 30u);
        for (int i = 0; i < 30; ++i) EXPECT_EQ(consumed[i], i);
    }
}

// count==0 — no-op, успех.
TEST(Pipeline, EmptyRun) {
    ii::Pipeline pipe(2);
    int calls = 0;
    bool ok = pipe.run(
        0,
        [&](int) { ++calls; return true; },
        [&](int) { ++calls; });
    EXPECT_TRUE(ok);
    EXPECT_EQ(calls, 0);
}

}  // namespace
