// Единый модуль распараллеливания всего проекта `ii`.
//
// Предоставляет примитив — parallel_for(count, min_grain, body) — для
// параллелизма ПО ДАННЫМ: «разрежь диапазон [0, count) на куски и посчитай
// каждый кусок, по возможности на разных ядрах». Этого достаточно для всех мест
// проекта: горячих ядер движка (engine/ops.cpp), сборки тайлов, декода выхода,
// препроцесса — везде, где есть N независимых единиц работы.
//
// Самодостаточность: модуль зависит только от STL и std::thread, поэтому и
// движок, и раннер остаются standalone (см. engine/ii.h).
//
// Требования: C++17 (thread-safe инициализация локальных статиков).

#pragma once

#include <cstdint>
#include <functional>

namespace ii {

// Сколько потоков задействует parallel_for, ВКЛЮЧАЯ вызывающий:
//   threads == 1  -> строго последовательно (потоки не создаются);
//   threads == 0  -> по числу аппаратных ядер (hardware_concurrency);
//   threads  > 1  -> ровно столько (1 вызывающий + threads-1 рабочих).
// Задаётся до инференса (бэкенд вызывает это в load() из Options::num_threads).
// Смена размера пересоздаёт пул лениво — при следующем параллельном вызове.
void set_num_threads(int threads);

// Фактическое число потоков (с разрешением 0 -> число аппаратных ядер).
int num_threads();

namespace detail {

// Сколько кусков нарезать для [0, count) при минимуме min_grain элементов на
// кусок. Возвращает 1 (= считать последовательно на месте), если дробить
// не нужно или нельзя:
//   • count < 2 * min_grain  — работы меньше, чем на два куска;
//   • включён 1-поточный режим (set_num_threads(1) / одно ядро);
//   • вызов вложен в другой parallel_for (рабочие потоки всегда «внутри»,
//     поэтому вложенного дробления и связанного с ним дедлока не возникает).
// Иначе возвращает число кусков в диапазоне [2, число потоков].
int plan_chunks(std::int64_t count, std::int64_t min_grain);

// Выполнить run_chunk(chunk_index) для chunk_index в [0, chunk_count):
// chunk 0 — на вызывающем потоке, остальные — на рабочих, — и дождаться
// завершения всех. Первое исключение из run_chunk прокидывается вызывающему.
// Вызывается только при chunk_count >= 2 (см. plan_chunks).
void run_chunks(int chunk_count, const std::function<void(int chunk_index)>& run_chunk);

}  // namespace detail

// Разрезать [0, count) на куски и выполнить body(begin, end) над каждым,
// по возможности параллельно. Тело ОБЯЗАНО писать только в «свой» диапазон
// выхода [begin, end), не пересекающийся с другими кусками, — тогда результат
// бит-в-бит совпадает с однопоточным.
//
// min_grain — минимальный размер куска (в единицах count), при котором
// распараллеливание окупается. Слишком мелкий тензор посчитается на месте
// (см. plan_chunks). Подбирается под стоимость одной единицы работы; в движке
// для этого есть ops.cpp:grain_for.
//
// Это шаблон, а не std::function, специально: на последовательном пути
// (1 поток / мелкий тензор / вложенный вызов) тело вызывается напрямую и
// инлайнится, без единой аллокации — одноядерная производительность не страдает.
template <class Body>
void parallel_for(std::int64_t count, std::int64_t min_grain, Body&& body) {
    if (count <= 0) return;

    const int chunk_count = detail::plan_chunks(count, min_grain);
    if (chunk_count <= 1) {                  // последовательный быстрый путь
        body(std::int64_t{0}, count);
        return;
    }

    // Равномерная статическая раскладка: первые (count % chunk_count) кусков на
    // один элемент длиннее остальных — разброс размеров не больше единицы.
    const std::int64_t base = count / chunk_count;
    const std::int64_t rem  = count % chunk_count;
    detail::run_chunks(chunk_count, [&](int chunk_index) {
        const std::int64_t begin = chunk_index * base +
                                   (chunk_index < rem ? chunk_index : rem);
        const std::int64_t len = base + (chunk_index < rem ? 1 : 0);
        if (len > 0) body(begin, begin + len);
    });
}

}  // namespace ii
