// Backend-agnostic интерфейс инференса.
//
// Назначение: оторвать ii.cpp от конкретного фреймворка (TFLite) и
// открыть путь к альтернативным бэкендам — TensorRT, DirectML, ONNX
// Runtime — без перетряхивания основного раннера. Один и тот же бинарь
// при сборке с разными опциями CMake может тянуть разные реализации
// Engine.
//
// Архитектура:
//   * Заголовок не подключает ни одной зависимости конкретного бэкенда —
//     ii.cpp, yolo.cpp и т.п. видят только эти типы.
//   * Реализации лежат в отдельных .cpp (inference_tflite.cpp,
//     в будущем inference_tensorrt.cpp, inference_directml.cpp),
//     каждая регистрируется в make_engine() через свою фабричную
//     функцию.
//   * Хот-пас (Invoke + доступ к буферу) — один виртуальный вызов на
//     инференс. На фоне миллисекундной латентности модели это ~1 нс
//     накладных, не измеримо.
//
// Жизненный цикл:
//   auto eng = inf::make_engine();              // дефолтный бэкенд
//   inf::Engine::Options opts;
//   opts.delegate_path = "/path/to/libdelegate.so";   // или "" — CPU
//   opts.num_threads   = 4;
//   if (!eng->load("model.tflite", opts)) ...;
//   void* in = eng->input_data(0);
//   fill_input(rgb, eng->inputs()[0], in);
//   eng->invoke();
//   const void* out = eng->output_data(0);
//   dequantize_output(eng->outputs()[0], out, float_buf);

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "half.h"  // inf::half_to_float — единая реализация на весь проект

namespace inf {

// Платформенно-нейтральные типы данных тензора. Покрывают весь набор,
// поддерживаемый TFLite + типовые случаи TensorRT/DirectML. Конкретный
// бэкенд маппит свой родной enum в этот при заполнении TensorDesc.
enum class DType {
    Unknown = 0,
    Float32,
    Float16,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    Bool,
};

const char* dtype_name(DType t);
std::size_t dtype_size(DType t);

// half_to_float теперь живёт в half.h (подключён выше) — одна реализация
// на весь проект. Имя и пространство имён (inf::half_to_float) сохранены.

// Срез информации о тензоре в backend-нейтральном виде.
// scale == 0 означает «не квантован» (или per-channel — в этом случае
// бэкенд выставляет 0, и потребитель не использует scalar-кванты).
struct TensorDesc {
    int              index = 0;        // индекс внутри бэкенда (для отладки)
    std::string      name;
    std::vector<int> shape;
    DType            dtype = DType::Unknown;
    float            scale = 0.0f;
    std::int32_t     zero_point = 0;
    std::size_t      bytes = 0;
};

// Абстрактный движок инференса. Один экземпляр — одна загруженная
// модель. Загрузка/инференс делаются последовательно (не thread-safe).
class Engine {
public:
    struct Options {
        // Путь к внешнему делегату/плагину ускорения. Пустая строка —
        // CPU fallback бэкенда. Для TFLite это путь к .so/.dll, для
        // TensorRT/DirectML пока зарезервировано (бэкенды сами решат,
        // что это значит).
        std::string delegate_path;
        // Число CPU-потоков для бэкенда. 0 — оставить дефолт.
        int         num_threads = 0;
    };

    virtual ~Engine() = default;

    // Загрузить модель и подготовить тензоры (включая делегата, если
    // задан). Печатает диагностику в stdout/stderr — те же сообщения,
    // что были в старом инлайн-движке. Возвращает false при любой
    // фатальной ошибке.
    virtual bool load(const std::string& model_path, const Options& opts) = 0;

    // Один проход инференса. False — фатальная ошибка бэкенда (стоит
    // прекратить цикл). Перед первым вызовом вход(ы) должны быть
    // заполнены через input_data().
    virtual bool invoke() = 0;

    // Список описаний входов/выходов в порядке индексирования
    // 0..size()-1. Возврат — ссылка на внутренний кэш Engine (живёт
    // до его уничтожения), копировать не нужно.
    virtual const std::vector<TensorDesc>& inputs()  const = 0;
    virtual const std::vector<TensorDesc>& outputs() const = 0;

    // Прямой доступ к памяти тензора. Возвращаемый указатель валиден
    // до следующего load(). bytes у дескриптора показывает реальный
    // размер буфера.
    //
    // Для TFLite это указатель внутрь arena интерпретатора — никакой
    // дополнительной копии. Для TensorRT/DirectML в будущем это может
    // быть pinned-host buffer, который перед invoke() сам бэкенд
    // выгрузит на устройство.
    virtual void*       input_data(int idx)        = 0;
    virtual const void* output_data(int idx) const = 0;

    // Человекочитаемое имя бэкенда — "tflite", "tensorrt" и т.п.
    // Печатается в логах/CSV-мете, помогает диагностике.
    virtual const char* backend_name() const = 0;
};

// Фабрика. backend = "" или "tflite" — TFLite (на текущий момент
// единственный собираемый); далее зарезервированы "tensorrt", "directml".
// Возвращает nullptr, если запрошенный бэкенд не собран в этом
// бинаре, и печатает причину в stderr.
std::unique_ptr<Engine> make_engine(const std::string& backend = "");

// Список бэкендов, собранных в текущий бинарь. Удобно для --help / UI.
std::vector<std::string> available_backends();

// Платформенный дефолт для пути к внешнему делегату TFLite. По умолчанию
// пуст ("") — раннер работает на CPU / выбранном бэкенде без делегата.
// Конкретный путь может подставить опциональный модуль делегата,
// собираемый отдельно (см. delegate.cpp + опции CMake), либо пользователь
// через --delegate. ii.cpp использует это как значение по-умолчанию.
const char* default_delegate_path();

}  // namespace inf
