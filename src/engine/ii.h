// Единая точка подключения встроенного движка `ii` как библиотеки.
//
// Хост-приложению, которому нужен движок (тензоры + операции + граф +
// загрузчик ONNX), достаточно одного #include "engine/ii.h" — без знания
// о внутреннем разбиении на модули. Для интеграции в качестве бэкенда
// ii::Engine используйте дополнительно фабрику make_ii_engine()
// (engine/backend.cpp) — она объявлена в inference.h-стеке, чтобы не тянуть
// контракт раннера в чисто библиотечные сборки.
//
// Всё API живёт в пространстве имён ii.

#pragma once

#include "engine/tensor.h"   // ii::Tensor, Shape, numel, ...
#include "engine/ops.h"      // математические ядра (conv/matmul/softmax/...)
#include "engine/graph.h"    // Graph, Node, OpRegistry, Executor, register_op
#include "engine/loader.h"   // load_onnx / parse_onnx
#include "parallel.h"        // parallel_for / set_num_threads / num_threads
