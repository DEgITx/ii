// Точки входа загрузчиков моделей в граф движка `ii`.
//
// Загрузчик наполняет ii::Graph (ноды + веса-константы + имена и формы
// входов/выходов) из файла модели. Форматы:
//   * ONNX  — ii_onnx.cpp  (основной);
//   * TFLite — позже        (второстепенный, flatbuffer).
// Парсеры самостоятельны (свой минимальный protobuf/flatbuffer-ридер),
// чтобы движок оставался zero-dependency и собирался на любой платформе.

#pragma once

#include <string>

#include "ii_graph.h"

namespace ii {

// Загрузить ONNX-модель в граф. true — успех; иначе false и текст ошибки
// в err. Граф остаётся пригодным к ii::Executor.
bool load_onnx(const std::string& path, Graph& g, std::string& err);

// Разобрать ONNX-модель прямо из буфера в памяти (сериализованный
// ModelProto). Тот же результат, что у load_onnx, но без файловой
// системы — удобно для юнит-тестов и встраивания.
bool parse_onnx(const void* data, std::size_t size, Graph& g,
                std::string& err);

}  // namespace ii
