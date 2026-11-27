// SPDX-License-Identifier: MIT
/// @file
/// Helpers shared by tests: random tensors, tolerant comparison, running a model with the public API.
#pragma once

#include <ii/model.h>
#include <ii/tensor.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ii_test {

/// A tensor of `dtype` filled with reproducible pseudo-random values: floats uniform in [low, high), integers in
/// [low, high] (truncated), bools 50/50.
ii::Tensor random_tensor(ii::DType dtype, const ii::Shape& shape, std::uint32_t seed, double low = -2.0,
                         double high = 2.0);

/// Element `i` of a (contiguous) tensor converted to double.
double element(const ii::Tensor& tensor, std::int64_t i);

struct Tolerance {
    double rtol = 1e-5;
    double atol = 1e-6;
};

/// Default tolerance for comparing results computed in `dtype` (accounts for the precision of the type).
Tolerance tolerance_for(ii::DType dtype);

/// Compares two tensors element-wise: |a - b| <= atol + rtol * |b|; NaN equals NaN; infinities must match exactly.
/// Returns an empty string when they match, otherwise a description of the first mismatches.
std::string compare(const ii::Tensor& actual, const ii::Tensor& expected, Tolerance tolerance);

/// Directory with test data (tests/data). The II_TEST_DATA_DIR environment variable overrides the built-in path
/// (used when tests run on another machine, e.g. an Android device).
std::string test_data_dir();

/// Path of the shared library loaded by the DynLib tests; II_TEST_DYNLIB_PATH overrides the built-in path.
std::string test_dynlib_path();

/// Path of the ii-backend-test-remote plugin; II_TEST_PLUGIN_PATH overrides the built-in path.
std::string test_plugin_path();

/// Creates a context with a small thread pool for tests.
ii::Context test_context();

/// Compiles `model` with `options`, binds `inputs` (by declaration order) and returns the outputs.
ii::Result<std::vector<ii::Tensor>> run_model(const ii::Model& model, const std::vector<ii::Tensor>& inputs,
                                              const ii::CompileOptions& options = {});

/// Runs `model` with backend `a` and with backend `b` (each with ii-ref as the fallback for nodes it does not
/// support) and compares all outputs; returns an empty string when they agree. The harness for differential backend
/// tests (every backend against ii-ref).
ii::Result<std::string> compare_backends(const ii::Model& model, const std::vector<ii::Tensor>& inputs,
                                         std::string_view a, std::string_view b, Tolerance tolerance);

}  // namespace ii_test
