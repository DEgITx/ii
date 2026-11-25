// SPDX-License-Identifier: MIT
#include "support/test_util.h"

#include "core/platform/platform.h"

#include <cmath>
#include <cstring>
#include <format>

namespace ii_test {

namespace {

template <class T>
T* typed(const ii::Tensor& tensor) {
    return static_cast<T*>(tensor.data());
}

}  // namespace

std::string test_data_dir() {
    return ii::platform::get_env("II_TEST_DATA_DIR").value_or(II_TEST_DATA_DIR);
}

std::string test_dynlib_path() {
    return ii::platform::get_env("II_TEST_DYNLIB_PATH").value_or(II_TEST_DYNLIB_PATH);
}

ii::Tensor random_tensor(ii::DType dtype, const ii::Shape& shape, std::uint32_t seed, double low, double high) {
    ii::Tensor tensor = ii::Tensor::empty(dtype, shape).value();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(low, high);
    const std::int64_t n = tensor.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        const double v = uniform(rng);
        switch (dtype) {
        case ii::DType::kF32:
            typed<float>(tensor)[i] = static_cast<float>(v);
            break;
        case ii::DType::kF64:
            typed<double>(tensor)[i] = v;
            break;
        case ii::DType::kF16:
            typed<ii::f16>(tensor)[i] = ii::f16(static_cast<float>(v));
            break;
        case ii::DType::kBF16:
            typed<ii::bf16>(tensor)[i] = ii::bf16(static_cast<float>(v));
            break;
        case ii::DType::kI8:
            typed<std::int8_t>(tensor)[i] = static_cast<std::int8_t>(std::llround(v));
            break;
        case ii::DType::kU8:
            typed<std::uint8_t>(tensor)[i] = static_cast<std::uint8_t>(std::llround(std::fabs(v)));
            break;
        case ii::DType::kI16:
            typed<std::int16_t>(tensor)[i] = static_cast<std::int16_t>(std::llround(v));
            break;
        case ii::DType::kU16:
            typed<std::uint16_t>(tensor)[i] = static_cast<std::uint16_t>(std::llround(std::fabs(v)));
            break;
        case ii::DType::kI32:
            typed<std::int32_t>(tensor)[i] = static_cast<std::int32_t>(std::llround(v));
            break;
        case ii::DType::kU32:
            typed<std::uint32_t>(tensor)[i] = static_cast<std::uint32_t>(std::llround(std::fabs(v)));
            break;
        case ii::DType::kI64:
            typed<std::int64_t>(tensor)[i] = std::llround(v);
            break;
        case ii::DType::kU64:
            typed<std::uint64_t>(tensor)[i] = static_cast<std::uint64_t>(std::llround(std::fabs(v)));
            break;
        case ii::DType::kBool:
            typed<std::uint8_t>(tensor)[i] = v >= (low + high) / 2 ? 1 : 0;
            break;
        default:
            break;
        }
    }
    return tensor;
}

double element(const ii::Tensor& tensor, std::int64_t i) {
    switch (tensor.dtype()) {
    case ii::DType::kF32:
        return typed<float>(tensor)[i];
    case ii::DType::kF64:
        return typed<double>(tensor)[i];
    case ii::DType::kF16:
        return typed<ii::f16>(tensor)[i].to_float();
    case ii::DType::kBF16:
        return typed<ii::bf16>(tensor)[i].to_float();
    case ii::DType::kI8:
        return typed<std::int8_t>(tensor)[i];
    case ii::DType::kU8:
        return typed<std::uint8_t>(tensor)[i];
    case ii::DType::kI16:
        return typed<std::int16_t>(tensor)[i];
    case ii::DType::kU16:
        return typed<std::uint16_t>(tensor)[i];
    case ii::DType::kI32:
        return typed<std::int32_t>(tensor)[i];
    case ii::DType::kU32:
        return typed<std::uint32_t>(tensor)[i];
    case ii::DType::kI64:
        return static_cast<double>(typed<std::int64_t>(tensor)[i]);
    case ii::DType::kU64:
        return static_cast<double>(typed<std::uint64_t>(tensor)[i]);
    case ii::DType::kBool:
        return typed<std::uint8_t>(tensor)[i] != 0 ? 1.0 : 0.0;
    default:
        return 0.0;
    }
}

Tolerance tolerance_for(ii::DType dtype) {
    switch (dtype) {
    case ii::DType::kF64:
        return {1e-12, 1e-12};
    case ii::DType::kF32:
        return {1e-5, 1e-6};
    case ii::DType::kF16:
        return {4e-3, 1e-3};
    case ii::DType::kBF16:
        return {3e-2, 1e-2};
    default:
        return {0, 0};
    }
}

std::string compare(const ii::Tensor& actual_input, const ii::Tensor& expected_input, Tolerance tolerance) {
    if (actual_input.dtype() != expected_input.dtype()) {
        return std::format("dtype {} != {}", ii::to_string(actual_input.dtype()),
                           ii::to_string(expected_input.dtype()));
    }
    if (!(actual_input.shape() == expected_input.shape())) {
        return std::format("shape {} != {}", actual_input.shape().to_string(), expected_input.shape().to_string());
    }
    const ii::Tensor actual = actual_input.contiguous().value();
    const ii::Tensor expected = expected_input.contiguous().value();
    std::string report;
    int mismatches = 0;
    for (std::int64_t i = 0; i < actual.numel(); ++i) {
        const double a = element(actual, i);
        const double e = element(expected, i);
        bool ok = false;
        if (std::isnan(a) || std::isnan(e)) {
            ok = std::isnan(a) && std::isnan(e);
        } else if (std::isinf(a) || std::isinf(e)) {
            ok = a == e;
        } else {
            ok = std::fabs(a - e) <= tolerance.atol + tolerance.rtol * std::fabs(e);
        }
        if (!ok) {
            if (++mismatches <= 5) {
                report += std::format("  [{}]: {} vs expected {}\n", i, a, e);
            }
        }
    }
    if (mismatches > 0) {
        report = std::format("{} of {} elements differ:\n", mismatches, actual.numel()) + report;
    }
    return report;
}

ii::Context test_context() {
    ii::ContextOptions options;
    options.num_threads = 2;
    return ii::Context::create(std::move(options)).value();
}

ii::Result<std::vector<ii::Tensor>> run_model(const ii::Model& model, const std::vector<ii::Tensor>& inputs,
                                              const ii::CompileOptions& options) {
    II_ASSIGN_OR_RETURN(ii::Program program, model.compile(options));
    II_ASSIGN_OR_RETURN(ii::Session session, program.new_session());
    const std::vector<ii::TensorInfo> infos = program.inputs();
    if (infos.size() != inputs.size()) {
        return ii::errors::invalid_argument("{} inputs given, the model has {}", inputs.size(), infos.size());
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        II_TRY(session.set_input(infos[i].name, inputs[i]));
    }
    II_TRY(session.run());
    std::vector<ii::Tensor> outputs;
    for (const ii::TensorInfo& info : program.outputs()) {
        II_ASSIGN_OR_RETURN(const ii::Tensor output, session.output(info.name));
        II_ASSIGN_OR_RETURN(ii::Tensor copy, output.clone());
        outputs.push_back(std::move(copy));
    }
    return outputs;
}

ii::Result<std::string> compare_backends(const ii::Model& model, const std::vector<ii::Tensor>& inputs,
                                         std::string_view a, std::string_view b, Tolerance tolerance) {
    // Each backend gets ii-ref as the fallback for the nodes it does not support.
    auto options_for = [](std::string_view backend) {
        ii::CompileOptions options;
        options.backends = {std::string(backend)};
        if (backend != "ii-ref") {
            options.backends.emplace_back("ii-ref");
        }
        return options;
    };
    const ii::CompileOptions options_a = options_for(a);
    const ii::CompileOptions options_b = options_for(b);
    II_ASSIGN_OR_RETURN(const std::vector<ii::Tensor> outputs_a, run_model(model, inputs, options_a));
    II_ASSIGN_OR_RETURN(const std::vector<ii::Tensor> outputs_b, run_model(model, inputs, options_b));
    std::string report;
    for (std::size_t i = 0; i < outputs_a.size(); ++i) {
        const std::string mismatch = compare(outputs_a[i], outputs_b[i], tolerance);
        if (!mismatch.empty()) {
            report += std::format("output {}: {}", i, mismatch);
        }
    }
    return report;
}

}  // namespace ii_test
