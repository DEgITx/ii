// SPDX-License-Identifier: MIT
// check_backends(): a quick differential self-test of backends (`ii check-backends`).
#include <ii/model.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <string_view>

namespace ii {

namespace {

struct CheckCase {
    std::string_view name;
    std::string_view text;  // static shapes; inputs are filled with uniform values in [low, high]
    double low = -2;
    double high = 2;
};

// One operation per case where possible, so that a case either runs on the backend or is skipped as unsupported.
constexpr CheckCase kCases[] = {
    {"add", R"(graph add {
  input %a : f32[4, 8]
  input %b : f32[8]
  %y : f32[4, 8] = add(%a, %b)
  output y = %y
})"},
    {"mul", R"(graph mul {
  input %a : f32[4, 8]
  input %b : f32[4, 1]
  %y : f32[4, 8] = mul(%a, %b)
  output y = %y
})"},
    {"div", R"(graph div {
  input %a : f32[4, 8]
  input %b : f32[4, 8]
  %y : f32[4, 8] = div(%a, %b)
  output y = %y
})",
     0.5, 2},
    {"exp", R"(graph exp {
  input %a : f32[64]
  %y : f32[64] = exp(%a)
  output y = %y
})"},
    {"tanh", R"(graph tanh {
  input %a : f32[64]
  %y : f32[64] = tanh(%a)
  output y = %y
})"},
    {"sigmoid", R"(graph sigmoid {
  input %a : f32[64]
  %y : f32[64] = sigmoid(%a)
  output y = %y
})"},
    {"relu", R"(graph relu {
  input %a : f32[64]
  %y : f32[64] = relu(%a)
  output y = %y
})"},
    {"gelu", R"(graph gelu {
  input %a : f32[4, 16]
  %y : f32[4, 16] = gelu(%a)
  output y = %y
})"},
    {"silu", R"(graph silu {
  input %a : f32[4, 16]
  %y : f32[4, 16] = silu(%a)
  output y = %y
})"},
    {"softmax", R"(graph softmax {
  input %a : f32[4, 16]
  %y : f32[4, 16] = softmax(%a) {axis = -1}
  output y = %y
})"},
    {"layer_norm", R"(graph layer_norm {
  input %x : f32[4, 16]
  input %s : f32[16]
  input %b : f32[16]
  %y : f32[4, 16] = layer_norm(%x, %s, %b) {axis = -1, epsilon = 1e-05}
  output y = %y
})"},
    {"rms_norm", R"(graph rms_norm {
  input %x : f32[4, 16]
  input %s : f32[16]
  %y : f32[4, 16] = rms_norm(%x, %s) {axis = -1, epsilon = 1e-06}
  output y = %y
})"},
    {"matmul", R"(graph matmul {
  input %a : f32[8, 16]
  input %b : f32[16, 12]
  %y : f32[8, 12] = matmul(%a, %b)
  output y = %y
})"},
    {"matmul batched", R"(graph matmul_batched {
  input %a : f32[2, 8, 16]
  input %b : f32[2, 12, 16]
  %y : f32[2, 8, 12] = matmul(%a, %b) {transpose_b = true}
  output y = %y
})"},
    {"matmul f16", R"(graph matmul_f16 {
  input %a : f16[8, 16]
  input %b : f16[16, 12]
  %y : f16[8, 12] = matmul(%a, %b)
  output y = %y
})"},
    {"linear", R"(graph linear {
  input %x : f32[8, 16]
  input %w : f32[12, 16]
  input %b : f32[12]
  %y : f32[8, 12] = linear(%x, %w, %b)
  output y = %y
})"},
    {"linear f16 weights", R"(graph linear_f16 {
  input %x : f32[8, 16]
  input %w : f16[12, 16]
  %y : f32[8, 12] = linear(%x, %w, _)
  output y = %y
})"},
    {"conv", R"(graph conv {
  input %x : f32[1, 3, 8, 8]
  input %w : f32[4, 3, 3, 3]
  input %b : f32[4]
  %y : f32[1, 4, 8, 8] = conv(%x, %w, %b) {pads = [1, 1, 1, 1]}
  output y = %y
})"},
    {"max_pool", R"(graph max_pool {
  input %x : f32[1, 2, 8, 8]
  %y : f32[1, 2, 4, 4] = max_pool(%x) {kernel = [2, 2], strides = [2, 2]}
  output y = %y
})"},
    {"reduce_sum", R"(graph reduce_sum {
  input %x : f32[4, 16]
  %y : f32[4] = reduce_sum(%x) {axes = [1]}
  output y = %y
})"},
    {"reduce_max", R"(graph reduce_max {
  input %x : f32[4, 16]
  %y : f32[4, 1] = reduce_max(%x) {axes = [1], keepdims = true}
  output y = %y
})"},
    {"permute", R"(graph permute {
  input %x : f32[2, 3, 4]
  %y : f32[4, 2, 3] = permute(%x) {perm = [2, 0, 1]}
  output y = %y
})"},
    {"reshape", R"(graph reshape {
  input %x : f32[4, 6]
  %y : f32[2, 12] = reshape(%x) {shape = [2, 12]}
  output y = %y
})"},
    {"concat", R"(graph concat {
  input %a : f32[2, 3]
  input %b : f32[2, 5]
  %y : f32[2, 8] = concat(%a, %b) {axis = 1}
  output y = %y
})"},
    {"slice", R"(graph slice {
  input %x : f32[4, 8]
  %y : f32[4, 4] = slice(%x) {axes = [1], starts = [2], ends = [6]}
  output y = %y
})"},
    {"gather", R"(graph gather {
  input %t : f32[10, 4]
  const %i : i64[3] = i64[3] {1, 7, 3}
  %y : f32[3, 4] = gather(%t, %i) {axis = 0}
  output y = %y
})"},
    {"cast", R"(graph cast {
  input %x : f32[64]
  %y : f16[64] = cast(%x) {to = f16}
  output y = %y
})"},
    {"sdpa", R"(graph sdpa {
  input %q : f32[1, 2, 4, 8]
  input %k : f32[1, 2, 6, 8]
  input %v : f32[1, 2, 6, 8]
  %y : f32[1, 2, 4, 8] = sdpa(%q, %k, %v, _, _) {causal = true}
  output y = %y
})"},
    {"rope", R"(graph rope {
  input %x : f32[1, 4, 2, 8]
  input %c : f32[4, 1, 4]
  input %s : f32[4, 1, 4]
  %y : f32[1, 4, 2, 8] = rope(%x, %c, %s)
  output y = %y
})"},
};

/// Deterministic uniform values (xorshift).
Result<Tensor> make_input(const TensorInfo& info, std::uint64_t seed, double low, double high) {
    Shape shape;
    for (const Dim& dim : info.shape) {
        const std::optional<std::int64_t> value = dim.constant();
        if (!value) {
            return errors::internal("check case input {} has a symbolic shape", info.name);
        }
        shape.push_back(*value);
    }
    II_ASSIGN_OR_RETURN(Tensor f32, Tensor::empty(DType::kF32, shape));
    std::uint64_t state = seed * 0x9E3779B97F4A7C15ull + 1;
    for (float& value : f32.values<float>()) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const double unit = static_cast<double>(state >> 11) / static_cast<double>(std::uint64_t{1} << 53);
        value = static_cast<float>(low + (high - low) * unit);
    }
    return info.dtype == DType::kF32 ? f32 : f32.to(info.dtype);
}

Result<std::vector<Tensor>> run(const Model& model, const std::vector<Tensor>& inputs, std::string backend) {
    CompileOptions options;
    options.backends = {std::move(backend)};
    II_ASSIGN_OR_RETURN(Program program, model.compile(options));
    II_ASSIGN_OR_RETURN(Session session, program.new_session());
    const std::vector<TensorInfo> infos = program.inputs();
    for (std::size_t i = 0; i < infos.size(); ++i) {
        II_TRY(session.set_input(infos[i].name, inputs[i]));
    }
    II_TRY(session.run());
    std::vector<Tensor> outputs;
    for (const TensorInfo& info : program.outputs()) {
        II_ASSIGN_OR_RETURN(Tensor output, session.output(info.name));
        II_ASSIGN_OR_RETURN(Tensor f32, output.to(DType::kF32));
        outputs.push_back(std::move(f32));
    }
    return outputs;
}

}  // namespace

Result<std::vector<BackendCheck>> check_backends(const Context& context, std::span<const std::string> backends) {
    std::vector<std::string> names;
    const std::vector<std::string> available = context.backend_names();
    if (backends.empty()) {
        for (const std::string& name : available) {
            if (name != "ii-ref") {
                names.push_back(name);
            }
        }
    } else {
        for (const std::string& name : backends) {
            if (std::find(available.begin(), available.end(), name) == available.end()) {
                return errors::not_found("no backend '{}'", name);
            }
            names.push_back(name);
        }
    }
    std::vector<BackendCheck> checks;
    const std::vector<DeviceDescription> devices = context.devices();
    for (const std::string& name : names) {
        BackendCheck check;
        check.backend = name;
        for (const DeviceDescription& device : devices) {
            if (std::find(device.backends.begin(), device.backends.end(), name) != device.backends.end()) {
                check.device = device.id;
            }
        }
        checks.push_back(std::move(check));
    }

    std::uint64_t seed = 1;
    for (const CheckCase& test : kCases) {
        II_ASSIGN_OR_RETURN(Model model, Model::from_text(context, test.text));
        std::vector<Tensor> inputs;
        bool half = false;  // f16 inputs or outputs: a looser tolerance
        for (const TensorInfo& info : model.inputs()) {
            II_ASSIGN_OR_RETURN(Tensor input, make_input(info, seed++, test.low, test.high));
            inputs.push_back(std::move(input));
            half = half || info.dtype == DType::kF16;
        }
        for (const TensorInfo& info : model.outputs()) {
            half = half || info.dtype == DType::kF16;
        }
        II_ASSIGN_OR_RETURN(std::vector<Tensor> expected, run(model, inputs, "ii-ref"));
        const double tolerance = half ? 1e-2 : 1e-4;
        for (BackendCheck& check : checks) {
            BackendCheckCase result;
            result.name = std::string(test.name);
            Result<std::vector<Tensor>> actual = run(model, inputs, check.backend);
            if (!actual.ok()) {
                const bool unsupported = actual.status().code() == StatusCode::kUnsupported;
                result.state = unsupported ? BackendCheckState::kSkipped : BackendCheckState::kFailed;
                result.message = std::string(actual.status().message());
            } else {
                result.state = BackendCheckState::kPassed;
                for (std::size_t o = 0; o < expected.size(); ++o) {
                    const std::span<const float> a = (*actual)[o].values<float>();
                    const std::span<const float> e = expected[o].values<float>();
                    for (std::size_t i = 0; i < e.size(); ++i) {
                        const double error = std::abs(static_cast<double>(a[i]) - e[i]) / (1.0 + std::abs(e[i]));
                        if (!(error <= tolerance)) {  // also catches NaN
                            if (result.state == BackendCheckState::kPassed) {
                                result.message = std::format("element {}: {} instead of {}", i, a[i], e[i]);
                            }
                            result.state = BackendCheckState::kFailed;
                        }
                        if (std::isfinite(error)) {
                            result.max_error = std::max(result.max_error, error);
                        }
                    }
                }
            }
            switch (result.state) {
            case BackendCheckState::kPassed:
                ++check.passed;
                break;
            case BackendCheckState::kFailed:
                ++check.failed;
                break;
            case BackendCheckState::kSkipped:
                ++check.skipped;
                break;
            }
            check.cases.push_back(std::move(result));
        }
    }
    return checks;
}

}  // namespace ii
