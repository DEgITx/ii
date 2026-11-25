// SPDX-License-Identifier: MIT
// Builds a small multi-layer perceptron with the C++ builder API, compiles it and classifies a batch of inputs.
//
//   logits = linear(relu(linear(x, w1, b1)), w2, b2)      x: [N, 4]  ->  logits: [N, 3]
//
// The batch size N is symbolic: the same compiled program runs any batch size.
#include <ii/ii.h>

#include <cstdio>
#include <random>
#include <vector>

namespace {

/// A tensor of `shape` with reproducible pseudo-random values in [-1, 1).
ii::Tensor random_weights(const ii::Shape& shape, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(shape.numel()));
    for (float& v : values) {
        v = uniform(rng);
    }
    return ii::Tensor::from(std::span<const float>(values), shape).value();
}

int fail(const ii::Status& status) {
    std::fprintf(stderr, "error: %s\n", status.to_string().c_str());
    return 1;
}

}  // namespace

int main() {
    auto context = ii::Context::create();
    if (!context.ok()) {
        return fail(context.status());
    }

    // --- Build the model ---
    ii::GraphBuilder b("mlp");
    const ii::Var x = b.input("x", ii::DType::kF32, {"N", 4});
    const ii::Var w1 = b.constant(random_weights({16, 4}, 1));  // [out, in], as in PyTorch
    const ii::Var b1 = b.constant(random_weights({16}, 2));
    const ii::Var w2 = b.constant(random_weights({3, 16}, 3));
    const ii::Var b2 = b.constant(random_weights({3}, 4));
    const ii::Var logits = ii::linear(ii::relu(ii::linear(x, w1, b1)), w2, b2);
    b.output("probabilities", ii::softmax(logits));
    b.output("class", ii::argmax(logits, -1));

    auto model = ii::Model::from_builder(*context, std::move(b));
    if (!model.ok()) {
        return fail(model.status());
    }
    auto program = model->compile();
    if (!program.ok()) {
        return fail(program.status());
    }
    std::printf("Execution plan:\n%s\n", program->describe().c_str());

    // --- Run a batch of 5 ---
    auto session = program->new_session();
    if (!session.ok()) {
        return fail(session.status());
    }
    const ii::Tensor batch = random_weights({5, 4}, 42);
    if (ii::Status status = session->set_input("x", batch); !status.ok()) {
        return fail(status);
    }
    if (ii::Status status = session->run(); !status.ok()) {
        return fail(status);
    }
    const ii::Tensor probabilities = session->output("probabilities").value();
    const ii::Tensor classes = session->output("class").value();
    const auto p = probabilities.values<float>();
    const auto c = classes.values<std::int64_t>();
    for (std::int64_t i = 0; i < batch.shape()[0]; ++i) {
        std::printf("sample %lld: class %lld  p = [%.3f, %.3f, %.3f]\n", static_cast<long long>(i),
                    static_cast<long long>(c[static_cast<std::size_t>(i)]), p[static_cast<std::size_t>(i * 3)],
                    p[static_cast<std::size_t>(i * 3 + 1)], p[static_cast<std::size_t>(i * 3 + 2)]);
    }
    return 0;
}
