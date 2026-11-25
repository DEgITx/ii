// SPDX-License-Identifier: MIT
#include "support/random_graph.h"

#include <ii/builder.h>

#include "support/test_util.h"

#include <random>

namespace ii_test {

namespace {

using ii::Dims;
using ii::DType;
using ii::Shape;
using ii::Var;

Shape static_shape(const Var& v) {
    Shape shape;
    for (const ii::Dim& d : v.shape()) {
        shape.push_back(d.value());
    }
    return shape;
}

bool broadcastable(const Shape& a, const Shape& b) {
    const int rank = std::max(a.rank(), b.rank());
    for (int i = 0; i < rank; ++i) {
        const std::int64_t da = i < rank - a.rank() ? 1 : a[i - (rank - a.rank())];
        const std::int64_t db = i < rank - b.rank() ? 1 : b[i - (rank - b.rank())];
        if (da != db && da != 1 && db != 1) {
            return false;
        }
    }
    return true;
}

}  // namespace

RandomModel random_model(const ii::Context& context, std::uint32_t seed, int nodes) {
    std::mt19937 rng(seed);
    auto pick = [&](std::size_t n) { return static_cast<std::size_t>(rng() % n); };
    const std::int64_t a = 2 + static_cast<std::int64_t>(rng() % 3);
    const std::int64_t b = 2 + static_cast<std::int64_t>(rng() % 4);
    const Shape base{a, b};

    ii::GraphBuilder builder("random");
    std::vector<Var> pool;
    RandomModel result{ii::Model::from_text(context, "graph {}").value(), {}};
    pool.push_back(builder.input("x", DType::kF32, {a, b}));
    pool.push_back(builder.input("y", DType::kF32, {b}));
    result.inputs.push_back(random_tensor(DType::kF32, base, seed * 7 + 1));
    result.inputs.push_back(random_tensor(DType::kF32, Shape{b}, seed * 7 + 2));
    const Var c0 = builder.constant(random_tensor(DType::kF32, base, seed * 7 + 3));
    const Var c1 = builder.constant(random_tensor(DType::kF32, Shape{1, b}, seed * 7 + 4));
    const Var c2 = builder.constant(random_tensor(DType::kF32, Shape{a, 1}, seed * 7 + 5, 0.5, 1.5));
    const Var w = builder.constant(random_tensor(DType::kF32, Shape{b, b}, seed * 7 + 6, -0.5, 0.5));
    // A purely constant subgraph that constant folding can evaluate.
    pool.push_back(ii::tanh(c0 * c1 + c2));
    pool.push_back(c1);
    pool.push_back(c2);

    for (int i = 0; i < nodes; ++i) {
        const Var x = pool[pick(pool.size())];
        const Shape xs = static_shape(x);
        Var y;
        switch (pick(12)) {
        case 0:
            y = ii::tanh(x);
            break;
        case 1:
            y = ii::sigmoid(x);
            break;
        case 2:
            y = pick(2) == 0 ? ii::gelu(x) : ii::silu(x);
            break;
        case 3:
            y = pick(2) == 0 ? ii::relu(x) : ii::abs(ii::neg(x));
            break;
        case 4:
        case 5: {
            // A binary operation with a broadcast-compatible partner.
            std::vector<Var> partners;
            for (const Var& candidate : pool) {
                if (broadcastable(xs, static_shape(candidate))) {
                    partners.push_back(candidate);
                }
            }
            const Var z = partners[pick(partners.size())];
            switch (pick(4)) {
            case 0:
                y = x + z;
                break;
            case 1:
                y = x - z;
                break;
            case 2:
                y = ii::tanh(x * z);
                break;
            default:
                y = pick(2) == 0 ? ii::maximum(x, z) : ii::minimum(x, z);
            }
            break;
        }
        case 6:
            if (xs.rank() == 2 && xs[1] == b) {
                y = ii::tanh(ii::matmul(x, w));
            } else {
                y = ii::reduce_mean(x);
            }
            break;
        case 7:
            if (xs.rank() == 2) {
                y = pick(2) == 0 ? ii::reduce_sum(x, {1}, true) : ii::reduce_max(x, {0}, true);
            } else {
                y = ii::softmax(x);
            }
            break;
        case 8:
            y = xs.rank() >= 1 ? ii::softmax(x) : ii::sigmoid(x);
            break;
        case 9:
            if (xs.rank() == 0) {
                y = ii::sigmoid(x);
            } else {
                y = xs[xs.rank() - 1] > 1 ? ii::layer_norm(x) : ii::rms_norm(x);
            }
            break;
        case 10:
            if (xs.rank() == 2) {
                // permute and reshape back and forth
                y = ii::permute(ii::reshape(ii::permute(x, {1, 0}), {xs[1], xs[0]}), {1, 0});
            } else {
                y = ii::neg(x);
            }
            break;
        default:
            if (xs.rank() >= 1 && xs[xs.rank() - 1] >= 2) {
                const int last = xs.rank() - 1;
                const std::int64_t size = xs[last];
                const std::int64_t split = 1 + static_cast<std::int64_t>(pick(static_cast<std::size_t>(size - 1)));
                const Var left = ii::slice(x, {last}, {0}, {split});
                const Var right = ii::slice(x, {last}, {split}, {size});
                y = ii::concat({ii::tanh(right), left}, last);
            } else {
                y = ii::sigmoid(x);
            }
        }
        if (!y.valid()) {
            break;  // an error is recorded in the builder and reported below
        }
        pool.push_back(y);
    }
    builder.output("out0", pool.back());
    builder.output("out1", pool[pool.size() / 2]);
    builder.output("out2", ii::reduce_sum(pool[pool.size() - 2] * pool[pool.size() / 3]));
    builder.output("folded", pool[2]);  // depends on constants only
    result.model = ii::Model::from_builder(context, std::move(builder)).value();
    return result;
}

}  // namespace ii_test
