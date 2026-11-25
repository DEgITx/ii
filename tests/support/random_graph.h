// SPDX-License-Identifier: MIT
/// @file
/// Random model generator for property tests: graph passes and backends must not change results.
#pragma once

#include <ii/model.h>

#include <cstdint>
#include <vector>

namespace ii_test {

struct RandomModel {
    ii::Model model;
    std::vector<ii::Tensor> inputs;  ///< matching inputs, in declaration order
};

/// Builds a random f32 model of about `nodes` operations: elementwise unary and binary operations with
/// broadcasting, matmul, reductions, normalizations, softmax, data movement (permute, reshape, slice + concat)
/// and composites, over two inputs and several constants (so that constant folding has work to do). Values stay
/// bounded, so results are comparable with a relative tolerance.
RandomModel random_model(const ii::Context& context, std::uint32_t seed, int nodes = 24);

}  // namespace ii_test
