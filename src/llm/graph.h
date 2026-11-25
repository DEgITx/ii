// SPDX-License-Identifier: MIT
/// @file
/// Builds the ii graph of a transformer decoder from its hyper-parameters and weights.
///
/// Inputs: "tokens" i32[1, S] and "pos" i64[] (the position of the first token, i.e. tokens already in the cache).
/// Output: "logits" f32[1, S, vocab] (all_logits) or f32[1, 1, vocab] (the last position).
/// States: "cache.<layer>.k" / "cache.<layer>.v" [1, kv_heads, context, head_size], written in place by
/// slice_update at `pos` and read by sdpa with kv_len = pos + S.
#pragma once

#include <ii/model.h>

#include "llm/hparams.h"

namespace ii::llm::detail {

struct GraphOptions {
    std::int64_t context = 0;
    DType cache_dtype = DType::kF32;
    bool all_logits = false;
};

/// Sizes of the weights used by the graph.
struct WeightStats {
    std::int64_t parameters = 0;
    std::size_t bytes = 0;
    std::string main_dtype;  ///< the dtype holding most bytes of matrix weights
};

Result<Model> build_model(const Context& context, const HParams& hparams, const WeightSource& weights,
                          const GraphOptions& options, WeightStats* stats);

}  // namespace ii::llm::detail
