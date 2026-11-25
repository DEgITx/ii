// SPDX-License-Identifier: MIT
/// @file
/// Hyper-parameters of the supported transformer architectures, read from GGUF metadata or a Hugging Face
/// config.json, and the source of the weights (GGUF names; Hugging Face names are mapped to them).
#pragma once

#include <ii/gguf.h>
#include <ii/status.h>
#include <ii/tensor.h>

#include "core/json.h"
#include "formats/safetensors.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ii::llm::detail {

enum class Activation { kSilu, kGeluTanh };

enum class RopeScaling { kNone, kLinear, kLlama3 };

struct HParams {
    std::string architecture;  ///< normalized: llama, qwen2, qwen3, gemma, gemma2, gemma3, phi3
    std::string name;
    std::int64_t vocab_size = 0;
    std::int64_t embedding = 0;
    std::int64_t layers = 0;
    std::int64_t heads = 0;
    std::int64_t kv_heads = 0;
    std::int64_t head_size = 0;
    std::int64_t feed_forward = 0;
    std::int64_t trained_context = 0;
    double norm_epsilon = 1e-5;

    // Rotary embedding
    double rope_base = 10000.0;
    double rope_base_local = 0;     ///< gemma3 sliding-window layers (0: rope_base)
    std::int64_t rope_dims = 0;     ///< rotated features per head (<= head_size)
    bool rope_interleaved = false;  ///< pairs (2i, 2i+1): GGUF llama (its q/k weights are permuted for it)
    RopeScaling rope_scaling = RopeScaling::kNone;
    double rope_factor = 1.0;
    double rope_low_freq_factor = 1.0;
    double rope_high_freq_factor = 4.0;
    std::int64_t rope_original_context = 0;

    // Layout and details
    Activation activation = Activation::kSilu;
    bool qkv_bias = false;           ///< qwen2
    bool qk_norm = false;            ///< qwen3, gemma3: rms_norm of q and k per head
    bool post_norms = false;         ///< gemma2, gemma3: norms after attention and feed-forward
    bool fused_qkv = false;          ///< phi3
    bool fused_gate_up = false;      ///< phi3
    bool norm_plus_one = false;      ///< Hugging Face gemma norms scale by (1 + w)
    double embedding_scale = 1.0;    ///< gemma: sqrt(embedding)
    double attention_scale = 0.0;    ///< 0: 1 / sqrt(head_size)
    double attention_softcap = 0.0;  ///< gemma2
    double final_softcap = 0.0;      ///< gemma2
    std::int64_t sliding_window = 0;
    std::vector<bool> sliding_layers;  ///< per layer: uses the sliding window
};

/// Reads the hyper-parameters of a GGUF model.
Result<HParams> hparams_from_gguf(const gguf::File& file);
/// Reads the hyper-parameters of a Hugging Face model (config.json).
Result<HParams> hparams_from_config(const Json& config);

/// Weights by their GGUF names ("blk.3.attn_q.weight").
class WeightSource {
public:
    virtual ~WeightSource() = default;
    virtual bool has(const std::string& name) const = 0;
    virtual Result<Tensor> get(const std::string& name) const = 0;
};

std::unique_ptr<WeightSource> gguf_weights(gguf::File file);
/// Hugging Face weights; names are translated (model.layers.3.self_attn.q_proj.weight -> blk.3.attn_q.weight).
std::unique_ptr<WeightSource> safetensors_weights(formats::SafetensorsModel model, const HParams& hparams);

}  // namespace ii::llm::detail
