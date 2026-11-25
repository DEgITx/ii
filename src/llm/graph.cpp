// SPDX-License-Identifier: MIT
#include "llm/graph.h"

#include <ii/builder.h>

#include <cmath>
#include <format>
#include <limits>
#include <map>
#include <numbers>
#include <vector>

namespace ii::llm::detail {

namespace {

constexpr std::int64_t kToEnd = std::numeric_limits<std::int64_t>::max();

class LlmBuilder {
public:
    LlmBuilder(GraphBuilder& b, const HParams& h, const WeightSource& w) : b_(b), h_(h), w_(w) {}

    /// A weight used as stored (matrices: f32, f16, bf16 or block-quantized).
    Result<Var> weight(const std::string& name) {
        II_ASSIGN_OR_RETURN(const Tensor t, w_.get(name));
        account(t);
        return b_.constant(t);
    }

    /// An optional weight (biases): an empty Var when absent.
    Result<Var> optional_weight(const std::string& name) {
        if (!w_.has(name)) {
            return Var();
        }
        return vector_f32(name, false);
    }

    /// A 1-D parameter (norm scales, biases) as f32; `plus_one` for Hugging Face gemma norms (scale 1 + w).
    Result<Var> vector_f32(const std::string& name, bool plus_one) {
        II_ASSIGN_OR_RETURN(const Tensor t, w_.get(name));
        account(t);
        Var v = b_.constant(t);
        if (t.dtype() != DType::kF32) {
            v = cast(v, DType::kF32);  // folded at compile time
        }
        if (plus_one) {
            v = v + 1.0;
        }
        return v;
    }

    Result<Var> norm(const Var& x, const std::string& name) {
        II_ASSIGN_OR_RETURN(const Var scale, vector_f32(name, h_.norm_plus_one));
        return rms_norm(x, scale, -1, h_.norm_epsilon);
    }

    WeightStats stats() const {
        WeightStats s = stats_;
        std::size_t best = 0;
        for (const auto& [dtype, bytes] : matrix_bytes_) {
            if (bytes > best) {
                best = bytes;
                s.main_dtype = dtype;
            }
        }
        return s;
    }

private:
    void account(const Tensor& t) {
        if (seen_.emplace(t.data(), true).second) {
            stats_.parameters += t.numel();
            stats_.bytes += t.nbytes();
            if (t.rank() >= 2) {
                matrix_bytes_[std::string(to_string(t.dtype()))] += t.nbytes();
            }
        }
    }

    GraphBuilder& b_;
    const HParams& h_;
    const WeightSource& w_;
    WeightStats stats_;
    std::map<std::string, std::size_t> matrix_bytes_;
    std::map<const void*, bool> seen_;
};

/// Inverse frequencies of the rotary embedding for `base`, with the model's scaling.
Result<std::vector<float>> inverse_frequencies(const HParams& h, double base, bool scaled,
                                               const WeightSource& weights) {
    const std::int64_t half = h.rope_dims / 2;
    std::vector<double> inv(static_cast<std::size_t>(half));
    for (std::int64_t i = 0; i < half; ++i) {
        inv[static_cast<std::size_t>(i)] =
            1.0 / std::pow(base, static_cast<double>(2 * i) / static_cast<double>(h.rope_dims));
    }
    if (scaled && h.rope_scaling == RopeScaling::kLinear) {
        for (double& f : inv) {
            f /= h.rope_factor;
        }
    } else if (scaled && h.rope_scaling == RopeScaling::kLlama3) {
        const auto old_context = static_cast<double>(h.rope_original_context);
        const double low_wavelength = old_context / h.rope_low_freq_factor;
        const double high_wavelength = old_context / h.rope_high_freq_factor;
        for (double& f : inv) {
            const double wavelength = 2.0 * std::numbers::pi / f;
            if (wavelength > low_wavelength) {
                f /= h.rope_factor;
            } else if (wavelength >= high_wavelength) {
                const double smooth = (old_context / wavelength - h.rope_low_freq_factor) /
                                      (h.rope_high_freq_factor - h.rope_low_freq_factor);
                f = (1.0 - smooth) * f / h.rope_factor + smooth * f;
            }
        }
    }
    // llama 3.x GGUF files carry the scaling as per-frequency divisors.
    if (scaled && weights.has("rope_freqs.weight")) {
        II_ASSIGN_OR_RETURN(const Tensor factors_tensor, weights.get("rope_freqs.weight"));
        if (factors_tensor.dtype() != DType::kF32 || factors_tensor.numel() != half) {
            return errors::data_corrupted("rope_freqs.weight must be f32[{}]", half);
        }
        const auto factors = factors_tensor.values<float>();
        for (std::int64_t i = 0; i < half; ++i) {
            inv[static_cast<std::size_t>(i)] /= static_cast<double>(factors[static_cast<std::size_t>(i)]);
        }
    }
    std::vector<float> result(inv.size());
    for (std::size_t i = 0; i < inv.size(); ++i) {
        result[i] = static_cast<float>(inv[i]);
    }
    return result;
}

}  // namespace

Result<Model> build_model(const Context& context, const HParams& h, const WeightSource& weights,
                          const GraphOptions& options, WeightStats* stats) {
    GraphBuilder b("llm");
    LlmBuilder w(b, h, weights);
    const std::int64_t hq = h.heads;
    const std::int64_t hkv = h.kv_heads;
    const std::int64_t d = h.head_size;
    const std::int64_t half = h.rope_dims / 2;
    const DType cache_dtype = options.cache_dtype;
    if (cache_dtype != DType::kF32 && cache_dtype != DType::kF16) {
        return errors::invalid_argument("the KV cache must be f32 or f16, got {}", to_string(cache_dtype));
    }

    const Var tokens = b.input("tokens", DType::kI32, {1, "S"});
    const Var pos = b.input("pos", DType::kI64, {});
    const Var positions = iota(b, {"S"}, DType::kI64) + pos;
    const Var kv_len = reduce_max(positions) + 1.0;

    // Token embeddings.
    II_ASSIGN_OR_RETURN(const Var table, w.weight("token_embd.weight"));
    Var x = gather(table, tokens, 0);
    if (is_block_quantized(table.dtype())) {
        x = dequantize(x);
    } else if (table.dtype() != DType::kF32) {
        x = cast(x, DType::kF32);
    }
    if (h.embedding_scale != 1.0) {
        x = x * h.embedding_scale;
    }

    // Rotary tables [S, 1, R/2], shared by the layers with the same base.
    auto rope_tables = [&](double base, bool scaled) -> Result<std::pair<Var, Var>> {
        II_ASSIGN_OR_RETURN(const std::vector<float> inv, inverse_frequencies(h, base, scaled, weights));
        II_ASSIGN_OR_RETURN(const Tensor inv_tensor, Tensor::from(std::span<const float>(inv), Shape{1, half}));
        const Var angles = reshape(cast(positions, DType::kF32), {"S", 1}) * b.constant(inv_tensor);
        return std::make_pair(reshape(cos(angles), {"S", 1, half}), reshape(sin(angles), {"S", 1, half}));
    };
    II_ASSIGN_OR_RETURN(const auto global_rope, rope_tables(h.rope_base, true));
    std::pair<Var, Var> local_rope = global_rope;
    if (h.rope_base_local > 0) {
        II_ASSIGN_OR_RETURN(local_rope, rope_tables(h.rope_base_local, false));
    }
    const double scale = h.attention_scale > 0 ? h.attention_scale : 1.0 / std::sqrt(static_cast<double>(d));

    for (std::int64_t l = 0; l < h.layers; ++l) {
        const std::string p = std::format("blk.{}.", l);
        const bool sliding = h.sliding_layers[static_cast<std::size_t>(l)];
        II_ASSIGN_OR_RETURN(Var hn, w.norm(x, p + "attn_norm.weight"));
        Var q;
        Var k;
        Var v;
        if (h.fused_qkv) {
            II_ASSIGN_OR_RETURN(const Var wqkv, w.weight(p + "attn_qkv.weight"));
            const std::vector<Var> parts = split(linear(hn, wqkv), {hq * d, hkv * d, hkv * d}, -1);
            q = parts[0];
            k = parts[1];
            v = parts[2];
        } else {
            II_ASSIGN_OR_RETURN(const Var wq, w.weight(p + "attn_q.weight"));
            II_ASSIGN_OR_RETURN(const Var wk, w.weight(p + "attn_k.weight"));
            II_ASSIGN_OR_RETURN(const Var wv, w.weight(p + "attn_v.weight"));
            II_ASSIGN_OR_RETURN(const Var bq, w.optional_weight(p + "attn_q.bias"));
            II_ASSIGN_OR_RETURN(const Var bk, w.optional_weight(p + "attn_k.bias"));
            II_ASSIGN_OR_RETURN(const Var bv, w.optional_weight(p + "attn_v.bias"));
            q = linear(hn, wq, bq);
            k = linear(hn, wk, bk);
            v = linear(hn, wv, bv);
        }
        q = reshape(q, {1, "S", hq, d});
        k = reshape(k, {1, "S", hkv, d});
        v = reshape(v, {1, "S", hkv, d});
        if (h.qk_norm) {
            II_ASSIGN_OR_RETURN(q, w.norm(q, p + "attn_q_norm.weight"));
            II_ASSIGN_OR_RETURN(k, w.norm(k, p + "attn_k_norm.weight"));
        }
        const auto& [cos_table, sin_table] = sliding && h.rope_base_local > 0 ? local_rope : global_rope;
        q = permute(rope(q, cos_table, sin_table, h.rope_interleaved), {0, 2, 1, 3});
        k = permute(rope(k, cos_table, sin_table, h.rope_interleaved), {0, 2, 1, 3});
        v = permute(v, {0, 2, 1, 3});
        if (cache_dtype != DType::kF32) {
            k = cast(k, cache_dtype);
            v = cast(v, cache_dtype);
        }
        const Var k_cache = b.state(std::format("cache.{}.k", l), cache_dtype, {1, hkv, options.context, d});
        const Var v_cache = b.state(std::format("cache.{}.v", l), cache_dtype, {1, hkv, options.context, d});
        const Var k_all = slice_update(k_cache, k, pos, 2);
        const Var v_all = slice_update(v_cache, v, pos, 2);
        b.update_state(k_cache, k_all);
        b.update_state(v_cache, v_all);
        Var a = sdpa(q, k_all, v_all, {}, true, scale, kv_len, sliding ? h.sliding_window : 0, h.attention_softcap);
        a = reshape(permute(a, {0, 2, 1, 3}), {1, "S", hq * d});
        II_ASSIGN_OR_RETURN(const Var wo, w.weight(p + "attn_output.weight"));
        Var o = linear(a, wo);
        if (h.post_norms) {
            II_ASSIGN_OR_RETURN(o, w.norm(o, p + "post_attention_norm.weight"));
        }
        x = x + o;

        II_ASSIGN_OR_RETURN(hn, w.norm(x, p + "ffn_norm.weight"));
        Var gate;
        Var up;
        if (h.fused_gate_up) {
            II_ASSIGN_OR_RETURN(const Var wgu, w.weight(p + "ffn_up.weight"));
            const std::vector<Var> parts = split(linear(hn, wgu), {h.feed_forward, h.feed_forward}, -1);
            gate = parts[0];
            up = parts[1];
        } else {
            II_ASSIGN_OR_RETURN(const Var wg, w.weight(p + "ffn_gate.weight"));
            II_ASSIGN_OR_RETURN(const Var wu, w.weight(p + "ffn_up.weight"));
            gate = linear(hn, wg);
            up = linear(hn, wu);
        }
        const Var activated = h.activation == Activation::kSilu ? silu(gate) : gelu(gate, true);
        II_ASSIGN_OR_RETURN(const Var wd, w.weight(p + "ffn_down.weight"));
        Var f = linear(activated * up, wd);
        if (h.post_norms) {
            II_ASSIGN_OR_RETURN(f, w.norm(f, p + "post_ffw_norm.weight"));
        }
        x = x + f;
    }

    II_ASSIGN_OR_RETURN(x, w.norm(x, "output_norm.weight"));
    if (!options.all_logits) {
        x = slice(x, {1}, {-1}, {kToEnd});
    }
    const std::string head = weights.has("output.weight") ? "output.weight" : "token_embd.weight";
    II_ASSIGN_OR_RETURN(const Var output_weight, w.weight(head));
    Var logits = linear(x, output_weight);
    if (h.final_softcap > 0) {
        logits = tanh(logits / h.final_softcap) * h.final_softcap;
    }
    b.output("logits", logits);
    if (stats != nullptr) {
        *stats = w.stats();
    }
    return Model::from_builder(context, std::move(b));
}

}  // namespace ii::llm::detail
