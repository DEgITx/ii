// SPDX-License-Identifier: MIT
#include "llm/hparams.h"

#include <cmath>
#include <format>
#include <map>

namespace ii::llm::detail {

namespace {

const char* const kArchitectures[] = {"llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "phi3"};

bool supported(const std::string& architecture) {
    for (const char* name : kArchitectures) {
        if (architecture == name) {
            return true;
        }
    }
    return false;
}

/// Architecture-specific defaults shared by both formats.
void apply_architecture(HParams& h) {
    if (h.architecture == "qwen2") {
        h.qkv_bias = true;
    } else if (h.architecture == "qwen3") {
        h.qk_norm = true;
    } else if (h.architecture == "gemma" || h.architecture == "gemma2" || h.architecture == "gemma3") {
        h.activation = Activation::kGeluTanh;
        h.embedding_scale = std::sqrt(static_cast<double>(h.embedding));
        h.post_norms = h.architecture != "gemma";
        h.qk_norm = h.architecture == "gemma3";
    } else if (h.architecture == "phi3") {
        h.fused_qkv = true;
        h.fused_gate_up = true;
    }
    if (h.sliding_layers.empty()) {
        h.sliding_layers.assign(static_cast<std::size_t>(h.layers), false);
        if (h.sliding_window > 0) {
            for (std::int64_t l = 0; l < h.layers; ++l) {
                bool sliding = true;  // mistral-style: every layer
                if (h.architecture == "gemma2") {
                    sliding = l % 2 == 0;
                } else if (h.architecture == "gemma3") {
                    sliding = (l + 1) % 6 != 0;
                }
                h.sliding_layers[static_cast<std::size_t>(l)] = sliding;
            }
        }
    }
}

Status validate(const HParams& h) {
    if (!supported(h.architecture)) {
        return errors::unsupported("architecture '{}' is not supported (supported: llama, qwen2, qwen3, gemma, "
                                   "gemma2, gemma3, phi3)",
                                   h.architecture);
    }
    if (h.vocab_size <= 0 || h.embedding <= 0 || h.layers <= 0 || h.heads <= 0 || h.kv_heads <= 0 || h.head_size <= 0 ||
        h.feed_forward <= 0 || h.heads % h.kv_heads != 0 || h.layers > 1024 || h.heads > 4096 || h.head_size > 4096 ||
        h.rope_dims <= 0 || h.rope_dims > h.head_size || h.rope_dims % 2 != 0) {
        return errors::data_corrupted(
            "invalid hyper-parameters: vocab {}, embedding {}, layers {}, heads {}/{}, head size {}, rope dims {}, "
            "feed-forward {}",
            h.vocab_size, h.embedding, h.layers, h.heads, h.kv_heads, h.head_size, h.rope_dims, h.feed_forward);
    }
    if (!(h.rope_base > 0) || !(h.norm_epsilon > 0) || !(h.rope_factor > 0)) {
        return errors::data_corrupted("invalid rope base, scaling factor or norm epsilon");
    }
    return ok_status();
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// GGUF

Result<HParams> hparams_from_gguf(const gguf::File& file) {
    HParams h;
    h.architecture = file.get_string("general.architecture").value_or("");
    if (h.architecture.empty()) {
        return errors::data_corrupted("GGUF model without general.architecture");
    }
    h.name = file.get_string("general.name").value_or("");
    const std::string& a = h.architecture;
    auto get_int = [&](const std::string& key) { return file.get_int(a + "." + key); };
    auto get_float = [&](const std::string& key) { return file.get_float(a + "." + key); };
    auto require = [&](const std::string& key) -> Result<std::int64_t> {
        if (const std::optional<std::int64_t> v = get_int(key)) {
            return *v;
        }
        return errors::data_corrupted("GGUF model without {}.{}", a, key);
    };
    if (!supported(a)) {
        return errors::unsupported("architecture '{}' is not supported (supported: llama, qwen2, qwen3, gemma, "
                                   "gemma2, gemma3, phi3)",
                                   a);
    }
    II_ASSIGN_OR_RETURN(h.embedding, require("embedding_length"));
    II_ASSIGN_OR_RETURN(h.layers, require("block_count"));
    II_ASSIGN_OR_RETURN(h.heads, require("attention.head_count"));
    II_ASSIGN_OR_RETURN(h.feed_forward, require("feed_forward_length"));
    h.kv_heads = get_int("attention.head_count_kv").value_or(h.heads);
    h.head_size = get_int("attention.key_length").value_or(h.heads > 0 ? h.embedding / h.heads : 0);
    h.trained_context = get_int("context_length").value_or(2048);
    h.norm_epsilon = get_float("attention.layer_norm_rms_epsilon").value_or(1e-5);
    h.rope_base = get_float("rope.freq_base").value_or(10000.0);
    h.rope_dims = get_int("rope.dimension_count").value_or(h.head_size);
    h.sliding_window = get_int("attention.sliding_window").value_or(0);
    h.attention_softcap = get_float("attn_logit_softcapping").value_or(0.0);
    h.final_softcap = get_float("final_logit_softcapping").value_or(0.0);
    if (const std::optional<std::int64_t> vocab = get_int("vocab_size")) {
        h.vocab_size = *vocab;
    } else if (const gguf::Value* tokens = file.find("tokenizer.ggml.tokens")) {
        h.vocab_size = static_cast<std::int64_t>(tokens->array_size());
    }
    // llama.cpp: architectures converted from Hugging Face with permuted q / k weights use interleaved pairs.
    h.rope_interleaved = a == "llama";
    const std::string scaling = file.get_string(a + ".rope.scaling.type").value_or("none");
    if (scaling == "linear") {
        h.rope_scaling = RopeScaling::kLinear;
        h.rope_factor = get_float("rope.scaling.factor").value_or(1.0);
    } else if (scaling != "none" && scaling != "") {
        // llama 3.x stores its frequency adjustments as the tensor rope_freqs.weight (handled by the graph);
        // yarn / longrope are not supported yet: use the base frequencies.
        h.rope_scaling = RopeScaling::kNone;
    }
    if (a == "gemma3") {
        h.rope_base_local = 10000.0;
        if (!get_int("attention.sliding_window")) {
            h.sliding_window = 512;
        }
    }
    if (a == "gemma2" && h.sliding_window == 0) {
        h.sliding_window = 4096;
    }
    apply_architecture(h);
    // GGUF gemma norms already include the +1 (added by the converter).
    h.norm_plus_one = false;
    II_TRY(validate(h));
    return h;
}

// ---------------------------------------------------------------------------------------------------------------
// Hugging Face config.json

Result<HParams> hparams_from_config(const Json& config) {
    // gemma3 multimodal checkpoints keep the language model's settings in text_config.
    const Json* text = config.find("text_config");
    const Json& c = text != nullptr && text->is_object() ? *text : config;
    auto get_int = [&](const char* key) -> std::optional<std::int64_t> {
        const Json* v = c.find(key);
        return v != nullptr ? v->as_int() : std::nullopt;
    };
    auto get_float = [&](const char* key) -> std::optional<double> {
        const Json* v = c.find(key);
        return v != nullptr && v->is_number() ? std::optional<double>(v->as_double()) : std::nullopt;
    };
    auto get_string = [&](const char* key) -> std::string {
        const Json* v = c.find(key);
        return v != nullptr && v->is_string() ? v->as_string() : std::string();
    };
    HParams h;
    std::string type = get_string("model_type");
    static const std::map<std::string, std::string> kTypes = {
        {"llama", "llama"},   {"mistral", "llama"},      {"qwen2", "qwen2"},   {"qwen3", "qwen3"}, {"gemma", "gemma"},
        {"gemma2", "gemma2"}, {"gemma3_text", "gemma3"}, {"gemma3", "gemma3"}, {"phi3", "phi3"},
    };
    const auto it = kTypes.find(type);
    if (it == kTypes.end()) {
        return errors::unsupported("model_type '{}' is not supported", type);
    }
    h.architecture = it->second;
    h.vocab_size = get_int("vocab_size").value_or(0);
    h.embedding = get_int("hidden_size").value_or(0);
    h.layers = get_int("num_hidden_layers").value_or(0);
    h.heads = get_int("num_attention_heads").value_or(0);
    h.kv_heads = get_int("num_key_value_heads").value_or(h.heads);
    h.head_size = get_int("head_dim").value_or(h.heads > 0 ? h.embedding / h.heads : 0);
    h.feed_forward = get_int("intermediate_size").value_or(0);
    h.trained_context = get_int("max_position_embeddings").value_or(2048);
    h.norm_epsilon = get_float("rms_norm_eps").value_or(1e-6);
    h.rope_base = get_float("rope_theta").value_or(10000.0);
    // transformers >= 5 writes rope_parameters: one dict, or one per layer type (gemma3).
    const Json* rope = c.find("rope_parameters");
    const Json* rope_global = rope;
    if (rope != nullptr && rope->is_object()) {
        if (const Json* full = rope->find("full_attention"); full != nullptr && full->is_object()) {
            rope_global = full;
            if (const Json* local = rope->find("sliding_attention"); local != nullptr && local->find("rope_theta")) {
                h.rope_base_local = local->find("rope_theta")->as_double();
            }
        }
        if (const Json* theta = rope_global->find("rope_theta"); theta != nullptr && theta->is_number()) {
            h.rope_base = theta->as_double();
        }
    } else {
        rope_global = nullptr;
    }
    double partial = get_float("partial_rotary_factor").value_or(1.0);
    if (rope_global != nullptr) {
        if (const Json* p = rope_global->find("partial_rotary_factor"); p != nullptr && p->is_number()) {
            partial = p->as_double();
        }
    }
    h.rope_dims = static_cast<std::int64_t>(static_cast<double>(h.head_size) * partial);
    h.rope_dims -= h.rope_dims % 2;
    h.rope_interleaved = false;  // Hugging Face weights rotate halves
    if (const Json* sliding = c.find("sliding_window"); sliding != nullptr && sliding->as_int()) {
        const Json* use = c.find("use_sliding_window");
        if (use == nullptr || use->as_bool() || h.architecture != "qwen2") {
            h.sliding_window = *sliding->as_int();
        }
    }
    h.attention_softcap = get_float("attn_logit_softcapping").value_or(0.0);
    h.final_softcap = get_float("final_logit_softcapping").value_or(0.0);
    if (h.architecture == "gemma2" || h.architecture == "gemma3") {
        if (const std::optional<double> scalar = get_float("query_pre_attn_scalar")) {
            h.attention_scale = 1.0 / std::sqrt(*scalar);
        }
    }
    if (h.architecture == "gemma3" && h.rope_base_local == 0) {
        h.rope_base_local = get_float("rope_local_base_freq").value_or(10000.0);
    }
    const Json* scaling = c.find("rope_scaling");
    if (scaling == nullptr || !scaling->is_object()) {
        scaling = rope_global;
    }
    if (scaling != nullptr && scaling->is_object()) {
        const Json* kind = scaling->find("rope_type");
        if (kind == nullptr) {
            kind = scaling->find("type");
        }
        const std::string name = kind != nullptr && kind->is_string() ? kind->as_string() : std::string();
        auto number = [&](const char* key, double fallback) {
            const Json* v = scaling->find(key);
            return v != nullptr && v->is_number() ? v->as_double() : fallback;
        };
        if (name == "linear") {
            h.rope_scaling = RopeScaling::kLinear;
            h.rope_factor = number("factor", 1.0);
        } else if (name == "llama3") {
            h.rope_scaling = RopeScaling::kLlama3;
            h.rope_factor = number("factor", 8.0);
            h.rope_low_freq_factor = number("low_freq_factor", 1.0);
            h.rope_high_freq_factor = number("high_freq_factor", 4.0);
            h.rope_original_context = static_cast<std::int64_t>(number("original_max_position_embeddings", 8192));
        } else if (name != "default" && !name.empty()) {
            return errors::unsupported("rope scaling '{}' is not supported", name);
        }
    }
    if (const Json* types = c.find("layer_types");
        types != nullptr && types->is_array() && static_cast<std::int64_t>(types->size()) == h.layers) {
        for (const Json& layer_type : types->as_array()) {
            h.sliding_layers.push_back(layer_type.as_string() == "sliding_attention" && h.sliding_window > 0);
        }
    }
    apply_architecture(h);
    h.norm_plus_one = h.architecture == "gemma" || h.architecture == "gemma2" || h.architecture == "gemma3";
    II_TRY(validate(h));
    return h;
}

// ---------------------------------------------------------------------------------------------------------------
// Weight sources

namespace {

class GgufWeights final : public WeightSource {
public:
    explicit GgufWeights(gguf::File file) : file_(std::move(file)) {}
    bool has(const std::string& name) const override { return file_.find_tensor(name) != nullptr; }
    Result<Tensor> get(const std::string& name) const override { return file_.tensor(name); }

private:
    gguf::File file_;
};

class SafetensorsWeights final : public WeightSource {
public:
    SafetensorsWeights(formats::SafetensorsModel model, const HParams& hparams)
        : model_(std::move(model)), gemma_norms_(hparams.post_norms) {}

    bool has(const std::string& name) const override { return model_.contains(translate(name)); }
    Result<Tensor> get(const std::string& name) const override { return model_.tensor(translate(name)); }

private:
    /// GGUF name -> Hugging Face name.
    std::string translate(const std::string& name) const {
        static const std::map<std::string, std::string> kGlobal = {
            {"token_embd.weight", "model.embed_tokens.weight"},
            {"output_norm.weight", "model.norm.weight"},
            {"output.weight", "lm_head.weight"},
        };
        if (const auto it = kGlobal.find(name); it != kGlobal.end()) {
            return it->second;
        }
        if (name.rfind("blk.", 0) != 0) {
            return name;
        }
        const std::size_t dot = name.find('.', 4);
        const std::string layer = name.substr(4, dot - 4);
        const std::string rest = name.substr(dot + 1);
        const std::size_t suffix_dot = rest.rfind('.');
        const std::string base = rest.substr(0, suffix_dot);
        const std::string suffix = rest.substr(suffix_dot);  // ".weight" / ".bias"
        static const std::map<std::string, std::string> kLayer = {
            {"attn_norm", "input_layernorm"},
            {"attn_q", "self_attn.q_proj"},
            {"attn_k", "self_attn.k_proj"},
            {"attn_v", "self_attn.v_proj"},
            {"attn_qkv", "self_attn.qkv_proj"},
            {"attn_output", "self_attn.o_proj"},
            {"attn_q_norm", "self_attn.q_norm"},
            {"attn_k_norm", "self_attn.k_norm"},
            {"ffn_gate", "mlp.gate_proj"},
            {"ffn_down", "mlp.down_proj"},
            {"post_attention_norm", "post_attention_layernorm"},
            {"post_ffw_norm", "post_feedforward_layernorm"},
        };
        std::string hf;
        if (base == "ffn_norm") {
            hf = gemma_norms_ ? "pre_feedforward_layernorm" : "post_attention_layernorm";
        } else if (base == "ffn_up") {
            hf = model_.contains("model.layers." + layer + ".mlp.gate_up_proj.weight") ? "mlp.gate_up_proj"
                                                                                       : "mlp.up_proj";
        } else if (const auto it = kLayer.find(base); it != kLayer.end()) {
            hf = it->second;
        } else {
            return name;
        }
        return "model.layers." + layer + "." + hf + suffix;
    }

    formats::SafetensorsModel model_;
    bool gemma_norms_;
};

}  // namespace

std::unique_ptr<WeightSource> gguf_weights(gguf::File file) {
    return std::make_unique<GgufWeights>(std::move(file));
}

std::unique_ptr<WeightSource> safetensors_weights(formats::SafetensorsModel model, const HParams& hparams) {
    return std::make_unique<SafetensorsWeights>(std::move(model), hparams);
}

}  // namespace ii::llm::detail
