#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generates tiny random language models and their reference logits for the LLM tests (tests/data/llm).

For every architecture (llama with llama-3 rope scaling, qwen2, qwen3, gemma2, gemma3, phi3) the script creates a
randomly initialized Hugging Face model with a small byte-level BPE tokenizer, and writes
    tests/data/llm/<arch>/            config.json, model.safetensors (f16), tokenizer.json, tokenizer_config.json
    tests/data/llm/<arch>.gguf        the same weights in GGUF (as llama.cpp's converter lays them out)
    tests/data/llm/<arch>/expected.npz  tokens and the float32 logits of transformers for every position
plus llama-q8_0.gguf (q8_0 matrices; expected logits from the dequantized weights) and tokenizer_cases.json (texts and
their token ids). The models are snapshots: regenerating changes the random weights only with a different torch.

Usage:
    python scripts/gen_llm_test_models.py

Requires torch, transformers, tokenizers, safetensors and gguf.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter, quants
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "tests" / "data" / "llm"

SPECIALS = ["<s>", "</s>", "<|im_start|>", "<|im_end|>"]
CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
          "'<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
CORPUS = [
    "Hello world! This is a tiny tokenizer for the tests of ii.",
    "The quick brown fox jumps over the lazy dog. 1234567890",
    "Language models predict the next token; they're trained on text.",
    "Привет, мир! Unicode text: café, naïve, 日本語, emoji 🙂.",
    "def main():\n    return 42\n\n\tTabs and   spaces.",
] * 20

TOKENIZER_CASES = [
    "Hello world!",
    " leading space and trailing ",
    "they're 2024 numbers 1234567",
    "Привет, мир! café 日本語 🙂",
    "line one\nline two\r\n\n  indented",
    "<|im_start|>user\nHi<|im_end|>",
    "",
]


def train_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=320, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(CORPUS, trainer)
    return tokenizer


def configs(vocab: int) -> dict:
    from transformers import (Gemma2Config, Gemma3TextConfig, LlamaConfig, Phi3Config, Qwen2Config, Qwen3Config)

    common = dict(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=128, bos_token_id=0, eos_token_id=1,
                  tie_word_embeddings=False)
    return {
        "llama": LlamaConfig(**common, num_hidden_layers=2, rms_norm_eps=1e-5, rope_theta=10000.0,
                             rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                           "high_freq_factor": 4.0, "original_max_position_embeddings": 16}),
        "qwen2": Qwen2Config(**common, num_hidden_layers=2, rms_norm_eps=1e-6, rope_theta=1000000.0),
        "qwen3": Qwen3Config(**{**common, "tie_word_embeddings": True}, num_hidden_layers=2, head_dim=16,
                             rms_norm_eps=1e-6),
        "gemma2": Gemma2Config(**{**common, "tie_word_embeddings": True}, num_hidden_layers=2, head_dim=8,
                               query_pre_attn_scalar=8, sliding_window=4, attn_logit_softcapping=5.0,
                               final_logit_softcapping=3.0, hidden_activation="gelu_pytorch_tanh"),
        "gemma3": Gemma3TextConfig(**{**common, "tie_word_embeddings": True}, num_hidden_layers=6, head_dim=8,
                                   query_pre_attn_scalar=8, sliding_window=4, rope_local_base_freq=10000.0,
                                   rope_theta=1000000.0, hidden_activation="gelu_pytorch_tanh"),
        "phi3": Phi3Config(**common, num_hidden_layers=2, rms_norm_eps=1e-5, pad_token_id=None),
    }


def make_model(config):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(1234)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            # Larger random weights than the default init, so that every term matters; f16-exact values.
            parameter.copy_((torch.randn_like(parameter) * 0.3).half().float())
    return model


def reference_logits(model, tokens: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.from_numpy(tokens[None].astype(np.int64)))
    return out.logits[0].float().numpy()


# --- GGUF --------------------------------------------------------------------------------------------------------

def permute(weights: np.ndarray, heads: int) -> np.ndarray:
    """llama.cpp's q/k permutation for interleaved rotary pairs."""
    return (weights.reshape(heads, 2, weights.shape[0] // heads // 2, *weights.shape[1:]).swapaxes(1, 2)
            .reshape(weights.shape))


def gguf_tensors(arch: str, model, config) -> dict[str, np.ndarray]:
    state = {k: v.float().numpy() for k, v in model.state_dict().items()}
    gemma = arch.startswith("gemma")
    out = {"token_embd.weight": state["model.embed_tokens.weight"],
           "output_norm.weight": state["model.norm.weight"] + (1.0 if gemma else 0.0)}
    if "lm_head.weight" in state and not config.tie_word_embeddings:
        out["output.weight"] = state["lm_head.weight"]
    names = {
        "input_layernorm": "attn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "self_attn.qkv_proj": "attn_qkv",
        "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm", "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down", "mlp.gate_up_proj": "ffn_up",
        "post_attention_layernorm": "post_attention_norm" if gemma else "ffn_norm",
        "pre_feedforward_layernorm": "ffn_norm", "post_feedforward_layernorm": "post_ffw_norm",
    }
    for key, value in state.items():
        if not key.startswith("model.layers."):
            continue
        parts = key.split(".")
        layer, rest = parts[2], ".".join(parts[3:])
        base, suffix = rest.rsplit(".", 1)
        name = names[base]
        if arch == "llama" and name == "attn_q":
            value = permute(value, config.num_attention_heads)
        if arch == "llama" and name == "attn_k":
            value = permute(value, config.num_key_value_heads)
        if gemma and name.endswith("norm"):
            value = value + 1.0
        out[f"blk.{layer}.{name}.{suffix}"] = value
    rope = getattr(config, "rope_parameters", None) or {}
    if arch == "llama" and rope.get("rope_type") == "llama3":
        # llama.cpp's converter stores llama 3 frequency scaling as per-frequency divisors.
        dim = config.head_dim
        freqs = 1.0 / (rope["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        factor, low, high = rope["factor"], rope["low_freq_factor"], rope["high_freq_factor"]
        old = rope["original_max_position_embeddings"]
        factors = []
        for freq in freqs:
            wavelength = 2 * np.pi / freq
            if wavelength < old / high:
                factors.append(1.0)
            elif wavelength > old / low:
                factors.append(factor)
            else:
                smooth = (old / wavelength - low) / (high - low)
                factors.append(1 / ((1 - smooth) / factor + smooth))
        out["rope_freqs.weight"] = np.array(factors, dtype=np.float32)
    return out


def write_gguf(path: Path, arch: str, config, tensors: dict[str, np.ndarray], tokenizer: Tokenizer,
               quantize: GGMLQuantizationType | None = None) -> dict[str, np.ndarray]:
    """Writes the GGUF file; returns the tensors as the file holds them (dequantized)."""
    writer = GGUFWriter(str(path), arch=arch)
    writer.add_name(f"tiny-{arch}")
    a = arch
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    writer.add_uint32(f"{a}.context_length", config.max_position_embeddings)
    writer.add_uint32(f"{a}.embedding_length", config.hidden_size)
    writer.add_uint32(f"{a}.block_count", config.num_hidden_layers)
    writer.add_uint32(f"{a}.feed_forward_length", config.intermediate_size)
    writer.add_uint32(f"{a}.attention.head_count", config.num_attention_heads)
    writer.add_uint32(f"{a}.attention.head_count_kv", config.num_key_value_heads)
    writer.add_uint32(f"{a}.attention.key_length", head_dim)
    writer.add_uint32(f"{a}.attention.value_length", head_dim)
    writer.add_float32(f"{a}.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    rope = getattr(config, "rope_parameters", None) or {}
    theta = rope.get("rope_theta") if isinstance(rope, dict) and "rope_theta" in rope else getattr(config, "rope_theta", 10000.0)
    if arch == "gemma3":
        theta = rope.get("full_attention", {}).get("rope_theta", 1000000.0) if isinstance(rope, dict) else 1000000.0
    writer.add_float32(f"{a}.rope.freq_base", float(theta))
    writer.add_uint32(f"{a}.rope.dimension_count", head_dim)
    if getattr(config, "sliding_window", None) and arch.startswith("gemma"):
        writer.add_uint32(f"{a}.attention.sliding_window", config.sliding_window)
    if arch == "gemma2":
        writer.add_float32(f"{a}.attn_logit_softcapping", config.attn_logit_softcapping)
        writer.add_float32(f"{a}.final_logit_softcapping", config.final_logit_softcapping)
    # Tokenizer
    vocab = tokenizer.get_vocab()
    texts = [""] * len(vocab)
    for text, index in vocab.items():
        texts[index] = text
    data = json.loads(tokenizer.to_str())
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "gpt-2")
    writer.add_array("tokenizer.ggml.tokens", texts)
    writer.add_array("tokenizer.ggml.token_type", [3 if t in SPECIALS else 1 for t in texts])
    writer.add_array("tokenizer.ggml.merges", [m if isinstance(m, str) else " ".join(m) for m in data["model"]["merges"]])
    writer.add_uint32("tokenizer.ggml.bos_token_id", 0)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 1)
    writer.add_bool("tokenizer.ggml.add_bos_token", True)
    writer.add_string("tokenizer.chat_template", CHATML)
    held = {}
    for name, value in tensors.items():
        value = np.ascontiguousarray(value.astype(np.float32))
        if value.ndim == 2 and quantize is not None and value.shape[1] % 32 == 0:
            q = quants.quantize(value, quantize)
            writer.add_tensor(name, q, raw_dtype=quantize)
            held[name] = quants.dequantize(q, quantize).reshape(value.shape)
        elif value.ndim == 2:
            writer.add_tensor(name, value.astype(np.float16))
            held[name] = value.astype(np.float16).astype(np.float32)
        else:
            writer.add_tensor(name, value)  # norms and biases stay f32
            held[name] = value
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return held


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    tokenizer = train_tokenizer()
    vocab = tokenizer.get_vocab_size()
    rng = np.random.default_rng(7)
    tokens = rng.integers(4, vocab, size=12).astype(np.int32)
    tokens[0] = 0  # BOS

    cases = []
    for text in TOKENIZER_CASES:
        cases.append({"text": text, "ids": tokenizer.encode(text).ids})
    (OUTPUT / "tokenizer_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")

    for arch, config in configs(vocab).items():
        model = make_model(config)
        directory = OUTPUT / arch
        directory.mkdir()
        config.save_pretrained(directory)
        state = {k: v.half().contiguous() for k, v in model.state_dict().items()
                 if not (config.tie_word_embeddings and k == "lm_head.weight")}
        save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
        tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "tokenizer_config.json").write_text(json.dumps({
            "bos_token": "<s>", "eos_token": "</s>", "add_bos_token": True, "chat_template": CHATML}, indent=1))
        logits = reference_logits(model, tokens)
        np.savez(directory / "expected.npz", tokens=tokens, logits=logits)
        write_gguf(OUTPUT / f"{arch}.gguf", arch, config, gguf_tensors(arch, model, config), tokenizer)
        print(f"{arch}: {sum(p.numel() for p in model.parameters())} parameters")

    # q8_0 llama: the expected logits come from the weights as stored (dequantized), undoing the q/k permutation.
    config = configs(vocab)["llama"]
    model = make_model(config)
    held = write_gguf(OUTPUT / "llama-q8_0.gguf", "llama", config, gguf_tensors("llama", model, config), tokenizer,
                      GGMLQuantizationType.Q8_0)
    inverse = {"attn_q": config.num_attention_heads, "attn_k": config.num_key_value_heads}
    with torch.no_grad():
        sd = model.state_dict()
        sd["model.embed_tokens.weight"].copy_(torch.from_numpy(held["token_embd.weight"]))
        sd["lm_head.weight"].copy_(torch.from_numpy(held["output.weight"]))
        for layer in range(config.num_hidden_layers):
            for hf, name in [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
                             ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_output"),
                             ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down")]:
                value = held[f"blk.{layer}.{name}.weight"]
                if name in inverse:
                    # The permutation is its own inverse up to the head grouping: apply its inverse explicitly.
                    heads = inverse[name]
                    value = value.reshape(heads, value.shape[0] // heads // 2, 2, -1).swapaxes(1, 2).reshape(value.shape)
                sd[f"model.layers.{layer}.{hf}.weight"].copy_(torch.from_numpy(value))
    np.savez(OUTPUT / "llama-q8_0.expected.npz", tokens=tokens, logits=reference_logits(model, tokens))
    print("llama-q8_0.gguf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
