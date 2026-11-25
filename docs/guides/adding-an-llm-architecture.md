# Adding a language model architecture

Language models are ordinary ii graphs built by `src/llm/graph.cpp` from hyper-parameters (`src/llm/hparams.h`).
An architecture that is a variant of the llama-style decoder (pre-norm blocks of attention and a gated feed-forward
network) usually needs only new hyper-parameters; anything else needs a flag in `HParams` and a few lines in the
graph builder. The architectures so far: llama, qwen2, qwen3, gemma, gemma2, gemma3, phi3.

## 1. Hyper-parameters

`src/llm/hparams.cpp` reads them from two sources, which must agree:

- **GGUF** (`hparams_from_gguf`): `general.architecture` names the architecture; keys are `<arch>.<name>` as written
  by llama.cpp's `convert_hf_to_gguf.py` (`embedding_length`, `block_count`, `attention.head_count`,
  `attention.head_count_kv`, `attention.key_length`, `rope.freq_base`, `rope.dimension_count`,
  `attention.layer_norm_rms_epsilon`, ...). Read llama.cpp's converter for the architecture: it also documents how
  the weights were transformed (e.g. q / k permuted for llama, `+1` added to gemma norms).
- **Hugging Face** (`hparams_from_config`): `model_type` and the fields of `config.json`, including
  `rope_parameters` (transformers 5) or `rope_theta` / `rope_scaling` (older files) and `layer_types`.

Architecture defaults that both formats share go into `apply_architecture` (biases, q / k norms, activation,
embedding scale, which layers use the sliding window).

## 2. Weight names

The graph builder uses GGUF tensor names (`token_embd.weight`, `blk.<i>.attn_q.weight`, ...).
`SafetensorsWeights::translate` maps them to Hugging Face names (`model.layers.<i>.self_attn.q_proj.weight`). Add the
architecture's names there if they differ (gemma2's four norms per layer, phi3's fused `qkv_proj` and `gate_up_proj`
are examples).

## 3. The graph

`build_model` is one loop over the layers. Keep architecture differences as `HParams` fields tested in the loop (not
`if (architecture == ...)`), so that a later model with the same feature needs no code: `qkv_bias`, `qk_norm`,
`post_norms`, `fused_qkv`, `fused_gate_up`, `attention_softcap`, `final_softcap`, `sliding_window` and
`sliding_layers`, `rope_base_local`, `embedding_scale`, `norm_plus_one`, the activation.

Only operations of the OpSet are used, so every backend runs the model; if a pattern is common and slow, add a fused
operation (with a decomposition) rather than special-casing a backend. The KV cache is two states per layer written
by `slice_update` and read by `sdpa` with `kv_len`: keep that structure (the runtime updates the states in place).

## 4. Tokenizer and chat template

GGUF files carry the tokenizer (`tokenizer.ggml.*`) and usually the chat template. A new byte-level BPE pre-tokenizer
(`tokenizer.ggml.pre`) needs a split pattern in `src/llm/unicode.cpp` (`SplitPattern`), written as a hand-made matcher
like the existing ones and tested against the Hugging Face tokenizer. Templates that use Jinja features the
interpreter lacks fail with a clear error: add the feature to `src/llm/jinja.cpp` and a case to
`scripts/gen_chat_template_cases.py`.

## 5. Tests

1. Add a configuration to `configs()` in `scripts/gen_llm_test_models.py` (small sizes, every feature switched on,
   e.g. a sliding window shorter than the test sequence) and the GGUF layout to `gguf_tensors` / `write_gguf`;
   regenerate with `python scripts/gen_llm_test_models.py`.
2. Add the architecture to `model_cases()` in `tests/unit/llm/llm_test.cpp`: the logits of every position must match
   transformers for the Hugging Face directory and for the GGUF file, also when the tokens go through the KV cache in
   chunks.
3. Check one real model: `ii generate <model> --save-logits out.npz -p "..."` then
   `python scripts/compare_transformers.py out.npz --reference <hf-dir> --text "..."`, greedy generation
   (`--save-tokens`, `--generated`) and perplexity (`ii perplexity` against `compare_transformers.py --perplexity`).
   Add it to the `llms` of `tests/models/manifest.json` if it is small enough for the nightly job.
4. Document the architecture in `docs/architecture.md` (§13) and in `include/ii/llm.h`.
