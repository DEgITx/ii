# Language models: speed and accuracy (stage 4)

`ii bench <model> --llm -p 512 -n 128 -r 3` (prompt processing: 512 tokens in one step; generation: 128 tokens one at a
time, from an empty cache), median of 3 runs after a warm-up run, all threads. Models as published (GGUF files of
the model authors, bartowski, unsloth, TheBloke).

## Intel Core Ultra 7 265KF (8 P + 12 E cores, AVX2 + AVX-VNNI), Windows 11

ii 0.1.0 (stage 4), MinGW-w64 GCC 14.2, kernel variant `avx_vnni`, 20 threads. Memory: 98 GB/s read (`ii info
--roofline`).

| model | weights | size | pp512, tokens/s | tg128, tokens/s | tg: weights read per second |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | q4_k_m (mostly q5_0 / q8_0: 896 is not a multiple of 256) | 0.45 GiB | 2567 | 157 | 76 GB/s |
| Qwen3-0.6B | q8_0 | 0.59 GiB | 2572 | 99 | 63 GB/s |
| TinyLlama-1.1B-Chat-v1.0 | q4_k_m | 0.62 GiB | 1170 | 99 | 66 GB/s |
| Llama-3.2-1B-Instruct | q4_k_m | 0.74 GiB | 1182 | 85 | 68 GB/s |
| gemma-3-1b-it | q4_k_m (output matrix of 262k tokens) | 0.74 GiB | 1480 | 84 | 67 GB/s |

- **Prompt processing** is compute-bound. Weights of the common types (`q4_0`, `q5_0`, `q8_0`, `q4_K`, `q6_K`) are
  repacked into panels of 8 rows when the program is compiled; the panel kernels reach 160-190 GMAC/s on one P-core
  (int8, `bench/micro/bench_quant.cpp`), 2.5-3× the row-by-row kernels, which doubled pp512 for every model above
  (TinyLlama: 640 → 1170 tokens/s, Qwen2.5-0.5B: 1330 → 2570). `q6_K` panels are the slowest (97 GMAC/s: decoding
  six-bit values costs more instructions).
- **Generation** reads every weight once per token: 63-76 GB/s of the measured 98 GB/s with all cores. It scales with
  the number of threads up to 16 (TinyLlama: 6 threads 69, 8: 78, 12: 90, 16: 100 tokens/s), so it is limited by the
  outstanding cache misses per core rather than by the kernels (one core decodes 42-50 GB/s of weights held in cache).
- Attention is ~2% of a generated token and ~12% of a 512-token prompt (one query row at a time; tiled flash attention
  is planned for long prompts).

### Against llama.cpp

Not measured: the comparison needs running llama.cpp's `llama-bench` on this machine (official release b12728,
`llama-b12728-bin-win-cpu-x64.zip`), which the sandbox of the session that produced these numbers did not allow.
Command for the same tests:

```sh
llama-bench -m <model>.gguf -p 512 -n 128 -r 3          # llama.cpp
ii bench <model>.gguf --llm -p 512 -n 128 -r 3 --ctx 1024
```

## Qualcomm Snapdragon 732G (2 × Cortex-A76 + 6 × Cortex-A55), Android 13

NDK r30 Clang, kernel variant `neon_dotprod`, 8 threads, through `scripts/android_run.py`.

| model | weights | pp128, tokens/s | tg64, tokens/s |
|---|---|---:|---:|
| Qwen2.5-0.5B-Instruct | q4_k_m | 134 | 20.7 |
| TinyLlama-1.1B-Chat-v1.0 | q4_k_m | 63 | 13.4 |

The dot-product variant (`sdot`, Armv8.2) with panels made prompt processing 3.3× and generation 1.8× faster than the
plain NEON kernels (Qwen2.5-0.5B: 40 → 134 and 11.7 → 20.7 tokens/s).

## Accuracy

| model | check | result |
|---|---|---|
| tiny random models (llama, qwen2, qwen3, gemma2, gemma3, phi3; HF and GGUF; q8_0) | logits of every position vs transformers, through the KV cache | ≤ 2e-4 relative (f16 weights), 3e-2 (q8_0 with 8-bit activations) |
| Qwen2.5-0.5B-Instruct (safetensors bf16) | logits vs transformers (f32) | 3.8e-6 relative, KL 1e-11 |
| | greedy generation | 64 of 64 tokens equal |
| TinyLlama-1.1B-Chat (safetensors bf16) | logits vs transformers | 2.2e-6 relative, KL 3.5e-12 |
| | greedy generation | 128 of 128 tokens equal |
| | perplexity, wikitext-2 test, 20 chunks of 512 | ii 9.2903, transformers 9.2903 (same tokens); 9.2915 with ii's tokenizer |
| TinyLlama-1.1B-Chat q4_k_m (GGUF) | perplexity, same text | 9.623 (+3.6% over float, typical of Q4_K_M for 1B models) |
| 11 chat templates × 6 conversations | rendered text vs transformers | 66 of 66 identical |
| Qwen2.5-0.5B-Instruct: `ii quantize` fp16 → q4_k_m | file, perplexity (8 chunks) | same size and type mix as the published q4_k_m (491 400 032 bytes); PPL 17.07 (published file 16.83, fp16 16.14); 7 s |

## Plan

1. Tiled (flash) attention for long prompts; prompt processing of attention is per query row today.
2. Panels for `q5_K` (Q5_K_M models) and the remaining types; `q6_K` panels with fewer instructions.
3. Generation: software prefetch of the next panels to raise the outstanding misses per core; thread count tuned per
   machine (fewer threads are as fast on some CPUs).
4. AVX-512 VNNI and Arm i8mm (`smmla`) variants.
