# ii — Testing and Benchmarks

Tests and benchmarks are part of the architecture, not an appendix to it. Two ground rules:

1. **Every fast implementation has a reference**, and the two are compared continuously (differential testing).
2. **Every hot path has a benchmark** with a target expressed as % of what the hardware can do (roofline).

---

## 1. Tools

| Task | Tool |
|------|------|
| unit and parameterized tests | GoogleTest (typed / value-parameterized tests fit the op × dtype × backend × ISA matrix) |
| micro-benchmarks | Google Benchmark |
| fuzzing | libFuzzer (Clang), corpora in `tests/fuzz/corpus/` |
| sanitizers | ASan, UBSan, TSan (thread pool, planner), MSan where possible |
| coverage | llvm-cov / gcov |
| references | Python scripts (`scripts/gen_op_references.py`; NumPy, PyTorch, later ONNX Runtime, transformers) — a developer tool only, never a library dependency |

Test dependencies are fetched via `FetchContent` only with `II_BUILD_TESTS` / `II_BUILD_BENCHMARKS` and never reach
library consumers.

---

## 2. Test pyramid

### 2.1 Unit (`tests/unit/`)
Core (Result, allocators, SmallVector, f16/bf16 conversions — exhaustively over all 65536 values, JSON, platform
layer on every OS), IR (construction, verify, text format round-trip), symbolic shapes, memory planner (no overlap of
simultaneously live values — a property checked on random graphs).

### 2.2 Op conformance (`tests/unit/ops/`)
For every op × dtype × set of shapes:

- edge-case shapes: empty tensors, size-1 dims, odd primes (7, 13, 127 — catch SIMD tail bugs), large sizes,
  non-contiguous strides, broadcasting in every combination, negative axes;
- edge-case values: ±0, ±Inf, NaN, denormals, very large/small;
- `ii-ref` is checked against **NumPy/PyTorch references**: `scripts/gen_op_references.py` writes
  `tests/data/op_references.json` (one-operation models in the IR text format + base64 tensors, expected values
  computed in float64 and rounded once); every case runs end to end through compile and a session, with and without
  decomposition of composites. `--check` verifies that the committed file is up to date;
- shape inference is checked against the actual result shapes.

### 2.3 Differential backend tests (`tests/backends/`)
The harness `ii_test::compare_backends(model, inputs, backend_a, backend_b, tolerance)` (`tests/support/`) executes
one model on two backends (each candidate against `ii-ref`) and compares the results. The same op tests automatically run for every backend — a new
backend gets thousands of tests on day one.

**Random graphs**: `ii_test::random_model(context, seed)` (seed → valid model + inputs) catches bugs at op
boundaries, in passes, in fusion, in the memory planner and in backend partitioning
(`tests/unit/compiler/pass_invariance_test.cpp`).

**ISA matrix**: the ii-cpu tests are parameterized over every kernel variant the machine can run
(`usable_isas()`: generic, avx2, avx512, neon): all conformance cases (`CpuConformanceTest`), differential tests
against ii-ref with the model compiled for ii-cpu alone, so nothing silently falls back (`tests/unit/backends/` —
GEMM edge shapes, layouts, batch broadcasting, f16/bf16, broadcasting patterns, every elementwise op, reductions,
softmax over every axis, normalizations, attention with GQA / causal / masks, permutations of 1-, 2-, 4- and 8-byte
elements), and sweeps of the vectorized math functions against double precision (max ulp error per function, special
values, f16/bf16 rounding of random bit patterns). Instruction sets missing on the machine (AVX-512) run under Intel
SDE in CI; ARM on ARM runners (Linux ARM64, macOS) and on Android (emulator in CI, phones over adb). `II_CPU_ISA`
forces a variant for any test or tool.

**Fake remote device**: the test backend `ii-test-remote` has a separate "memory domain" (copies through a separate
arena) — cross-device transfers, events and offload are tested without a GPU.

**GPU backends** (`ii-cuda`, TensorRT, later Vulkan/Metal) run the same differential suite on GPU runners; TensorRT
additionally tests engine-cache invalidation (different shapes / precision / version produce a rebuild).

### 2.3a Importers and formats (`tests/unit/formats/`, `tests/onnx/`, `tests/models/`)
- Protobuf decoding (round trips, malformed input), ONNX import of models built in memory (symbolic shape
  arithmetic, conv + batch norm + Gemm against the builder API, functions, constant `If`, external data and its
  directory confinement, constant inputs, truncated and garbage models), `.npy` / `.npz` (NumPy-written files,
  compressed archives, every dtype) and DEFLATE.
- **ONNX conformance**: `scripts/onnx_node_tests.py` writes the operator test cases of the `onnx` package (model +
  inputs + expected outputs); `ii_onnx_tests <dir>` runs each on ii-ref and ii-cpu, reports pass / fail / unsupported
  and with `--expected tests/onnx/expected_passes.txt` fails when a listed case regresses (CI job
  `onnx-conformance`). Parameter inputs (axes, shapes) are turned into constants when a case needs them.
- **Model tests**: `tests/models/manifest.json` lists real models (URL + SHA-256, or a checkpoint exported with
  ultralytics); `scripts/fetch_models.py` downloads them into a cache, `scripts/model_tests.py` compares every output
  with ONNX Runtime on the same inputs (`scripts/compare_onnxruntime.py`) and runs `examples/yolo` on a test image.
  Suite `pr` (MobileNetV2, YOLOv8n) runs in CI on every change, `nightly` adds ResNet-50 and BERT-base.
- **GGUF, safetensors and quantization** (stage 4): files written by gguf-py and the safetensors package
  (`scripts/gen_format_samples.py`), the GGUF writer round trip, truncated and corrupted files; the block codecs
  against gguf-py (`scripts/gen_quant_references.py`: dequantization of every type gguf-py implements, byte-exact
  quantization of `q4_0` … `q8_0`, round-trip error per bit width).

### 2.4 Pass tests (`tests/unit/compiler/`)
- "input text IR → expected text IR" (structural check), e.g. which nodes CSE merges, which chains elementwise
  fusion groups and that fused programs decompose back;
- **numerical equivalence**: the graph is executed before and after the pass on random inputs and results must match
  within tolerance. Applied to every pass automatically.

### 2.5 Gradients (`tests/grad/`, training stage)
For every op with a gradient rule — central finite differences in f64 on `ii-ref`; the "overfit a single batch"
test (loss → ~0); training curve of a small model compared to PyTorch with identical deterministic initialization.

### 2.6 Whole models (`tests/models/`)
- **Tiny language models** (`tests/data/llm`, `scripts/gen_llm_test_models.py`): randomly initialized transformers
  models of every supported architecture (llama with llama-3 rope scaling, qwen2, qwen3, gemma2 with soft-capping and
  a sliding window shorter than the test, gemma3 with both layer kinds, phi3), a small byte-level BPE tokenizer and
  a chat template, saved as Hugging Face directories (f16 safetensors) and as GGUF (laid out like llama.cpp's
  converter, q / k permuted for llama), plus a q8_0 GGUF. `tests/unit/llm/llm_test.cpp` compares the logits of every
  position with transformers, once for the whole sequence and once token chunk by chunk through the KV cache, on
  ii-cpu and on ii-ref; tokenization of test strings equals the Hugging Face tokenizer's. `jinja_test.cpp` renders 66
  cases of 11 real chat templates (`scripts/gen_chat_template_cases.py`) and requires the output of transformers.
- **Real language models** (`tests/models/manifest.json` "llms", `scripts/llm_tests.py`): Qwen2.5-0.5B-Instruct
  q4_k_m answers a question in every PR run; nightly, Qwen2.5-0.5B-Instruct and TinyLlama-1.1B-Chat (safetensors)
  must match transformers (logits within 1e-4 relative, greedy generation token for token for 64 / 128 tokens,
  perplexity on wikitext-2 within 0.1%), and TinyLlama q4_k_m GGUF must stay within 6% of the float perplexity.
  `scripts/compare_transformers.py` does the comparisons (`ii generate --save-logits / --save-tokens`).
- **Other real models** (nightly): ResNet-50, MobileNetV2, BERT-base compared with ONNX Runtime (above).

### 2.7 Runtime invariants
- **Zero allocations in `run()`**: the test swaps in a counting allocator, warms up the session and checks that
  subsequent runs allocate nothing.
- **Determinism**: in deterministic mode two runs are bitwise identical.
- **Thread safety**: N threads × their own `Session` over one `Program`, under TSan.
- **Errors**: invalid inputs produce a `Status`, never a crash. Missing plugin runtimes (e.g. TensorRT DLLs absent)
  make the backend unavailable, never break the core.

### 2.8 Fuzzing (`tests/fuzz/`)
Targets: the IR text format and JSON (stage 1); ONNX import (`fuzz_onnx`: protobuf decoding, tensor conversion,
operator mapping and shape tracking; accepted models must compile) and `.npy` / `.npz` with DEFLATE (`fuzz_npy`)
(stage 3); GGUF and safetensors (`fuzz_gguf`: a parsed GGUF is also read as a language model — hyper-parameters,
tokenizer, graph construction) and tokenizers plus Jinja templates (`fuzz_tokenizer`) (stage 4); later `.ii`.
With `II_BUILD_FUZZERS=ON` (preset `linux-clang-fuzz`) each target is a libFuzzer executable and ctest fuzzes it for
60 s in every PR; every other build replays the committed corpora
(`tests/fuzz/corpus/<target>/`) as regression tests. Crashing inputs found by fuzzing are added to the corpus.
`nightly.yml` fuzzes every target for 30 minutes.

---

## 3. Numerical tolerances

| Case | Tolerance |
|------|-----------|
| exact f32 elementwise (add, mul, …) | bitwise (or ≤ 1 ULP with FMA) |
| transcendental f32 (exp, tanh, erf) | ≤ the documented max error of the approximation (typically 2–4 ULP) |
| reductions / matmul f32 | `|err| ≤ c · K · eps · (|A|·|B|)` — a bound depending on the summation length, not a single `atol` |
| f16 / bf16 | compared to an f32 reference with that type's eps; f32 accumulation is mandatory |
| quantized | reference = `ii-ref` on the **same** quantized weights (dequantization is bitwise exact) → tolerance only on accumulation; ii-cpu also quantizes the activations to 8 bits (as llama.cpp), so its kernels are checked exactly on the same quantized activations, and models through perplexity |
| TensorRT / reduced precision modes | per-precision tolerances (fp16/int8/fp8), plus model-level metrics below |
| whole models | logit cosine similarity ≥ 0.999 (f32), top-1 / greedy match, perplexity delta |

Tolerances live in one place (`tests/support/tolerance.h`), each justified by a comment.

---

## 4. Benchmarks

### 4.1 Machine profile (roofline)
`ii info --roofline` measures peak FLOPS per dtype (an FMA loop on all cores) and memory bandwidth (a STREAM-like
test) and stores the profile. Every benchmark reports not only absolute time but also **% of peak**.

### 4.2 Micro-benchmarks (`bench/micro/`, Google Benchmark)

| Group | Metric | Target (ii-cpu) |
|-------|--------|-----------------|
| GEMM f32 (large sizes) | GFLOPS, % of peak | ≥ 85% of peak single-threaded, ≥ 75% multi-threaded; parity ±10% with OpenBLAS/oneDNN (stage 2: parity with OpenBLAS reached, see `bench/results/`) |
| quantized GEMV (q4_k, q8_0) | GB/s, % of bandwidth | ≥ 85% of memory bandwidth |
| quantized GEMM (prefill) | GOPS | parity ±10% with llama.cpp |
| elementwise / reductions | GB/s | ≥ 90% of memcpy |
| sdpa | tokens/s, % of peak | baseline first, then improvements |
| conv (3×3, 1×1, depthwise) | GFLOPS | parity with ORT CPU EP |
| ThreadPool `parallel_for` | µs per launch | < 5 µs |
| memory planner / compilation | ms | guard against startup-time regressions |

GPU backends get the same groups with their own targets (% of GPU peak; parity with cuBLAS/TensorRT where applicable).

### 4.3 End-to-end benchmarks (`ii bench`, `bench/e2e/`)
- **General models**: latency p50/p90/p99, throughput (batched), peak memory, time to first result (load + compile +
  first run; for TensorRT reported separately with cold and warm engine cache) — ResNet-50, MobileNetV2, BERT-base,
  YOLO-n.
- **LLMs**: `pp512` (prefill, tokens/s), `tg128` (generation, tokens/s) — methodology compatible with `llama-bench`
  for a fair comparison; plus time to first token and memory. Models: TinyLlama-1.1B, Llama-3.2-1B/3B, Qwen2.5-1.5B
  in f16 / q8_0 / q4_k_m.
- **Baselines**: llama.cpp, ONNX Runtime, OpenBLAS/oneDNN, TensorRT (`trtexec`) — run scripts in `bench/baselines/`.

`ii bench <model>` measures whole models (latency percentiles); `bench/results/stage3-onnx.md` compares
ResNet-50, MobileNetV2, YOLOv8n and BERT-base with ONNX Runtime.

`bench/micro/bench_cpu.cpp` measures ii-cpu end to end (one-node compiled models): `BM_Linear` (prepacked f32 /
f16 / bf16 weights: square GEMMs, GEMV, prefill shapes), `BM_Matmul` (both operands at run time),
`BM_Elementwise` (one op vs a fused chain), `BM_Softmax`, `BM_LayerNorm`; `II_CPU_ISA` selects the variant.
`scripts/bench_openblas.py` measures OpenBLAS (through NumPy) on the same shapes; results with the machine
description go to `bench/results/`.

### 4.4 Methodology
- warm-up, repeated runs, median and confidence interval, outlier rejection;
- fixed thread count and (optionally) core pinning; on hybrid CPUs — separate measurements for P-cores only / all cores;
- every report includes: CPU/GPU, ISA, frequencies, OS, compiler, driver versions, ii version and git hash, build
  options;
- results are JSON; `scripts/bench_compare.py` compares two runs and flags regressions above a threshold.

### 4.5 Regressions
A nightly benchmark run on a fixed machine with history; a regression > 5% on key metrics is a signal (blocking for
releases). PRs get a fast subset as an informational report.

---

## 5. CI matrix

Summary only; workflows, triggers, runners and security rules are defined in [ci.md](ci.md).

| Platform | Compilers | What runs | From |
|----------|-----------|-----------|------|
| Windows x86-64 | MSVC, clang-cl, MinGW-GCC | all tests, C ABI interop across compilers | stage 0 |
| Linux x86-64 | GCC, Clang | all tests, sanitizers, short fuzzing, coverage, SDE for AVX-512/AMX | stage 0 |
| Linux ARM64 | GCC, Clang | all tests, NEON/dotprod/i8mm paths | stage 0 |
| macOS ARM64 / x86-64 | Apple Clang | all tests (+ Metal later) | stage 0 |
| Android ARM64 / x86-64 | NDK Clang | build, tests on emulator (CI) and devices (`ctest --preset android-arm64-release` over adb) | stage 2 (done) |
| GPU runner (self-hosted, NVIDIA) | MSVC / GCC + CUDA | differential tests for ii-cuda and TensorRT, GPU benchmarks | stage 5 |
| iOS, WebAssembly | Apple Clang, Emscripten | build, tests where possible | stage 6 |
| Nightly | — | real models, long fuzzing, benchmarks with history | stage 1+ |

Every roadmap stage has completion criteria expressed through these tests and benchmarks.
