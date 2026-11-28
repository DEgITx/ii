# The CUDA and TensorRT backends: speed and accuracy (stages 5.3-5.5)

`ii bench <model> --repeat 100` (CPU: 20-30 runs), median latency of one run, after warm-up runs. A run includes
copying the inputs to the GPU and the outputs back (pageable host memory) and runs the plan the planner chose with
all backends available (the plugin found through `II_PLUGIN_PATH`). Models in f32 from `tests/models/manifest.json`.

## NVIDIA GeForce RTX 5080 (Blackwell, sm_120, 16 GB), Intel Core Ultra 7 265KF, Windows 11

ii 0.1.0 (stage 5.3) built with MSVC 14.44, CUDA 13.2 (cuBLASLt 13.4, NVRTC 13.2), driver 610.88. GPU: f32 computes
in f32; "TF32" is `II_PLUGIN_OPTIONS=cuda.tf32=1`. CPU: ii-cpu, kernel variant `avx_vnni`, 20 threads.

| model | inputs | GPU, ms | GPU TF32, ms | CPU, ms | GPU vs CPU |
|---|---|---:|---:|---:|---:|
| ResNet-50 v1 | 2×3×224×224 | 3.30 | 2.32 | 21.7 | 6.6× |
| ResNet-50 v1 | 16×3×224×224 | 14.4 | 10.8 | 175 | 12× |
| YOLOv8n | 1×3×640×640 | 3.65 | 3.50 | 17.6 | 4.8× |
| BERT-base-uncased | 2×32 tokens | 3.97 | 3.65 | 18.0 | 4.5× |
| BERT-base-uncased | 8×128 tokens | 16.9 | 15.0 | 212 | 12.5× |
| MobileNetV2 | 2×3×224×224 | 1.45 | 1.29 | 3.38 | 2.3× |

Accuracy against ONNX Runtime 1.31 (CPU) on the same random inputs (`scripts/compare_onnxruntime.py`), GPU without
TF32: ResNet-50 max |error| 7.9e-6 (9e-7 of the largest value), YOLOv8n 6.7e-4 of values up to 640 (1e-6), BERT
3.8e-5 of logits up to 20 (2e-6); cosine similarity 1.0000000 for all three.

Where the time goes (`ii run --profile`, which waits for every GPU step):

- **ResNet-50, N = 2**: convolutions are 96% of the run (`fused_conv` 3.6 ms in 49 steps, `conv` 0.19 ms), copies
  0.12 ms. Convolutions go through im2col and an f32 GEMM on the CUDA cores (about 5 TFLOP/s of the 58 TFLOP/s
  FP32 peak at this size); TF32 tensor cores cut the GEMM part. Implicit-GEMM convolutions without the im2col copy,
  f16 / bf16 inference and CUDA Graphs are the next steps for CNNs.
- **YOLOv8n**: convolutions 72%, copying the 4.9 MB input and the output 12%, `concat` / `split` 12%.
- **BERT, 8×128**: the output logits (8 × 128 × 30522 floats, 125 MB) are copied to pageable host memory; that copy is
  a large part of the run. Matrix products use cuBLASLt.
- The planner keeps BERT's attention-mask arithmetic (a few hundred bytes, i64) on the CPU and copies its result to
  the GPU once; every other step of the three models runs on the GPU.

Stage 5.4 (CUDA graphs: a run with the same memory as the previous one replays a recording of its kernels;
calibrated planner figures), same machine and settings, GPU without TF32:

| model | inputs | 5.3, ms | 5.4, ms | 5.4 without graphs, ms |
|---|---|---:|---:|---:|
| ResNet-50 v1 | 2×3×224×224 | 3.30 | 2.97 | 3.28 |
| ResNet-50 v1 | 16×3×224×224 | 14.4 | 13.6 | 13.9 |
| YOLOv8n | 1×3×640×640 | 3.65 | 2.62 | 3.63 |
| BERT-base-uncased | 2×32 tokens | 3.97 | 2.55 | 4.42 |
| BERT-base-uncased | 8×128 tokens | 16.9 | 18.4 | 19.1 |
| MobileNetV2 | 2×3×224×224 | 1.45 | 1.02 | 1.43 |

(BERT 8×128 is bound by copying its 125 MB of logits to pageable host memory, whose speed varies between runs.)

## TensorRT (stage 5.5)

Both plugins found (`II_PLUGIN_PATH`), the planner's plan: each of these models becomes one TensorRT engine (the
part's estimate beats ii-cuda's kernels), ii copies the inputs to the GPU and the outputs back. `ii bench <model>
--repeat 200 --warmup 20` with fixed dimensions (`--dims`) against `trtexec --onnx=<model> --useCudaGraph
--iterations=500 --warmUp=500` with the same shapes (BERT: the same token ids), TensorRT 10.16.1, run back to back on
the same GPU (other applications were using it, so absolute times are a little noisy). f32: `--noTF32` for trtexec,
no options for ii (strongly typed network, TF32 off); TF32: trtexec's default and
`II_PLUGIN_OPTIONS=tensorrt.tf32=1;cuda.tf32=1`. Medians in ms; trtexec's latency includes copying the inputs and
outputs (pinned host memory), its compute time is the engine alone.

| model | inputs | ii-cuda 5.4 | ii 5.5 | trtexec latency | trtexec compute | ii 5.5 TF32 | trtexec TF32 latency |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 v1 | 2×3×224×224 | 2.97 | 1.61 | 1.40 | 1.37 | 1.47 | 1.09 |
| ResNet-50 v1 | 16×3×224×224 | 13.6 | 6.45 | 5.78 | 5.60 | 4.95 | 4.20 |
| YOLOv8n | 1×3×640×640 | 2.62 | 1.92 | 1.23 | 1.06 | 1.56 | 0.92 |
| BERT-base-uncased | 2×32 tokens | 2.55 | 1.93 | 1.38 | 1.18 | 1.58 | 1.12 |
| BERT-base-uncased | 8×128 tokens | 18.4 | 16.8 | 11.3 | 8.0 | 14.3 | 9.2 |
| MobileNetV2 | 2×3×224×224 | 1.02 | 0.52 | 0.38 | 0.35 | 0.43 | 0.31 |

The engines are TensorRT's, so the GPU part runs at trtexec's speed; the rest of the difference is copying between
host and GPU, which ii does from and to pageable memory (`ii run --profile`, which runs every step without CUDA
graphs: BERT 8×128 spends 8.5 ms in the engine — trtexec 8.0 — and 8.5 ms copying 125 MB of logits, trtexec 3.3 ms
from pinned memory; YOLOv8n 0.64 ms copying its 4.9 MB input and 2.8 MB output). Pinned staging buffers for inputs and
outputs are the next step. A single TensorRT step is recorded into a CUDA graph like a run of ii-cuda kernels
(without it ResNet-50 ×2 took 2.06 ms and MobileNetV2 1.00 ms: the engine's kernels launched one by one).

Accuracy against ONNX Runtime 1.31 (CPU), `scripts/compare_onnxruntime.py`, f32 plans: ResNet-50 max |error| 6.8e-6
(8e-7 of the largest value), YOLOv8n 7.3e-4 of values up to 640 (1.1e-6), BERT 3.4e-5 of logits up to 11.8 (2.9e-6),
MobileNetV2 9.5e-6 (1.2e-6); cosine similarity 1.0000000 for all.

Engine builds: the first run of a new model (or of shapes outside the engine's profile) builds the engine —
ResNet-50 and YOLOv8n about 22 s, BERT 17 s, small parts 0.2-2 s; later runs load it from the cache
(`<ii cache>/tensorrt`) in a second or a few (BERT's 420 MB of weights are hashed for the key). Language models are
unaffected: their quantized products, rope, KV-cache writes and attention with `kv_len` stay with ii-cuda, and the
small pieces between them are not worth an engine (Qwen2.5-0.5B q4_k_m: the same plan and speed with and without the
TensorRT plugin).

TensorRT 10.16.1 computes a softmax followed by a sum over all elements of a small tensor wrong when it fuses them
(reproduced with trtexec on a two-node ONNX model, [4, 4]); the translator reduces several axes in two layers, which
avoids it.

## Language models (stage 5.4)

`ii bench model.gguf --llm -r 5 --kv-type f16` against `llama-bench -p 512 -n 128 -ngl 99 -r 5 -ctk f16 -ctv f16`
of llama.cpp b12728 (the release build `win-cuda-13.5-x64`), same GGUF files, same machine: tokens per second of a
512-token prompt (pp512) and of generating 128 tokens one by one (tg128); ii runs the plan the planner chose (the
whole model on the GPU, the token embedding gathered on the CPU), autotuning cache filled.

| model | weights | test | ii | llama.cpp | ii / llama.cpp |
|---|---|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | q4_k_m (mostly q5_0) | pp512 | 57 580 | 63 374 | 0.91 |
| | | tg128 | 569 | 874 | 0.65 |
| Qwen3-0.6B | q8_0 | pp512 | 39 184 | 50 886 | 0.77 |
| | | tg128 | 380 | 677 | 0.56 |
| TinyLlama-1.1B-Chat | q4_k_m | pp512 | 31 255 | 41 537 | 0.75 |
| | | tg128 | 508 | 759 | 0.67 |
| Llama-3.2-1B-Instruct | q4_k_m | pp512 | 32 019 | 44 639 | 0.72 |
| | | tg128 | 510 | 720 | 0.71 |
| gemma-3-1b-it | q4_k_m | pp512 | 22 277 | 41 016 | 0.54 |
| | | tg128 | 334 | 517 | 0.65 |
| Qwen2.5-0.5B-Instruct | f16 | pp512 | 36 106 | 78 478 | 0.46 |
| | | tg128 | 432 | 593 | 0.73 |

For comparison, the CPU (ii-cpu, 20 threads, stage 4): Qwen2.5-0.5B q4_k_m pp512 ≈ 2 570, tg128 ≈ 157;
Llama-3.2-1B q4_k_m ≈ 1 180 / 85.

Where ii stands and why:

- **Decoding** (tg) is a CUDA graph of ~17 kernels per layer replayed once per token (Qwen2.5-0.5B: 1.55 ms of GPU
  time per token, of which the weights' memory traffic is ~0.5 ms). The kernels are short (1.5–6 µs) and their
  number, not the bandwidth, sets the time: llama.cpp fuses more per layer (normalization into the products,
  gate and up projections into one product with the activation, rope with the cache write). Planned: fused linear
  products sharing an input, residual additions in the products' epilogue.
- **Prompts** (pp): block-quantized weights are decoded to f16 and multiplied with x in f16 on cuBLASLt with f32
  accumulation (~90 TFLOP/s on this GPU); llama.cpp multiplies the quantized blocks with 8-bit activations on int8
  tensor cores (MMQ), which needs neither the decoding pass nor f16 accumulation. 16-bit weights multiply two f16
  parts of x (f32 precision) where llama.cpp accumulates in f16. gemma-3 has heads of 256 features, which the
  tensor-core attention does not cover yet (the CUDA-core kernel runs instead).
- Attention: the tensor-core kernel for prompts is 6–9× faster than the CUDA-core one at 512 tokens (Qwen2.5-0.5B:
  62 µs instead of 388 per layer; heads of 128: 124 instead of 717); decoding attention with split keys takes
  ~7 µs per layer at short contexts.

Accuracy (`ii generate --save-logits`, 64-71 positions of a fixed English text):

| model | GPU vs CPU: KL | GPU vs ii-ref (f32): KL | CPU vs ii-ref: KL |
|---|---:|---:|---:|
| Qwen2.5-0.5B q4_k_m | 3.2e-3 | 3.4e-6 | 3.8e-3 |
| Qwen3-0.6B q8_0 | 7.0e-3 | 2.4e-6 | 1.1e-2 |
| Qwen2.5-0.5B f16 | 3.0e-11 | | |

The GPU multiplies quantized weights with f32 (decoding) or f16 (prompt) activations, so it stays close to the
exact f32 reference; ii-cpu, like llama.cpp, quantizes activations to 8 bits, which is what separates GPU and CPU
logits of quantized models. With Hugging Face models (bf16 weights) the GPU matches transformers like the CPU:
`scripts/llm_tests.py --suite nightly` with `II_PLUGIN_PATH` set passes — logits within 8.2e-6 (Qwen2.5-0.5B) and
3.5e-6 (TinyLlama) of transformers, greedy generation equal for 64 / 128 tokens, TinyLlama's wikitext-2 perplexity
9.1393 (transformers 9.1365; the CPU's is the same), TinyLlama q4_k_m +3.76% over the float model.

Offload: Llama-3.2-1B q4_k_m with a context of 2048 needs ~1.2 GB on the GPU. With `--device-memory cuda:0=500M` the
planner moves the front of the model (108 operations, the KV cache included) to the CPU: the same greedy text,
133 tokens/s (GPU alone 509, CPU alone 85); with 250 MB, 103 tokens/s.
