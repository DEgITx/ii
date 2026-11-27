# The CUDA backend: speed and accuracy (stage 5.3)

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

Not compared yet: TensorRT (`trtexec`, part 5.5) and llama.cpp CUDA (LLMs, part 5.4).
