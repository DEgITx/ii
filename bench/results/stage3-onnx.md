# ONNX models: ii vs ONNX Runtime (stage 3)

Latency of one inference (batch 1) measured with `ii bench <model>` and with ONNX Runtime 1.31.0 (CPU execution
provider, default graph optimizations) on the same machine: median of 50–100 runs after warm-up. Single-thread runs
are pinned to one performance core (see `bench/results/stage2-cpu.md`). Accuracy against ONNX Runtime is checked by
`scripts/model_tests.py`: every model below agrees to ~1e-6 relative to its largest output value (argmax agreement
100% for classifiers; YOLOv8n finds the same objects with the same scores on `bus.jpg`).

## Intel Core Ultra 7 265KF (8 P + 12 E cores, AVX2), Windows 11

ii 0.1.0 (stage 3), MinGW-w64 GCC 14.2, `-O3`, kernel variant `avx2`.

| model | input | ii, all cores | ORT, all cores | ii, 1 P-core | ORT, 1 P-core |
|---|---|---:|---:|---:|---:|
| ResNet-50 (torchvision export) | 1×3×224×224 | 8.1 ms | 5.5 ms | 77 ms | 54 ms |
| ResNet-50 v1-12 (ONNX Model Zoo, with BatchNormalization) | 1×3×224×224 | 7.8 ms | 5.3 ms | 75 ms | 51 ms |
| MobileNetV2 (torchvision export) | 1×3×224×224 | 1.70 ms | 0.89 ms | 10.5 ms | 4.9 ms |
| YOLOv8n (ultralytics export) | 1×3×640×640 | 12.6 ms | 10.1 ms | 112 ms | 68 ms |
| BERT-base (transformers export) | 1×128 tokens | 19.4 ms | 17.4 ms | 180 ms | 163 ms |

- ii is 1.1–1.25× slower than ONNX Runtime on transformer encoders and YOLOv8n with all cores, 1.5× on ResNet-50
  and 1.9× on MobileNetV2.
- Where the time goes (profiles from `ii run --profile`): convolutions are 90% of the CNNs. The GEMM micro-kernel
  reaches ~140 GFLOP/s on one P-core for large matrices but only ~90 GFLOP/s when a convolution has few output
  channels (M = 64: each packed B panel is reused by only 11 row panels), and the im2col panels have to be written
  and read once per image. ONNX Runtime (MLAS) uses direct convolution kernels in the blocked NCHWc layout, which
  avoid both.
- Depthwise convolutions (MobileNet) run at 8–34 GFLOP/s: small planes (7×7, 14×14) waste most of each vector, which
  a channel-blocked layout avoids.
- BERT: the `linear` GEMMs are 83% of the time; attention is computed by separate matmul / softmax / permute
  operations (no fused attention pattern yet).

## Qualcomm Snapdragon 732G (2 × Cortex-A76 + 6 × Cortex-A55), Android 13

NDK r30 Clang, kernel variant `neon`, all 8 cores, run through `scripts/android_run.py`: YOLOv8n 640×640 264 ms,
MobileNetV2 (Model Zoo) 28 ms. `examples/yolo` finds the same five objects with the same scores and boxes as on the PC.

## Plan

1. **NCHWc layout for convolutions** (a layout pass that keeps activations channel-blocked between convolutions,
   direct kernels for 1×1 and 3×3 with the output channels in vector lanes, depthwise across channels). This is the
   main gap to MLAS for CNNs.
2. **Attention pattern → `sdpa`** in the importer / fusion passes (BERT, ViT); also LayerNorm / GELU written with
   primitives in older exports.
3. **GEMM for small M**: larger NC per item and B-panel reuse across row panels of several output channel groups.
4. Winograd F(2,3) for 3×3 convolutions with many channels (after NCHWc).
