# ii-cpu benchmarks (stage 2)

Measured with `bench/micro/bench_cpu.cpp` (one-node compiled models: runtime overhead included) and
`scripts/bench_openblas.py` (OpenBLAS sgemm through NumPy, `x @ w.T`, the same shapes). `BM_Linear` multiplies
`x[M, K]` by constant weights `w[N, K]` (prepacked when the program is compiled). Best of repeated runs.

Single-thread numbers are pinned to one performance core (Windows: process affinity mask 1; Android:
`taskset 80`, a Cortex-A76) — on hybrid CPUs an unpinned single-thread benchmark may run on an efficiency core and
report about 35% less. "All" uses every logical CPU (P and E cores, big and LITTLE).

## Intel Core Ultra 7 265KF (8 P + 12 E cores, AVX2, no AVX-512), DDR5, Windows 11

ii 0.1.0 (stage 2), MinGW-w64 GCC 14.2, `-O3`, kernel variant `avx2`; OpenBLAS 0.3.31 (NumPy 2.4.6, Haswell
kernels). Peak of one P-core: 2 FMA × 8 lanes × 2 flops × ~5.5 GHz ≈ 176 GFLOP/s.

### GEMM f32, GFLOP/s

| M × N × K | ii-cpu, 1 thread | OpenBLAS, 1 thread | ii-cpu, all | OpenBLAS, all |
|---|---:|---:|---:|---:|
| 256 × 256 × 256 | 122 | 147 | 1403 | 401 |
| 512 × 512 × 512 | 143 | 138 | 1637 | 726 |
| 1024 × 1024 × 1024 | 144 | 140 | 1637 | 1146 |
| 2048 × 2048 × 2048 | 141 | 144 | 1659 | 1357 |
| 64 × 4096 × 4096 | 126 | 108 | 1399 | 822 |
| 512 × 4096 × 4096 | 140 | 145 | 1541 | 1347 |
| 1 × 4096 × 4096 (GEMV) | 16.6 (33 GB/s) | 21.2 (42 GB/s) | 78 (156 GB/s) | 101 (201 GB/s) |

- Large GEMMs: parity with OpenBLAS on one core (~81% of peak), ahead on all cores (dynamic distribution of work
  items balances P and E cores).
- MSVC 19.44 build of the same code: 121–134 GFLOP/s single-threaded (GCC's register allocation of the micro-kernel is
  better); clang-cl to be measured.
- Both operands at run time (`BM_Matmul`, B packed per work item): 123 GFLOP/s (1 thread), 1337 GFLOP/s (all), 1024³.

### GEMV with 16-bit weights (1 × 4096 × 4096, decode-like)

| weights | 1 thread | all |
|---|---:|---:|
| f16 | 21.9 GB/s | 234 GB/s |
| bf16 | 20.4 GB/s | 210 GB/s |

(GB/s of weights read per run; 32 MiB of weights mostly stay in the 30 MiB L3 between runs, so "all" exceeds DRAM
bandwidth.)

### Memory-bound kernels (f32, 4M elements unless noted)

| kernel | 1 thread | all |
|---|---:|---:|
| add (3 arrays) | 24.7 GB/s | 328 GB/s |
| gelu(x·y + x)·0.5, fused into one pass | 14.1 GB/s | 177 GB/s |
| same chain without fusion (before) | 5.0 GB/s | 55 GB/s |
| softmax 1024 × 1024 | 15.2 GB/s | 186 GB/s |
| layer_norm 1024 × 4096 | 20.4 GB/s | 319 GB/s |

## Qualcomm Snapdragon 732G (2 × Cortex-A76 2.3 GHz + 6 × Cortex-A55 1.8 GHz), Android 13

NDK r30 Clang 21, `-O3`, kernel variant `neon`; tests and benchmarks run through `scripts/android_run.py`. Peak of
one A76: 2 FMA × 4 lanes × 2 flops × 2.3 GHz = 36.8 GFLOP/s.

| M × N × K | ii-cpu, 1 thread (A76) | ii-cpu, all 8 cores |
|---|---:|---:|
| 256 × 256 × 256 | 26.0 | — |
| 512 × 512 × 512 | 28.3 | 55 |
| 1024 × 1024 × 1024 | 27.7 | 71 |
| 64 × 4096 × 4096 | 28.2 | 65 |
| 512 × 4096 × 4096 | 28.3 | 79 |
| 1 × 4096 × 4096 (GEMV) | 5.7 (11.4 GB/s) | 3.8 (7.7 GB/s) |

77% of peak on a big core. f16 GEMV: 7.6 GB/s (1 thread), 7.2 GB/s (all).

## Gaps and plan

- **f32 GEMV** is ~80% of OpenBLAS (one core and all cores). Plan (stage 4, together with quantized GEMV, which is
  where decode time goes): several B panels in flight per micro-kernel call (more memory-level parallelism), tuned
  prefetch distance, and work items sized by bandwidth rather than flops.
- **GEMV on big.LITTLE**: all 8 cores are slower than one A76 — memory-bound work on the LITTLE cores adds contention
  without bandwidth. Plan: pick the thread count for memory-bound kernels by measurement (autotuning, stage 4) and
  prefer performance cores.
- **Small GEMMs on one core** (256³: 122 vs 147 GFLOP/s): packing and per-item overhead dominate; to be revisited
  with direct (unpacked) kernels for small K.
- **MSVC** is 8–15% slower than GCC on the same kernels; options: compile ISA kernels with clang-cl on MSVC builds,
  or hand-written micro-kernels.
