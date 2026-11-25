# ii — Architecture

> Status: design draft. This document describes the target architecture. Implementation order lives in
> [roadmap.md](roadmap.md); the testing and measurement strategy lives in
> [testing-and-benchmarks.md](testing-and-benchmarks.md).

---

## 1. Goals, non-goals, principles

### Goals

1. **Neural network inference** (first and foremost) and **training** — one library, one internal representation.
2. **Runs everywhere**: from weak x86/ARM machines without a GPU (and without AVX) to many-core servers and discrete
   accelerators. Cross-platform from day one: Windows, Linux, macOS; Android next; then iOS and WebAssembly.
   A single x86-64 binary runs on any x86-64 CPU (runtime ISA dispatch).
3. **Chooses the backend itself**: own `ii-*` backends and third-party ones (TensorRT, ONNX Runtime, OpenVINO,
   CoreML, oneDNN, …), taking into account device capabilities, op costs, memory capacity and transfer costs.
4. **Own backend ii** that runs everywhere, is correct (has a reference) and fast (targets expressed as % of the
   hardware roofline).
5. **Easy to embed**: zero mandatory dependencies, no global state, injectable allocator / threads / logger, stable
   C ABI, static and shared builds, `find_package(ii)` / `add_subdirectory`.
6. **Well designed and documented**: explicit layers, explicit invariants, every significant decision recorded.
7. **`ii` CLI** to run, inspect, benchmark, convert and quantize models.

### Non-goals (at least initially)

- Replacing PyTorch as a research framework driven by dynamic Python code.
- Multi-node distributed training. Multi-device training on one machine — later.
- Big-endian platforms (little-endian only, stated explicitly).

### Principles (used to settle design disputes)

| # | Principle | Consequence |
|---|-----------|-------------|
| P1 | **Correctness before speed** | Every fast path has a reference implementation and a differential test against it. |
| P2 | **One IR for everything** | Import, optimization, inference, backward pass and optimizer step are all graphs of the same IR. Every optimization and every backend automatically works for training too. |
| P3 | **A backend only has to implement primitives** | Composite ops (softmax, layer_norm, attention, …) are decomposed into primitives unless the backend implements them natively. New backends are cheap. |
| P4 | **Always runs** | `ii-ref` implements every op on any CPU; the planner can always fall back to it. |
| P5 | **Zero mandatory core dependencies** | Everything third-party is optional and, as a rule, shipped as a plugin (separate .so/.dll/.dylib). |
| P6 | **No global state** | Everything lives in `ii::Context`; allocator, thread pool and logger are injectable. No registration through static initializers. |
| P7 | **Zero allocations in steady-state `run()`** | All memory is planned at compile time; verified by a test with a counting allocator. |
| P8 | **Measurability** | Every hot path has a benchmark; performance targets are expressed as % of peak FLOPS / memory bandwidth. |
| P9 | **Explainability** | It is always possible to find out why a backend was chosen, how the graph was partitioned and where the time went (`ii inspect --plan`, profiler). |

---

## 2. Layers

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  tools/ii (CLI)                     bindings: C, Python*, C#*, Rust* (*later)   │
├──────────────────────────────────────────────────────────────────────────────┤
│  Public API         C++: include/ii/*.h            C ABI: include/ii/c_api.h    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Domain             ii::llm   — architectures, tokenizers, KV cache, sampling    │
│                     ii::nn / ii::train — modules, autodiff, optimizers, loops    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Runtime            Context · Model · Program · Session                         │
│                     Planner (backend selection) · MemoryPlanner · Executor ·     │
│                     Profiler                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Compiler           IR · OpSet · ShapeInference · Passes · Autodiff · Patterns  │
├──────────────────────────────────────────────────────────────────────────────┤
│  Formats            GGUF · safetensors (+HF config) · ONNX · .ii (native)       │
├──────────────────────────────────────────────────────────────────────────────┤
│  Backend API ──┬── own:     ii-ref · ii-cpu · ii-cuda · ii-vulkan · ii-metal     │
│                └── plugins: tensorrt · onnxruntime · openvino · coreml · …       │
├──────────────────────────────────────────────────────────────────────────────┤
│  Core               Status/Result · Log · Allocator · ThreadPool · CpuInfo ·     │
│                     MappedFile · DynLib · f16/bf16 · SmallVector · Json · Hash  │
│  Platform           Windows · Linux · macOS · Android · iOS · Wasm               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Dependency rules** (enforced by CMake targets — each layer is its own object library):

- Dependencies point downward only. `core` knows nothing about anything above it.
- All OS-specific code lives in `core/platform/` behind small interfaces (memory mapping, dynamic library loading,
  CPU topology, thread affinity, high-resolution clocks, cache directories). No `#ifdef _WIN32` outside of it.
- `formats` depends on `ir`, not on `runtime` (a loader just builds a graph plus references to weights).
- A backend depends on `backend_api`, `ir` (read-only) and `core`. Never on `runtime` internals or on another backend.
- `llm` and `train` use only public / semi-public interfaces of `runtime` and `ir` — this guarantees the API is
  expressive enough for third-party users.

---

## 3. Public API object model

```
Context      devices, backends, thread pool, allocators, logger, tuning cache   (1 per application)
  └─ Model   graph + weights (mmap), immutable                                  (shared)
      └─ Program   compiled plan for concrete devices, immutable                (shared)
           └─ Session   execution state: activation arena, KV cache,          (1 per thread/request)
                        input/output bindings
```

| Object | Thread safety | Creation cost |
|--------|---------------|---------------|
| `Context` | thread-safe | expensive (device discovery) |
| `Model` | immutable → safe | cheap (mmap, lazy reads) |
| `Program` | immutable → safe | expensive (passes, planning, weight packing, engine builds) — cached |
| `Session` | **not** thread-safe, one per thread | cheap (arena allocation) |

This is what matters most for embedding and servers: weights and the compiled plan are shared, while concurrent
requests each run in their own `Session`.

### Example: generic inference (C++)

```cpp
#include <ii/ii.h>

int main() {
    auto ctx     = ii::Context::create().value();                 // discovers devices and backends
    auto model   = ii::Model::load(ctx, "resnet50.onnx").value();
    auto program = model.compile({.policy = ii::Policy::Latency}).value();
    auto session = program.new_session().value();

    ii::Tensor image = ii::Tensor::wrap(pixels, ii::f32, {1, 3, 224, 224});  // zero-copy
    auto out = session.run({{"input", image}}).value();
    std::span<const float> logits = out[0].data<float>();
}
```

### Example: LLM

```cpp
auto llm  = ii::llm::Llm::load(ctx, "qwen2.5-1.5b-instruct-q4_k_m.gguf").value();
auto chat = llm.new_chat({.temperature = 0.7f, .top_p = 0.9f});
chat.send("Hi! Who are you?", [](std::string_view piece) { std::cout << piece << std::flush; });
```

### Example: training

```cpp
namespace nn = ii::nn;

struct Mlp : nn::Module {
    nn::Linear fc1{this, 784, 256}, fc2{this, 256, 10};
    ii::Var forward(ii::Var x) { return fc2(ii::relu(fc1(x))); }
};

Mlp net;
auto step = ii::train::compile_step(ctx, net, ii::loss::cross_entropy, ii::optim::AdamW{.lr = 1e-3f}).value();
for (auto& [x, y] : loader) step.run(x, y);   // forward + backward + AdamW — one compiled Program
```

### Example: C ABI

```c
ii_context* ctx;  ii_model* model;  ii_program* prog;  ii_session* s;
II_TRY(ii_context_create(NULL, &ctx));
II_TRY(ii_model_load(ctx, "model.onnx", &model));
II_TRY(ii_program_compile(model, NULL, &prog));
II_TRY(ii_session_create(prog, &s));
II_TRY(ii_session_bind_input(s, "input", &tensor_desc));
II_TRY(ii_session_run(s));
```

### Error handling

- The library **never throws across its boundaries** (many embedding applications build with `-fno-exceptions`):
  `ii::Status` (code + message) and `ii::Result<T>` (an `std::expected`-like type of our own, so C++23 is not
  required). `II_TRY(expr)` propagates errors.
- `II_ASSERT` checks invariants in debug builds; `II_CHECK` always, for "impossible" states.
- In the C ABI — `ii_status` codes + `ii_last_error_message()` (thread-local).

---

## 4. Tensors and data types

### DType

| Class | Types |
|-------|-------|
| float | `f32`, `f16`, `bf16`, `f64` (ref/tests only), `f8e4m3`, `f8e5m2` (later) |
| int | `i8`, `u8`, `i16`, `i32`, `i64`, `bool` |
| block-quantized | `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q4_k`, `q5_k`, `q6_k`, `q8_k`, … — byte-compatible with GGUF layouts so weights load zero-copy via mmap |
| per-channel quantized | `i8`/`u8`/`i4` + separate scale/zero_point tensors (ONNX QDQ, GPTQ/AWQ) |

Block types are described by `DTypeTraits { block_size, bytes_per_block, … }`; only the last dimension may be
quantized, and strides along it are measured in blocks.

### Shape, Tensor, Buffer, Device

- `Shape` — `SmallVector<int64_t, 8>` (no heap), rank ≤ 8. In the IR a dimension may be symbolic (see §5).
- `Buffer` — a reference-counted memory region on a `Device`. Sources: own allocation, file mmap, external user
  memory (zero-copy `Tensor::wrap`), backend memory (GPU).
- `Tensor` = `{Buffer, offset, dtype, shape, strides}`. `reshape`/`permute`/`slice`/`expand` are copy-free views.
- Default alignment is 64 bytes (cache line / AVX-512).
- A public `Tensor` is always logically row-major; physical layouts (NCHWc, packed GEMM panels) are a backend's
  internal business and never leak out.

---

## 5. IR (intermediate representation)

- **`Graph`** — an SSA graph in topological order: `Node`s (operations) produce `Value`s.
- **`Value`** — `{id, TensorType{dtype, SymShape}, producer, users}`. Kinds: input, output, constant (references a
  `Buffer`, possibly mmap-ed), intermediate, **state**.
- **State values** are read at the start of `run()` and written at the end; they live in the `Session` between runs.
  One mechanism covers transformer KV caches, RNN hidden states and, for training, parameters and optimizer moments.
- **Symbolic dimensions**: a `Dim` is a constant or an expression over symbols (`N`, `seq`, `past`): affine
  expressions and products — enough for reshaping attention heads. Symbols may have upper bounds (`seq ≤ 4096`).
- **Attributes** — a typed `variant` (int, float, string, lists, tensor, subgraph).
- **Subgraphs** — for `if`/`loop` (required by some ONNX models). The LLM generation loop lives in the runtime, not
  in the graph.
- **`Graph::verify()`** — checks SSA, types, shapes, topological order. Runs after every pass in debug builds.
- **Text format** for debugging and pass tests:

  ```
  %3 : f32[N, 4096] = matmul(%x, %w1) {trans_b = true}
  %4 : f32[N, 4096] = silu(%3)
  ```
  Pass tests are written as "input → expected text" (FileCheck style) plus numerical equivalence.

---

## 6. Operation set (OpSet)

### Primitives and composites

**Primitives** (every backend implements them or declines the node):

| Group | Ops |
|-------|-----|
| unary | neg, abs, exp, log, sqrt, rsqrt, recip, sin, cos, tanh, sigmoid, erf, relu, floor, ceil, round, sign, not |
| binary (broadcast) | add, sub, mul, div, pow, max, min, mod, eq, ne, lt, le, gt, ge, and, or, xor |
| other elementwise | where, cast, clamp |
| reductions | sum, mean, max, min, prod, argmax, argmin |
| linear algebra | matmul (batched, broadcast, transposes, quantized B) |
| convolution | conv (N-d, groups, dilation, padding), conv_transpose, max_pool, avg_pool |
| data movement | reshape, permute, slice, expand, concat, split, pad, gather, scatter, tile, flip, cumsum |
| creation | iota, fill, random_uniform/normal (dropout/initialization) |
| quantization | quantize, dequantize |
| control flow | if, loop |

**Composites** (have a decomposition rule; a backend may implement them natively — which is what is done for
speed): gelu, silu, softmax, log_softmax, layer_norm, rms_norm, group_norm, batch_norm, linear (matmul + bias),
rope, sdpa (scaled dot-product attention: mask, causal, GQA, KV cache), swiglu, cross_entropy, adamw_step, …

### OpSchema — the single source of truth about an op

```cpp
struct OpSchema {
    std::string_view name;
    int since_version;
    Arity inputs, outputs;
    TypeConstraints types;
    ShapeFn     infer_shape;   // symbolic shape inference
    DecomposeFn decompose;     // nullptr for primitives
    GradFn      gradient;      // backward rule, expressed in ii ops
    CostFn      cost;          // flops and bytes — for the planner and roofline reports
    std::string_view doc;      // the op reference documentation is generated from this
};
```

**Adding an op** = schema + reference kernel in `ii-ref` + conformance test + (if differentiable) gradient rule and
a finite-difference check. Checklist: `docs/guides/adding-an-op.md`.

---

## 7. Compiler: pass pipeline

```
import → canonicalize → shape inference → constant folding → DCE/CSE → algebraic simplification
      → pattern matching / fusion → [autodiff, when training] → backend partitioning
      → backend lowering (layouts, weight packing, decomposition of unsupported composites)
      → memory planning → ExecutionPlan
```

- **PassManager** with explicit pass dependencies; `verify()` after every pass in debug builds.
- **Constant folding** executes subgraphs through `ii-ref` — no separate "evaluator".
- **Pattern matching** — a small declarative DSL: `matmul → add(bias) → gelu` ⇒ `linear{act=gelu}`;
  recognizing "unrolled" attention from ONNX ⇒ `sdpa`; `x * rsqrt(mean(x²) + eps) * w` ⇒ `rms_norm`.
- **Pass invariant**: a pass never changes results beyond tolerance — verified automatically in tests (the graph is
  executed before and after the pass on random inputs).

---

## 8. Backends

### Two kinds of backends behind one interface

1. **Kernel backends** (`ii-ref`, `ii-cpu`, `ii-cuda`, `ii-vulkan`, `ii-metal`, plus oneDNN/XNNPACK/cuBLAS wrappers):
   provide kernels per op; scheduling and memory are owned by the ii runtime. Implemented on top of the helper base
   class `KernelBackend` with a kernel table `(op, dtype constraints, ISA) → kernel`.
2. **Graph backends** (TensorRT, ONNX Runtime, OpenVINO, CoreML): take whole subgraphs and compile them with their own
   engine (similar to ORT `GetCapability` / TFLite delegates).

```cpp
namespace ii::backend {

struct DeviceInfo {
    DeviceKind   kind;            // cpu, gpu, npu, …
    MemoryDomain domain;          // which memory this device addresses (see below)
    std::string  name;
    uint64_t     memory_bytes;
    PerfModel    perf;            // peak FLOPS per dtype, memory and link bandwidth (estimated or measured)
};

enum class Support { No, Native, Fused };

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string_view name() const noexcept = 0;
    virtual std::span<Device* const> devices() noexcept = 0;

    /// Whether the backend can execute the node on the device (dtype, attributes, shapes).
    virtual Support supports(const ir::Node&, const Device&) const = 0;
    /// Graph backends: the maximal subgraphs they are willing to take as a whole.
    virtual std::vector<ir::NodeSet> claim(const ir::Graph&, const Device&) const { return {}; }
    /// Time estimate (by default — roofline from OpSchema::cost and the device PerfModel).
    virtual Cost estimate(const ir::Node&, const Device&) const;
    /// Compiles a partition into an executable object.
    virtual Result<std::unique_ptr<Executable>> compile(const ir::Partition&, Device&, const CompileOptions&) = 0;
};

class Executable {
public:
    virtual ~Executable() = default;
    virtual MemoryRequirements memory() const = 0;   // scratch/workspace — accounted for by the memory planner
    virtual Status execute(ExecContext&) = 0;         // allocation-free
};

} // namespace ii::backend
```

### Devices, memory domains, synchronization

- A `Device` allocates `Buffer`s, copies data (host↔device, device↔device), provides `Stream`s and `Event`s.
- **Memory domains.** Several backends may work on the same physical memory: `ii-ref`, `ii-cpu` and ORT-CPU share
  the host domain; `ii-cuda`, TensorRT and cuBLAS share the domain of one CUDA device. Buffers move between backends
  of the same domain **without copies** — only stream synchronization. Unified-memory devices (Apple, iGPUs) expose
  the host domain as well.
- Cross-domain transitions are explicit `transfer` steps in the plan, with synchronization events.

### TensorRT backend (plugin)

- Translates an ii IR partition directly into a TensorRT `INetworkDefinition` through the C++ API — no ONNX
  round-trip, so no information is lost and no ONNX exporter is required.
- Dynamic shapes map to TensorRT optimization profiles derived from symbol bounds (`seq ≤ 4096`).
- Engine builds are slow, so built engines are serialized to `<cache>/ii/trt/` keyed by
  `(subgraph hash, shape profile, GPU, driver, TensorRT version, precision)`.
- Shares the CUDA device, streams and memory domain with `ii-cuda`: a plan can interleave TensorRT partitions with
  `ii-cuda` kernels (e.g. ops TensorRT does not support, KV-cache updates, sampling) without host round-trips.
- Precision modes (fp16/bf16/int8/fp8) follow `CompileOptions` precision policy; results are verified against
  `ii-ref` by the same differential tests as every other backend.

### Plugins

- Third-party backends are built as separate `ii-backend-<name>` shared libraries so the core never depends on
  their SDKs.
- The stable boundary is a C struct `ii_backend_plugin_v1` (a table of function pointers) + an exported
  `ii_backend_plugin_entry()`; ABI version handshake. Inside the plugin a C++ wrapper implements `Backend`.
- Plugins are looked up next to the library, in `II_PLUGIN_PATH` and in paths given in `Context::Options`.
  A missing SDK runtime (e.g. no TensorRT DLLs on the user's machine) makes the plugin report itself unavailable —
  never a crash or a load failure of the core.
- Built-in backends are registered **explicitly** in `Context::create()` (no static initializers — they break with
  static linking and cause initialization-order problems).

### Planned backends

| Backend | Kind | Devices | Stage |
|---------|------|---------|-------|
| `ii-ref` | own, kernel | any CPU | 1 — reference and last-resort fallback |
| `ii-cpu` | own, kernel | x86-64 (SSE2…AVX-512/AMX), ARM64 (NEON…SVE), generic C++ | 2 |
| `ii-cuda` | own, kernel (+ cuBLASLt) | NVIDIA | 5 |
| tensorrt | plugin, graph | NVIDIA | 5 |
| `ii-vulkan` | own, kernel | any GPU with Vulkan 1.2+ (NVIDIA/AMD/Intel/mobile) | 6 |
| `ii-metal` | own, kernel | Apple | 6 |
| onnxruntime | plugin, graph | whatever its EPs support — covers the long tail of ONNX ops | 6 |
| openvino | plugin, graph | Intel CPU/iGPU/NPU | 6 |
| coreml | plugin, graph | Apple ANE/GPU | 6 |
| onednn / xnnpack / kleidiai | plugin, kernel | x86 (AMX) / ARM | 6, driven by benchmark data |

---

## 9. Automatic backend selection (Planner)

Goal: minimize estimated time (or energy — per policy) subject to memory and capability constraints.

**Algorithm:**

1. **Discovery.** `Context` queries backends for devices, `DeviceInfo` and supported dtypes. The `PerfModel` comes
   from a table of known devices or is measured by a micro-benchmark on first run and cached.
2. **Candidates.** For each node: `{(backend, device) | supports ≠ No}`; `ii-ref@cpu` is always present.
   Graph backends claim subgraphs via `claim()` — these become "super-nodes" with their own estimate.
3. **Cost.** `t(node, backend, device)` from `estimate()` (default roofline: `max(flops/peak, bytes/bw)` + launch
   overhead); a transfer on an edge crossing memory domains costs `bytes / link_bw + latency`; edges within one
   domain cost only synchronization.
4. **Initial assignment** — greedy in topological order (HEFT-like): minimize node time + transfers from already
   assigned predecessors.
5. **Refinement** — local search: merge "islands" (a lone node on another device between large partitions) when this
   reduces total time; eliminates transfer ping-pong.
6. **Memory constraint.** Weights + peak activations per device ≤ `capacity × (1 − reserve)`. On overflow, repeated
   blocks (transformer layers) are moved one by one to the next cheapest device — **automatic layer offload** when
   the model does not fit into VRAM.
7. **Autotuning (optional).** `TuneMode::Measure`: run the top-K plans on real data and pick the best. Results are
   cached by `(model hash, shapes, device, driver version, ii version)` in `<cache>/ii/tuning.json`. Kernel
   parameters (block sizes, decode thread count) are tuned the same way.

**Policies and overrides:** `Policy::{Latency, Throughput, LowPower, LowMemory}`, allow/deny lists of backends and
devices, memory limits; overridable via `CompileOptions`, the CLI (`--backend tensorrt,cuda,cpu`) and `II_BACKENDS`.

**Explanation:** `ii inspect model --plan` prints partitions, devices, estimates and the reasons backends declined
nodes ("ii-vulkan: op `scatter` with dtype i64 is not supported").

---

## 10. Runtime: memory and execution

- **Memory planning**: value lifetimes → offsets in an arena via "greedy by size" (as in TFLite); in-place for
  elementwise ops on dying inputs; one arena per device; kernel workspaces are planned like ordinary values.
  Constants (weights) live outside the arena (mmap or packed copies).
- **ExecutionPlan** — a flat array of steps with pre-bound arguments (pointers, shapes, strides); execution is a walk
  over the array with no per-op virtual dispatch and no allocations.
- **Dynamic shapes**: the plan contains a small "shape program" (bytecode) computing concrete sizes from symbol
  values. If symbols have upper bounds, the arena is planned once for the maximum; otherwise the memory plan is
  recomputed when exceeded (rare, amortized), with a cache of shape buckets.
- **I/O binding**: the user may pass their own input and output buffers (zero-copy), including GPU buffers.
- **Profiler**: time per step / partition / transfer, exported as Chrome trace JSON (viewable in Perfetto):
  `ii run model --profile trace.json`.

---

## 11. The `ii-cpu` backend

### Runtime ISA dispatch

- Kernels in `src/backends/cpu/kernels/<isa>/` are compiled **multiple times** with different flags:
  `generic` (SSE2 / NEON / plain C++), `avx2` (AVX2 + FMA + F16C), `avx_vnni`, `avx512` (F/BW/VL/DQ),
  `avx512_vnni_bf16`, `amx`, `neon_dotprod`, `neon_i8mm`, `sve`; later `rvv` and `wasm_simd128`.
- `CpuInfo` (cpuid / `getauxval` / `sysctl` / Windows APIs) picks the best level; the kernel table is filled with
  function pointers. Forced level: `II_CPU_ISA=generic|avx2|…` — so all paths can be tested on one machine.
- **ODR trap**: inline functions / templates instantiated in a TU built with `-mavx2` may be picked by the linker for
  the generic path → SIGILL on older CPUs. Fix: all ISA-specific code (templates included) lives in a dedicated
  namespace `ii::cpu::II_ISA_NS`; shared headers enter those TUs only through it.
- Compiler differences (GCC/Clang `-m…` flags vs MSVC `/arch:…`, missing per-ISA flags on MSVC for some extensions)
  are handled in one CMake helper; on MSVC, Clang-only kernels are compiled with clang-cl where required.

### SIMD layer

A thin in-house abstraction `Vec<T, Isa>` (load/store/fma/reduce/convert) on top of intrinsics — for elementwise ops
and reductions. Hot micro-kernels (GEMM, quantized dot products) are written directly with intrinsics.

### GEMM (the heart of performance)

- BLIS approach: pack A and B into panels, an `MR×NR` register micro-kernel (e.g. AVX2 f32 6×16, AVX-512 14×32,
  NEON 8×12), `MC/KC/NC` cache blocking from cache sizes reported by `CpuInfo`.
- **Weights are pre-packed at compile time** (B is a constant): packing is not repeated on every run.
- f16/bf16 weights — widened to f32 during packing or on the fly (F16C), f32 accumulation.
- Parallelization over M/N blocks with awareness of the shared L3.

### Quantized kernels (the key to weak hardware)

- **LLM decode (GEMV, batch 1)** is memory-bound: dequantize on the fly inside the dot product; target ≥ 85–90% of
  memory bandwidth.
- **Prefill / batch** is compute-bound: activations are quantized to int8 per block, then int8×int4/int8 dot products
  (`vpdpbusd` AVX-VNNI/AVX512-VNNI, `sdot`/`smmla` on ARM).

### Attention, normalizations, elementwise

- `sdpa`: tiled flash-attention for CPU (online softmax, attention matrix never materialized), GQA, KV cache.
- `rms_norm`, `layer_norm`, `rope`, `softmax`, `swiglu` — native fused kernels.
- Chains of elementwise ops are fused into a single pass over memory (loop-level fusion).
- Approximations (`exp`, `tanh`, `erf`) are explicit, with a documented max error and tests; `-ffast-math` is **not**
  used globally (it breaks NaN/Inf).

### Threads

- Own `ThreadPool`: `parallel_for(range, grain, fn)`; the calling thread participates; brief spinning, then sleeping
  on a futex / condition variable; target launch overhead < 5 µs.
- **Hybrid CPUs** (P/E cores: Intel Core Ultra, Apple M-series, big.LITTLE on Android): static equal splits leave fast
  cores idle, so work is handed out dynamically in chunks via an atomic counter. Optional pinning to performance cores
  for latency-sensitive work (per-OS topology: `GetLogicalProcessorInformationEx`, sysfs, `hw.perflevel*`).
- For memory-bound decode the optimal thread count is often lower than the core count — picked by autotuning.
- External pool: an embedding application may supply its own implementation of the `ii::Executor` interface (an
  engine's job system).
- **Deterministic mode**: fixed reduction order (static split) → bitwise reproducibility for the same thread count.

---

## 12. Own GPU backends

- **ii-cuda** — maximum performance on NVIDIA: own kernels (quantized matmul, attention, fused elementwise) +
  cuBLASLt for GEMM, NVRTC for generated fused kernels, CUDA Graphs to cut launch overhead in decode. Shares its
  memory domain with the TensorRT plugin.
- **ii-vulkan** — the universal GPU backend (Windows/Linux/Android, all vendors). GLSL → SPIR-V at build time; heavy
  ops (matmul with cooperative matrices, attention, quantized matmul) are hand-written; elementwise chains are
  generated and compiled at runtime (pipeline cache on disk).
- **ii-metal** — Apple, unified memory (no host↔device copies).

Shared GPU part: a generator of fused elementwise kernels from an IR subgraph (one codegen, several dialects:
CUDA C / GLSL / MSL).

---

## 13. LLM subsystem (`ii::llm`)

Built **on top of** the common IR and runtime rather than as a separate engine — so it automatically gets every
backend, the planner, offload and the profiler.

- **Architectures as code**: `src/llm/arch/llama.cpp`, `qwen.cpp`, `gemma.cpp`, `phi.cpp`, … build the graph with the
  builder API from hyper-parameters in GGUF metadata or HF `config.json`. MoE (Mixtral, Qwen-MoE) — later.
- **KV cache** — session state values; later a paged KV cache for continuous batching.
- **Prefill and decode** — one graph with a `seq` symbol; the planner may choose different kernels / thread counts for
  `seq = 1`.
- **Tokenizers**: byte-level BPE (GPT-2 / Llama 3 / Qwen), SentencePiece (BPE / Unigram); from GGUF or
  `tokenizer.json`.
- **Chat templates**: an interpreter for the subset of Jinja2 that real GGUF templates use.
- **Sampling**: greedy / temperature / top-k / top-p / min-p / repetition penalty chain; later grammars (JSON),
  speculative decoding.
- **Token streaming** via callback; generation cancellation.

---

## 14. Training (`ii::nn`, `ii::train`)

Training is implemented last (see roadmap), but the core is designed for it from the start: state values, gradient
rules in `OpSchema`, and the "one IR" principle.

- **Autodiff as a compiler pass**: `ir::differentiate(graph, loss, params)` builds the backward graph from
  `OpSchema::gradient`. Gradients are ii ops too ⇒ they run on every backend and go through the same optimizations.
  Fused backward ops for hot spots (`softmax_ce_grad`, `conv_backward_*`, `sdpa_backward`).
- **A training step is one Program**: forward + backward + optimizer step (`adamw_step` — a fused kernel); parameters
  and moments are session state values ⇒ zero allocations per step.
- **Frontend**: `nn::Module` (Linear, Conv2d, Embedding, LayerNorm, RMSNorm, Attention, …), `ii::Var` — a symbolic
  builder-API variable; user functions are traced into a graph (like `torch.compile` / JAX `jit`).
- **Later**: mixed precision (bf16 compute + f32 master weights, loss scaling for f16), gradient checkpointing (a pass
  inserting recomputation), **LoRA/QLoRA** — training adapters on top of a frozen quantized base (especially valuable
  on weak hardware), checkpoints in safetensors.

---

## 15. Model formats and security

| Format | Purpose | Implementation |
|--------|---------|----------------|
| GGUF | quantized LLMs | header/metadata parser, tensor mmap, GGML type → ii dtype mapping |
| safetensors + `config.json` | HF models | JSON header + mmap; architecture from `config.json` |
| ONNX | general models (CV, audio, encoders) | **own protobuf wire-format decoder** (no libprotobuf dependency), external data, `onnx op@opset → ii ops` mapping table |
| `.ii` | native, fast startup | header + sections; graph in a compact TLV encoding, weights aligned for mmap, optional weights pre-packed for a specific ISA; versioned format |

**Loader security**: a model is untrusted input. All sizes and offsets are checked with overflow-safe arithmetic,
no out-of-bounds reads; every parser (ONNX, GGUF, safetensors, `.ii`, JSON, tokenizers, Jinja) is fuzzed
(libFuzzer) in CI.

---

## 16. Platforms, embedding and build

### Platform tiers

| Tier | Platforms | Meaning |
|------|-----------|---------|
| 1 (from stage 0) | Windows x86-64 (MSVC, clang-cl, MinGW-GCC), Linux x86-64 and ARM64 (GCC, Clang), macOS ARM64 + x86-64 (Apple Clang) | every commit builds and passes all tests in CI |
| 2 (from stage 2) | Android ARM64 (+ x86-64 emulator), NDK/Clang | builds in CI; tests run on an emulator / device |
| 3 (later) | iOS, WebAssembly, Linux RISC-V, Windows ARM64 | builds in CI; tests as infrastructure allows |

Minimum versions (proposed): Windows 10 1809+, glibc 2.31+ (Ubuntu 20.04), macOS 13.3+ (C++20 `<format>` in libc++), Android API 26+,
CUDA compute capability 7.5+ (matches TensorRT 10).

### Build

- **CMake ≥ 3.25**, `CMakePresets.json` with presets per platform/compiler; targets `ii::ii` (library), `ii` (CLI);
  `find_package(ii)`, `add_subdirectory`, `FetchContent`, pkg-config.
- **Options**: `II_BUILD_SHARED`, `II_BUILD_CLI`, `II_BUILD_TESTS`, `II_BUILD_BENCHMARKS`, `II_WITH_TRAINING`,
  `II_WITH_LLM`, `II_BACKEND_CUDA|TENSORRT|VULKAN|METAL`, `II_CPU_ISAS="generic;avx2;avx512;…"`, `II_MINIMAL`.
- **Minimal build** for embedded targets: inference only, CPU only, only the ops required by given models (like the
  ORT minimal build) — to control binary size.
- **ABI**: the C ABI is stable within a major version; the C++ API lives in inline namespace `ii::v1`; symbols are
  hidden by default (`-fvisibility=hidden`, explicit `II_API`); public headers contain no third-party types (pimpl).
  The C ABI also solves MinGW ↔ MSVC interop: a DLL built with one compiler is usable from the other.
- **Configuration without globals**: `Context::Options { allocator, executor (thread pool), log_sink, cache_dir,
  plugin_paths, … }`; `II_*` environment variables only override defaults and are documented.
- **Windows naming detail**: the library and the CLI are both called `ii`, so the CMake targets differ
  (`ii` / `ii_cli`) and PDB names are set explicitly to avoid `ii.pdb` collisions.

---

## 17. The `ii` CLI

```
ii info                          hardware, ISA, devices, backends, plugins; --roofline measures peaks
ii run <model> [-i name=file]    run any model (.npy/.npz/raw/images), output .npy/JSON/top-k
ii generate <llm> -p "…"         text generation
ii chat <llm>                    interactive chat
ii bench <model> [--llm]         benchmark: latency p50/p90/p99, throughput, memory; for LLMs pp/tg tokens/s
ii inspect <model> [--plan]      graph, ops, parameters, shapes, backend partitioning and reasons
ii convert <in> <out.ii>         conversion to the native format (+ pre-packing)
ii quantize <in> <out> --type q4_k
ii check-backends [<model>]      verify every available backend against the reference on this machine
ii train <config>                training from a config (later)
```

Common flags: `--backend`, `--device`, `--threads`, `--policy`, `--profile trace.json`, `--log-level`, `--json`.

---

## 18. Repository layout

```
ii/
├── CMakeLists.txt, CMakePresets.json, cmake/      build, toolchains, package export
├── include/ii/                                     public headers (ii.h, c_api.h, tensor.h, …)
├── src/
│   ├── core/          status, log, alloc, thread_pool, cpu_info, half, json, small_vector
│   │   └── platform/  mapped_file, dynlib, topology, affinity, clock, paths — per OS
│   ├── ir/            graph, value, node, types, sym_shape, verify, text format, builder
│   ├── ops/           OpSchemas: shapes, decompositions, gradients, costs
│   ├── passes/        canonicalization, constant folding, DCE/CSE, fusion, autodiff, layouts
│   ├── runtime/       context, model, program, session, planner, memory_planner, executor, profiler
│   ├── backend_api/   Backend, Device, Buffer, Stream, KernelBackend, plugin ABI
│   ├── backends/
│   │   ├── ref/       reference kernels
│   │   ├── cpu/       ii-cpu: simd/, gemm/, quant/, kernels/<isa>/
│   │   ├── cuda/  vulkan/  metal/
│   │   └── plugins/   tensorrt/, onnxruntime/, openvino/, coreml/, …
│   ├── formats/       gguf/, safetensors/, onnx/ (+ protobuf wire), native/
│   ├── llm/           arch/, tokenizer/, sampling/, chat_template/
│   ├── train/         nn/, optim/, loss/, data/
│   └── capi/          C ABI implementation
├── tools/ii/          CLI
├── tests/             unit/, ops/, passes/, backends/, models/, fuzz/, support/ (comparison harness)
├── bench/             micro/ (Google Benchmark), e2e/, baselines/
├── examples/          minimal embedding examples (built in CI)
├── scripts/           reference generation (PyTorch/NumPy), benchmark comparison
└── docs/              architecture, roadmap, testing-and-benchmarks, guides/, adr/
```

---

## 19. Code style and documentation

- **Language**: all code, comments, commit messages and documentation are in English
  (the planning document `roadmap.md` is the only exception).
- **C++20** (concepts, `std::span`, designated initializers); no C++ modules (tooling is immature).
- Naming: types `PascalCase`, functions and variables `snake_case`, constants `kPascalCase`, files
  `snake_case.h/.cpp`. Headers use `.h` for C++ too; the C ABI header is `ii/c_api.h`.
- `clang-format` (LLVM-based, 120 columns), `clang-tidy` (bugprone, performance, modernize, selected
  cppcoreguidelines), `-Wall -Wextra -Wpedantic -Werror` / `/W4 /WX` in CI.
- No RTTI dependence and no exceptions in the core; ownership via `std::unique_ptr` / intrusive refcounts for buffers.
- API docs — Doxygen `///` comments in public headers; conceptual docs in `docs/`: architecture, guides ("adding an
  op", "adding a backend", "adding an LLM architecture", "writing a kernel", "performance tuning"), an ADR for every
  significant decision.
- Examples in `examples/` are built and run in CI — documentation cannot silently rot.
- License: **MIT**. Every source file starts with an SPDX header: `// SPDX-License-Identifier: MIT`.

---

## 20. Key decisions (ADR summary)

| ADR | Decision | Why | Alternative |
|-----|----------|-----|-------------|
| 0001 | C++20, CMake | balance between modern features and toolchain support (NDK, older GCC, MSVC) | C++23 (`std::expected`) — later |
| 0002 | `Result<T>`, no exceptions across the API | embedding into `-fno-exceptions` code, C ABI | exceptions |
| 0003 | Own IR: primitives + composites with decomposition | cheap new backends, single path for training | ONNX as the IR — heavyweight, lacks the needed symbolic shapes and state |
| 0004 | `ii-ref` implements everything and is the oracle | P1, P4 | — |
| 0005 | Third-party backends are plugins with a C ABI | P5; third-party SDKs are huge and optional at runtime | static linking — optional |
| 0006 | Own protobuf decoder for ONNX | no heavy libprotobuf dependency | libprotobuf |
| 0007 | Byte-compatible quantization blocks with GGUF | zero-copy loading of the model ecosystem | own layouts + conversion |
| 0008 | Autodiff as a pass over the IR | training on every backend "for free" | tape-based eager autograd |
| 0009 | Explicit backend registration, no static initializers | robustness with static linking | auto-registration |
| 0010 | Kernels compiled per ISA + runtime dispatch | one binary for any x86/ARM | building with `-march=native` |
| 0011 | TensorRT integration via the network-definition API, sharing the CUDA memory domain with `ii-cuda` | no ONNX round-trip, mixed TRT/own-kernel plans without copies | export partitions to ONNX + TRT ONNX parser |
| 0012 | Tier-1 platforms from stage 0: Windows, Linux, macOS; OS code isolated in `core/platform` | portability is cheap early and expensive late | Windows-first, port later |
| 0013 | MIT license | maximally permissive for embedding | Apache-2.0 |

---

## 21. Risks

| Risk | Mitigation |
|------|------------|
| The project scope is huge | vertical slices: every stage ends with a working, measured result |
| Speed parity with mature libraries (llama.cpp, oneDNN) | focus on the few kernels that take 90% of the time; roofline targets; delegation to third-party backends |
| Long tail of ONNX ops | ONNX Runtime plugin as a fallback for exotic ops |
| Cost of GPU backends | shared elementwise codegen; TensorRT covers dense NVIDIA workloads early |
| TensorRT engine build time and version churn | on-disk engine cache with a strict key; plugin isolated behind a C ABI |
| Chat template compatibility (Jinja) | a subset + tests on real templates of popular models |
| Toolchain differences (MSVC / MinGW / Clang / Apple Clang / NDK) | CI on all of them from stage 0; C ABI at the boundary; OS code isolated in `core/platform` |
