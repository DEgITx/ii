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
    auto model   = ii::Model::load(ctx, "resnet50.onnx").value(); // stage 3; today: Model::from_text / GraphBuilder
    auto program = model.compile().value();                       // passes, backend choice, kernels
    auto session = program.new_session().value();

    ii::Tensor image = ii::Tensor::wrap(pixels, ii::DType::kF32, {1, 3, 224, 224});  // zero-copy
    auto out = session.run({{"input", image}}).value();
    std::span<const float> logits = out[0].values<float>();
}
```

### Example: LLM

```cpp
auto llm  = ii::llm::Llm::load(ctx, "qwen2.5-1.5b-instruct-q4_k_m.gguf").value();
auto chat = llm.new_chat({.temperature = 0.7f, .top_p = 0.9f}).value();
chat.send("Hi! Who are you?", [](std::string_view piece) { std::cout << piece << std::flush; return true; });
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
ii_context* ctx = NULL; ii_model* model = NULL; ii_program* prog = NULL; ii_session* s = NULL;
if (ii_context_create(NULL, &ctx) != II_OK ||
    ii_model_from_text(ctx, text, strlen(text), &model) != II_OK ||   /* ii_model_load: stage 3 */
    ii_model_compile(model, NULL, &prog) != II_OK ||
    ii_session_create(prog, &s) != II_OK ||
    ii_session_set_input(s, "input", &tensor) != II_OK ||             /* zero-copy */
    ii_session_run(s) != II_OK) {
    fprintf(stderr, "%s
", ii_last_error());
}
```

### Error handling

- The library **never throws across its boundaries** (many embedding applications build with `-fno-exceptions`):
  `ii::Status` (code + message) and `ii::Result<T>` (an `std::expected`-like type of our own, so C++23 is not
  required). `II_TRY(expr)` propagates errors.
- `II_ASSERT` checks invariants in debug builds; `II_CHECK` always, for "impossible" states.
- In the C ABI — `ii_status` codes + `ii_last_error()` (thread-local).

---

## 4. Tensors and data types

### DType

| Class | Types |
|-------|-------|
| float | `f32`, `f16`, `bf16`, `f64` (ref/tests only), `f8e4m3`, `f8e5m2` (later) |
| int | `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `bool` |
| block-quantized | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, `q2_k` … `q6_k`, `q8_k` — byte-compatible with GGUF layouts so weights load zero-copy via mmap |
| per-channel quantized | `i8`/`u8`/`i4` + separate scale/zero_point tensors (ONNX QDQ, GPTQ/AWQ) |

Every dtype is described by a constexpr `DTypeInfo { name, kind, block_size, block_bytes }` (`include/ii/dtype.h`).
Block types pack `block_size` consecutive elements of the innermost dimension into `block_bytes` bytes. Strides are
counted in elements for every dtype; a block-quantized tensor must be contiguous in its innermost dimension and every
view must start on a block boundary.

### Shape, Tensor, Buffer, Device

- `Shape` — up to 8 `int64_t` stored inline (no heap), rank ≤ 8; also used for strides. In the IR a dimension may be
  symbolic (`Dim`, see §5).
- `Buffer` — a reference-counted memory region on a `Device`. Sources: own allocation, file mmap, external user
  memory (zero-copy `Tensor::wrap`), backend memory (GPU).
- `Tensor` = `{Buffer, offset, dtype, shape, strides}`. `reshape`/`permute`/`slice`/`expand` are copy-free views.
- Default alignment is 64 bytes (cache line / AVX-512).
- A public `Tensor` is always logically row-major; physical layouts (NCHWc, packed GEMM panels) are a backend's
  internal business and never leak out.

---

## 5. IR (intermediate representation)

- **`Graph`** — an SSA graph in topological order: `Node`s (operations) produce `Value`s.
- **`Value`** — `{id, TensorType{dtype, Dims}, producer, uses}`. Kinds: input, constant (a `Tensor`, possibly
  mmap-ed), state, node output. Graph outputs are named references to values.
- **State values** are read at the start of `run()` and written at the end; they live in the `Session` between runs.
  One mechanism covers transformer KV caches, RNN hidden states and, for training, parameters and optimizer moments.
  A state updated by an in-place operation that reads it last (`slice_update` writing new keys into a preallocated
  KV cache) is updated in its own memory: no arena buffer and no copy at the end of the run (§10).
- **Symbolic dimensions**: a `Dim` (public, `include/ii/dim.h`) is a constant or an integer expression over symbols
  (`N`, `seq`, `past`) kept in a canonical polynomial form whose variables are symbols and floor divisions by
  constants — so `N*2 + N == 3*N` and conv/pool output sizes `(H + 2p - k) / s + 1` stay symbolic. Symbols are bound
  from the input shapes when a session runs (directly, or by solving affine dimensions such as `2*N + 1`); upper
  bounds (`seq ≤ 4096`) come with the LLM work.
- **Attributes** — a typed `variant` (bool, int, float, string, int/float/string lists, dtype, dims, tensor);
  subgraphs arrive with `if`/`loop`. Every attribute is declared in the operation schema with its kind and default;
  nodes always carry the complete, canonical attribute set.
- **Subgraphs** — ONNX `If` nodes whose condition is known when the model is loaded are inlined by the importer;
  general `if`/`loop` (control flow at run time) is planned together with data-dependent shapes. The LLM generation
  loop lives in the runtime, not in the graph.
- **`Graph::verify()`** — checks SSA, types, shapes, topological order. Runs after every pass in debug builds.
- **Text format** for debugging and pass tests:

  ```
  graph mlp {
    input %x : f32[N, 4096]
    const %w1 : f32[4096, 4096] = f32[4096, 4096] {...}
    %3 : f32[N, 4096] = matmul(%x, %w1) {transpose_a = false, transpose_b = true}
    %4 : f32[N, 4096] = silu(%3)
    output y = %4
  }
  ```
  `ir::print_graph` / `ir::parse_graph` (`src/ir/text.h`) round-trip exactly; `ii inspect model.ii.txt` reads it.
  Pass tests are written as "input → expected text" plus numerical equivalence.

---

## 6. Operation set (OpSet)

### Primitives and composites

**Primitives** (every backend implements them or declines the node):

| Group | Ops |
|-------|-----|
| unary | neg, abs, exp, log, sqrt, rsqrt, recip, sin, cos, tanh, sigmoid, erf, relu, floor, ceil, round, sign, not |
| binary (broadcast) | add, sub, mul, div, pow, max, min, mod, eq, ne, lt, le, gt, ge, and, or, xor |
| other elementwise | where, cast, clamp |
| reductions | reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod, argmax, argmin |
| linear algebra | matmul (batched, broadcast, transposes, quantized B) |
| convolution | conv (N-d, groups, dilation, padding), conv_transpose, max_pool, avg_pool |
| image, selection | resize (nearest / linear / cubic, ONNX coordinate modes), topk |
| data movement | reshape, permute, slice, expand, concat, split, pad, gather, scatter, tile, flip, cumsum |
| creation | iota, fill, random_uniform/normal (dropout/initialization) |
| quantization | quantize, dequantize |
| control flow | if, loop (later: with data-dependent shapes; constant ONNX `If` is inlined when importing) |

**Composites** (have a decomposition rule; a backend may implement them natively — which is what is done for
speed): gelu, silu, softmax, log_softmax, layer_norm, rms_norm, linear (matmul + bias), sdpa (scaled dot-product
attention: mask, causal, GQA; for LLMs `kv_len` over a preallocated KV cache, sliding `window`, `softcap`, 16-bit
keys and values with f32 queries) — OpSet v1; rope (rotary embedding, halves or interleaved pairs, partial) and
slice_update (write into a tensor at a run-time offset; in place) — stage 4; later group_norm, batch_norm, swiglu,
cross_entropy, adamw_step, …

`matmul` and `linear` take weights in a narrower type than the activations: f16 / bf16 with f32 activations
(widened while computing) and the GGUF block-quantized types (`q4_0` … `q6_k`).

The generated reference of every operation (signature, semantics, attributes with defaults) is
[opset.md](opset.md) (`ii ops --markdown`).

### OpSchema — the single source of truth about an op

```cpp
struct OpSchema {                        // src/ir/op_schema.h
    std::string_view name, category, doc; // the op reference (docs/opset.md) is generated from this
    int since_version;
    std::vector<InputSpec>  inputs;      // names, optional / variadic
    std::vector<OutputSpec> outputs;
    std::vector<AttrSpec>   attrs;       // name, kind, default (none = required)
    InferFn     infer;                   // symbolic shape and type inference + validation
    DecomposeFn decompose;               // nullptr for primitives
    CostFn      cost;                    // flops and bytes — for the planner and roofline reports
    GradFn      gradient;                // backward rule, expressed in ii ops (stage 7)
    uint32_t    flags;                   // elementwise, commutative, random, ...
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
- **CSE** merges nodes with the same operation, inputs and attributes (random operations excepted).
- **Pattern fusion** (`compiler/fusion.cpp`): `add(matmul(x, w), bias)` ⇒ `linear(x, w, bias)` (bias in the GEMM
  epilogue). Planned with the ONNX importer (stage 3), where such graphs appear: "unrolled" attention ⇒ `sdpa`,
  `x * rsqrt(mean(x²) + eps) * w` ⇒ `rms_norm`, layer norm, GELU / SiLU written with primitives.
- **Elementwise fusion** groups chains of elementwise operations of one shape (float and bool tensors; values used only
  inside the chain) into an internal `fused_elementwise` node whose `program` attribute holds the subgraph in the IR
  text format. It is a composite: backends without a fused kernel get the chain back by decomposition, so the pass is
  safe for every backend.
- **Convolution epilogues**: a convolution whose only user is an elementwise operation or a fused chain (an
  activation, SiLU, a residual addition, a per-channel scale and shift — batch normalization after import) becomes an
  internal `fused_conv` with the chain as its `program`; extra inputs must be per pixel, per channel or scalars.
  ii-cpu applies the program to each output block while it is still in cache; other backends decompose it.
  `CompileOptions::fuse` turns CSE and all fusions off.
- **Pass invariant**: a pass never changes results beyond tolerance — verified automatically in tests (the graph is
  executed before and after the pass on random inputs).

---

## 8. Backends

### Two kinds of backends behind one interface

1. **Kernel backends** (`ii-ref`, `ii-cpu`, `ii-cuda`, `ii-vulkan`, `ii-metal`, plus oneDNN/XNNPACK/cuBLAS wrappers):
   provide kernels per op; scheduling and memory are owned by the ii runtime. Implemented on top of the helper base
   class `KernelBackend` with a kernel table `op → (factory, support check)`; ISA variants come with `ii-cpu`.
2. **Graph backends** (TensorRT, ONNX Runtime, OpenVINO, CoreML): take whole subgraphs and compile them with their own
   engine (similar to ORT `GetCapability` / TFLite delegates).

```cpp
namespace ii::backend {  // src/backend/backend.h

class Backend {
public:
    virtual std::string_view name() const noexcept = 0;
    /// The device whose memory its kernels read and write, with capacity and speed (the planner's cost model).
    virtual const DeviceInfo& device_info() const noexcept = 0;
    /// Memory and queues of a device other than the host (nullptr for host backends).
    virtual DeviceApi* device_api() const noexcept;
    /// OK, or kUnsupported with the reason (dtype, attributes, shapes) — shown by `ii inspect --plan`.
    virtual Status supports(const ir::Node&) const = 0;
    /// Estimated time of a node with concrete shapes; default: roofline of device_info() + launch overhead.
    virtual double estimate_us(const ir::CostContext&) const;
    virtual Result<std::unique_ptr<Kernel>> create_kernel(const ir::Node&) const = 0;
    // Graph backends (TensorRT, part 5.5): claim(graph) → subgraphs, each compiled into one Kernel.
};

/// Executes one node (or a fused subgraph). Created once per Program and shared by all its sessions.
class Kernel {
public:
    /// Scratch memory for concrete shapes; planned into the session arena (of the kernel's device).
    virtual std::size_t workspace_size(std::span<const Shape> inputs, std::span<const Shape> outputs) const;
    /// Dense row-major inputs/outputs with concrete shapes. Thread-safe and allocation-free. Kernels of a device
    /// other than the host enqueue their work on `context.queue` (a CUDA stream).
    virtual Status run(const KernelContext&) const = 0;
    /// True if the kernel copied a constant input when it was created (packed layout, device memory) and never
    /// reads it: the runtime then does not copy that constant to the device.
    virtual bool captures_input(int input) const;
};

} // namespace ii::backend
```

### Devices, memory domains, synchronization

`src/backend/device.h`:

- **`DeviceInfo`** — the device (`cpu`, `cuda:0`, `test:0`), its name, the memory ii may use, peak GFLOP/s, memory
  bandwidth, kernel launch overhead and the host link (bandwidth, latency): the inputs of the planner's cost model.
  Host figures are a rough estimate from the core count and ISA (`ii info --roofline` measures the real ones).
- **`DeviceApi`** (one per device, thread-safe) allocates device memory as reference-counted `Buffer`s whose data
  pointer is a device address, and creates **`Queue`s** — ordered, possibly asynchronous work, one per session (a
  CUDA stream): `copy_to_device` (the host source may be reused as soon as the call returns), `copy_to_host` (the data
  is there after `synchronize()`), device copies, zero fills.
- **Memory domains.** Backends on the same device share its memory: `ii-ref` and `ii-cpu` share the host; `ii-cuda`
  and TensorRT will share a GPU. Values move between backends of one device **without copies**.
- **Transfers.** Where the plan reads a value on another device than the one holding it, the compiler inserts an
  internal `transfer` node (`{from = "cpu", to = "test:0"}`, one per value and destination; between two devices
  through the host). The runtime executes it itself: `copy_to_device` on the destination's queue, or `copy_to_host`
  followed by a synchronization, because the host steps after it read the data. Constants read by device kernels
  are copied once, when the program is compiled (unless the kernel `captures_input` them); graph outputs always come
  back to the host; states live on the device the planner gives them (a KV cache next to the attention that reads
  it; `Session::state` then returns a host copy). `run()` ends by synchronizing every queue.
- A tensor in device memory can be viewed but not read by the host functions of `Tensor` (they fail or terminate).
- **`ii-test-remote`** (`src/backends/test_remote/`) simulates a device without a GPU: its own tracked memory, filled
  with NaN when allocated; kernels (those of ii-ref) check that every input, output and workspace lies in that
  memory, so a missing transfer is an error; work runs only when the queue is synchronized, like a GPU stream, so a
  missing synchronization yields NaN or stale data. Operations, memory size and speed are configurable
  (`ContextOptions::test_devices`, `II_TEST_DEVICES`); the whole test suite runs on it in CI.

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

*Implemented in stage 5.2; guide: [docs/guides/backend-plugins.md](guides/backend-plugins.md).*

- Backends with large SDKs (ii-cuda, TensorRT, …) are built as separate `ii-backend-<name>` shared libraries so the
  core never depends on their SDKs, and an application linked with ii runs where the SDK is absent.
- The stable boundary is plain C (`include/ii/plugin.h`): the plugin exports `ii_backend_plugin_v1(host_abi_version)`
  returning a table of function pointers (`ii_plugin_v1`: `abi_version` + `struct_size` handshake, backends,
  device info, `supports` / `estimate_us` / `create_kernel`, kernels, device memory and queues). Errors are status
  codes with a fixed-size message buffer; no C++ type or exception crosses it, so a plugin built with MSVC works with
  an ii built with MinGW and vice versa.
- Nodes cross as a one-node graph in the IR text format; constant data is elided (`{...}`) and passed by pointer
  (no copy). Graph backends will receive larger subgraphs through the same functions.
- Inside a plugin, the SDK (`src/plugin/sdk.h`, target `ii_plugin_sdk`) adapts ordinary C++ `Backend`s to the C
  table; on the ii side `src/plugin/host.cpp` adapts the table back to `Backend`, `Kernel`, `DeviceApi` and `Queue`,
  so the planner and runtime treat plugin backends like built-in ones. `ii-test-remote` is also built as the plugin
  `ii-backend-test-remote`; CI runs the whole test suite with the simulated devices coming from it.
- Plugins are looked up in `ContextOptions::plugin_paths`, in `II_PLUGIN_PATH` and next to the ii library; libraries
  are loaded once per process and never unloaded. A missing SDK runtime (e.g. no TensorRT DLLs on the user's machine)
  makes the plugin report itself unavailable — never a crash or a load failure of the core. `Context::plugins()`,
  `ii info` and `ii check-backends` show what was loaded.
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

Implemented (part 5.1, `src/runtime/planner.h`): steps 2–6 below with the roofline estimates of the backends;
estimates use concrete shapes with symbols from `CompileOptions::plan_dims` (others = 1, e.g. one generated token).
A composite node left alone on another device than all of its neighbours, where no backend implements it, is
decomposed and the graph planned again, so that its primitives run next to the neighbours (P3). With all backends on
the host the plan is simply the fastest backend per node (ii-cpu, ii-ref for the rest). Autotuning (step 7) comes
with ii-cuda (part 5.4); policies and graph backends' `claim()` later.

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
6. **Memory constraint.** Weights + states + peak activations per device ≤ `capacity × (1 − reserve)` (capacity:
   the device's, or `CompileOptions::device_memory`; reserve 10%). On overflow the nodes at the front of the graph
   move one by one to their best other device, so the graph is cut once — for a transformer the first layers run on
   the CPU and the rest on the GPU: **automatic layer offload** when the model does not fit into VRAM. The local
   search respects the limits; a model that fits nowhere is an `OutOfMemory` error that names the device.
7. **Autotuning (optional).** `TuneMode::Measure`: run the top-K plans on real data and pick the best. Results are
   cached by `(model hash, shapes, device, driver version, ii version)` in `<cache>/ii/tuning.json`. Kernel
   parameters (block sizes, decode thread count) are tuned the same way.

**Policies and overrides:** `Policy::{Latency, Throughput, LowPower, LowMemory}`, allow/deny lists of backends and
devices, memory limits; overridable via `CompileOptions`, the CLI (`--backend tensorrt,cuda,cpu`) and `II_BACKENDS`.

**Explanation:** `ii inspect model --plan` (`Program::describe_plan`) prints the devices, partitions (runs of steps
on one backend) with estimated times, the copies between devices, memory per device, decisions forced by memory
limits and the reasons backends declined nodes ("ii-test-remote: no kernel for 'relu' (12 nodes)").

---

## 10. Runtime: memory and execution

- **Memory planning**: value lifetimes → offsets in one arena per session and device via "greedy by size" (as in TFLite);
  kernel workspaces and state-update staging are planned like ordinary values. Constants (weights) live outside the
  arena (mmap or packed copies). Values share buffers: a view (`reshape`, flag `kOpView`) aliases its input and its
  step is skipped; a kernel that declares `can_run_in_place(i)` writes into input `i`'s buffer when that buffer dies
  at the step and has the same size. A chain of elementwise steps then needs one buffer instead of one per step.
- **ExecutionPlan** — a flat array of steps with pre-built kernel arguments (shapes, dtypes; data pointers patched per
  run); execution is a walk over the array — one virtual `Kernel::run` call per step, no allocations.
- **Dynamic shapes**: a `Program` is compiled once for symbolic shapes. When a session sees new input shapes it binds
  the symbols, re-runs (concrete) shape inference node by node — checking the results against the symbolic types —
  and re-plans memory; the arena is reallocated only when it must grow. Unchanged shapes reuse the plan, so the steady
  state performs zero allocations (verified by a test that counts every `operator new`); the last four plans are kept,
  so alternating shapes (an LLM's prompt steps and single-token steps) do not re-plan. Planned later: symbol upper
  bounds (`seq ≤ 4096`) to plan once for the maximum.
- **In-place state updates**: when the step producing a state's update runs in place on the state itself and nothing
  reads the old state afterwards, its output is the state's memory. A KV cache [1, heads, context, head size] written
  by `slice_update` therefore costs only the new rows per step.
- **I/O binding**: the user may pass their own input and output buffers (zero-copy), including GPU buffers.
- **Profiler**: `Session::set_profiling(true)` times every step (two clock reads, no allocations); `profile()` returns
  the steps with operation, backend, shapes and times; `to_chrome_trace()` exports runs as Chrome trace JSON
  (viewable in Perfetto): `ii run model --profile trace.json` also prints a per-operation summary. Partitions and
  transfers join the trace with multiple devices (stage 5).

---

## 11. The `ii-cpu` backend

`src/backends/cpu/`: drivers (shapes, broadcasting, blocking, threading) are ordinary C++; kernels live in
`isa/` and are compiled once per ISA variant. ii-cpu is registered before ii-ref, so the planner gives it every node it
supports and leaves the rest (other dtypes, rare operations) to ii-ref; `ii inspect` shows the split.

### Runtime ISA dispatch

- The sources in `src/backends/cpu/isa/` are compiled **once per variant** (`ii_add_isa_component`) into namespace
  `ii::cpu::<isa>` with that variant's flags: `generic` (baseline C++: SSE2 / NEON), `avx2` (AVX2 + FMA + F16C),
  `avx_vnni` (the AVX2 kernels plus `vpdpbusd` int8 dot products), `avx512` (F/BW/VL/DQ), `neon` (Arm64 intrinsics),
  `neon_dotprod` (NEON plus the Armv8.2 `sdot` instructions). An extension variant defines its base's
  `II_ISA_IS_<BASE>` too, so it compiles all of the base's code paths. Each variant exposes one `KernelTable` of
  function pointers (`kernel_table.h`); the backend picks the table at creation. Later: `avx512_vnni`, `neon_i8mm`.
- `CpuInfo` picks the best usable variant; `II_CPU_ISA=generic|avx2|…` or `ContextOptions::cpu_isa` forces one, so all
  paths are tested on one machine (the tests are parameterized over every usable variant; AVX-512 runs under Intel SDE
  in CI).
- **ODR trap**: inline functions / templates instantiated in a TU built with `-mavx2` may be picked by the linker for
  the generic path → SIGILL on older CPUs. Rules: all ISA code (templates included) lives in `ii::cpu::II_ISA_NS`;
  ISA translation units use no inline code of shared headers (not even the standard library's — `<cmath>` only in the
  generic variant); the table header holds only plain structs, enums and declarations. Rare scalar functions (`pow`,
  `fmod`, `sin`, `cos`) are called through `scalar_math.cpp`, compiled without ISA flags.
- Compiler differences (GCC/Clang `-m…` flags vs MSVC `/arch:…`) are handled in one CMake helper (`IiIsa.cmake`).

### SIMD layer

`isa/simd.h` — a thin vector abstraction (`VF` float, `VI` int32, `VM` mask, `VD` double vectors with load/store,
arithmetic, FMA, compares, selects, rounding, bit operations, f16/bf16 conversions) over the intrinsics of one ISA;
the generic variant uses 4-lane arrays that the compiler vectorizes for the baseline. `isa/vmath.h` builds `exp`,
`log`, `tanh`, `sigmoid`, `erf`, `gelu`, `silu` on top of it once for every ISA (Cephes-style range reduction and
minimax polynomials). Accuracy is stated in the header and verified by sweeps against double precision (2–8 ulp;
`-ffast-math` is **not** used: NaN and infinities behave as in ii-ref).

### GEMM (the heart of performance)

- BLIS approach: an `MR×NR` register micro-kernel written once with the SIMD layer (AVX2 6×16, AVX-512 14×32,
  NEON 8×12, generic 4×8) with variants for every row count `1..MR` (GEMV and edge tiles do only the work they need),
  `MC/KC/NC` cache blocking from the cache sizes reported by `CpuInfo`.
- **Constant weights are packed when the program is compiled** (B panels of NR columns over the full depth); a
  non-constant B is packed per work item into per-thread scratch. A work item is an `MC×NC` block of C (of one batch
  matrix); items are handed out dynamically to the threads (several per thread, so P/E cores and big.LITTLE balance).
- f16/bf16 weights stay 16-bit in their panels and are widened in the micro-kernel (F16C / NEON / shift); f32
  accumulation; 16-bit outputs are accumulated in f32 scratch and converted once.
- `linear` adds the bias in the GEMM epilogue; `matmul` supports transposes and batch broadcasting. Single-row
  panels (decode) stream B over the whole depth with software prefetch.
- The driver (`gemm_driver.h`) is shared with convolution: B may also be packed by a callback (implicit im2col),
  A may be prepacked (constant convolution weights), and an epilogue callback sees every finished f32 block.
- f16 / bf16 weights with f32 activations use the same driver (B widened in the micro-kernel). Not yet: activations
  in the epilogue of `matmul` / `linear` (a following activation is a separate fused elementwise pass).

### Convolution, pooling, resize (stage 3)

- **Convolution** (NCHW, f32, 1-D and 2-D) is a GEMM per image and group: weights [M/groups, C/groups·KH·KW]
  (prepacked into MR panels when the program is compiled) times the im2col matrix, which is never materialized as a
  whole: it is packed straight from the image into NR-column GEMM panels (zero padding included) once per image, in
  parallel, and the micro-kernel reads the panels in place. 1×1 stride-1 convolutions pack a plain copy of the image.
  Biases initialize the output blocks; a `fused_conv` epilogue runs on each finished block.
- **Depthwise** convolutions (one input channel per group, MobileNet) are direct stencils: each plane is copied into a
  zero-padded L1 buffer and the taps are accumulated with vector FMAs (stride 1) or gathered columns (other strides).
- `max_pool` / `avg_pool` (row-wise vector max / sum over the window taps), `resize` (nearest and linear over the last
  two dimensions with precomputed taps). Cubic resize, 3-D windows and `conv_transpose` run on ii-ref.
- Gap to ONNX Runtime (MLAS: direct NCHWc kernels) and the plan: `bench/results/stage3-onnx.md`.

### Quantized kernels (the key to weak hardware) — stage 4

`linear` / `matmul` with GGUF block-quantized weights (`qmatmul.cpp`, kernels in `isa/quant.cpp` and `isa/qgemm.cpp`):

- The activation rows are quantized to 8 bits first, as llama.cpp does: blocks of 32 (`d`, `d * sum`, 32 int8) for
  `q4_0` … `q8_0` weights, ggml's `q8_K` (256 values with sums of groups of 16) for the K-quants. Integer dot products
  then replace dequantization: `maddubs` / `madd` on AVX2, `vpdpbusd` on AVX-VNNI, `sdot` on Armv8.2. Unsigned weight
  codes are multiplied directly and their offset (`q4_0`: 8, `q6_K`: 32, …) and minimums come from the activation sums.
- **Row kernels** (every type, every variant; scalar in `generic`) multiply one weight row with up to four activation
  rows, so a weight block is decoded once per four rows.
- **Panel kernels** (`q4_0`, `q5_0`, `q8_0`, `q4_K`, `q6_K` on AVX2 / AVX-VNNI / NEON dotprod): constant weights are
  repacked when the program is compiled into panels of 8 rows interleaved by 4 values (layouts in `kernel_table.h`).
  Four activation bytes are broadcast against 32 weight bytes (8 rows), so one dot-product instruction does 32
  multiply-adds without horizontal sums and every decoded weight byte serves four activation rows: 2.5-3× the row
  kernels for prompt processing (160-190 GMAC/s on one Arrow Lake core), and contiguous 8-column outputs.
- Work items are blocks of 64 activation rows × columns (several per thread); the activation quantization is
  parallel too. Decode (one row) reads the weights once: ~65-75% of the measured memory bandwidth with all cores;
  per core it is limited by the outstanding cache misses rather than by the kernel (one core decodes 45-50 GB/s of
  weights held in cache).
- Exactness: the kernels of every variant are checked against a double-precision computation on the same quantized
  activations, and the panel kernels against the row kernels (`tests/unit/backends/cpu_quant_test.cpp`); models are
  checked end to end against transformers (§13).

### Elementwise operations and fusion

- Every elementwise operation (float and bool tensors: f32, f16, bf16, bool) compiles to a small **program** of
  instructions over registers that each hold a tile of 128 f32 values; an interpreter in `isa/elementwise.cpp`
  evaluates it tile by tile. A single operation is a one-instruction program; a `fused_elementwise` node (created by
  the fusion pass, §7) is a longer one, so a chain of operations reads its inputs and writes its output once while the
  intermediates stay in L1. The driver collapses dimensions and handles broadcasting (an input is either contiguous
  or a repeated scalar along the innermost run) and splits the work over threads.
- 16-bit inputs are widened on load and rounded once on store; inside a fused chain intermediates stay f32 (more
  accurate than rounding every step; a `cast` to f16/bf16 inside a chain still rounds).

### Reductions, normalizations, attention

- Reductions over adjacent axes, `softmax`/`log_softmax` (any axis; vectorized across the inner dimension when the
  axis is not the last), `layer_norm`, `rms_norm`: row kernels; sums are accumulated in double precision.
- `sdpa`: one query row at a time (scores, softmax and the weighted sum of values in a per-thread row of scratch, so
  the attention matrix is never materialized); grouped-query attention, causal masking, bool and additive masks, and
  for LLMs the keys in use (`kv_len` of a preallocated cache), sliding windows, soft-capping and f16 caches (only the
  keys a query can attend are visited). Later: tiled flash attention for long prompts.
- `rope` and `slice_update` (in place) for the LLM graphs.
- `reshape`, `permute` (tiled transposes), `concat`: dtype-agnostic copies. Everything else falls back to ii-ref
  (`ii inspect` shows which nodes).

### Threads

- Own `ThreadPool`: `parallel_for(range, grain, fn)`; the calling thread participates; brief spinning, then sleeping
  on a condition variable. Kernels parallelize through `parallel_range` (skipped below ~32K units of work) or
  `parallel_slots` (a fixed number of workers, each with its own scratch, pulling work items from an atomic counter).
- **Hybrid CPUs** (P/E cores: Intel Core Ultra, Apple M-series, big.LITTLE on Android): static equal splits leave fast
  cores idle, so work is handed out dynamically. Optional pinning to performance cores for latency-sensitive work
  (per-OS topology: `GetLogicalProcessorInformationEx`, sysfs, `hw.perflevel*`) — later.
- For memory-bound decode the optimal thread count is often lower than the core count (measured on a Snapdragon 732G:
  GEMV is not faster on 8 threads than on one big core) — picked by autotuning (stage 4).
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
backend, the planner, offload and the profiler. Public API: `include/ii/llm.h` (`Llm`, `Sequence`, `Chat`,
`Tokenizer`, `Sampler`); implementation in `src/llm/`.

- **Loading**: a GGUF file (`gguf::File`, tensors used in place from the mapping) or a Hugging Face directory
  (`config.json`, safetensors shards, `tokenizer.json`, `tokenizer_config.json`). Hugging Face tensor names are mapped
  to GGUF names so one graph builder serves both (`hparams.cpp`).
- **Architectures as data**: `graph.cpp` builds the decoder for llama (Llama 2/3, Mistral, TinyLlama, SmolLM), qwen2,
  qwen3, gemma, gemma2 and gemma3 (logit soft-capping, sliding windows with their own rope base, extra norms, scaled
  embeddings) and phi3 (fused qkv and gate/up) from the hyper-parameters. Rotary embeddings: interleaved pairs for
  GGUF llama (whose converter permutes q / k) and halves otherwise; llama-3 scaling from `rope_scaling` or the
  `rope_freqs` tensor; linear scaling.
- **One graph for prompt and decode**: inputs `tokens` i32[1, S] and `pos` i64[] (tokens already cached), output
  `logits` (the last position, or all with `LoadOptions::all_logits`). Positions `pos + iota(S)` feed the rotary
  tables; `kv_len = pos + S`.
- **KV cache**: per layer two states [1, kv heads, context, head size] (f32 or f16), written by `slice_update` at
  `pos` (in place, §10) and read by `sdpa` with `kv_len` and the causal mask. A `Sequence` owns a session (its cache)
  and the token list; `rewind(n)` drops tokens, which `Chat` uses to reuse the cached prefix of a conversation.
- **Tokenizers** (`tokenizer.cpp`): SentencePiece merging by score (GGUF "llama" vocabularies; token order when a
  file has no scores), byte-level BPE by merge rank with the GPT-2, Llama 3 and Qwen 2 pre-tokenizer splits
  (hand-written matchers over Unicode category tables generated from Python's `unicodedata`), BPE over characters
  with byte fallback (Hugging Face exports of SentencePiece models); special tokens, end-of-generation tokens,
  streaming-safe UTF-8 decoding.
- **Chat templates** (`jinja.cpp`): an interpreter for the Jinja subset of real templates, rendered like transformers
  (trim_blocks, lstrip_blocks, `tojson`, `raise_exception`, namespaces, loop controls, macros); resource limits for
  untrusted templates. ChatML when a model has none.
- **Sampling**: repetition penalty, top-k, top-p, min-p, temperature (llama.cpp's order), deterministic seeds; greedy
  at temperature 0. Token streaming via callback (returning false stops).
- **Checked against transformers**: tiny random models of every architecture (both formats, `tests/data/llm`) match
  per position, through the KV cache; real models (`scripts/llm_tests.py`): logits within ~4e-6, greedy generation
  equal for 64-128 tokens, perplexity equal to 4 digits.
- Later: MoE, paged KV cache and batching of sequences, speculative decoding, grammars, multimodal inputs.

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
| GGUF | quantized LLMs | `include/ii/gguf.h`: reader (versions 2-3, typed metadata, tensors as views of the mapping) and writer (`ii quantize`); GGML type ↔ ii dtype |
| safetensors + `config.json` | HF models | JSON header + mmap, sharded models (`model.safetensors.index.json`); architecture from `config.json` |
| ONNX | general models (CV, audio, encoders) | **own protobuf wire-format decoder** (no libprotobuf dependency), external data, `onnx op@opset → ii ops` mapping table (stage 3, see below) |
| `.npy` / `.npz` | tensors (inputs, outputs, test data) | `ii/io.h`: NumPy format 1.0–3.0, stored and DEFLATE-compressed archives (own inflate) |
| `.ii` | native, fast startup | header + sections; graph in a compact TLV encoding, weights aligned for mmap, optional weights pre-packed for a specific ISA; versioned format |

**ONNX import** (`src/formats/onnx/`, `Model::load` / `Model::from_onnx`):
- `onnx_proto.cpp` decodes ModelProto, GraphProto, NodeProto, TensorProto, AttributeProto and FunctionProto with the
  wire-format reader (`src/formats/protobuf.h`); raw tensor data is referenced from the memory-mapped file when it
  is suitably aligned, otherwise copied. External data must stay inside the model's directory.
- `onnx_ops.cpp` maps ~130 operators of the default domain (opsets 1–23 and newer where semantics are unchanged) to
  ii operations; model-local functions are inlined; `If` with a condition known at load time is inlined.
- ONNX passes many parameters as tensors (Reshape targets, Slice bounds, reduction axes, Resize scales), ii keeps them
  in attributes. The importer therefore tracks the **contents of small integer tensors symbolically**: `Shape(x)` of
  an input [N, 3, 224, 224] is known to be [N, 3, 224, 224], and Gather / Slice / Concat / Unsqueeze / arithmetic /
  Equal / Where over such values are evaluated while importing, so dynamic-batch and dynamic-sequence models import
  with symbolic dimensions. Parameters computed from constants are evaluated with ii-ref on demand. When contents
  cannot be expressed (e.g. slicing by a run-time value), `LoadOptions::dims` fixes input dimensions and
  `LoadOptions::constant_inputs` replaces parameter inputs by constants.
- Verified against ONNX Runtime on real models (`tests/models/manifest.json`) and on the operator test cases shipped
  with the `onnx` package (`tests/onnx`, run in CI).

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
- **Configuration without globals**: `ContextOptions { allocator, executor (thread pool), logger, plugin_paths,
  plugin_options, … }`; `II_*` environment variables only override defaults and are documented.
- **Windows naming detail**: the library and the CLI are both called `ii`, so the CMake targets differ
  (`ii` / `ii_cli`) and PDB names are set explicitly to avoid `ii.pdb` collisions.

---

## 17. The `ii` CLI

```
ii info                          hardware, ISA, devices, backends, plugins; --roofline measures peaks
ii run <model> [--dims N=8]      run an ONNX or ii text model; inputs from --inputs a.npz, --input x=a.npy or
                                 images (--input images=cat.jpg, --letterbox, --mean/--std), otherwise random;
                                 --save-outputs out.npz, --top K, --profile trace.json
ii generate <llm> -p "…"         text generation (--chat applies the chat template; --save-logits, --profile)
ii chat <llm>                    interactive chat
ii perplexity <llm> -f text      perplexity like llama-perplexity (chunks of --ctx tokens)
ii tokenize <llm> -p "…"         the tokens of a text
ii bench <model> [--llm]         benchmark: latency p50/p90/p99 and throughput; for LLMs pp / tg tokens per second
ii inspect <model> [--plan]      inputs and outputs, the execution plan with backends and devices; --plan explains
                                 it (estimates, copies, memory, declines); --graph, --imported; for LLMs too
ii convert <in> <out.ii>         conversion to the native format (+ pre-packing)
ii quantize <in.gguf> <out.gguf> --type q4_k_m   GGUF weights to q8_0 … q6_k or the llama.cpp mixes
ii check-backends                small models of common operations on every backend (and plugin) compared with
                                 the reference on this machine; --backends, --plugin, --json
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
│   ├── backend/       Backend, Kernel, KernelBackend; device.h: DeviceInfo, DeviceApi, Queue
│   ├── plugin/        plugin loading (host.cpp) and the plugin SDK (sdk.h); the C ABI is include/ii/plugin.h
│   ├── backends/
│   │   ├── ref/       reference kernels
│   │   ├── cpu/       ii-cpu: drivers; isa/ (SIMD layer, math, kernels compiled per ISA)
│   │   ├── test_remote/  ii-test-remote: a simulated device for multi-device tests
│   │   ├── cuda/  vulkan/  metal/
│   │   └── plugins/   tensorrt/, onnxruntime/, openvino/, coreml/, …
│   ├── formats/       onnx/ (+ protobuf wire decoder), npy/npz, gguf, safetensors; later native/
│   ├── llm/           hparams (+ weight names), graph, tokenizer, unicode, jinja, sampler, llm (API)
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
