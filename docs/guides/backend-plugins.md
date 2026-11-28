# Backend plugins

A backend plugin is a shared library that adds backends to ii at run time. Backends that need a large SDK (CUDA,
TensorRT, …) are plugins, so an application linked with ii runs on machines without that SDK, and ii itself never
depends on it. The interface between ii and a plugin is plain C (`include/ii/plugin.h`), so a plugin built with one
compiler works with an ii built with another (MSVC and MinGW, say).

## Finding and loading plugins

A context loads plugins when it is created (`Context::create`):

| Source | Notes |
|---|---|
| `ContextOptions::plugin_paths` | plugin files or directories; a plugin named here that cannot be loaded makes `Context::create` fail |
| `II_PLUGIN_PATH` | directories (or files), separated by `;` on Windows and `:` elsewhere |
| next to the ii library | the directory of `ii.dll` / `libii.so`, or of the executable for a static ii |

The last two are skipped with `ContextOptions::search_plugins = false`. In directories, files named
`ii-backend-<name>.dll` / `libii-backend-<name>.dll` (Windows), `libii-backend-<name>.so` (Linux) or
`libii-backend-<name>.dylib` / `.so` (macOS) are loaded. A library is loaded once per process and stays loaded.

`Context::plugins()` and `ii info` list every plugin found, with its state:

- **loaded** — its backends are available (possibly none, e.g. when no GPU is present);
- **unavailable** — the plugin cannot work on this machine (no driver, missing SDK libraries); not an error;
- **failed** — not a plugin, another ABI version, or the name of a plugin already loaded.

```
$ ii info
...
Device    cpu      Intel(R) Core(TM) Ultra 7 265KF (ii-cpu, ii-ref)
Device    test:0   ii test device 0, 2 GiB (ii-test-remote)
Plugin    ii-backend-test-remote 0.1.0 (loaded)
          F:\ii\build\windows-mingw-gcc\plugins\Release\libii-backend-test-remote.dll
```

Plugins take configuration from `ContextOptions::plugin_options` and the `II_PLUGIN_OPTIONS` environment variable
(`key=value` pairs separated by `;`, e.g. `II_PLUGIN_OPTIONS=cuda.tf32=1;cuda.devices=0`); the context's options win.
Each plugin documents its keys.

The plugins built with ii:

| Plugin | Backends | Built |
|---|---|---|
| `ii-backend-cuda` | `ii-cuda` (one per NVIDIA GPU: `ii-cuda`, `ii-cuda.1`, …) | where the CUDA Toolkit is found; on Windows by MSVC ([building.md](building.md#the-cuda-backend)) |
| `ii-backend-test-remote` | `ii-test-remote` (simulated devices, `test_devices`) | with the tests |

`ii check-backends` runs small models of common operations on every backend and compares the results with ii-ref —
a quick check that a plugin works on a machine:

```
$ II_TEST_DEVICES=default ii check-backends --plugin build/plugins/Release
ii-cpu             cpu      27 passed, 0 failed,  2 skipped
ii-cuda            cuda:0   29 passed, 0 failed,  0 skipped
ii-test-remote     test:0   29 passed, 0 failed,  0 skipped
```

## The interface

The plugin exports one function:

```c
const ii_plugin_v1* ii_backend_plugin_v1(uint32_t host_abi_version);
```

It returns a table of functions (or NULL if it does not support `host_abi_version`). ii checks the table's
`abi_version` and `struct_size` and that the required functions exist, then calls `create_backends` once per
context. Each backend then:

- describes its device (`device_info`: kind, index, memory, speed — the inputs of the planner's cost model);
- answers whether it can execute a node (`supports`) and, optionally, how long it takes (`estimate_us`; without it
  ii uses the roofline of the device info);
- creates kernels (`create_kernel`) which ii runs (`run`), with `kernel_flags` (in-place, captured inputs, and with
  input -1 whether the kernel's work can be recorded) and `workspace_size`;
- optionally compiles parts of a model (a graph backend: `backend_flags` returns `II_PLUGIN_BACKEND_SUBGRAPHS`; it
  then gets `subgraph` nodes, see below);
- for a device other than the host: allocates memory and creates queues of ordered work (copies, `synchronize`);
  optionally a queue records work (`begin_recording`, `end_recording`, `replay`, `destroy_recording`, appended to the
  table later; ii-cuda records CUDA graphs): ii then records a run of recordable kernels once their arguments repeat
  and replays it instead of calling them.
  Optionally it allocates page-locked host memory (`allocate_host`, `deallocate_host`; `cudaHostAlloc`): ii keeps the
  host values sessions copy to and from the device in it and hands it to applications
  (`Context::pinned_allocator()`). Copies from such memory may be asynchronous: ii keeps the source unchanged until
  the queue is synchronized.

The semantics are those of the C++ interfaces `ii::backend::Backend`, `Kernel`, `DeviceApi` and `Queue`
(`src/backend/backend.h`, `src/backend/device.h`).

Nodes are passed as a one-node graph in the ii IR text format: inputs are graph inputs `%x0`, `%x1`, … or constants
`%c0`, … whose data is elided (`{...}`) and passed by pointer next to the text. For example, a `linear` node with
constant weights:

```
graph node {
  input %x0 : f32[N, 256]
  const %c0 : f32[256, 256] = f32[256, 256] {...}
  const %c1 : f32[256] = f32[256] {...}
  %0 : f32[N, 256] = linear(%x0, %c0, %c1) {transpose_w = true}
  output y0 = %0
}
```

Weights are therefore never copied or serialized; in `create_kernel` their data stays valid while the kernel lives.
Later ABI versions may pass larger subgraphs (graph backends such as TensorRT) through the same functions.

Rules of the interface:

- functions return a status code (`ii_status` values) and fill an `ii_plugin_error` with a message on failure;
- element types are `ii::DType` values and device kinds `ii::DeviceKind` values (both only grow);
- no C++ type and no exception crosses the boundary; strings are UTF-8;
- `run` must not allocate host memory (the steady state of a session allocates nothing).

## Writing a plugin with the SDK

A plugin built from the ii sources uses the plugin SDK (`src/plugin/sdk.h`, CMake target `ii_plugin_sdk`): it
implements ordinary C++ backends and lets the SDK do the rest — parsing nodes, wrapping constant data, forwarding
kernels, memory and queues, converting errors and exceptions.

```cpp
#include "plugin/sdk.h"

namespace {

ii::Result<ii::plugin::BackendList> create_backends(const ii::plugin::PluginHost& host) {
    if (!driver_present()) {
        return ii::errors::unavailable("no CUDA driver");  // listed as "unavailable", not an error
    }
    ii::plugin::BackendList backends;
    backends.push_back(std::make_unique<MyBackend>(host.option("my.device")));
    return backends;
}

constexpr ii::plugin::PluginDefinition kPlugin{"ii-backend-my", "1.0", &create_backends};

}  // namespace

II_DEFINE_BACKEND_PLUGIN(kPlugin)
```

In CMake, `ii_add_backend_plugin(<target> SOURCES …)` builds a plugin with the SDK into `<build>/plugins/<config>`.
The plugin links `ii_internal` statically, so it carries its own copy of the parts of ii it uses (IR, operation
set, shape inference); it must not exchange C++ objects with the loading ii.

`src/backends/test_remote/plugin.cpp` is a complete example: the simulated device of `ii-test-remote` as a plugin.
When it is loaded (`II_PLUGIN_PATH=<build>/plugins/<config>`), it provides the simulated devices of
`ContextOptions::test_devices` / `II_TEST_DEVICES` instead of the built-in backend; CI runs the whole test suite
that way. The plugin tests (`tests/unit/plugin/plugin_test.cpp`) also run ii-ref and the simulated device through an
in-process plugin, so every node crosses the interface as text.

## Compatibility

- **Graph backends** (ii-tensorrt): after planning, ii offers them parts of the graph on their device as `subgraph`
  nodes — one node whose `program` attribute holds the part in the IR text format (graph input i is node input i,
  the model's constants among them; graph output i is node output i). `supports` decides whether the backend takes a
  part, `estimate_us` whether it is worth it (the backend should forward estimates: `custom_estimates`), and
  `create_kernel` compiles it. `ops::subgraph_program()` returns the part with the constants inlined. The SDK forwards
  `Backend::compiles_subgraphs()`.
- **Sharing a device**: when several plugins have backends on one device (ii-cuda and TensorRT on a GPU), ii uses the
  memory and queues of the first kernel backend for all of them. A kernel then gets `queue = NULL` and the queue's
  native handle in `native_queue` (from `queue_native_handle`, a `cudaStream_t` for CUDA); the SDK passes it as a
  queue whose `native_handle()` returns it. Kernels enqueue on that stream; device pointers are valid across the
  plugins (one CUDA context).
- ii loads plugins whose `abi_version` equals its `II_PLUGIN_ABI_VERSION`; a breaking change gets a new entry point
  (`ii_backend_plugin_v2`). New functions are appended to the table; `struct_size` tells which exist (a table that
  ends before the recording functions loads, and its queues simply do not record; one without `backend_flags` has no
  graph backends; one without `allocate_host` has no page-locked memory). Fields appended to `ii_plugin_run_context` exist when ii's `ii_plugin_host::run_context_size`
  covers them.
- A plugin built with MSVC works with an ii built with MinGW GCC and vice versa (checked for the test plugin: the
  whole test suite passes both ways).
- The text format of nodes follows the ii version of the plugin's sources: an operation unknown to the plugin is
  reported as unsupported, and the planner places it elsewhere.
