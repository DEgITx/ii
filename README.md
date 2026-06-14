<p align="center">
  <img src="docs/github-logo.png" alt="ii logo" width="600">
</p>

# ii — run neural networks anywhere, simply

`ii` is a small, fast **inference library and command-line runner** for neural
networks. Point it at a model and an image and it preprocesses the input, runs
inference, and gives you the result — raw outputs, YOLO detections drawn on
screen, or a decoded image written back to a file. It runs the same models the
same way across very different hardware: a laptop, a workstation GPU, or a tiny
embedded board.

Two things make it convenient:

- **One interface, many backends.** TensorFlow Lite, NVIDIA TensorRT, ONNX
  Runtime / DirectML, and a **built-in pure-C++ engine** all sit behind a single
  `ii::Engine` abstraction. Pick what you build, choose at runtime with
  `--backend`, and every feature (detection, image-to-image, video,
  benchmarking) works the same regardless of which one runs underneath.
- **No SDK required to get started.** The built-in `ii` engine has zero external
  dependencies — it compiles on any platform with just a C++20 compiler and runs
  ONNX models out of the box. It is designed for **embedded targets, efficiency,
  and simplicity**: a single static library, deterministic results, and
  allocation-free parallel kernels. Reach for a heavier backend only when you
  want vendor acceleration.

`ii` is equally usable as a **CLI tool** and as an **embeddable library**: the
core knows nothing about any specific framework, so you can drop the whole runner
— or just its inference layer — into a larger application without dragging in
dependencies you don't use.

## Why ii

- **Runs everywhere** — the same binary covers CPU, GPU, and NPU; the same
  source builds on Linux, Windows, and macOS.
- **Starts with nothing** — the built-in engine needs no vendor SDK, so you can
  build and run a model in one `cmake` invocation.
- **Scales up** — when you have the hardware, link TensorRT (NVIDIA GPUs),
  DirectML (any Windows D3D12 GPU/iGPU/NPU), or TFLite with an NPU/GPU delegate,
  and switch with a single flag.
- **Embeddable** — one narrow `ii::Engine` interface, one static library
  (`ii_core`), and reusable helpers for preprocessing, YOLO decoding,
  image-to-image output and parallelism.

## What it does

- **Object detection** — built-in YOLOv8 decoding + NMS, boxes drawn over the
  frame. COCO-80 labels by default, or supply your own class list.
- **Image-to-image models** — super-resolution, low-light enhancement,
  denoising. Show the result on screen or save it to PNG, with optional
  tiling / sliding-window for small-input models and seam feathering.
- **Live video** — camera capture (V4L2 on Linux, Media Foundation on Windows)
  with on-the-fly inference and an on-screen FPS / jitter overlay.
- **On-screen display** — a zero-copy window (Wayland + EGL + GLES2 on Linux,
  Direct3D 11 on Windows) that letterboxes the frame and updates a single
  texture per frame.
- **Benchmarking & comparison** — warmup + timed runs, CPU-vs-delegate
  comparison, and cross-model comparison (e.g. float32 vs INT8, or one backend
  against another).
- **Telemetry** — CPU / memory / thread monitoring and CSV export of benchmark,
  FPS, comparison and system-monitor data.
- **Quantization-aware** — INT8 / UInt8 / Float16 / Float32 tensors; the
  scale / zero-point are read directly from the model.

## Backends

| Backend            | Model format        | CMake option (default) | Notes                                                        |
|--------------------|---------------------|------------------------|-------------------------------------------------------------|
| **ii** (built-in)  | `.onnx`             | `USE_II_ENGINE` (ON)   | Pure C++20, **no SDK**. Always on; great for embedded.       |
| TensorFlow Lite    | `.tflite`           | `USE_TFLITE` (OFF)     | CPU + optional NPU/GPU delegate (`--delegate`).             |
| NVIDIA TensorRT    | `.engine` / `.onnx` | `USE_TENSORRT` (OFF)   | Requires TensorRT 10+ and the CUDA runtime.                |
| ONNX Runtime / DML | `.onnx`             | `USE_DIRECTML` (OFF)   | DirectML EP on Windows (GPU/iGPU/NPU); CPU EP elsewhere.    |

Only the built-in `ii` engine is enabled by default — it needs no external SDK,
so the project builds and runs out of the box. The other backends are opt-in:
turn them on with their `USE_*` option once you have the corresponding SDK. A
single binary can carry any combination of backends; choose at runtime with
`--backend <name>`. When `--backend` is omitted, the runner **auto-selects the
best working backend**: it tries the compiled-in backends in priority order
(`tensorrt` → `directml` → `tflite` → `ii`) and keeps the first one that actually
loads the model with the given options — so the chosen backend matches both the
model format and the available hardware/delegate, falling back all the way to the
always-present pure-CPU `ii` engine. The compiled-in backends are listed in
`--help`.

### The built-in `ii` engine

The native engine (`src/engine/`) is a self-contained graph executor written in
plain C++20 — no TensorFlow, no ONNX Runtime, no CUDA. It loads ONNX models
directly, supports a broad op set (Conv, Gemm/MatMul, the common activations,
pooling, normalization, resize, concat/slice/gather/reshape, reductions, …), and
fans heavy kernels out across cores via an allocation-free parallel primitive
that stays **bit-identical to serial**. It is the path to take when you want a
small, dependency-free runner on an embedded device.

## Building

Requires CMake ≥ 3.16 and a C++20 compiler. [stb](https://github.com/nothings/stb)
(image load / resize) is fetched automatically. With only the default options you
get a working binary immediately — no SDK to install.

### Linux

```sh
cmake -S . -B build
cmake --build build -j
```

That's it — the built-in `ii` engine needs no SDK, so this produces a working
binary immediately. To add the TensorFlow Lite backend, enable it and point the
build at your install:

```sh
cmake -S . -B build -DUSE_TFLITE=ON -DTFLITE_ROOT=/usr
# or, for a non-standard location:
cmake -S . -B build -DUSE_TFLITE=ON \
    -DTFLITE_INCLUDE_DIR=/opt/tflite/include \
    -DTFLITE_LIB=/opt/tflite/lib/libtensorflowlite.so
```

Headless host (no display / camera):

```sh
cmake -S . -B build -DUSE_DISPLAY=OFF -DUSE_CAMERA=OFF
```

### Windows (MSVC)

```sh
cmake -S . -B build
cmake --build build --config Release -j
```

Builds the `ii` engine with no extra dependencies; display (Direct3D 11) and
camera (Media Foundation) are on by default. For native GPU acceleration, add
`-DUSE_DIRECTML=ON -DORT_ROOT=...` (any D3D12 GPU/iGPU/NPU) or
`-DUSE_TENSORRT=ON -DTRT_ROOT=...` (NVIDIA). TFLite can be linked too with
`-DUSE_TFLITE=ON -DTFLITE_ROOT=C:/path/to/tflite-prebuilt`.

### macOS

CPU-only dev host (`ii` engine + optional TFLite); display and camera are
stubbed. Useful for development and cross-backend regression.

### Build options

| Option            | Default | Purpose                                                       |
|-------------------|---------|---------------------------------------------------------------|
| `USE_II_ENGINE`   | ON      | Built-in pure-C++ engine (no SDK). Always available.         |
| `USE_TFLITE`      | OFF     | TensorFlow Lite backend (`-DTFLITE_ROOT=...`).               |
| `USE_TENSORRT`    | OFF     | TensorRT backend (`-DTRT_ROOT=...`).                          |
| `USE_DIRECTML`    | OFF     | ONNX Runtime / DirectML backend (`-DORT_ROOT=...`).         |
| `USE_DISPLAY`     | ON¹     | On-screen output (Wayland on Linux, Direct3D 11 on Windows).  |
| `USE_CAMERA`      | ON¹     | Camera capture (V4L2 on Linux, Media Foundation on Windows).  |

¹ `ON` on Linux and Windows, `OFF` elsewhere.

## Quick start

```sh
# Run an ONNX model on the built-in engine — works out of the box, no SDK
./ii model.onnx image.jpg

# The examples below use TFLite models — build with -DUSE_TFLITE=ON first.
# Run TFLite on CPU (no delegate); or with an NPU/GPU delegate:
./ii model.tflite image.jpg --backend tflite --no-delegate
./ii model.tflite image.jpg --backend tflite --delegate /lib/libneutron_delegate.so

# Object detection — draw YOLOv8 boxes in a window
./ii yolov8m_int8.tflite image.jpg --display --yolo --conf 0.4 --iou 0.5

# Live camera detection
./ii yolov8m_int8.tflite --camera --display --yolo --stats

# Super-resolution / enhance — show or save the output image
./ii fsrcnn_qat.tflite image.jpg --display --show-output
./ii fsrcnn_qat.tflite image.jpg --save-output out.png

# Benchmark (with an image, or with random input)
./ii model.tflite image.jpg --benchmark --runs 100
./ii model.tflite --random-input --benchmark --runs 100

# Cross-check: compare INT8 against a float32 reference
./ii yolov8m_int8.tflite --random-input --compare model_fp32.tflite --random-runs 50
```

Run `./ii --help` for the full option list and the backends compiled into your
binary.

## Common options

| Option | Description |
|--------|-------------|
| `--backend <name>`   | Inference backend (`ii`, `tflite`, `tensorrt`, `directml`). |
| `--delegate <path>`  | Path to an external acceleration delegate / plugin (TFLite). |
| `--no-delegate`      | Run on CPU without a delegate.                         |
| `--benchmark`        | Warmup + timed run; `--runs N`, `--warmup N`.         |
| `--threads <N>`      | CPU threads for inference.                            |
| `--display`          | Open a window (Wayland on Linux, D3D11 on Windows).   |
| `--stats`            | FPS / jitter counter (overlay + stdout).             |
| `--yolo`             | Decode output as YOLOv8 and draw boxes.              |
| `--conf <p>` `--iou <p>` | Confidence / NMS-IoU thresholds.                 |
| `--classes <path>`   | Class-name file (one per line; default COCO-80).     |
| `--show-output`      | Render the model's output as an image (SR / enhance). |
| `--save-output <p>`  | Save the decoded output to PNG.                       |
| `--tile`             | Tiling / sliding window for small-input models.      |
| `--camera [dev]`     | Camera capture (`/dev/videoN` on Linux, index on Windows). |
| `--compare <path>`   | Run a reference model and compare outputs.            |
| `--export <prefix>`  | Write benchmark / FPS / comparison data to CSV.      |
| `--sysmon`           | Monitor process & system CPU / memory (Linux).        |

## Use as a library

Everything except the CLI is compiled into one static library, **`ii_core`**,
designed to be embedded in other applications. The `ii::Engine` interface
(`inference.h`) is fully self-contained — it pulls in no backend SDK headers — so
a host program depends on just that one narrow abstraction and links whichever
backend(s) it built.

```cpp
#include "inference.h"

auto eng = ii::make_engine("ii");            // or "tflite" / "tensorrt" / "directml"
ii::Engine::Options opts;
opts.delegate_path = "";                      // "" = CPU
opts.num_threads   = 4;
eng->load("model.onnx", opts);

std::memcpy(eng->input_data(0), rgb, eng->inputs()[0].bytes);
eng->invoke();
const void* out = eng->output_data(0);        // read with outputs()[0] desc
```

Supporting pieces are equally decoupled and reusable on their own: `preprocess.*`
(letterbox), `yolo.*` (decode + NMS), `image_proc.*` / `tile.*` (image-to-image
decode and tiling), `parallel.*` (deterministic intra-op data parallelism), and
`tensor_utils.*` (quantize / dequantize). The built-in engine's core
(`src/engine/`, umbrella header `engine/ii.h`) is also usable entirely
standalone. To embed, link the `ii_core` target (e.g. via `add_subdirectory()`) —
it propagates the `src/` include path.

## Architecture

The whole codebase lives in one namespace, **`ii`**, and compiles into the
`ii_core` static library plus a thin `ii` executable (`main()` in `ii.cpp` + arg
parsing in `cli.*`). Inside the library the code is split into small,
single-purpose modules so backends and platform features turn on and off
independently:

- **`inference.*`** — the backend-agnostic `ii::Engine` interface and the
  `make_engine()` factory. Each backend lives in its own translation unit
  (`inference_tflite.cpp`, `inference_tensorrt.cpp`, `inference_directml.cpp`,
  `engine/backend.cpp`) and is linked in only when its CMake option is enabled.
- **`engine/`** — the built-in pure-C++ engine: tensors, math kernels, the graph
  + op registry + executor, the ONNX loader, and the `ii::Engine` adapter. The
  op table (`kernels.cpp`) is the extension point for new layers.
- **`parallel.*`** — the project-wide parallelism primitive,
  `parallel_for(count, min_grain, body)`: splits a range across cores, backed by
  a reused allocation-free worker pool, bit-identical to serial, and collapsing
  to a direct call on a single core.
- **`preprocess.*` / `image_proc.*` / `tile.*`** — letterbox preprocessing,
  output decoding, and tiling for image-to-image models.
- **`yolo.*` / `yolo_render.*`** — YOLOv8 decoding, NMS and box rendering.
- **`display.*`** — abstract on-screen output (Wayland on Linux, D3D11 on
  Windows, stub elsewhere).
- **`camera.*` / `frame_source.*`** — abstract video source (V4L2 on Linux, Media
  Foundation on Windows, stub elsewhere) feeding a unified inference loop.
- **`delegate.*`** — selection of an optional external TFLite delegate, kept out
  of the inference core so the runner is not tied to any accelerator.
- **`stats.* / sysmon.* / csv_export.*`** — FPS statistics, resource monitoring
  and CSV export.

Adding a backend is local: implement `ii::Engine`, register a branch in
`make_engine()`, add the `target_sources`/`INF_HAS_*` block in CMake, and map the
native dtype enum to `ii::DType` — no changes to the runner.

## License

Released under the [MIT License](LICENSE).
