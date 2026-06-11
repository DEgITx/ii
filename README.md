# ii — a backend-agnostic neural network model runner

`ii` is a small, fast C++ command-line runner for neural-network models. Point
it at a model and an image and it will preprocess, run inference, and either
print the raw outputs, draw detections on screen, or write the result back to a
file — across several inference backends behind a single interface.

It is designed to be **embeddable**: the core knows nothing about any specific
framework. Backends (TensorFlow Lite, NVIDIA TensorRT, ONNX Runtime / DirectML)
are selected at build time and dispatched at runtime through one narrow
`inf::Engine` abstraction, so you can drop the runner — or just its inference
layer — into a larger application without dragging in dependencies you don't use.

That same `inf::Engine` abstraction makes `ii` more than a CLI: it is the
foundation of a small, embeddable **inference engine** in its own right. Today
it wraps third-party runtimes behind a uniform interface; the longer-term
direction is to grow into a self-contained inference engine — a native execution
path that runs models directly, with the existing backends as fallbacks. The
architecture is already laid out for that: add a backend, and every feature here
(detection, image-to-image, video, benchmarking) comes along for free.

## Features

- **Multiple inference backends** behind one interface — TensorFlow Lite,
  NVIDIA TensorRT (10+), and ONNX Runtime / DirectML. Pick at build time, choose
  at runtime with `--backend`.
- **Optional external acceleration delegate** for TFLite (NPU / GPU) via
  `--delegate <path>` — or run on CPU with `--no-delegate`.
- **Object detection** — built-in YOLOv8 decoding + NMS, boxes drawn over the
  frame. COCO-80 labels by default, or supply your own class list.
- **Image-to-image models** — super-resolution, low-light enhancement,
  denoising. Show the result on screen or save it to PNG, with optional
  tiling / sliding-window for small-input models and seam feathering.
- **Live video** — V4L2 camera capture (UVC / CSI), YUYV and MJPEG, with
  on-the-fly inference and an on-screen FPS / jitter overlay.
- **On-screen display** — a zero-copy Wayland + EGL + GLES2 window that
  letterboxes the frame and updates a single texture per frame.
- **Benchmarking & comparison** — warmup + timed runs, CPU-vs-delegate
  comparison, and reference-model comparison (e.g. float32 vs INT8).
- **Telemetry** — CPU / memory / thread monitoring and CSV export of
  benchmark, FPS, comparison and system-monitor data.
- **Quantization-aware** — INT8 / UInt8 / Float16 / Float32 tensors; the
  scale / zero-point are read directly from the model.
- **Cross-platform** — Linux (full feature set), plus Windows / macOS as
  CPU-only dev hosts for regression and development.

## Supported backends

| Backend            | CMake option (default) | Notes                                              |
|--------------------|------------------------|----------------------------------------------------|
| TensorFlow Lite    | `USE_TFLITE` (ON)      | CPU + optional external delegate (`--delegate`).   |
| NVIDIA TensorRT    | `USE_TENSORRT` (OFF)   | Requires TensorRT 10+ and the CUDA runtime.        |
| ONNX Runtime / DML | `USE_DIRECTML` (OFF)   | DirectML EP on Windows; CPU EP elsewhere.          |

A single binary can be built with any combination of backends. Use
`--backend <name>` to choose at runtime; built-in backends are listed in
`--help`.

## Building

Requires CMake ≥ 3.16 and a C++20 compiler. [stb](https://github.com/nothings/stb)
(image load / resize) is fetched automatically.

### Linux

```sh
cmake -S src -B build -DTFLITE_ROOT=/usr
cmake --build build -j
```

If TensorFlow Lite is installed in a non-standard location, point the build at
it explicitly:

```sh
cmake -S src -B build \
    -DTFLITE_INCLUDE_DIR=/opt/tflite/include \
    -DTFLITE_LIB=/opt/tflite/lib/libtensorflowlite.so
```

Without an on-screen display or camera (e.g. a headless host):

```sh
cmake -S src -B build -DUSE_DISPLAY=OFF -DUSE_CAMERA=OFF
```

### Windows (MSVC, e.g. via vcpkg)

```sh
cmake -S src -B build -DTFLITE_ROOT=C:/path/to/tflite-prebuilt
cmake --build build --config Release -j
```

The Wayland display and V4L2 camera are Linux-only and default to `OFF` on
other platforms; the runner there works in CPU mode (or with an alternative
backend such as TensorRT / DirectML).

### Build options

| Option            | Default | Purpose                                            |
|-------------------|---------|----------------------------------------------------|
| `USE_TFLITE`      | ON      | Build the TensorFlow Lite backend.                 |
| `USE_TENSORRT`    | OFF     | Build the TensorRT backend (`-DTRT_ROOT=...`).     |
| `USE_DIRECTML`    | OFF     | Build the ONNX Runtime / DirectML backend (`-DORT_ROOT=...`). |
| `USE_DISPLAY`     | ON¹     | Wayland / EGL / GLES2 on-screen output (Linux).    |
| `USE_CAMERA`      | ON¹     | V4L2 camera capture (Linux).                       |

¹ `ON` on Linux, `OFF` elsewhere.

## Quick start

```sh
# Single inference, print raw outputs
./ii model.tflite image.jpg

# Run on CPU (no delegate)
./ii model.tflite image.jpg --no-delegate

# Object detection — draw YOLOv8 boxes in a window
./ii yolov8m_int8.tflite image.jpg --display --yolo --conf 0.4 --iou 0.5

# Live camera detection
./ii yolov8m_int8.tflite --camera --display --yolo --stats

# Super-resolution / enhance — show or save the output image
./ii fsrcnn_qat.tflite image.jpg --display --show-output
./ii fsrcnn_qat.tflite image.jpg --save-output out.png

# Benchmark
./ii model.tflite image.jpg --benchmark --runs 100

# Benchmark without an image (random input)
./ii model.tflite --random-input --benchmark --runs 100

# Compare INT8 against a float32 reference on CPU
./ii yolov8m_int8.tflite --random-input --compare model_fp32.tflite --random-runs 50
```

Run `./ii --help` for the full option list.

## Common options

| Option | Description |
|--------|-------------|
| `--backend <name>`   | Inference backend (`tflite`, `tensorrt`, `directml`). |
| `--delegate <path>`  | Path to an external acceleration delegate / plugin.   |
| `--no-delegate`      | Run on CPU without a delegate.                         |
| `--benchmark`        | Warmup + timed run; `--runs N`, `--warmup N`.         |
| `--threads <N>`      | CPU threads for the interpreter.                      |
| `--display`          | Open a window (Wayland / EGL / GLES2).                |
| `--stats`            | FPS / jitter counter (overlay + stdout).             |
| `--yolo`             | Decode output as YOLOv8 and draw boxes.              |
| `--conf <p>` `--iou <p>` | Confidence / NMS-IoU thresholds.                 |
| `--classes <path>`   | Class-name file (one per line; default COCO-80).     |
| `--show-output`      | Render the model's output as an image (SR / enhance). |
| `--save-output <p>`  | Save the decoded output to PNG.                       |
| `--tile`             | Tiling / sliding window for small-input models.      |
| `--camera [dev]`     | Capture from a V4L2 camera (default `/dev/video0`).   |
| `--compare <path>`   | Run a reference model on CPU and compare outputs.    |
| `--export <prefix>`  | Write benchmark / FPS / comparison data to CSV.      |
| `--sysmon`           | Monitor process & system CPU / memory.               |

## Architecture

The runner is split into small, single-purpose modules so that backends and
platform features can be turned on and off independently:

- **`inference.*`** — the backend-agnostic `inf::Engine` interface and the
  `make_engine()` factory. Each backend lives in its own translation unit
  (`inference_tflite.cpp`, `inference_tensorrt.cpp`, `inference_directml.cpp`)
  and is linked in only when its CMake option is enabled.
- **`delegate.*`** — selection of an optional external TFLite delegate, kept
  separate from the inference core so the runner is not tied to any particular
  accelerator. A delegate path can always be supplied via `--delegate`.
- **`preprocess.*` / `image_proc.*` / `tile.*`** — letterbox preprocessing,
  output decoding, and tiling for image-to-image models.
- **`yolo.*` / `yolo_render.*`** — YOLOv8 decoding, NMS and box rendering.
- **`display.*`** — abstract on-screen output (Wayland backend on Linux, a stub
  elsewhere).
- **`camera.*` / `frame_source.*`** — abstract video source (V4L2 backend on
  Linux, a stub elsewhere) feeding a unified inference loop.
- **`parallel.*` / `stats.*` / `sysmon.*` / `csv_export.*`** — thread pool,
  FPS statistics, resource monitoring and CSV export.

## Use as a library

`ii` is not only a CLI — the inference core is designed to be **embedded in
other applications**. The `inf::Engine` interface (`inference.h`) is fully
self-contained: it pulls in no backend SDK headers, so a host program can depend
on just that one narrow abstraction and link whichever backend(s) it built.

```cpp
#include "inference.h"

auto eng = inf::make_engine("tflite");      // or "tensorrt" / "directml"
inf::Engine::Options opts;
opts.delegate_path = "";                     // "" = CPU
opts.num_threads   = 4;
eng->load("model.tflite", opts);

std::memcpy(eng->input_data(0), rgb, eng->inputs()[0].bytes);
eng->invoke();
const void* out = eng->output_data(0);       // read with outputs()[0] desc
```

Supporting pieces are equally decoupled and reusable on their own:
`preprocess.*` (letterbox), `yolo.*` (decode + NMS), `image_proc.*` / `tile.*`
(image-to-image decode and tiling), and `tensor_utils.*` (quantize / dequantize).
Because backends register behind `make_engine()`, you can drop in the whole
runner or just its inference layer without dragging in dependencies you don't
use. There is no installed library target today — embed by adding the relevant
sources (and the backend's `INF_HAS_*` define) to your own build.

## License

Released under the [MIT License](LICENSE).
