# ii

An efficient, embeddable C++ library for neural network inference (first and foremost) and training.

- **Runs everywhere** — from weak CPUs without a GPU to discrete accelerators; Windows, Linux, macOS, Android.
  One binary for any x86-64 / ARM64 CPU: SIMD kernels for every instruction set level, chosen at run time.
- **Chooses the backend itself** — own backends (`ii-cpu`, `ii-cuda`, `ii-vulkan`, …) and third-party ones
  (TensorRT, ONNX Runtime, OpenVINO, CoreML), based on cost, memory and transfers; splits a model across devices
  automatically.
- **Correct by construction** — the reference backend `ii-ref` and differential tests for every fast path.
- **Easy to embed** — zero mandatory dependencies, no global state, stable C ABI.
- **Formats** — GGUF, safetensors, ONNX, native `.ii`.
- **`ii` CLI** — `info`, `run`, `chat`, `bench`, `inspect`, `quantize`, `convert`.

## Example

```cpp
#include <ii/ii.h>

auto context = ii::Context::create().value();
ii::GraphBuilder b("mlp");
ii::Var x = b.input("x", ii::DType::kF32, {"N", 4});              // symbolic batch size
ii::Var h = ii::relu(ii::linear(x, b.constant(w1), b.constant(b1)));
b.output("probabilities", ii::softmax(ii::linear(h, b.constant(w2))));

auto program = ii::Model::from_builder(context, std::move(b)).value().compile().value();
auto session = program.new_session().value();
auto outputs = session.run({{"x", batch}}).value();
```

ONNX models load directly — `ii::Model::load(context, "yolov8n.onnx")` — and from the command line:

```sh
ii run resnet50.onnx --input data=cat.jpg --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225 --top 5
ii run bert.onnx --inputs inputs.npz --save-outputs outputs.npz
ii bench yolov8n.onnx
ii_example_yolo yolov8n.onnx bus.jpg result.jpg        # examples/yolo: detection with NMS, boxes drawn
```

More: [examples/](examples) (C++ and C), `ii inspect model.onnx`, `ii run model.onnx --profile trace.json`, `ii ops`.

## Build

```sh
cmake --workflow --preset linux-gcc-release        # or windows-msvc-release, macos-clang-release, ...
./build/linux-gcc/bin/Release/ii info --roofline
```

Details: [building guide](docs/guides/building.md), [contributing](CONTRIBUTING.md).

> Status: stages 0 (foundation), 1 (tensors, IR, OpSet v1, the reference backend `ii-ref`, runtime, C++ and C
> APIs), 2 (the optimized CPU backend `ii-cpu`: SIMD for generic / AVX2 / AVX-512 / NEON, GEMM on par with
> OpenBLAS, operation fusion, profiler, Android) and 3 (ONNX import, convolutions, `.npy`/`.npz`, images in the CLI,
> YOLOv8 example, model tests against ONNX Runtime) are implemented. See [architecture](docs/architecture.md), [OpSet reference](docs/opset.md),
> [roadmap](docs/roadmap.md) (in Russian), [testing and benchmarks](docs/testing-and-benchmarks.md),
> [CI/CD](docs/ci.md), [adding an operation](docs/guides/adding-an-op.md).

## License

[MIT](LICENSE)
