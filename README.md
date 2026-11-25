# ii

An efficient, embeddable C++ library for neural network inference (first and foremost) and training.

- **Runs everywhere** — from weak CPUs without a GPU to discrete accelerators; Windows, Linux, macOS, with Android
  next. One binary for any x86-64 / ARM64 CPU.
- **Chooses the backend itself** — own backends (`ii-cpu`, `ii-cuda`, `ii-vulkan`, …) and third-party ones
  (TensorRT, ONNX Runtime, OpenVINO, CoreML), based on cost, memory and transfers; splits a model across devices
  automatically.
- **Correct by construction** — the reference backend `ii-ref` and differential tests for every fast path.
- **Easy to embed** — zero mandatory dependencies, no global state, stable C ABI.
- **Formats** — GGUF, safetensors, ONNX, native `.ii`.
- **`ii` CLI** — `info`, `run`, `chat`, `bench`, `inspect`, `quantize`, `convert`.

> Status: design. See [architecture](docs/architecture.md), [roadmap](docs/roadmap.md) (in Russian),
> [testing and benchmarks](docs/testing-and-benchmarks.md), [CI/CD](docs/ci.md).

## License

[MIT](LICENSE)
