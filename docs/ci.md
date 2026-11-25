# ii — CI/CD (GitHub Actions)

> Status: plan. Workflows are introduced stage by stage (see §8). Test content is defined in
> [testing-and-benchmarks.md](testing-and-benchmarks.md); this document defines *where and when* it runs.

## 1. Principles

1. **Same commands locally and in CI.** Every CI build is a CMake *workflow preset*
   (`cmake --workflow --preset ci-linux-gcc-release` = configure + build + test). A developer reproduces any CI job
   with one command; YAML contains no build logic.
2. **Fast PR feedback.** Target ≤ 15 min wall clock for the PR pipeline: compiler cache, parallel matrix, docs-only
   changes skip builds, superseded runs are cancelled.
3. **Tier-1 platforms are gating.** Windows, Linux (x86-64, ARM64) and macOS must be green to merge.
4. **Hosted runners for correctness, a fixed self-hosted machine for GPU and benchmarks.** Hosted runners are noisy
   and have no NVIDIA GPU; benchmark history is only meaningful on one stable machine.
5. **Supply-chain hygiene.** Third-party actions are pinned by commit SHA and updated by Dependabot; least-privilege
   `permissions:` per workflow; self-hosted runners never execute untrusted code (§5).

## 2. Repository layout

```
.github/
├── workflows/
│   ├── ci.yml          PR + push: lint, tier-1 build/test matrix, sanitizers, ISA matrix, fuzz smoke
│   ├── android.yml     PR + push: Android NDK build, tests on emulator              (stage 2)
│   ├── gpu.yml         self-hosted NVIDIA: ii-cuda, TensorRT differential tests      (stage 5)
│   ├── nightly.yml     schedule: real models, long fuzzing, benchmarks with history (stage 1+)
│   ├── release.yml     tag v*: packages for all platforms, GitHub Release           (after stage 3)
│   ├── docs.yml        master: Doxygen + docs site → GitHub Pages                     (stage 1)
│   └── codeql.yml      PR + weekly: CodeQL C++ security analysis                    (stage 0)
├── actions/
│   └── setup/action.yml   composite: toolchain, Ninja, ccache/sccache, Python deps
├── dependabot.yml         weekly updates for github-actions
├── CODEOWNERS
├── pull_request_template.md
└── ISSUE_TEMPLATE/        bug report (with `ii info --json` output), feature request
```

## 3. `ci.yml` — the main pipeline

Triggers: `pull_request`, `push` to `master`, `workflow_dispatch`.
`concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }`.
A `changes` job (paths filter) lets docs-only changes skip build jobs while still reporting required checks.

### Jobs

| Job | Runner | What it does |
|-----|--------|--------------|
| `lint` | `ubuntu-24.04` | clang-format check, SPDX header check, CMake formatting, typos, clang-tidy on changed files |
| `build-test` (matrix below) | hosted | configure, build, `ctest`, upload JUnit + logs on failure |
| `isa-matrix` | `ubuntu-24.04` | `ii-cpu` tests with `II_CPU_ISA=generic`, `avx2`, `avx_vnni`; under Intel SDE: `avx512` (`-skx`), `avx512_vnni_bf16`/`amx` (`-spr`) |
| `sanitizers` | `ubuntu-24.04` | Clang ASan+UBSan (all tests), TSan (threading/runtime tests) |
| `msvc-asan` | `windows-2025` | MSVC `/fsanitize=address` — catches Windows-specific memory bugs |
| `fuzz-smoke` | `ubuntu-24.04` | preset `linux-clang-fuzz`: each libFuzzer target for 60 s on its corpus (`tests/fuzz/corpus/`); part of the `sanitizers` job |
| `abi-interop` | `windows-2025` | `ii.dll` built with MinGW consumed from an MSVC-built C program, and vice versa |
| `examples` | `ubuntu-24.04`, `windows-2025`, `macos-15` | builds `examples/` against the *installed* package (`find_package(ii)`) — verifies packaging |

### `build-test` matrix

| OS image | Compiler | Configs |
|----------|----------|---------|
| `windows-2025` | MSVC (latest) | Debug, Release |
| `windows-2025` | clang-cl | Release |
| `windows-2025` | MinGW-w64 GCC (MSYS2 UCRT64) | Release |
| `ubuntu-24.04` | GCC 14 | Debug, Release |
| `ubuntu-24.04` | Clang (latest stable) | Release |
| `ubuntu-24.04-arm` | GCC 14, Clang | Release |
| `macos-15` (ARM64) | Apple Clang | Debug, Release |
| `macos-15-intel` (x86-64) | Apple Clang | Release |

Notes:
- Image labels are pinned explicitly (not `*-latest`) and bumped deliberately.
- Debug builds enable `II_ASSERT` and `Graph::verify()` after every pass — they catch logic errors Release hides.
- Warnings are errors in CI (`-Werror` / `/WX`).
- Compiler cache: `ccache` for every toolchain (MSVC included), keyed by OS + preset, restored from `master`.

### Required checks for `master`

`lint`, all `build-test` entries, `sanitizers`, `isa-matrix`, `abi-interop`, `examples`; later `android` (stage 2)
and model tests (stage 3). Branch protection: PR required, linear history, required checks green.

## 4. Android and AVX-512 jobs (stage 2, in `ci.yml`)

- `android`: NDK from the runner image (`$ANDROID_NDK_LATEST_HOME`); presets `android-arm64` and `android-x86_64`
  (`ANDROID_PLATFORM=26`). `arm64-v8a` is built only (its NEON code paths run natively on `ubuntu-24.04-arm` and
  macOS ARM64, and on real devices through the same presets); `x86_64` is tested on an emulator
  (`reactivecircus/android-emulator-runner`, KVM enabled). For Android builds CMake sets
  `CMAKE_CROSSCOMPILING_EMULATOR` to `scripts/android_run.py`, which pushes each executable (and any host paths in its
  arguments and `II_*` variables) with `adb` and runs it, so the job simply calls `ctest`. The same works locally
  with a phone connected over adb.
- `avx512-sde`: the AVX-512 kernels on hosted runners without AVX-512 — the ii-cpu tests run under Intel SDE
  (`-spr`, `petarpetrovt/setup-sde`), where the ISA-parameterized tests include `avx512`.

## 5. `gpu.yml` — self-hosted NVIDIA runner (stage 5)

Runner: the development machine (Windows 11, RTX 5080, CUDA 13.2, TensorRT 10.16, MSVC), labels
`[self-hosted, windows, x64, gpu, cuda, tensorrt]`. A Linux GPU runner (WSL2 or a second machine) can be added later.

Jobs: build with `II_BACKEND_CUDA=ON`, `II_BACKEND_TENSORRT=ON`; differential tests of `ii-cuda` and TensorRT vs
`ii-ref`; TensorRT engine-cache tests; GPU micro-benchmarks.

**Security (mandatory for a public repository):**
- Triggers: `push` to `master`, `schedule`, `workflow_dispatch`, and `pull_request` **only** when
  `github.event.pull_request.head.repo.full_name == github.repository` **and** the PR has the `gpu` label (applied by
  a maintainer). Pull requests from forks never reach the self-hosted runner.
- Repository setting: "Require approval for all outside collaborators" for workflow runs.
- The runner is a service under a dedicated non-admin Windows account, assigned to a runner group restricted to this
  repository; workspace cleaned after each job; no long-lived secrets on the machine beyond what the job needs.

## 6. `nightly.yml` (from stage 1, growing)

Triggers: `schedule` (e.g. 02:00 UTC), `workflow_dispatch`.

| Job | Runner | What it does |
|-----|--------|--------------|
| `models` | hosted (small models), self-hosted (large) | `scripts/fetch_models.py` downloads models listed in `tests/models/manifest.json` (URL + sha256), cached by manifest hash; logits / detections / perplexity vs references (ONNX Runtime, transformers) |
| `fuzz-long` | `ubuntu-24.04` | 30 min per target; corpus persisted between runs (cache/artifact); crashes uploaded as artifacts and turned into regression tests |
| `bench` | self-hosted (fixed machine) | micro + e2e benchmarks → JSON; history and charts via `github-action-benchmark` on the `gh-pages` branch (`/bench`); alert comment / issue on > 5% regression |
| `coverage` | `ubuntu-24.04` | llvm-cov report (optionally uploaded to Codecov) |

Benchmark hygiene on the self-hosted machine: runs only at night, "High performance" power plan, fixed thread
counts, P-core-only and all-core variants, machine profile (`ii info --json --roofline`) stored with every result.

Large repository caches are limited (GitHub cache quota per repo), so big models (LLMs) live in a persistent local
cache directory on the self-hosted runner; hosted runners only fetch small models.

## 7. `release.yml` (first release after stage 3)

Trigger: tag `v*.*.*`.

| Artifact | Built on | Notes |
|----------|----------|-------|
| `ii-<ver>-windows-x64.zip` | `windows-2025`, MSVC | `ii.exe`, `ii.dll`, import lib, headers, CMake config |
| `ii-<ver>-linux-x86_64.tar.gz`, `-aarch64` | `manylinux_2_28` containers | old glibc baseline → runs on most distros |
| `ii-<ver>-macos-universal.tar.gz` | `macos-15` + `macos-15-intel` | per-arch builds merged with `lipo` (ISA-specific kernels need per-arch flags) |
| `ii-<ver>-android.zip` | `ubuntu-24.04` | prebuilt `.so` per ABI + headers |
| `ii-backend-cuda`, `ii-backend-tensorrt` plugins | self-hosted GPU runner | separate archives; core packages stay SDK-free |

Every release: full test suite on the packaged binaries, `SHA256SUMS`, build provenance attestations
(`actions/attest-build-provenance`), release notes generated from merged PRs.

## 8. Introduction by stage

| Stage | CI additions |
|-------|--------------|
| 0 | `ci.yml` (`lint`, `build-test` tier-1 matrix, `sanitizers`, `msvc-asan`), `codeql.yml`, composite `setup` action, Dependabot, PR/issue templates, branch protection |
| 1 | done: `examples` (package installed, examples built with `find_package(ii)` and run on Linux, Windows, macOS), fuzz smoke (`linux-clang-fuzz` preset: libFuzzer on the IR text parser and JSON, 60 s each; other builds replay the corpora). Moved to stage 3: `abi-interop`, `docs.yml`, `nightly.yml` |
| 2 | done: `avx512-sde` (ii-cpu tests under Intel SDE), `android` (arm64-v8a build, x86_64 tests on an emulator through `scripts/android_run.py`), `generic` kernel variant tested on Linux. Moved: self-hosted runner for nightly benchmarks — with `nightly.yml` (stage 3) |
| 3 | model tests (YOLOv8, ResNet, MobileNet, BERT) in PR (tiny/small) and nightly (full); ONNX/protobuf fuzzers; `release.yml` → v0.1 |
| 4 | LLM model tests (tiny-llama in PR, real GGUF nightly on self-hosted), GGUF/tokenizer/Jinja fuzzers, LLM benchmarks vs llama.cpp |
| 5 | `gpu.yml` (ii-cuda, TensorRT), GPU benchmarks, GPU plugin artifacts in releases |
| 6+ | iOS, WebAssembly (Emscripten), Windows ARM64 (`windows-11-arm`), Vulkan (lavapipe software driver on hosted Linux), Metal on macOS |

## 9. Cost notes

- For a **public** repository hosted runner minutes (including macOS and ARM64 Linux) are free; for a **private**
  one macOS minutes are billed at a multiplier and the matrix should be trimmed (e.g. macOS only on `master`).
- Self-hosted runner minutes are free but use the development machine; heavy jobs are scheduled at night.
