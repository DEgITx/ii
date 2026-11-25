# Contributing to ii

## Ground rules

- **Correctness first.** Every fast path has a reference implementation and a test comparing them.
- **Measure.** Performance changes come with benchmark numbers (before/after, machine, build).
- **English** for code, comments, commit messages and documentation.
- **No platform `#ifdef`s** in library code outside `src/core/platform/`.
- **No exceptions** across the library boundary; use `ii::Status` / `ii::Result<T>`.
- **No global mutable state**; configuration flows through objects (`Context`, options structs).

The architecture and its reasoning are described in [docs/architecture.md](docs/architecture.md).

## Workflow

```sh
cmake --workflow --preset <your-platform>-debug     # build and test, see docs/guides/building.md
python scripts/check_format.py --fix                # clang-format 19
python scripts/check_spdx.py                        # every source file starts with an SPDX header
```

Pull requests run the CI pipeline described in [docs/ci.md](docs/ci.md); all required checks must pass.

## Code style

- C++20. Formatting is defined by `.clang-format` (enforced in CI), static analysis by `.clang-tidy`.
- Naming: types `PascalCase`, functions and variables `snake_case`, constants `kPascalCase`, members `name_`,
  files `snake_case.h` / `snake_case.cpp`.
- Every source file starts with `// SPDX-License-Identifier: MIT`.
- Public headers (`include/ii/`) document every declaration with Doxygen `///` comments and contain no third-party
  types.
- ISA-specific kernel files follow the rules in `cmake/IiIsa.cmake` (own namespace, no shared inline code).

## Tests

- Unit tests live in `tests/unit/<area>/<name>_test.cpp` (GoogleTest).
- A new public function or a bug fix comes with a test. A bug fix test fails without the fix.
- Tests must be deterministic: fixed seeds, no dependence on timing beyond generous upper bounds.
