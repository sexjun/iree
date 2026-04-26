# IREE Copilot Instructions

IREE (Intermediate Representation Execution Environment) is an MLIR-based end-to-end ML compiler and runtime. It lowers ML models to a unified IR targeting everything from datacenter GPUs to embedded devices.

## Architecture Overview

The repo is split into two major independent components:

- **`compiler/`** — MLIR-based compiler. Uses LLVM/MLIR coding conventions. Lives under namespace `mlir::iree_compiler`. Must not depend on the runtime.
- **`runtime/`** — C/C++ runtime. Uses Google style. Lives under namespace `iree`. Must not depend on the compiler.

Key compiler dialects (under `compiler/src/iree/compiler/Dialect/`):
- **Flow** — dispatch region formation and workload analysis
- **Stream** — async execution and resource management
- **HAL** — hardware abstraction layer (executables, devices, commands)
- **VM** — virtual machine bytecode for portable execution
- **Encoding / LinalgExt** — data layout and extended Linalg ops

The compiler pipeline: Input dialects (StableHLO / TOSA / Linalg) → Flow → Stream → HAL → VM bytecode or native executables.

Codegen backends are in `compiler/src/iree/compiler/Codegen/` organized by target: `LLVMCPU`, `LLVMGPU`, `SPIRV`, `ROCDL`, etc.

## Build System

IREE uses **both Bazel and CMake**. CMakeLists.txt files in most directories are **auto-generated** from `BUILD.bazel` files via `build_tools/bazel_to_cmake/bazel_to_cmake.py`. When modifying build rules, edit the `BUILD.bazel` file and regenerate:

```shell
python build_tools/bazel_to_cmake/bazel_to_cmake.py path/to/BUILD.bazel
# Or regenerate all:
python build_tools/bazel_to_cmake/bazel_to_cmake.py
```

Auto-generated CMakeLists.txt files have a header comment indicating this. Content below `### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###` is preserved during regeneration.

### CMake Build

```shell
# Standard build (uses Ninja + lld for speed)
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_LLD=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build

# Or use the CI script (handles all flags):
./build_tools/cmake/build_all.sh build
```

Key CMake options:
- `-DIREE_BUILD_COMPILER=ON/OFF` — build the compiler (default ON)
- `-DIREE_BUILD_TESTS=ON/OFF` — build tests (default ON)
- `-DIREE_BUILD_PYTHON_BINDINGS=ON/OFF` — Python bindings (default OFF)
- `-DIREE_HAL_DRIVER_CUDA=ON/OFF`, `-DIREE_HAL_DRIVER_HIP=ON/OFF` — GPU backends

## Testing

### Run all tests (CMake)

```shell
# Build test deps first (needed for e2e tests)
cmake --build build --target iree-test-deps

cd build && ctest --output-on-failure -j$(nproc)
# Or use the CI script:
./build_tools/cmake/ctest_all.sh build
```

### Run a single test (CMake)

```shell
# Compiler lit test (MLIR FileCheck)
cd build && ctest -R "iree/compiler/Dialect/VM/Conversion/MathToVM/test/arithmetic_ops.mlir.test"

# Runtime C++ unit test
cd build && ctest -R "iree/base/bitfield_test"

# E2E check test
cd build && ctest -R "tests/e2e/stablehlo_ops/check_vmvx_local-task_floor.mlir"
```

### Run a single test (Bazel)

```shell
bazel test //compiler/src/iree/compiler/Dialect/VM/Conversion/MathToVM/test:arithmetic_ops.mlir.test
bazel test //runtime/src/iree/base:bitfield_test
```

### Test types

| Type | Tooling | Notes |
|------|---------|-------|
| Compiler lit tests (`iree_lit_test`) | MLIR FileCheck via `iree-opt` | `.mlir` files in `test/` dirs |
| Runtime unit tests (`iree_cc_test`) | GoogleTest | `foo_test.cc` next to `foo.cc` |
| E2E tests (`iree_check_test`) | Custom `check` framework | In `tests/e2e/`; use `util.unfoldable_constant` to prevent compile-time folding |

## Linting and Formatting

Uses [pre-commit](https://pre-commit.com/). Install and run:

```shell
pip install pre-commit
pre-commit install   # run on each commit automatically
pre-commit run --all-files  # run manually
```

Formatters used: `clang-format` (C/C++), `black` (Python), `markdownlint` (docs).

## Key Conventions

### Build rules

Use these CMake functions (matching Bazel counterparts):
- `iree_cc_library` / `iree_cc_binary` / `iree_cc_test`
- `iree_lit_test_suite` for compiler FileCheck tests
- `iree_bytecode_module` for compiling MLIR to IREE modules

In `iree_cc_test` deps, use `iree::testing::gtest_main` instead of the bare `gtest_main`.
In test source files, `#include "iree/testing/gtest.h"` rather than gtest/gmock headers directly.

### Compiler (MLIR/LLVM style)

- Follows [MLIR style guide](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide) / [LLVM coding standards](https://llvm.org/docs/CodingStandards.html)
- **IREE deviations**: always use braces with single-line `if`/loops; `static` functions are allowed in anonymous namespaces
- TableGen pass *declarations* use `Pass` suffix (e.g. `def FooBarPass`); implementation files do *not* (e.g. `FooBar.cpp`)
- Dialect op tests live in `.../IR/test/`: `{op_category}_ops.mlir` for printing/parsing, `{op_category}_folding.mlir` for folding/canonicalization

### Runtime (Google style)

- Follows [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

### Commits / PRs

- DCO sign-off required: either cryptographic commit signing or `Signed-off-by:` trailer
- PRs should be small and focused on a single issue

## Repo Structure

```
compiler/   # MLIR compiler (must not depend on runtime)
  src/iree/compiler/
    Dialect/        # Flow, HAL, Stream, VM, etc.
    Codegen/        # Target-specific code generation
    Pipelines/      # End-to-end compilation pipelines
runtime/    # C/C++ runtime (must not depend on compiler)
  src/iree/
    base/           # Utilities, allocators, status codes
    hal/            # Hardware Abstraction Layer
    vm/             # Virtual machine
tests/      # E2E integration tests
build_tools/
  cmake/    # CMake helper functions and build scripts
  bazel_to_cmake/  # Auto-generates CMakeLists.txt from BUILD.bazel
```
