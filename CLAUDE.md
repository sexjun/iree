# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IREE (Intermediate Representation Execution Environment, pronounced "eerie") is an MLIR-based end-to-end compiler and runtime for Machine Learning models. It lowers ML models to a unified intermediate representation that scales from datacenter servers down to mobile and edge deployments.

- **Website**: https://iree.dev/
- **License**: Apache 2.0 with LLVM Exceptions
- **C++ Standard**: C++17 (no extensions)
- **C Standard**: C11

## Quick Start

```bash
git clone https://github.com/iree-org/iree.git
cd iree
git submodule update --init
```

## Build System

IREE uses **CMake (primary)** with **Ninja** generator. CMake files are auto-generated from Bazel BUILD files via `build_tools/bazel_to_cmake/bazel_to_cmake.py`.

### Standard Build

```bash
# Configure
cmake -G Ninja -B build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

# Build
cmake --build build/

# Run all tests
ctest --test-dir build/ --output-on-failure
```

### Common CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `IREE_BUILD_COMPILER` | ON | Build the IREE compiler |
| `IREE_BUILD_TESTS` | ON | Build unit tests |
| `IREE_BUILD_PYTHON_BINDINGS` | OFF | Build Python bindings |
| `IREE_BUILD_DOCS` | OFF | Build documentation |
| `IREE_BUILD_SAMPLES` | ON | Build sample projects |
| `IREE_ENABLE_ASSERTIONS` | OFF | Enable assertions |
| `IREE_ENABLE_LLD` | OFF | Use LLD linker |
| `IREE_HAL_DRIVER_CUDA` | OFF | Enable CUDA HAL driver |
| `IREE_HAL_DRIVER_VULKAN` | OFF | Enable Vulkan HAL driver |
| `IREE_HAL_DRIVER_METAL` | OFF | Enable Metal HAL driver |
| `IREE_TARGET_BACKEND_CUDA` | OFF | Enable CUDA target backend |
| `IREE_TARGET_BACKEND_ROCM` | OFF | Enable ROCm target backend |

### Convenience Build Scripts

Located in `build_tools/cmake/`:
- `build_all.sh` - Full build with CI-like settings
- `ctest_all.sh` - Run all tests with environment-aware filtering
- `build_and_test_asan.sh` - Build/test with AddressSanitizer
- `build_and_test_tsan.sh` - Build/test with ThreadSanitizer
- `build_and_test_ubsan.sh` - Build/test with UndefinedBehaviorSanitizer

### Python Bindings

```bash
# Compiler
python -m pip wheel compiler/
# Runtime
python -m pip wheel runtime/
```

## Running Tests

```bash
# All tests
ctest --test-dir build/ --output-on-failure

# Run tests matching a pattern
ctest --test-dir build/ -R "some_test_name"

# Run only tests matching a label
ctest --test-dir build/ --label-regex "driver=vulkan"

# Run with sanitizer build
./build_tools/cmake/build_and_test_asan.sh
```

### Test Frameworks

- **GoogleTest (gtest/gmock)** for C++ unit tests
- **LLVM lit** for `.mlir.test` files (ShTest format)
- **CTest** as the test runner

### Test File Locations

- `tests/e2e/` - End-to-end tests (matmul, convolution, linalg, etc.)
- `tools/test/` - Tool-specific tests
- `runtime/src/iree/hal/cts/` - HAL Conformance Test Suite
- Inline tests: `*.mlir.test` files throughout the codebase

## Code Architecture

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `compiler/` | MLIR-based compiler - MLIR dialects, passes, pipelines |
| `runtime/` | Runtime with HAL (Hardware Abstraction Layer) and VM |
| `tools/` | CLI tools: iree-compile, iree-run-module, iree-benchmark-module |
| `tests/` | End-to-end and integration tests |
| `samples/` | Example projects demonstrating IREE usage |
| `build_tools/` | Build infrastructure, CMake helpers, CI tooling |
| `docs/` | Website documentation (MkDocs-based) |
| `third_party/` | Vendored dependencies (llvm-project, stablehlo, etc.) |
| `integrations/` | External framework integrations (PJRT, TensorFlow) |

### Compiler (`compiler/src/iree/compiler/`)

- **`Dialect/`** - IREE-specific MLIR dialects:
  - `Flow/` - Tensor program modeling, compute workload partitioning
  - `HAL/` - Hardware Abstraction Layer for buffer and execution management
  - `Stream/` - Device placement and asynchronous scheduling
  - `VM/` - Abstract Virtual Machine
  - `Util/` - Common types across IREE dialects
- **`Codegen/`** - Device code generation (LLVMCPU, LLVMGPU, SPIRV, etc.)
- **`DispatchCreation/`** - Dispatch formation
- **`GlobalOptimization/`** - Global optimization passes
- **`InputConversion/`** - Frontend dialect conversions (StableHLO, Torch, TOSA)
- **`Pipelines/`** - Compilation pipeline definitions
- **`PluginAPI/`** - Plugin infrastructure

### Compiler Plugins (`compiler/plugins/`)

- **Input**: `input/StableHLO/`, `input/Torch/`, `input/TOSA/`
- **Targets**: `target/CUDA/`, `target/LLVMCPU/`, `target/MetalSPIRV/`, `target/ROCM/`, `target/VMVX/`, `target/VulkanSPIRV/`, `target/WebGPUSPIRV/`

### Runtime (`runtime/src/iree/`)

- **`hal/`** - Hardware Abstraction Layer (allocator, buffer, command_buffer, device, drivers)
  - `drivers/` - amdgpu, cuda, hip, local-sync, local-task, metal, null, vulkan
- **`vm/`** - Virtual Machine (bytecode interpreter, native modules, shims, stack)
- **`base/`** - Base utilities (logging, status, arena, etc.)
- **`io/`** - I/O utilities (parameter handling, streams)
- **`task/`** - Task scheduling system
- **`tokenizer/`** - Text tokenization
- **`tooling/`** - Tooling utilities

## Development Workflow

### Code Formatting

Pre-commit hooks are configured (`.pre-commit-config.yaml`). Run before committing:

```bash
pre-commit run --all-files
```

This runs:
- **clang-format** for C/C++ files
- **black** for Python files
- **markdownlint** for documentation
- Trailing whitespace and EOF fixers

### Bazel to CMake Sync

After modifying BUILD files, regenerate CMake files:

```bash
./build_tools/bazel_to_cmake/bazel_to_cmake.py
```

### Key Maintainers

| Component | Maintainer |
|-----------|------------|
| Project direction | Stella Laurenzo (@stellaraccident) |
| Runtime, HAL | Ben Vanik (@benvanik) |
| Codegen, Optimizations | Mahesh Ravishankar (@MaheshRavishankar) |
| CI, Docs, Tools | Scott Todd (@ScottTodd) |
| Torch Input Pipeline | Rob Suderman (@rsuderman) |
| Default HAL Drivers | Lei Zhang (@antiagainst) |

See `.github/CODEOWNERS` for path-based reviewers.

## Important Notes

- **third_party/ is vendored** - Do not modify files in `third_party/` directly
- **CMake files are generated from Bazel** - Edit BUILD files first, then run bazel_to_cmake
- **Large codebase** - Use parallel agents for exploration; be selective about context loading
- **MLIR dependency** - IREE is built on top of MLIR/LLVM; familiarity with MLIR dialects and pass infrastructure is essential
