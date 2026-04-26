# IREE — Agent Instructions

IREE ("eerie") is an MLIR-based end-to-end ML compiler and runtime. This is a large C++/Python project with dual CMake + Bazel build systems.

## Build Systems (pick one)

**CMake** (primary for most developers):
```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```
- Use `cmake --preset new-{os}-dev` if you create a `CMakeUserPresets.json` including `build_tools/cmake/presets/all.json`
- Run tests: `ctest --test-dir build --output-on-failure` or use `./build_tools/cmake/ctest_all.sh build`
- Tests default to disabling GPU drivers. Set `IREE_VULKAN_DISABLE=0`, `IREE_CUDA_ENABLE=1`, etc. to enable.

**Bazel** (version 8.6.0, pinned in `.bazelversion`):
```bash
python configure_bazel.py   # generates configured.bazelrc
bazel build //compiler/...
bazel test //...
```
- Run `python configure_bazel.py` before first Bazel build — it writes `configured.bazelrc` based on your compiler.
- Set `CC` and `CXX` env vars before running `configure_bazel.py` for clang builds.
- Use `--config=localdev` for faster local iteration (enables Skymeld).

**Both systems coexist.** BUILD.bazel files are the source of truth; CMakeLists.txt files are auto-generated via `bazel_to_cmake.py`. Edit BUILD.bazel, then run the bazel-to-cmake conversion (pre-commit does this automatically).

## Submodules

- **Full build**: `git submodule update --init --recursive` (large — llvm-project is ~GBs)
- **Runtime-only**: `bash build_tools/scripts/git/update_runtime_submodules.sh` (much smaller set)
- CMake will error if submodules are missing (controlled by `IREE_ERROR_ON_MISSING_SUBMODULES`)

## Linting & Formatting

Run `pre-commit run --all-files` or install the hook with `pre-commit install`.
- C/C++: clang-format (LLVM style)
- Python: Black
- Bazel: buildifier
- Markdown: markdownlint (config in `docs/.markdownlint.yml`)
- Typos: typos (config in `build_tools/linters/typos.toml`)
- CI lint workflow requires `third_party/llvm-project` submodule initialized

## Testing

- **CMake**: `ctest --test-dir <build-dir>` — tests are labeled with directory paths and driver tags (e.g., `driver=vulkan`)
- **Bazel**: `bazel test //...` — tests use tags for filtering
- GPU tests (CUDA/HIP/Vulkan/Metal) are **disabled by default** in ctest
- Windows and macOS have known test exclusions in `build_tools/cmake/ctest_all.sh`

## Key Directories

| Path | Purpose |
|------|---------|
| `compiler/` | IREE compiler (MLIR dialects, pipelines, target backends) |
| `runtime/` | Runtime (HAL drivers, VM, base libraries, Python bindings) |
| `build_tools/` | CMake functions, CI scripts, Bazel config, linters |
| `tests/` | End-to-end and integration tests |
| `samples/` | Example projects and compiler plugins |
| `integrations/` | TensorFlow, PJRT integrations |
| `third_party/` | Submodules: llvm-project, stablehlo, torch-mlir, etc. |
| `llvm-external-projects/` | IREE dialects as LLVM external project |

## Python

- Compiler bindings: `pip install compiler/` (from source tree) or `pip wheel compiler/`
- Runtime build deps: `pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt`
- Python 3.10+ required. Use virtual environments.

## Conventions

- C++17 standard, C11 standard
- Files named `BUILD.bazel` (not `BUILD`) — enforced by pre-commit
- `third_party/` excluded from pre-commit hooks
- Commit formatting: Black for Python, LLVM style for C++ — use `.git-blame-ignore-revs` to skip format-only commits
- DCO sign-off required on commits (`.github/dco.yml`)

## CI

- GitHub Actions: `ci.yml` orchestrates builds across Linux/macOS/Windows
- Lint runs on every PR via `lint.yml`
- Package CI via `pkgci.yml` for Python wheel builds
- Many platform-specific workflows in `.github/workflows/`

## Maintainers

See `MAINTAINERS.md` and `.github/CODEOWNERS` for component owners. Key contacts: @benvanik (runtime), @stellaraccident (compiler C API, Python), @MaheshRavishankar (codegen), @ScottTodd (CI/docs/tools).
