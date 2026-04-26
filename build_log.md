# Build Log

## Context
- Date: 2026-04-26
- Workspace: /Users/alanchen/workspace/iree
- Requested action: Build the project, fix problems if any, and record issue/cause/fix log.

## Initial Build Attempt
- Tool: Build_CMakeTools
- Result: Failed during early LLVM/MLIR compilation.
- First failing unit: `third_party/llvm-project/mlir/lib/IR/BuiltinDialectBytecode.cpp`
- Observed compiler: `/usr/bin/clang++` (Apple clang 21.0.0)
- Failure summary: `clang++` crashed with `Illegal instruction: 4` while compiling, indicating a compiler/toolchain failure instead of a source-level compile error.

## Current Hypothesis
- Root cause is the local Apple clang 21 toolchain crashing on this machine/build configuration.
- Smallest practical remediation is to reconfigure the build to use Homebrew LLVM 19 instead of `/usr/bin/clang++`.

## Actions Log
1. Loaded repository and C++ instructions.
2. Ran CMake build with Build_CMakeTools.
3. Collected diagnostics and inspected `build/CMakeCache.txt`.
4. Confirmed CMake cache was using `/usr/bin/clang` and `/usr/bin/clang++`.
5. Confirmed Homebrew LLVM 19 is installed at `/usr/local/opt/llvm@19/bin/clang++`.
6. Preparing to reconfigure the build with Homebrew LLVM 19.

## Status
- In progress.
