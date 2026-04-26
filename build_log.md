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
6. Reconfigured a clean `build/` directory to use Homebrew LLVM 19.
7. User applied additional fixes and requested a clean rebuild.
8. Removed `build/` again and restarted the build from a clean directory with CMake Tools.
9. Verified the rebuilt tree produced `build/tools/iree-compile` successfully.

## Issue
- Initial build failed while compiling `third_party/llvm-project/mlir/lib/IR/BuiltinDialectBytecode.cpp`.
- The compiler process crashed with `Illegal instruction: 4`.

## Cause
- The build cache was configured to use the system Apple toolchain:
	- `CMAKE_C_COMPILER=/usr/bin/clang`
	- `CMAKE_CXX_COMPILER=/usr/bin/clang++`
- This was a compiler/toolchain crash, not a source-level compile error.
- The failure disappeared after switching the build to Homebrew LLVM 19, which confirms the problem was the local Apple clang 21 toolchain on this machine/build configuration.

## Fix
- Deleted the existing `build/` directory to avoid reusing the old compiler cache.
- Reconfigured the project with:
	- `-DCMAKE_C_COMPILER=/usr/local/opt/llvm@19/bin/clang`
	- `-DCMAKE_CXX_COMPILER=/usr/local/opt/llvm@19/bin/clang++`
- Rebuilt from a clean directory.

## Rebuild Result
- `Build_CMakeTools` completed without the previous compiler crash.
- Follow-up verification with `cmake --build build --target iree-compile -j1 --verbose` reported `ninja: no work to do`.
- Verified output binary:
	- `build/tools/iree-compile: Mach-O 64-bit executable x86_64`

## Remaining Warnings
- `ocloc not found` for MLIR XeVM native binary compilation support.
- StableHLO CMake deprecation warning for policy `CMP0116`.
- StableHLO builder tests unavailable because `gtest` not found.
- `Thin archives requested but not supported by ar`.
- These warnings did not block the build.

## Status
- Completed successfully.
