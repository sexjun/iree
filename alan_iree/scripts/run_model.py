#!/usr/bin/env python3
"""Run a compiled IREE .vmfb model with synthetic test input.

Usage:
    python run_model.py <model.vmfb> [random|ones]

Generates random or ones input data, runs inference, and prints output info.
Uses SystemContext + Config(driver_name='local-task') matching iree-run-module.
"""

import sys
import numpy as np


def generate_input(shape, dtype=np.float32, mode="random"):
    """Generate synthetic input data."""
    resolved_shape = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            resolved_shape.append(dim)
        else:
            resolved_shape.append(1)

    if mode == "ones":
        return np.ones(resolved_shape, dtype=dtype)
    else:
        return np.random.randn(*resolved_shape).astype(dtype)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <model.vmfb> [random|ones]")
        sys.exit(1)

    vmfb_path = sys.argv[1]
    data_mode = sys.argv[2] if len(sys.argv) > 2 else "random"

    print(f"  Loading: {vmfb_path}")
    print(f"  Input data mode: {data_mode}")

    try:
        from iree.runtime import (
            Config,
            SystemContext,
            VmModule,
        )
    except ImportError:
        print("Error: iree.runtime not available.")
        print("The runtime Python bindings are not installed or not in PYTHONPATH.")
        print(f"\nTo run this step, ensure your locally built runtime is installed:")
        print(f"  pip install /Users/alanchen/workspace/iree/build/runtime")
        print(f"\nThe .vmfb file was produced successfully:")
        print(f"  File: {vmfb_path}")
        return

    print("  Initializing IREE runtime (local-task)...")
    ctx = SystemContext(config=Config(driver_name="local-task"))

    with open(vmfb_path, "rb") as f:
        vmfb_data = f.read()

    vm_module = VmModule.from_buffer(
        ctx.instance, vmfb_data, warn_if_copy=False
    )
    ctx.add_vm_module(vm_module)

    module_name = vm_module.name
    print(f"  Module: {module_name}")

    bound = ctx.modules.get(module_name)
    if bound is None:
        print(f"  Available modules: {list(ctx.modules.keys())}")
        print("Error: module not found in context")
        return

    # Discover functions from vm_module.function_names
    func_names = list(vm_module.function_names)
    print(f"  Available functions: {func_names}")

    # Pick the main inference function (skip __init and $async variants)
    func_name = None
    for candidate in func_names:
        if candidate == "__init":
            continue
        if "$async" in candidate:
            continue
        func_name = candidate
        break
    if func_name is None:
        # Fallback: pick the first non-init function
        for candidate in func_names:
            if candidate != "__init":
                func_name = candidate
                break
    if func_name is None:
        print("Error: no callable functions found in module")
        return

    func = getattr(bound, func_name)
    vm_func = func.vm_function
    print(f"  Calling: {func_name}")
    print(f"  Function: {vm_func}")
    print(f"  Inputs: 1x float32[1, 3, 224, 224] (ResNet50 default)")

    inputs = []
    shape = [1, 3, 224, 224]
    arr = generate_input(shape, np.float32, data_mode)
    inputs.append(arr)
    print(f"    shape={arr.shape}, dtype={arr.dtype}")

    import time
    print("  Executing...")
    t0 = time.time()
    results = func(*inputs)
    elapsed = time.time() - t0
    print(f"  Execution time: {elapsed:.3f}s")

    if isinstance(results, (list, tuple)):
        print(f"  Output count: {len(results)}")
        for i, out in enumerate(results):
            arr = np.asarray(out)
            print(f"  Output[{i}]: shape={arr.shape}, dtype={arr.dtype}")
            flat = arr.flatten()
            print(f"    Values (first 10): {flat[:10]}")
            if len(flat) > 10:
                print(f"    min={flat.min():.6f}, max={flat.max():.6f}, mean={flat.mean():.6f}")
    else:
        arr = np.asarray(results)
        print(f"  Output: shape={arr.shape}, dtype={arr.dtype}")
        flat = arr.flatten()
        print(f"    Values (first 10): {flat[:10]}")

    print("  Inference completed successfully.")


if __name__ == "__main__":
    main()
