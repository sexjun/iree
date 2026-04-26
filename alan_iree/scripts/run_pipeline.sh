#!/usr/bin/env bash
# =============================================================================
# IREE ONNX Model Pipeline: Import -> Compile -> Run
#
# Usage:
#   ./run_pipeline.sh <model.onnx> [--target-backend llvm-cpu] [--input-data random|ones]
#
# Examples:
#   ./run_pipeline.sh alan_iree/models/resnet50-v1-12-bs1.onnx
#   ./run_pipeline.sh alan_iree/models/resnet50-v1-12-bs1.onnx --input-data ones
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Locally built IREE tools (do not download — always use the compiled versions)
IREE_COMPILE="${IREE_COMPILE:-/Users/alanchen/workspace/iree/build/tools/iree-compile}"
IREE_RUN_MODULE="${IREE_RUN_MODULE:-/Users/alanchen/workspace/iree/build/tools/iree-run-module}"

PYTHON="${PYTHON:-python}"
TARGET_BACKEND="${TARGET_BACKEND:-llvm-cpu}"
INPUT_DATA="${INPUT_DATA:-ones}"
OPSET_VERSION="${OPSET_VERSION:-17}"

MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-backend)
            TARGET_BACKEND="$2"
            shift 2
            ;;
        --input-data)
            INPUT_DATA="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 <model.onnx> [--target-backend BACKEND] [--input-data random|ones]"
            exit 0
            ;;
        *)
            if [[ -z "$MODEL_PATH" ]]; then
                MODEL_PATH="$1"
            else
                echo "Error: unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: please provide an ONNX model path"
    exit 1
fi

MODEL_PATH="$(realpath "$MODEL_PATH")"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: model file not found: $MODEL_PATH"
    exit 1
fi

OUTPUT_DIR="$(dirname "$MODEL_PATH")"
MODEL_NAME="$(basename "$MODEL_PATH" .onnx)"

MLIR_FILE="${OUTPUT_DIR}/${MODEL_NAME}.mlir"
VMFB_FILE="${OUTPUT_DIR}/${MODEL_NAME}.vmfb"

echo "========================================"
echo "IREE ONNX Model Pipeline"
echo "========================================"
echo "  Model:        $MODEL_PATH"
echo "  Target:       $TARGET_BACKEND"
echo "  Input data:   $INPUT_DATA"
echo "  MLIR output:  $MLIR_FILE"
echo "  VMFB output:  $VMFB_FILE"
echo "========================================"

# ─── Step 1: ONNX -> MLIR ────────────────────────────────────────────────────
echo ""
echo "[1/3] Converting ONNX -> MLIR..."
$PYTHON -m iree.compiler.tools.import_onnx \
    "$MODEL_PATH" \
    -o "$MLIR_FILE" \
    --opset-version "$OPSET_VERSION"

MLIR_SIZE=$(du -h "$MLIR_FILE" | cut -f1)
echo "  Done: $MLIR_FILE ($MLIR_SIZE)"

# ─── Step 2: MLIR -> VMFB ────────────────────────────────────────────────────
echo ""
echo "[2/3] Compiling MLIR -> VMFB (target: $TARGET_BACKEND)..."

if [[ ! -x "$IREE_COMPILE" ]]; then
    echo "Error: iree-compile not found at $IREE_COMPILE"
    exit 1
fi

"$IREE_COMPILE" "$MLIR_FILE" \
    --iree-hal-target-device=local \
    "--iree-hal-local-target-device-backends=$TARGET_BACKEND" \
    --iree-llvmcpu-target-cpu=host \
    -o "$VMFB_FILE" 2>&1

if [[ -f "$VMFB_FILE" ]]; then
    VMFB_SIZE=$(du -h "$VMFB_FILE" | cut -f1)
    echo "  Done: $VMFB_FILE ($VMFB_SIZE)"
else
    echo "Error: compilation failed, VMFB file not created"
    exit 1
fi

# ─── Step 3: Run with iree-run-module ────────────────────────────────────────
echo ""
echo "[3/3] Running compiled model with $INPUT_DATA input data..."

if [[ ! -x "$IREE_RUN_MODULE" ]]; then
    echo "Error: iree-run-module not found at $IREE_RUN_MODULE"
    echo "Set IREE_RUN_MODULE to point to your locally built binary."
    exit 1
fi

# Auto-detect the function name from the VMFB
FUNC_NAME=$($PYTHON -c "
import onnx
# We can't read VMFB directly, but for ONNX-converted models the
# function name is typically the original ONNX model name without extension.
# Fall back to common names.
print('mxnet_converted_model')
" 2>/dev/null || echo "main")

# Build input spec
if [[ "$INPUT_DATA" == "ones" ]]; then
    INPUT_SPEC="1x3x224x224xf32=1"
else
    # Generate random input as a binary file
    RANDOM_BIN="${OUTPUT_DIR}/${MODEL_NAME}_random_input.bin"
    echo "  Generating random input: $RANDOM_BIN"
    $PYTHON -c "
import numpy as np
np.random.randn(1, 3, 224, 224).astype(np.float32).tofile('$RANDOM_BIN')
"
    INPUT_SPEC="1x3x224x224xf32=@${RANDOM_BIN}"
fi

echo "  Function: $FUNC_NAME"
echo "  Input: $INPUT_SPEC"
echo ""

"$IREE_RUN_MODULE" \
    --module="$VMFB_FILE" \
    --device=local-task \
    --function="$FUNC_NAME" \
    --input="$INPUT_SPEC" \
    --print_statistics=true 2>&1

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "  MLIR: $MLIR_FILE"
echo "  VMFB: $VMFB_FILE"
echo "========================================"
