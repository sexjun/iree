# CDS - ONNX Model Tools

Tools for modifying and running ONNX models through the IREE compiler pipeline.

## Quick Start

```bash
# Step 1: Fix batch size (dynamic N -> 1)
python cds/fix_batch_size.py

# Step 2: Full pipeline (ONNX -> MLIR -> VMFB -> Run)
cds/run_pipeline.sh cds/resnet50-v1-12-bs1.onnx --input-data random
```

## Scripts

| Script | Purpose |
|--------|---------|
| `fix_batch_size.py` | Fix dynamic batch dimension to 1 |
| `extract_layer.py` | Extract a specific layer as a standalone ONNX model |
| `run_pipeline.sh` | One-command pipeline: ONNX -> MLIR -> VMFB -> inference |
| `run_model.py` | Run a compiled .vmfb with synthetic input (random or ones) |

## Examples

```bash
# Extract a specific layer
python cds/extract_layer.py cds/resnet50-v1-12-bs1.onnx resnetv17_pool0_fwd

# Compile and run with ones input
cds/run_pipeline.sh cds/resnet50-v1-12-bs1.onnx --input-data ones

# Run only inference (skip compilation)
python cds/run_model.py cds/resnet50-v1-12-bs1.vmfb random
```

See [cds_doc/onnx_to_iree_pipeline.md](../cds_doc/onnx_to_iree_pipeline.md) for detailed documentation.
