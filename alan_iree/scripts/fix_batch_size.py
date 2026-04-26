#!/usr/bin/env python3
"""Fix batch size to 1 in an ONNX model."""

import onnx

MODEL_IN = "cds/resnet50-v1-12.onnx"
MODEL_OUT = "cds/resnet50-v1-12-bs1.onnx"

def fix_batch(model, batch_size=1):
    """Set the first dimension of all input/output tensor shapes to batch_size."""
    for tensor in list(model.graph.input) + list(model.graph.output):
        if tensor.type.tensor_type.HasField("shape"):
            dim = tensor.type.tensor_type.shape.dim
            if len(dim) > 0:
                dim[0].dim_value = batch_size
                dim[0].ClearField("dim_param")

    for tensor in model.graph.value_info:
        if tensor.type.tensor_type.HasField("shape"):
            dim = tensor.type.tensor_type.shape.dim
            if len(dim) > 0 and dim[0].HasField("dim_param"):
                dim[0].dim_value = batch_size
                dim[0].ClearField("dim_param")

    return model

if __name__ == "__main__":
    print(f"Loading {MODEL_IN}...")
    model = onnx.load(MODEL_IN)
    print(f"  Inputs: {[i.name for i in model.graph.input]}")

    print("Fixing batch dimension to 1...")
    model = fix_batch(model, batch_size=1)

    print(f"Saving to {MODEL_OUT}...")
    onnx.save(model, MODEL_OUT)
    print("Done.")
