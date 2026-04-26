#!/usr/bin/env python3
"""Extract a specific layer (intermediate output) from an ONNX model as a standalone model.

Usage:
    python extract_layer.py <input_model.onnx> <layer_name>

Example:
    python extract_layer.py cds/resnet50-v1-12.onnx resnetv17_stage2_conv3_fwd

The extracted model will have the same inputs as the original model,
but its output will be the specified intermediate tensor.
"""

import sys
import onnx
from onnx import helper
from collections import deque


def find_nodes_upstream(graph, target_names):
    """Find all nodes that contribute to producing the target tensor names."""
    needed = set(target_names)
    produced_by = {}
    node_needed = set()

    for node in graph.node:
        for out in node.output:
            produced_by[out] = node

    queue = deque(needed)
    while queue:
        name = queue.popleft()
        if name not in produced_by:
            continue
        node = produced_by[name]
        nid = id(node)
        if nid not in node_needed:
            node_needed.add(nid)
            for inp in node.input:
                queue.append(inp)

    return node_needed, produced_by


def extract_layer(model_path, layer_name):
    model = onnx.load(model_path)
    graph = model.graph

    all_output_names = set()
    for node in graph.node:
        all_output_names.update(node.output)

    if layer_name not in all_output_names:
        print(f"Error: layer '{layer_name}' not found in model.")
        print("\nAvailable intermediate layers (first 30):")
        candidates = sorted(all_output_names)
        for name in candidates[:30]:
            print(f"  {name}")
        print("  ...")
        sys.exit(1)

    node_needed, produced_by = find_nodes_upstream(graph, {layer_name})

    new_nodes = [node for node in graph.node if id(node) in node_needed]
    new_inputs = []
    model_input_names = {inp.name for inp in graph.input}

    needed_inputs = set()
    for node in new_nodes:
        for inp in node.input:
            if inp in model_input_names:
                needed_inputs.add(inp)

    for inp in graph.input:
        if inp.name in needed_inputs:
            new_inputs.append(inp)

    new_outputs = []
    for orig_out in graph.output:
        if orig_out.name == layer_name:
            new_outputs.append(orig_out)
            break
    else:
        for vi in graph.value_info:
            if vi.name == layer_name:
                shape_proto = vi.type.tensor_type.shape
                shape = [
                    d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in shape_proto.dim
                ]
                new_out = helper.make_tensor_value_info(
                    layer_name,
                    vi.type.tensor_type.elem_type,
                    shape,
                )
                new_outputs.append(new_out)
                break
        else:
            new_out = helper.make_tensor_value_info(layer_name, onnx.TensorProto.FLOAT, None)
            new_outputs.append(new_out)

    new_initializers = []
    for init in graph.initializer:
        for node in new_nodes:
            if init.name in node.input:
                new_initializers.append(init)
                break

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name + "_extracted",
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=new_initializers,
    )

    for vi in graph.value_info:
        if any(vi.name in node.output for node in new_nodes):
            new_graph.value_info.append(vi)

    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    onnx.checker.check_model(new_model)

    output_path = model_path.rsplit(".", 1)[0] + f"_{layer_name}.onnx"
    onnx.save(new_model, output_path)

    print(f"Extracted model saved to: {output_path}")
    print(f"  Inputs: {[i.name for i in new_inputs]}")
    print(f"  Output: {layer_name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <model.onnx> <layer_name>")
        print(f"\nExample: python {sys.argv[0]} cds/resnet50-v1-12.onnx resnetv17_stage2_conv3_fwd")
        sys.exit(1)

    extract_layer(sys.argv[1], sys.argv[2])
