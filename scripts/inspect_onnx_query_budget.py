#!/usr/bin/env python3
"""Inspect ONNX TopK nodes and constants related to RT-DETR query selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(str(args.model), load_external_data=False)
    initializers = {
        value.name: numpy_helper.to_array(value)
        for value in model.graph.initializer
    }
    constants: dict[str, np.ndarray] = {}
    producers: dict[str, onnx.NodeProto] = {}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for output in node.output:
            producers[output] = node
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    array = numpy_helper.to_array(attribute.t)
                    for output in node.output:
                        constants[output] = array

    print(f"model: {args.model}")
    print("inputs:", [value.name for value in model.graph.input])
    topk_nodes = [node for node in model.graph.node if node.op_type == "TopK"]
    print(f"TopK nodes: {len(topk_nodes)}")
    for index, node in enumerate(topk_nodes):
        k_name = node.input[1] if len(node.input) > 1 else ""
        k_value = initializers.get(k_name)
        if k_value is None:
            k_value = constants.get(k_name)
        source = producers.get(node.input[0])
        downstream = sorted(
            {
                consumer.op_type
                for output in node.output
                for consumer in consumers.get(output, [])
            }
        )
        print(
            f"[{index}] name={node.name or '<unnamed>'} "
            f"data={node.input[0]} source={source.op_type if source else 'graph'} "
            f"k={k_name} value={k_value.tolist() if k_value is not None else '?'} "
            f"outputs={list(node.output)} consumers={downstream}"
        )


if __name__ == "__main__":
    main()
