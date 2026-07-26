#!/usr/bin/env python3
"""Expose RT-DETR decoder and postprocessor TopK K as an ONNX input.

The source exports used by this project contain two fixed ``TopK(K=300)``
nodes: uncertainty-minimal encoder query selection and final prediction
selection. This tool rewires both K inputs to one scalar-vector INT64 graph
input so a single ONNX Runtime session can change the query count per frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-name", default="query_budget")
    parser.add_argument("--source-budget", type=int, default=300)
    parser.add_argument(
        "--expected-topk",
        type=int,
        default=2,
        help="Fail unless this many matching fixed-budget TopK nodes are found.",
    )
    return parser.parse_args()


def constant_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                array = numpy_helper.to_array(attribute.t)
                for output in node.output:
                    values[output] = array
    return values


def main() -> None:
    args = parse_args()
    if args.source_budget < 1:
        raise ValueError("--source-budget must be positive")
    if args.model.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --model")

    model = onnx.load(str(args.model))
    existing_inputs = {value.name for value in model.graph.input}
    if args.input_name in existing_inputs:
        raise ValueError(f"Graph input already exists: {args.input_name}")
    values = constant_values(model)
    matched: list[str] = []
    for node in model.graph.node:
        if node.op_type != "TopK" or len(node.input) < 2:
            continue
        k_value = values.get(node.input[1])
        if k_value is None:
            continue
        flattened = np.asarray(k_value).reshape(-1)
        if flattened.size != 1 or int(flattened[0]) != args.source_budget:
            continue
        node.input[1] = args.input_name
        matched.append(node.name or node.output[0])

    if len(matched) != args.expected_topk:
        raise RuntimeError(
            f"Expected {args.expected_topk} TopK nodes with K={args.source_budget}, "
            f"found {len(matched)}: {matched}"
        )

    model.graph.input.append(
        helper.make_tensor_value_info(
            args.input_name,
            TensorProto.INT64,
            [1],
        )
    )
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata.update(
        {
            "dynamic_query_budget": "true",
            "dynamic_query_budget_input": args.input_name,
            "dynamic_query_budget_max": str(args.source_budget),
            "dynamic_query_budget_nodes": ",".join(matched),
        }
    )
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"Saved dynamic-query ONNX: {args.output}")
    print(f"Input: {args.input_name} INT64[1], valid range 1..{args.source_budget}")
    print("Rewired TopK nodes:")
    for name in matched:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
