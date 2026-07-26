"""Validate that an RT-DETR ONNX graph exposes both dynamic query TopK nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto


def validate(path: Path, input_name: str = "query_budget") -> tuple[str, str]:
    model = onnx.load(str(path))
    inputs = {value.name: value for value in model.graph.input}
    query_input = inputs.get(input_name)
    if query_input is None:
        raise RuntimeError(f"missing graph input {input_name!r}")
    tensor_type = query_input.type.tensor_type
    if tensor_type.elem_type != TensorProto.INT64:
        raise RuntimeError(f"{input_name!r} must be INT64")
    dims = [dim.dim_value for dim in tensor_type.shape.dim]
    if dims != [1]:
        raise RuntimeError(f"{input_name!r} must have shape [1], got {dims}")
    decoder = None
    final = None
    for node in model.graph.node:
        if node.op_type != "TopK" or len(node.input) < 2:
            continue
        if node.name == "/model/decoder/TopK":
            decoder = node
        elif node.name == "/postprocessor/TopK":
            final = node
    if decoder is None or decoder.input[1] != input_name:
        raise RuntimeError("decoder TopK does not use the query_budget input")
    if final is None or final.input[1] != input_name:
        raise RuntimeError("final prediction TopK does not use the query_budget input")
    return decoder.name, final.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-name", default="query_budget")
    args = parser.parse_args()
    decoder, final = validate(args.model, args.input_name)
    print(f"valid dynamic query graph: {args.model}")
    print(f"  decoder TopK: {decoder}")
    print(f"  final TopK:   {final}")


if __name__ == "__main__":
    main()
