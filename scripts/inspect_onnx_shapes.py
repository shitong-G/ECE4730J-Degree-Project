#!/usr/bin/env python3
"""Print ONNX model input/output shapes."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ONNX input/output shapes")
    parser.add_argument("models", nargs="+", type=Path)
    return parser.parse_args()


def _shape(value) -> str:
    return "[" + ", ".join(str(item) for item in value) + "]"


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    for model in args.models:
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        print(model)
        for item in session.get_inputs():
            print(f"  input  {item.name}: {_shape(item.shape)} {item.type}")
        for item in session.get_outputs():
            print(f"  output {item.name}: {_shape(item.shape)} {item.type}")


if __name__ == "__main__":
    main()
