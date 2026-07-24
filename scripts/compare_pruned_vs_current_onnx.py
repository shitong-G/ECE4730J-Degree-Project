#!/usr/bin/env python3
"""Compare current Pi ONNX models vs sjx L1-pruned ONNX."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    (
        320,
        ROOT / "models" / "rtdetr_r18_lite_pi4_320.onnx",
        ROOT / "experiments" / "results" / "cnn_pruning_smoke" / "rtdetr_r18_lite_l1_r20_320.onnx",
    ),
    (480, ROOT / "models" / "rtdetr_r18_lite_pi4_480.onnx", None),
    (
        640,
        ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx",
        ROOT
        / "experiments"
        / "results"
        / "cnn_pruning_smoke_640"
        / "rtdetr_r18_lite_l1_r20_640.onnx",
    ),
]


def onnx_stats(path: Path) -> dict:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    n_params = 0
    for init in model.graph.initializer:
        dims = list(init.dims) or [1]
        n = 1
        for d in dims:
            n *= int(d)
        n_params += n
    return {
        "nodes": len(model.graph.node),
        "params": n_params,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "outputs": [o.name for o in model.graph.output],
    }


def bench(path: Path, size: int, warmup: int = 3, runs: int = 10) -> float:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    images = np.random.rand(1, 3, size, size).astype("float32")
    feeds = {}
    for inp in sess.get_inputs():
        if "image" in inp.name.lower():
            feeds[inp.name] = images
        else:
            dtype = np.int64 if "int64" in str(inp.type) else np.float32
            feeds[inp.name] = np.array([[size, size]], dtype=dtype)
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        sess.run(outs, feeds)
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(outs, feeds)
    return (time.perf_counter() - t0) * 1000.0 / runs


def main() -> None:
    print(f"{'res':>4}  {'tag':<16}  {'file':<42}  {'MB':>7}  {'params':>12}  {'nodes':>6}  {'ORT_ms':>8}")
    print("-" * 110)
    for size, current, pruned in PAIRS:
        base_st = base_ms = None
        for tag, path in (("current", current), ("pruned_l1_r20", pruned)):
            if path is None or not path.exists():
                print(f"{size:>4}  {tag:<16}  {'(missing)':<42}")
                continue
            st = onnx_stats(path)
            ms = bench(path, size)
            print(
                f"{size:>4}  {tag:<16}  {path.name:<42}  "
                f"{st['size_mb']:7.1f}  {st['params']:12,d}  {st['nodes']:6d}  {ms:8.1f}"
            )
            if tag == "current":
                base_st, base_ms = st, ms
            elif base_st is not None and base_ms:
                d_mb = (1 - st["size_mb"] / base_st["size_mb"]) * 100
                d_p = (1 - st["params"] / base_st["params"]) * 100
                d_ms = (1 - ms / base_ms) * 100
                print(
                    f"{'':>4}  {'delta':<16}  {'':<42}  "
                    f"{d_mb:+6.1f}%  {d_p:+10.1f}%  {'':>6}  {d_ms:+7.1f}%"
                )


if __name__ == "__main__":
    main()
