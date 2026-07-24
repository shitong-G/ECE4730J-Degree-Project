#!/usr/bin/env python3
"""Run structured CNN pruning experiments for RT-DETRv2 R18-lite.

The goal is to create Pi-transferable artifacts: structurally pruned ONNX
models whose convolution channels are actually removed, not merely zeroed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RTDETR_ROOT = ROOT / "third_party" / "RT-DETR" / "rtdetrv2_pytorch"
DEFAULT_CONFIG = RTDETR_ROOT / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_sp1_120e_coco.yml"


@dataclass
class ExperimentResult:
    method: str
    ratio: float
    input_size: int
    params: int
    conv_params: int
    pytorch_ms: float
    onnx_ms: float | None
    onnx_path: str
    status: str
    note: str


def _prepare_imports() -> None:
    if not RTDETR_ROOT.exists():
        raise FileNotFoundError(
            f"RT-DETR source not found at {RTDETR_ROOT}. Run scripts/clone_rtdetr.sh first."
        )
    sys.path.insert(0, str(RTDETR_ROOT))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))


def _load_checkpoint_state(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _load_compatible_state(model: Any, state: dict[str, Any]) -> list[str]:
    """Load checkpoint tensors whose names and shapes match the current model.

    RT-DETR stores eval-size-dependent decoder buffers such as anchors. When
    exporting 320/480 variants from a 640 checkpoint, those buffers have a
    different shape and should be regenerated from the current config.
    """
    current = model.state_dict()
    compatible = {}
    skipped = []
    for name, tensor in state.items():
        if name not in current:
            skipped.append(name)
            continue
        if tuple(current[name].shape) != tuple(tensor.shape):
            skipped.append(name)
            continue
        compatible[name] = tensor
    model.load_state_dict(compatible, strict=False)
    return skipped


def build_model(config: Path, input_size: int, checkpoint: Path | None) -> Any:
    """Build an RT-DETR model; disable backbone auto-download unless checkpoint is absent."""
    import torch
    from src.core import YAMLConfig

    cfg = YAMLConfig(
        str(config),
        eval_spatial_size=[input_size, input_size],
        PResNet={"pretrained": False},
    )
    model = cfg.model.eval()
    if checkpoint is not None:
        state = _load_checkpoint_state(checkpoint)
        skipped = _load_compatible_state(model, state)
        if skipped:
            print(f"Skipped {len(skipped)} checkpoint tensors with incompatible shape/name.")
    model.eval()
    return model, cfg.postprocessor.eval()


class ExportModel:
    """Match upstream export_onnx.py: model + postprocessor wrapper."""

    def __init__(self, model: Any, postprocessor: Any) -> None:
        import torch.nn as nn

        class _Wrapper(nn.Module):
            def __init__(self, model: Any, postprocessor: Any) -> None:
                super().__init__()
                self.model = model.deploy()
                self.postprocessor = postprocessor.deploy()

            def forward(self, images: Any, orig_target_sizes: Any) -> Any:
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)

        self.module = _Wrapper(model, postprocessor).eval()


def count_params(model: Any) -> tuple[int, int]:
    import torch.nn as nn

    params = sum(p.numel() for p in model.parameters())
    conv_params = sum(p.numel() for m in model.modules() if isinstance(m, nn.Conv2d) for p in m.parameters())
    return params, conv_params


def _backbone_convs(model: Any) -> list[Any]:
    import torch.nn as nn

    return [
        module
        for name, module in model.named_modules()
        if name.startswith("backbone.") and isinstance(module, nn.Conv2d)
    ]


def apply_pruning(model: Any, method: str, ratio: float, input_size: int, round_to: int) -> None:
    """Apply dependency-aware structured channel pruning to backbone conv roots."""
    if ratio <= 0.0 or method == "baseline":
        return

    import torch
    import torch_pruning as tp

    method = method.lower()
    if method == "l1":
        importance = tp.importance.MagnitudeImportance(p=1)
    elif method == "l2":
        importance = tp.importance.MagnitudeImportance(p=2)
    elif method == "bn_scale":
        importance = tp.importance.BNScaleImportance()
    elif method == "fpgm":
        importance = tp.importance.FPGMImportance()
    elif method == "random":
        importance = tp.importance.RandomImportance()
    else:
        raise ValueError(f"Unsupported pruning method: {method}")

    backbone_convs = _backbone_convs(model)
    if not backbone_convs:
        raise RuntimeError("No backbone Conv2d layers found to prune.")

    ratio_dict = {module: ratio for module in backbone_convs}
    example_inputs = torch.randn(1, 3, input_size, input_size)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance,
        global_pruning=True,
        pruning_ratio=0.0,
        pruning_ratio_dict=ratio_dict,
        iterative_steps=1,
        round_to=round_to,
        root_module_types=[torch.nn.Conv2d],
    )
    pruner.step()
    model.eval()


def benchmark_pytorch(model: Any, input_size: int, warmup: int, runs: int) -> float:
    import torch

    model.eval()
    data = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(data)
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(data)
        return (time.perf_counter() - t0) * 1000.0 / max(runs, 1)


def export_onnx(
    model: Any,
    postprocessor: Any,
    output_file: Path,
    input_size: int,
    opset: int,
) -> None:
    import torch

    output_file.parent.mkdir(parents=True, exist_ok=True)
    wrapped = ExportModel(model, postprocessor).module
    data = torch.rand(1, 3, input_size, input_size)
    size = torch.tensor([[input_size, input_size]])
    with torch.no_grad():
        _ = wrapped(data, size)
    torch.onnx.export(
        wrapped,
        (data, size),
        str(output_file),
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
        opset_version=opset,
        verbose=False,
        do_constant_folding=True,
        external_data=False,
    )


def benchmark_onnx(path: Path, input_size: int, warmup: int, runs: int) -> float:
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    images = np.random.rand(1, 3, input_size, input_size).astype("float32")
    sizes = np.array([[input_size, input_size]], dtype="int64")
    feeds = {"images": images, "orig_target_sizes": sizes}
    output_names = [out.name for out in sess.get_outputs()]
    for _ in range(warmup):
        sess.run(output_names, feeds)
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(output_names, feeds)
    return (time.perf_counter() - t0) * 1000.0 / max(runs, 1)


def parse_plan(values: list[str]) -> list[tuple[str, float]]:
    plan: list[tuple[str, float]] = []
    for value in values:
        if ":" in value:
            method, ratio_text = value.split(":", 1)
            ratio = float(ratio_text)
        else:
            method, ratio = value, 0.0
        plan.append((method.strip(), ratio))
    return plan


def write_outputs(results: list[ExperimentResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExperimentResult.__annotations__))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    md_path = out_dir / "summary.md"
    lines = [
        "# CNN Pruning Experiment Summary",
        "",
        "| method | ratio | input | params | conv params | PyTorch ms | ONNX ms | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        onnx_ms = "" if r.onnx_ms is None else f"{r.onnx_ms:.3f}"
        lines.append(
            f"| {r.method} | {r.ratio:.2f} | {r.input_size} | {r.params} | "
            f"{r.conv_params} | {r.pytorch_ms:.3f} | {onnx_ms} | {r.status} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(args: argparse.Namespace, method: str, ratio: float) -> ExperimentResult:
    import torch

    torch.manual_seed(args.seed)
    label = method if method == "baseline" else f"{method}_r{int(ratio * 100):02d}"
    onnx_path = args.out_dir / f"rtdetr_r18_lite_{label}_{args.input_size}.onnx"
    note = ""
    try:
        model, postprocessor = build_model(args.config, args.input_size, args.checkpoint)
        apply_pruning(model, method, ratio, args.input_size, args.round_to)
        params, conv_params = count_params(model)
        pytorch_ms = benchmark_pytorch(model, args.input_size, args.warmup, args.runs)
        onnx_ms: float | None = None
        if not args.no_export:
            export_onnx(model, postprocessor, onnx_path, args.input_size, args.opset)
            onnx_ms = benchmark_onnx(onnx_path, args.input_size, args.warmup, args.runs)
        return ExperimentResult(
            method=method,
            ratio=ratio,
            input_size=args.input_size,
            params=params,
            conv_params=conv_params,
            pytorch_ms=pytorch_ms,
            onnx_ms=onnx_ms,
            onnx_path=str(onnx_path),
            status="ok",
            note=note,
        )
    except Exception as exc:  # keep matrix runs going and record failures
        return ExperimentResult(
            method=method,
            ratio=ratio,
            input_size=args.input_size,
            params=0,
            conv_params=0,
            pytorch_ms=0.0,
            onnx_ms=None,
            onnx_path=str(onnx_path),
            status="failed",
            note=repr(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "results" / "cnn_pruning")
    parser.add_argument(
        "--plan",
        nargs="+",
        default=["baseline", "l1:0.10", "l1:0.20", "l2:0.20", "fpgm:0.20", "bn_scale:0.20", "random:0.20"],
        help="Items like baseline, l1:0.20, l2:0.20, fpgm:0.20, bn_scale:0.20, random:0.20.",
    )
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4730)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    _prepare_imports()
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for method, ratio in parse_plan(args.plan):
        print(f"==> {method}:{ratio:.2f}")
        result = run_one(args, method, ratio)
        print(f"    {result.status} pytorch={result.pytorch_ms:.3f}ms onnx={result.onnx_ms}")
        if result.note:
            print(f"    note: {result.note}")
        results.append(result)

    write_outputs(results, args.out_dir)
    print(f"Saved {args.out_dir / 'summary.csv'}")
    print(f"Saved {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
