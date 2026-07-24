#!/usr/bin/env python3
"""Smoke-test the sjx CNN structured-pruning experiment.

Pulls only the minimal path needed to decide whether pruning is effective:
  1) dependency / source checks
  2) baseline vs L1 20% backbone prune
  3) param reduction + optional ONNX Runtime round-trip

This intentionally avoids the full multi-method matrix in
scripts/run_cnn_pruning_experiment.py.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "third_party" / "checkpoints" / "rtdetrv2_r18vd_sp1_120e_coco.pth"
DEFAULT_OUT = ROOT / "experiments" / "results" / "cnn_pruning_smoke"


def _check_deps(require_export: bool) -> list[str]:
    # RT-DETR import path also pulls tensorboard via torch.utils.tensorboard.
    required = ["torch", "torchvision", "torch_pruning", "yaml", "tensorboard"]
    if require_export:
        required.extend(["onnx", "onnxruntime", "numpy"])
    missing: list[str] = []
    for name in required:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--ratio", type=float, default=0.20)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export/ORT check.")
    parser.add_argument(
        "--min-param-reduction",
        type=float,
        default=0.05,
        help="Fail if pruned model does not reduce total params by at least this fraction.",
    )
    args = parser.parse_args()

    print("==> CNN pruning smoke test")
    print(f"    root={ROOT}")
    print(f"    input_size={args.input_size} ratio={args.ratio:.2f}")

    missing = _check_deps(require_export=not args.no_export)
    if missing:
        print("FAIL: missing packages:")
        for name in missing:
            print(f"  - {name}")
        print(
            "Install in WSL conda env 4903, e.g.:\n"
            "  source activate 4903\n"
            "  pip install torch torchvision onnx onnxruntime torch-pruning "
            "numpy opencv-python PyYAML tensorboard"
        )
        return 2

    if not args.checkpoint.exists():
        print(f"FAIL: checkpoint not found: {args.checkpoint}")
        return 2

    # Import after deps check so failure messages stay clear.
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_cnn_pruning_experiment import (  # noqa: E402
        apply_pruning,
        benchmark_onnx,
        benchmark_pytorch,
        build_model,
        count_params,
        export_onnx,
        _prepare_imports,
    )

    _prepare_imports()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t_build = time.perf_counter()
    model, postprocessor = build_model(
        ROOT
        / "third_party"
        / "RT-DETR"
        / "rtdetrv2_pytorch"
        / "configs"
        / "rtdetrv2"
        / "rtdetrv2_r18vd_sp1_120e_coco.yml",
        args.input_size,
        args.checkpoint,
    )
    print(f"    model built in {time.perf_counter() - t_build:.1f}s")

    base_params, base_conv = count_params(model)
    base_ms = benchmark_pytorch(model, args.input_size, args.warmup, args.runs)
    print(f"    baseline params={base_params:,} conv={base_conv:,} pytorch={base_ms:.2f}ms")

    apply_pruning(model, "l1", args.ratio, args.input_size, round_to=8)
    pruned_params, pruned_conv = count_params(model)
    pruned_ms = benchmark_pytorch(model, args.input_size, args.warmup, args.runs)
    reduction = 1.0 - (pruned_params / max(base_params, 1))
    conv_reduction = 1.0 - (pruned_conv / max(base_conv, 1))
    print(
        f"    pruned   params={pruned_params:,} conv={pruned_conv:,} "
        f"pytorch={pruned_ms:.2f}ms "
        f"(param_reduction={reduction * 100:.1f}%, conv_reduction={conv_reduction * 100:.1f}%)"
    )

    onnx_path = args.out_dir / f"rtdetr_r18_lite_l1_r{int(args.ratio * 100):02d}_{args.input_size}.onnx"
    onnx_ms = None
    if not args.no_export:
        export_onnx(model, postprocessor, onnx_path, args.input_size, opset=18)
        onnx_ms = benchmark_onnx(onnx_path, args.input_size, args.warmup, args.runs)
        print(f"    onnx ok path={onnx_path} latency={onnx_ms:.2f}ms size={onnx_path.stat().st_size:,}B")

    ok = reduction >= args.min_param_reduction and pruned_conv < base_conv
    summary = {
        "status": "ok" if ok else "failed",
        "input_size": args.input_size,
        "ratio": args.ratio,
        "baseline_params": base_params,
        "pruned_params": pruned_params,
        "param_reduction": reduction,
        "baseline_conv_params": base_conv,
        "pruned_conv_params": pruned_conv,
        "conv_reduction": conv_reduction,
        "baseline_pytorch_ms": base_ms,
        "pruned_pytorch_ms": pruned_ms,
        "onnx_ms": onnx_ms,
        "onnx_path": str(onnx_path) if not args.no_export else None,
        "min_param_reduction": args.min_param_reduction,
    }
    summary_path = args.out_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"    wrote {summary_path}")

    if not ok:
        print(
            f"FAIL: pruning did not reduce params enough "
            f"(got {reduction * 100:.1f}%, need >= {args.min_param_reduction * 100:.1f}%)"
        )
        return 1

    print("PASS: structured pruning reduced model size and completed smoke checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
