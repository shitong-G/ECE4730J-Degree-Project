#!/usr/bin/env python3
"""Export L1-pruned RT-DETR ONNX, then static-INT8 quantize for given resolutions.

Pipeline per resolution:
  1) structured L1 channel prune + FP32 ONNX export
  2) static INT8 quantization via tools/quantize_onnx_to_c.py

Example (WSL / conda 4903):
  python scripts/export_pruned_quantized_onnx.py --sizes 320 480
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "third_party" / "checkpoints" / "rtdetrv2_r18vd_sp1_120e_coco.pth"
DEFAULT_CALIB = ROOT / "data" / "calibration_frames"
DEFAULT_OUT = ROOT / "experiments" / "results" / "cnn_pruning_quantized"


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[320, 480])
    parser.add_argument("--ratio", type=float, default=0.20)
    parser.add_argument("--method", default="l1", choices=["l1", "l2", "fpgm", "bn_scale", "random"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-calib", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-prune", action="store_true", help="Reuse existing FP32 pruned ONNX.")
    parser.add_argument("--skip-quant", action="store_true", help="Only prune/export FP32.")
    parser.add_argument(
        "--reuse-fp32",
        action="store_true",
        help="If FP32 pruned ONNX already exists under --out-dir, skip re-pruning that size.",
    )
    args = parser.parse_args()

    python = sys.executable
    prune_script = ROOT / "scripts" / "run_cnn_pruning_experiment.py"
    quant_script = ROOT / "tools" / "quantize_onnx_to_c.py"

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.skip_quant and not args.calib_dir.exists():
        raise FileNotFoundError(f"Calibration dir missing: {args.calib_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{args.method}_r{int(args.ratio * 100):02d}"
    produced: list[Path] = []

    for size in args.sizes:
        size_dir = args.out_dir / f"{size}"
        size_dir.mkdir(parents=True, exist_ok=True)
        fp32_name = f"rtdetr_r18_lite_{label}_{size}.onnx"
        fp32_path = size_dir / fp32_name
        int8_path = size_dir / f"rtdetr_r18_lite_{label}_{size}_int8.onnx"

        if not args.skip_prune:
            if args.reuse_fp32 and fp32_path.exists():
                print(f"[skip] reuse FP32 {fp32_path}")
            else:
                _run(
                    [
                        python,
                        str(prune_script),
                        "--checkpoint",
                        str(args.checkpoint),
                        "--input-size",
                        str(size),
                        "--out-dir",
                        str(size_dir),
                        "--warmup",
                        str(args.warmup),
                        "--runs",
                        str(args.runs),
                        "--plan",
                        f"{args.method}:{args.ratio}",
                    ]
                )
                if not fp32_path.exists():
                    raise FileNotFoundError(f"Expected pruned FP32 missing: {fp32_path}")

        if not args.skip_quant:
            _run(
                [
                    python,
                    str(quant_script),
                    "--model",
                    str(fp32_path),
                    "--out-onnx",
                    str(int8_path),
                    "--mode",
                    "static",
                    "--calib-dir",
                    str(args.calib_dir),
                    "--resolution",
                    str(size),
                    "--max-calib",
                    str(args.max_calib),
                ]
            )
            produced.append(int8_path)
            print(
                f"[ok] {size}: FP32={fp32_path.stat().st_size / 1e6:.1f}MB  "
                f"INT8={int8_path.stat().st_size / 1e6:.1f}MB"
            )
        else:
            produced.append(fp32_path)

    print("\n==> outputs")
    for path in produced:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
