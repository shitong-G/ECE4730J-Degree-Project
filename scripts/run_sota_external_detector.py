#!/usr/bin/env python3
"""Run one external detector baseline on a video and write a comparable summary."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def csv_path_for(output_dir: Path, detector: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{detector}.csv"


def write_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def run_ultralytics(args: argparse.Namespace) -> dict[str, object]:
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    rows: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for frame_id, result in enumerate(
        model.predict(
            source=str(args.video),
            imgsz=args.imgsz,
            device=args.device,
            stream=True,
            save=args.save_video,
            project=str(args.visualization_dir.parent),
            name=args.visualization_dir.name,
            verbose=False,
        ),
        start=1,
    ):
        if args.max_frames and frame_id > args.max_frames:
            break
        speed = result.speed
        detections = 0 if result.boxes is None else len(result.boxes)
        rows.append(
            {
                "frame": frame_id,
                "preprocess_ms": float(speed.get("preprocess", 0.0)),
                "inference_ms": float(speed.get("inference", 0.0)),
                "postprocess_ms": float(speed.get("postprocess", 0.0)),
                "detection_count": int(detections),
            }
        )
    wall = time.perf_counter() - start
    if not rows:
        raise RuntimeError("Ultralytics produced no frames")

    frame_csv = args.output_csv.with_name(args.output_csv.stem + "_frames.csv")
    with frame_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    return {
        "detector": args.detector,
        "backend": "ultralytics_onnxruntime",
        "model": str(args.model),
        "video": str(args.video),
        "frames": len(rows),
        "wall_sec": round(wall, 6),
        "throughput_fps": round(len(rows) / wall, 6) if wall > 0 else 0.0,
        "preprocess_ms_mean": round(mean("preprocess_ms"), 6),
        "inference_ms_mean": round(mean("inference_ms"), 6),
        "postprocess_ms_mean": round(mean("postprocess_ms"), 6),
        "detection_count_mean": round(mean("detection_count"), 6),
        "frame_csv": str(frame_csv),
    }


def run_subprocess_baseline(
    args: argparse.Namespace,
    command: list[str],
    backend: str,
) -> dict[str, object]:
    env = os.environ.copy()
    env["TORCH_HOME"] = str(ROOT / "tmp" / "torch")
    env["PADDLE_HOME"] = str(ROOT / "tmp" / "paddle")
    env["XDG_CACHE_HOME"] = str(ROOT / "tmp" / "cache")
    env["HOME"] = str(ROOT / "tmp" / "home")
    env["USERPROFILE"] = str(ROOT / "tmp" / "home")
    env["APPDATA"] = str(ROOT / "tmp" / "appdata")
    env["MPLCONFIGDIR"] = str(ROOT / ".plot_mplconfig")
    env["YOLO_CONFIG_DIR"] = str(ROOT / "Ultralytics")
    env["FLAGS_use_mkldnn"] = "0"
    env["FLAGS_use_onednn"] = "0"
    env["ONEDNN_VERBOSE"] = "0"
    Path(env["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["PADDLE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["APPDATA"]).mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, check=False, env=env)
    wall = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            f"{args.detector} command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )
    frames = args.max_frames or video_frame_count(args.video)
    return {
        "detector": args.detector,
        "backend": backend,
        "model": str(args.model),
        "video": str(args.video),
        "frames": frames,
        "wall_sec": round(wall, 6),
        "throughput_fps": round(frames / wall, 6) if wall > 0 and frames else 0.0,
        "preprocess_ms_mean": "",
        "inference_ms_mean": "",
        "postprocess_ms_mean": "",
        "detection_count_mean": "",
        "frame_csv": "",
    }


def video_frame_count(video: Path) -> int:
    try:
        import cv2
    except ImportError:
        return 0
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return count


def picodet_command(args: argparse.Namespace) -> list[str]:
    script = ROOT / "third_party" / "PaddleDetection" / "deploy" / "python" / "infer.py"
    if not script.exists():
        raise FileNotFoundError(
            "PaddleDetection deploy runner not found. Clone PaddleDetection first: "
            "git clone --depth 1 --branch release/2.7 "
            "https://github.com/PaddlePaddle/PaddleDetection.git third_party/PaddleDetection"
        )
    argv = [
        str(script),
        f"--model_dir={args.model}",
        f"--video_file={args.video}",
        "--device=CPU",
        f"--cpu_threads={args.threads}",
        f"--output_dir={args.visualization_dir}",
    ]
    wrapper = f"""
import runpy
import sys

import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {{
        "float": [np.float16, np.float32, np.float64],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_],
    }}

sys.argv = {argv!r}
runpy.run_path({str(script)!r}, run_name="__main__")
"""
    return [sys.executable, "-c", wrapper]


def nanodet_command(args: argparse.Namespace) -> list[str]:
    script = ROOT / "third_party" / "nanodet" / "demo" / "demo.py"
    if not script.exists():
        raise FileNotFoundError(
            "NanoDet demo runner not found. Clone NanoDet first: "
            "git clone --depth 1 https://github.com/RangiLyu/nanodet.git third_party/nanodet"
        )
    nanodet_config = nanodet_no_pretrain_config(args.nanodet_config)
    argv = [
        str(script),
        "video",
        "--config",
        str(nanodet_config),
        "--model",
        str(args.model),
        "--path",
        str(args.video),
    ]
    wrapper = f"""
import runpy
import sys
import types

import torch

m = types.ModuleType("torch._six")
m.string_classes = (str, bytes)
sys.modules["torch._six"] = m

_torch_load = torch.load

def _compat_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)

torch.load = _compat_load

_module_to = torch.nn.Module.to

def _cpu_safe_to(self, *args, **kwargs):
    if args and str(args[0]).startswith("cuda") and not torch.cuda.is_available():
        args = ("cpu",) + tuple(args[1:])
    if str(kwargs.get("device", "")).startswith("cuda") and not torch.cuda.is_available():
        kwargs["device"] = "cpu"
    return _module_to(self, *args, **kwargs)

torch.nn.Module.to = _cpu_safe_to

_tensor_to = torch.Tensor.to

def _tensor_cpu_safe_to(self, *args, **kwargs):
    if args and str(args[0]).startswith("cuda") and not torch.cuda.is_available():
        args = ("cpu",) + tuple(args[1:])
    if str(kwargs.get("device", "")).startswith("cuda") and not torch.cuda.is_available():
        kwargs["device"] = "cpu"
    return _tensor_to(self, *args, **kwargs)

torch.Tensor.to = _tensor_cpu_safe_to

sys.argv = {argv!r}
runpy.run_path({str(script)!r}, run_name="__main__")
"""
    return [sys.executable, "-c", wrapper]


def nanodet_no_pretrain_config(config_path: Path) -> Path:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    backbone = config.setdefault("model", {}).setdefault("arch", {}).setdefault(
        "backbone",
        {},
    )
    backbone["pretrain"] = False
    output = ROOT / "tmp" / "sota_baselines" / (config_path.stem + "_no_pretrain.yml")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector",
        choices=["yolov8n", "pp_picodet_s_320", "nanodet_plus_m_320"],
        required=True,
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=ROOT / "experiments" / "visualizations" / "sota_baselines" / "external",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--nanodet-config",
        type=Path,
        default=ROOT / "third_party" / "nanodet" / "config" / "nanodet-plus-m_320.yml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.visualization_dir.mkdir(parents=True, exist_ok=True)
    if args.detector == "yolov8n":
        summary = run_ultralytics(args)
    elif args.detector == "pp_picodet_s_320":
        summary = run_subprocess_baseline(
            args,
            picodet_command(args),
            "paddle_inference_cpu",
        )
    else:
        summary = run_subprocess_baseline(
            args,
            nanodet_command(args),
            "nanodet_pytorch_demo",
        )
    write_summary(args.output_csv, summary)
    print(f"summary: {args.output_csv}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
