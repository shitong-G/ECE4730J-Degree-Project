#!/usr/bin/env python3
"""Run RT-DETR proposed software and external detector baselines on Raspberry Pi."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RTDETR_SPECS = [
    {
        "key": "rtdetr_native_fp32",
        "title": "Native RT-DETR FP32, every frame",
        "strategy": "native_rtdetr",
        "model_mode": "native",
    },
    {
        "key": "proposed_lk_tracking",
        "title": "Proposed detector with event-triggered LK tracking",
        "strategy": "scene_track_lk",
        "model_mode": "family",
    },
    {
        "key": "proposed_software",
        "title": "Full proposed software policy: scene + LK + thermal interval control",
        "strategy": "scene_thermal_interval_lk",
        "model_mode": "family",
    },
]


EXTERNAL_SPECS = [
    {
        "key": "yolov8n_640",
        "detector": "yolov8n",
        "model": ROOT / "models" / "baselines" / "yolov8n_640.onnx",
        "imgsz": 640,
    },
    {
        "key": "pp_picodet_s_320",
        "detector": "pp_picodet_s_320",
        "model": ROOT / "models" / "baselines" / "picodet_s_320_coco_lcnet_portable",
        "imgsz": 320,
    },
    {
        "key": "nanodet_plus_m_320",
        "detector": "nanodet_plus_m_320",
        "model": ROOT / "models" / "baselines" / "nanodet-plus-m_320.ckpt",
        "imgsz": 320,
    },
]


def cpu_temp_c() -> float | None:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip()) / 1000.0
    except Exception:
        return None


def wait_for_cooldown(target_c: float, poll_sec: float, max_wait_min: float) -> None:
    deadline = time.time() + max_wait_min * 60.0
    while True:
        temp = cpu_temp_c()
        if temp is None:
            print("cooldown: temperature sensor unavailable; continuing")
            return
        if temp <= target_c:
            print(f"cooldown: {temp:.1f} C <= {target_c:.1f} C")
            return
        if time.time() >= deadline:
            print(f"cooldown: timeout at {temp:.1f} C; continuing")
            return
        print(f"cooldown: {temp:.1f} C, waiting for <= {target_c:.1f} C")
        time.sleep(poll_sec)


def run_command(command: list[str], *, plan_only: bool) -> int:
    print("\n$ " + " ".join(str(item) for item in command))
    if plan_only:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def rtdetr_command(args: argparse.Namespace, spec: dict[str, str], output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--config",
        str(args.config),
        "--strategy",
        spec["strategy"],
        "--video",
        str(args.video),
        "--loop-video",
        "--duration-min",
        str(args.duration_min),
        "--max-frames",
        str(args.max_frames),
        "--output",
        str(output),
        "--log-detections",
        "--no-stage-summary",
    ]
    if args.enable_thread_sessions:
        command.extend(["--enable-thread-sessions", "--thread-session-counts", args.thread_session_counts])
    if args.apply_runtime_actions:
        command.append("--apply-runtime-actions")
    if spec["model_mode"] == "native":
        command.extend(["--model", str(args.native_model)])
    else:
        command.extend(
            [
                "--model-paths-by-resolution",
                f"320={args.proposed_model_320},640={args.proposed_model_640}",
            ]
        )
    return command


def external_command(args: argparse.Namespace, spec: dict[str, object], output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_sota_external_detector.py"),
        "--detector",
        str(spec["detector"]),
        "--video",
        str(args.video),
        "--model",
        str(spec["model"]),
        "--output-csv",
        str(output),
        "--visualization-dir",
        str(args.visualization_dir / str(spec["key"])),
        "--imgsz",
        str(spec["imgsz"]),
        "--threads",
        str(args.threads),
        "--max-frames",
        str(args.max_frames),
    ]
    if args.save_external_video:
        command.append("--save-video")
    return command


def append_manifest(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--duration-min", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--thread-session-counts", default="1,2,3,4")
    parser.add_argument("--enable-thread-sessions", action="store_true")
    parser.add_argument("--apply-runtime-actions", action="store_true")
    parser.add_argument("--skip-rtdetr", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--save-external-video", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cooldown-temp-c", type=float, default=55.0)
    parser.add_argument("--cooldown-poll-sec", type=float, default=10.0)
    parser.add_argument("--max-cooldown-min", type=float, default=30.0)
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("pi_sota_comparison_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "logs" / "sota_baselines",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=ROOT / "experiments" / "visualizations" / "sota_baselines",
    )
    parser.add_argument(
        "--native-model",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx",
    )
    parser.add_argument(
        "--proposed-model-320",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_320_int8.onnx",
    )
    parser.add_argument(
        "--proposed-model-640",
        type=Path,
        default=ROOT / "models" / "rtdetr_r18_lite_pi4_640_int8.onnx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    manifest = run_dir / "manifest.csv"
    if not args.video.exists():
        raise FileNotFoundError(args.video)

    failures = 0
    if not args.skip_rtdetr:
        for spec in RTDETR_SPECS:
            output = run_dir / f"{spec['key']}.csv"
            wait_for_cooldown(args.cooldown_temp_c, args.cooldown_poll_sec, args.max_cooldown_min)
            command = rtdetr_command(args, spec, output)
            started = time.time()
            rc = run_command(command, plan_only=args.plan_only)
            elapsed = time.time() - started
            failures += int(rc != 0)
            append_manifest(
                manifest,
                {
                    "key": spec["key"],
                    "kind": "rtdetr_policy",
                    "title": spec["title"],
                    "output": str(output),
                    "returncode": rc,
                    "elapsed_sec": round(elapsed, 3),
                },
            )

    if not args.skip_external:
        for spec in EXTERNAL_SPECS:
            output = run_dir / f"{spec['key']}.csv"
            wait_for_cooldown(args.cooldown_temp_c, args.cooldown_poll_sec, args.max_cooldown_min)
            command = external_command(args, spec, output)
            started = time.time()
            rc = run_command(command, plan_only=args.plan_only)
            elapsed = time.time() - started
            failures += int(rc != 0)
            append_manifest(
                manifest,
                {
                    "key": spec["key"],
                    "kind": "external_detector",
                    "title": spec["detector"],
                    "output": str(output),
                    "returncode": rc,
                    "elapsed_sec": round(elapsed, 3),
                },
            )

    print(f"\nmanifest: {manifest}")
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


if __name__ == "__main__":
    main()
