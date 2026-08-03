#!/usr/bin/env python3
"""Run the formal Ours experiment and sequential live dashboard comparisons.

The dashboards are intentionally run one at a time on the same port.  Each
dashboard exits after its configured duration, then the next condition starts.
Run this script from the project root on the Raspberry Pi.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
DYNAMIC_320 = "models/rtdetr_r18_lite_pi4_320_int8_dynamic_q.onnx"
DYNAMIC_480 = "models/rtdetr_r18_lite_pi4_480_int8_dynamic_q.onnx"
DYNAMIC_640 = "models/rtdetr_r18_lite_pi4_640_int8_dynamic_q.onnx"
DYNAMIC_FAMILY = f"320={DYNAMIC_320},480={DYNAMIC_480},640={DYNAMIC_640}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "experiment", "dashboards"], default="all")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample3.mp4")
    parser.add_argument("--experiment-duration-min", type=float, default=20.0)
    parser.add_argument("--dashboard-duration-min", type=float, default=20.0)
    parser.add_argument("--cooldown-min", type=float, default=0.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jpeg-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument(
        "--apply-runtime-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the same best-effort governor/affinity actions as the formal run.",
    )
    parser.add_argument("--fan-control", choices=["config", "enabled", "disabled"], default="disabled")
    parser.add_argument("--skip-yolov8", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-quantized-native", action="store_true")
    parser.add_argument("--skip-ours-dashboard", action="store_true")
    return parser.parse_args()


def _command(args: list[str]) -> list[str]:
    return [str(PYTHON), *args]


def _run(command: list[str]) -> None:
    print("\n$ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _runtime_action_flags(enabled: bool) -> list[str]:
    return ["--apply-runtime-actions"] if enabled else []


def ours_experiment_command(args: argparse.Namespace, output: Path) -> list[str]:
    return _command(
        [
            "scripts/run_experiment.py",
            "--config",
            "configs/raspberry_pi4.yaml",
            "--strategy",
            "scene_thermal_interval_lk",
            "--video",
            str(args.video),
            "--loop-video",
            "--duration-min",
            str(args.experiment_duration_min),
            "--output",
            str(output),
            "--log-detections",
            "--fan-control",
            args.fan_control,
            "--query-budget-mode",
            "strict",
            "--model-paths-by-resolution",
            DYNAMIC_FAMILY,
            "--temperature-query-budget",
            "--query-budget-normal",
            "64",
            "--query-budget-warm",
            "48",
            "--query-budget-hot",
            "40",
            "--query-budget-critical",
            "32",
            "--query-budget-hysteresis-c",
            "4",
            "--no-stage-summary",
            *_runtime_action_flags(args.apply_runtime_actions),
        ]
    )


def common_dashboard_flags(args: argparse.Namespace) -> list[str]:
    return [
        "--config",
        "configs/raspberry_pi4.yaml",
        "--video",
        str(args.video),
        "--loop-video",
        "--duration-min",
        str(args.dashboard_duration_min),
        "--fan-control",
        args.fan_control,
        "--jpeg-width",
        str(args.jpeg_width),
        "--jpeg-quality",
        str(args.jpeg_quality),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-detections",
        "--no-stage-summary",
        *_runtime_action_flags(args.apply_runtime_actions),
    ]


def dashboard_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    common = common_dashboard_flags(args)
    commands: list[tuple[str, list[str]]] = []
    if not args.skip_native:
        commands.append(
            (
                "native",
                _command(
                    [
                        "scripts/run_live_dashboard.py",
                        *common,
                        "--strategy",
                        "native_rtdetr",
                        "--model",
                        "models/rtdetr_r18_lite_pi4_640.onnx",
                    ]
                ),
            )
        )
    if not args.skip_ours_dashboard:
        commands.append(
            (
                "ours",
                _command(
                    [
                        "scripts/run_live_dashboard.py",
                        *common,
                        "--strategy",
                        "scene_thermal_interval_lk",
                        "--model-paths-by-resolution",
                        DYNAMIC_FAMILY,
                        "--query-budget-mode",
                        "strict",
                        "--temperature-query-budget",
                        "--query-budget-normal",
                        "64",
                        "--query-budget-warm",
                        "48",
                        "--query-budget-hot",
                        "40",
                        "--query-budget-critical",
                        "32",
                        "--query-budget-hysteresis-c",
                        "4",
                    ]
                ),
            )
        )
    if not args.skip_quantized_native:
        commands.append(
            (
                "quantized_native",
                _command(
                    [
                        "scripts/run_live_dashboard.py",
                        *common,
                        "--strategy",
                        "native_rtdetr",
                        "--model",
                        DYNAMIC_640,
                        "--query-budget-mode",
                        "strict",
                        "--query-budget-override",
                        "300",
                    ]
                ),
            )
        )
    if not args.skip_yolov8:
        commands.append(
            (
                "yolov8",
                _command(
                    [
                        "scripts/run_yolov8_live_dashboard.py",
                        "--model",
                        "models/baselines/yolov8n_640.onnx",
                        "--video",
                        str(args.video),
                        "--loop-video",
                        "--duration-min",
                        str(args.dashboard_duration_min),
                        "--jpeg-width",
                        str(args.jpeg_width),
                        "--jpeg-quality",
                        str(args.jpeg_quality),
                        "--host",
                        args.host,
                        "--port",
                        str(args.port),
                    ]
                ),
            )
        )
    return commands


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.experiment_duration_min <= 0 or args.dashboard_duration_min <= 0:
        raise ValueError("Experiment and dashboard durations must be positive")
    if args.cooldown_min < 0:
        raise ValueError("--cooldown-min cannot be negative")

    if args.mode in {"all", "experiment"}:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "experiments" / "logs" / f"ours_remeasure_{stamp}.csv"
        _run(ours_experiment_command(args, output))

    if args.mode in {"all", "dashboards"}:
        commands = dashboard_commands(args)
        for index, (name, command) in enumerate(commands):
            print(f"\n===== dashboard: {name} ({index + 1}/{len(commands)}) =====", flush=True)
            _run(command)
            if index + 1 < len(commands) and args.cooldown_min > 0:
                seconds = args.cooldown_min * 60.0
                print(f"Cooling down for {args.cooldown_min:.1f} min", flush=True)
                time.sleep(seconds)


if __name__ == "__main__":
    main()
