#!/usr/bin/env python3
"""Run a timed experiment with selectable strategy and logging."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.runtime.config import load_config
from scene_runtime.runtime.loop import RuntimeLoop
from scene_runtime.utils.video import FrameSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scene-runtime experiment runner")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument(
        "--strategy",
        default="scene_thermal_coadaptive",
        choices=[
            "native_rtdetr",
            "default",
            "static_affinity",
            "fixed_low_power",
            "fixed_frame_skip",
            "thermal_only",
            "scene_only",
            "scene_thermal_coadaptive",
        ],
    )
    p.add_argument("--video", type=Path, default=None, help="Video path or omit for synthetic")
    p.add_argument(
        "--loop-video",
        action="store_true",
        help="Loop the input video until duration-min is reached",
    )
    p.add_argument("--dry-run", action="store_true", help="Simulate inference without ONNX model")
    p.add_argument("--duration-min", type=float, default=15.0)
    p.add_argument("--output", type=Path, default=None, help="CSV log output path")
    p.add_argument(
        "--thermal-state",
        choices=["normal", "warm", "hot", "critical", "unknown"],
        default=None,
        help="Override detected thermal state for policy testing on non-Pi machines",
    )
    p.add_argument(
        "--thermal-temp-c",
        type=float,
        default=None,
        help="Override detected temperature in Celsius for thermal policy testing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.strategy)
    if args.thermal_state is not None:
        config.setdefault("thermal", {})["override_state"] = args.thermal_state
    if args.thermal_temp_c is not None:
        config.setdefault("thermal", {})["override_temp_c"] = args.thermal_temp_c

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            ROOT / "experiments" / "logs" / f"{args.strategy}_{stamp}.csv"
        )

    duration_sec = args.duration_min * 60.0
    max_frames = int(duration_sec * 10) if args.dry_run else None

    source = FrameSource(
        args.video,
        synthetic=args.video is None,
        max_frames=max_frames,
        loop=args.loop_video,
    )

    loop = RuntimeLoop(
        config,
        source,
        dry_run=args.dry_run,
        duration_sec=duration_sec,
        log_path=args.output,
    )

    log_path = loop.run()
    print(f"Experiment finished.")
    print(f"  strategy: {args.strategy}")
    print(f"  dry_run:  {args.dry_run}")
    print(f"  log:      {log_path}")


if __name__ == "__main__":
    main()
