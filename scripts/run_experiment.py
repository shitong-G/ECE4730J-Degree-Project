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
            "thermal_balanced",
            "thermal_interval_first",
            "scene_only",
            "scene_track_lk",
            "scene_thermal_coadaptive",
            "scene_thermal_interval_first",
            "scene_thermal_interval_lk",
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
        "--log-detections",
        action="store_true",
        help="Write per-frame detection boxes to <output>_detections.jsonl",
    )
    p.add_argument("--detection-output", type=Path, default=None)
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
    p.add_argument(
        "--enable-thread-sessions",
        action="store_true",
        help="Preload ONNX Runtime sessions for configured cpu thread counts",
    )
    p.add_argument(
        "--thread-session-counts",
        default=None,
        help="Comma-separated thread counts, e.g. 1,2,3,4",
    )
    p.add_argument(
        "--apply-runtime-actions",
        action="store_true",
        help="Best-effort apply governor and CPU affinity from RuntimeAction",
    )
    p.add_argument(
        "--enable-lk-tracking",
        action="store_true",
        help="Use Lucas-Kanade tracking to update boxes on skipped detector frames",
    )
    p.add_argument(
        "--lk-force-refresh-on-failure",
        action="store_true",
        help="Run RT-DETR immediately when LK tracking quality degrades",
    )
    p.add_argument("--lk-max-failure-ratio", type=float, default=None)
    p.add_argument("--lk-min-valid-points", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.strategy)
    if args.thermal_state is not None:
        config.setdefault("thermal", {})["override_state"] = args.thermal_state
    if args.thermal_temp_c is not None:
        config.setdefault("thermal", {})["override_temp_c"] = args.thermal_temp_c
    if args.enable_thread_sessions:
        config.setdefault("inference", {})["enable_thread_sessions"] = True
    if args.thread_session_counts:
        counts = [int(item.strip()) for item in args.thread_session_counts.split(",") if item.strip()]
        config.setdefault("inference", {})["thread_session_counts"] = counts
    if args.apply_runtime_actions:
        config.setdefault("os_control", {})["apply_runtime_actions"] = True
    if args.enable_lk_tracking:
        config.setdefault("tracking", {})["enable_lk_tracking"] = True
    if args.lk_force_refresh_on_failure:
        config.setdefault("tracking", {})["lk_force_refresh_on_failure"] = True
    if args.lk_max_failure_ratio is not None:
        config.setdefault("tracking", {})["lk_max_failure_ratio"] = args.lk_max_failure_ratio
    if args.lk_min_valid_points is not None:
        config.setdefault("tracking", {})["lk_min_valid_points"] = args.lk_min_valid_points

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
        detection_log_path=(
            args.detection_output
            if args.detection_output is not None
            else args.output.with_name(args.output.stem + "_detections.jsonl")
            if args.log_detections
            else None
        ),
    )

    log_path = loop.run()
    print(f"Experiment finished.")
    print(f"  strategy: {args.strategy}")
    print(f"  dry_run:  {args.dry_run}")
    print(f"  log:      {log_path}")


if __name__ == "__main__":
    main()
