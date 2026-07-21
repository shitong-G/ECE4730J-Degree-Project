#!/usr/bin/env python3
"""Benchmark IMX219 raw capture and conversion latency on Raspberry Pi."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.utils.video import FrameSource


STAGES = [
    ("source_total_ms", "total"),
    ("source_wait_ms", "wait"),
    ("capture_ms", "capture"),
    ("isp_ms", "convert"),
    ("source_resize_ms", "resize"),
    ("source_save_ms", "save"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--imx219-media-device", default="/dev/media3")
    parser.add_argument("--imx219-video-device", default="/dev/video0")
    parser.add_argument("--imx219-sensor-entity", default="imx219 10-0010")
    parser.add_argument("--imx219-width", type=int, default=1640)
    parser.add_argument("--imx219-height", type=int, default=1232)
    parser.add_argument("--imx219-stride-pixels", type=int, default=1648)
    parser.add_argument("--imx219-frame-width", type=int, default=640)
    parser.add_argument("--imx219-frame-height", type=int, default=480)
    parser.add_argument("--imx219-raw-output", type=Path, default=Path("/dev/shm/camera_bench.raw"))
    parser.add_argument("--imx219-jpg-output", type=Path, default=Path("/dev/shm/camera_bench.jpg"))
    parser.add_argument(
        "--imx219-save-latest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save latest JPEG while benchmarking. Default off to measure runtime cost only.",
    )
    parser.add_argument(
        "--imx219-capture-interval-sec",
        type=float,
        default=0.0,
        help="Minimum interval between raw captures.",
    )
    parser.add_argument(
        "--imx219-runtime-mode",
        choices=["lite-isp", "gray"],
        default="lite-isp",
        help="lite-isp measures color tuning; gray measures fast grayscale runtime input.",
    )
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _print_summary(rows: list[dict[str, float]], wall_s: float) -> None:
    print(f"Measured frames: {len(rows)}")
    print(f"Wall time s:     {_fmt(wall_s)}")
    if wall_s > 0:
        print(f"Camera FPS:      {_fmt(len(rows) / wall_s)}")
    print()
    print(f"{'stage':16s} {'mean':>10s} {'median':>10s} {'p95':>10s} {'max':>10s}")
    print("-" * 60)
    for key, label in STAGES:
        values = [float(row.get(key, 0.0)) for row in rows if float(row.get(key, 0.0)) > 0]
        if not values:
            continue
        print(
            f"{label:16s} {_fmt(mean(values)):>10s} {_fmt(median(values)):>10s} "
            f"{_fmt(_percentile(values, 0.95)):>10s} {_fmt(max(values)):>10s}"
        )


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_index", *[key for key, _label in STAGES]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"frame_index": index, **{key: row.get(key, 0.0) for key, _ in STAGES}})
    print(f"CSV written:     {path}")


def main() -> None:
    args = parse_args()
    total_frames = max(1, args.count) + max(0, args.warmup)
    source = FrameSource(
        None,
        synthetic=False,
        max_frames=total_frames,
        camera_backend="imx219-raw",
        imx219_media_device=args.imx219_media_device,
        imx219_video_device=args.imx219_video_device,
        imx219_sensor_entity=args.imx219_sensor_entity,
        imx219_width=args.imx219_width,
        imx219_height=args.imx219_height,
        imx219_stride_pixels=args.imx219_stride_pixels,
        imx219_frame_width=args.imx219_frame_width,
        imx219_frame_height=args.imx219_frame_height,
        imx219_raw_output=args.imx219_raw_output,
        imx219_jpg_output=args.imx219_jpg_output,
        imx219_save_latest=args.imx219_save_latest,
        imx219_capture_interval_sec=args.imx219_capture_interval_sec,
        imx219_runtime_mode=args.imx219_runtime_mode,
        frame_source_mode="serial",
    )

    measured: list[dict[str, float]] = []
    started = None
    try:
        for index, _frame in enumerate(source):
            if index < args.warmup:
                continue
            if started is None:
                started = time.perf_counter()
            measured.append(source.last_profile)
    finally:
        source.release()

    wall_s = 0.0 if started is None else time.perf_counter() - started
    print(f"IMX219 runtime mode: {args.imx219_runtime_mode}")
    print(f"Output frame size:   {args.imx219_frame_width}x{args.imx219_frame_height}")
    print(f"Save latest JPEG:    {args.imx219_save_latest}")
    print()
    _print_summary(measured, wall_s)
    if args.csv_output is not None:
        _write_csv(args.csv_output, measured)


if __name__ == "__main__":
    main()
