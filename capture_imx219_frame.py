#!/usr/bin/env python3
"""Capture one IMX219 RG10 raw frame and convert it with the lightweight ISP."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from convert_imx219_rg10_lite_isp import convert_file


def _run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(
            f"Required command not found: {name}. Install v4l-utils/media-ctl first."
        )


def capture_raw(args: argparse.Namespace) -> Path:
    """Configure the CSI media graph and capture one RG10 raw frame."""
    if not args.skip_capture:
        _require_tool("media-ctl")
        _require_tool("v4l2-ctl")

    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    sensor_fmt = (
        f'"{args.sensor_entity}":0 '
        f"[fmt:{args.sensor_format}/{args.width}x{args.height} field:none]"
    )
    _run_command(
        [
            "media-ctl",
            "-d",
            args.media_device,
            "--set-v4l2",
            sensor_fmt,
        ],
        dry_run=args.skip_capture,
    )
    _run_command(
        [
            "v4l2-ctl",
            "-d",
            args.video_device,
            f"--set-fmt-video=width={args.width},height={args.height},pixelformat={args.pixel_format}",
        ],
        dry_run=args.skip_capture,
    )
    _run_command(
        [
            "v4l2-ctl",
            "-d",
            args.video_device,
            f"--stream-mmap={args.stream_mmap}",
            f"--stream-count={args.stream_count}",
            f"--stream-to={raw_path}",
        ],
        dry_run=args.skip_capture,
    )

    if not args.skip_capture and (not raw_path.exists() or raw_path.stat().st_size == 0):
        raise RuntimeError(f"Raw capture failed or produced an empty file: {raw_path}")
    return raw_path


def capture_and_convert(args: argparse.Namespace) -> Path:
    raw_path = capture_raw(args)
    jpg_path = Path(args.jpg_output)
    convert_file(
        input_path=raw_path,
        output_path=jpg_path,
        width=args.width,
        height=args.height,
        stride_pixels=args.stride_pixels,
        black_level=args.black_level,
        r_gain=args.r_gain,
        g_gain=args.g_gain,
        b_gain=args.b_gain,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        gamma=args.gamma,
        saturation=args.saturation,
        sharpen_amount=args.sharpen,
        resize_original=args.resize_original,
    )
    return jpg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-device", default="/dev/media3")
    parser.add_argument("--video-device", default="/dev/video0")
    parser.add_argument("--sensor-entity", default="imx219 10-0010")
    parser.add_argument("--sensor-format", default="SRGGB10_1X10")
    parser.add_argument("--pixel-format", default="RG10")
    parser.add_argument("--width", type=int, default=1640)
    parser.add_argument("--height", type=int, default=1232)
    parser.add_argument("--stride-pixels", type=int, default=1648)
    parser.add_argument("--stream-mmap", type=int, default=3)
    parser.add_argument("--stream-count", type=int, default=1)
    parser.add_argument("--raw-output", default="camera_latest.raw")
    parser.add_argument("--jpg-output", default="camera_latest.jpg")
    parser.add_argument("--skip-capture", action="store_true", help="Reuse an existing raw file")
    parser.add_argument("--black-level", type=float, default=64.0)
    parser.add_argument("--r-gain", type=float, default=1.2)
    parser.add_argument("--g-gain", type=float, default=0.8)
    parser.add_argument("--b-gain", type=float, default=1.2)
    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--saturation", type=float, default=1.8)
    parser.add_argument("--sharpen", type=float, default=0.2)
    parser.add_argument(
        "--resize-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize lite ISP output back to the original raw resolution.",
    )
    return parser.parse_args()


def main() -> None:
    jpg_path = capture_and_convert(parse_args())
    print(f"Saved camera image: {jpg_path}")


if __name__ == "__main__":
    main()
