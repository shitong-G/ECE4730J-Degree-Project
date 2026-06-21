#!/usr/bin/env python3
"""Run adaptive runtime controller with scene and device monitors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.runtime.config import load_config
from scene_runtime.runtime.loop import RuntimeLoop
from scene_runtime.utils.video import FrameSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive scene-thermal runtime controller")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument(
        "--strategy",
        default="scene_thermal_coadaptive",
        choices=[
            "default",
            "static_affinity",
            "fixed_low_power",
            "fixed_frame_skip",
            "thermal_only",
            "scene_only",
            "scene_thermal_coadaptive",
        ],
    )
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--loop-video", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--duration-min", type=float, default=15.0)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.strategy)
    duration_sec = args.duration_min * 60.0
    source = FrameSource(
        args.video,
        synthetic=args.video is None,
        max_frames=int(duration_sec * 10) if args.dry_run else None,
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
    print(f"Controller run complete. Log: {log_path}")


if __name__ == "__main__":
    main()
