#!/usr/bin/env python3
"""Run an experiment and expose a live browser dashboard."""

from __future__ import annotations

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.dashboard import LiveDashboardServer, LiveDashboardState
from scene_runtime.runtime.config import load_config
from scene_runtime.runtime.loop import RuntimeLoop
from scene_runtime.utils.video import FrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scene-runtime with live web dashboard")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument(
        "--strategy",
        default="thermal_balanced",
        choices=[
            "native_rtdetr",
            "default",
            "static_affinity",
            "fixed_low_power",
            "fixed_frame_skip",
            "thermal_only",
            "thermal_balanced",
            "scene_only",
            "scene_thermal_coadaptive",
        ],
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--loop-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--duration-min", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--history", type=int, default=600, help="Number of dashboard samples to keep")
    parser.add_argument("--jpeg-width", type=int, default=960, help="Live stream width; lower reduces network load")
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--no-video-stream", action="store_true")
    parser.add_argument(
        "--thermal-state",
        choices=["normal", "warm", "hot", "critical", "unknown"],
        default=None,
    )
    parser.add_argument("--thermal-temp-c", type=float, default=None)
    parser.add_argument("--enable-thread-sessions", action="store_true")
    parser.add_argument("--thread-session-counts", default=None)
    parser.add_argument("--apply-runtime-actions", action="store_true")
    parser.add_argument("--enable-lk-tracking", action="store_true")
    parser.add_argument("--lk-force-refresh-on-failure", action="store_true")
    parser.add_argument("--lk-max-failure-ratio", type=float, default=None)
    parser.add_argument("--lk-min-valid-points", type=int, default=None)
    return parser.parse_args()


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
        config.setdefault("inference", {})["thread_session_counts"] = [
            int(item.strip()) for item in args.thread_session_counts.split(",") if item.strip()
        ]
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
        args.output = ROOT / "experiments" / "logs" / f"{args.strategy}_live_{stamp}.csv"

    state = LiveDashboardState(
        max_history=args.history,
        jpeg_quality=args.jpeg_quality,
        jpeg_width=args.jpeg_width,
        score_threshold=args.score_threshold,
        show_stream=not args.no_video_stream,
    )
    server = LiveDashboardServer(state, host=args.host, port=args.port)
    server.start()

    ip = _best_effort_ip()
    bind_url = f"http://{args.host}:{args.port}"
    lan_url = f"http://{ip}:{args.port}" if ip else bind_url
    print("Live dashboard started.")
    print(f"  local/bind: {bind_url}")
    print(f"  LAN URL:    {lan_url}")
    print(f"  log:        {args.output}")
    print("Press Ctrl+C to stop.")

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
        live_callback=state.publish,
    )

    try:
        log_path = loop.run()
        print(f"Experiment finished. Log: {log_path}")
    finally:
        server.stop()


def _best_effort_ip() -> str | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


if __name__ == "__main__":
    main()
