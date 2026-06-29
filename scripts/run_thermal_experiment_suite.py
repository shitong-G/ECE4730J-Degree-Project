#!/usr/bin/env python3
"""Run a thermal-aware experiment suite and summarize each run."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STRATEGIES = [
    "native_rtdetr",
    "fixed_frame_skip",
    "fixed_low_power",
    "thermal_only",
    "thermal_balanced",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thermal baseline/adaptive experiment suite")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--duration-min", type=float, default=15.0)
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--repeats", type=int, default=1, help="Repeat the full strategy list N times")
    parser.add_argument("--run-id", default=datetime.now().strftime("thermal_%Y%m%d_%H%M%S"))
    parser.add_argument("--log-dir", type=Path, default=ROOT / "experiments" / "logs" / "thermal_suite")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "experiments" / "results" / "thermal_suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop-video", action="store_true")
    parser.add_argument("--log-detections", action="store_true")
    parser.add_argument("--enable-thread-sessions", action="store_true")
    parser.add_argument("--thread-session-counts", default=None)
    parser.add_argument("--apply-runtime-actions", action="store_true")
    parser.add_argument("--enable-lk-tracking", action="store_true")
    parser.add_argument("--lk-force-refresh-on-failure", action="store_true")
    parser.add_argument("--lk-max-failure-ratio", type=float, default=None)
    parser.add_argument("--lk-min-valid-points", type=int, default=None)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run each experiment through run_live_dashboard.py for browser monitoring",
    )
    parser.add_argument("--dashboard-host", default="0.0.0.0")
    parser.add_argument("--dashboard-port", type=int, default=8000)
    parser.add_argument("--dashboard-jpeg-width", type=int, default=960)
    parser.add_argument("--dashboard-jpeg-quality", type=int, default=78)
    parser.add_argument("--dashboard-no-video-stream", action="store_true")
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=0.0,
        help="Minimum wall-clock cooldown between runs, even if temperature is already low",
    )
    parser.add_argument(
        "--cooldown-temp-c",
        type=float,
        default=None,
        help="Wait until CPU temperature is at or below this value before the next run",
    )
    parser.add_argument(
        "--cooldown-poll-sec",
        type=float,
        default=10.0,
        help="Temperature polling interval while waiting for cooldown",
    )
    parser.add_argument(
        "--max-cooldown-min",
        type=float,
        default=30.0,
        help="Maximum temperature-based cooldown wait before continuing anyway",
    )
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--thermal-state",
        choices=["normal", "warm", "hot", "critical", "unknown"],
        default=None,
        help="Pass a fixed thermal state override to run_experiment.py",
    )
    parser.add_argument(
        "--thermal-temp-c",
        type=float,
        default=None,
        help="Pass a fixed temperature override to run_experiment.py",
    )
    return parser.parse_args()


def read_cpu_temp_c() -> float | None:
    """Read Raspberry Pi/Linux CPU temperature from thermal sysfs."""
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        if not path.exists():
            return None
        return float(path.read_text(encoding="utf-8").strip()) / 1000.0
    except (OSError, ValueError):
        return None


def wait_for_cooldown(args: argparse.Namespace) -> None:
    """Wait fixed time and then, if available, wait for CPU temp threshold."""
    if args.cooldown_sec > 0:
        print(f"Cooling down for at least {args.cooldown_sec:.1f}s...", flush=True)
        time.sleep(args.cooldown_sec)

    if args.cooldown_temp_c is None:
        return

    temp = read_cpu_temp_c()
    if temp is None:
        print(
            "CPU temperature is unavailable; skipping temperature-based cooldown.",
            flush=True,
        )
        return

    deadline = time.monotonic() + max(0.0, args.max_cooldown_min) * 60.0
    while temp > args.cooldown_temp_c:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            print(
                f"Cooldown timeout reached at {temp:.1f}C; continuing anyway.",
                flush=True,
            )
            return
        print(
            f"Waiting for cooldown: {temp:.1f}C > {args.cooldown_temp_c:.1f}C "
            f"(timeout in {remaining / 60.0:.1f} min)",
            flush=True,
        )
        time.sleep(max(1.0, args.cooldown_poll_sec))
        temp = read_cpu_temp_c()
        if temp is None:
            print("CPU temperature became unavailable; continuing.", flush=True)
            return

    print(f"Cooldown complete: {temp:.1f}C <= {args.cooldown_temp_c:.1f}C", flush=True)


def run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def print_runtime_action_probe(enabled: bool) -> None:
    """Print OS runtime-action capability once before the suite."""
    if not enabled:
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_runtime_action_support.py"),
    ]
    print("Runtime action support probe:", flush=True)
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"Runtime action support probe failed: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    print_runtime_action_probe(args.apply_runtime_actions)

    logs: list[Path] = []
    run_number = 0
    total_runs = max(1, args.repeats) * len(strategies)
    for repeat in range(1, max(1, args.repeats) + 1):
        for strategy_index, strategy in enumerate(strategies, start=1):
            run_number += 1
            if run_number > 1:
                wait_for_cooldown(args)

            label = (
                f"{args.run_id}_r{repeat:02d}_{strategy_index:02d}_"
                f"{strategy}"
            )
            output = args.log_dir / f"{label}.csv"
            print(
                f"Starting run {run_number}/{total_runs}: repeat={repeat}, strategy={strategy}",
                flush=True,
            )
            start_temp = read_cpu_temp_c()
            if start_temp is not None:
                print(f"Start CPU temperature: {start_temp:.1f}C", flush=True)

            cmd = [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / ("run_live_dashboard.py" if args.dashboard else "run_experiment.py")
                ),
                "--config",
                str(args.config),
                "--strategy",
                strategy,
                "--duration-min",
                str(args.duration_min),
                "--output",
                str(output),
            ]
            if args.video is not None:
                cmd.extend(["--video", str(args.video)])
            if args.loop_video:
                cmd.append("--loop-video")
            if args.dry_run:
                cmd.append("--dry-run")
            if args.log_detections:
                cmd.append("--log-detections")
            if args.enable_thread_sessions:
                cmd.append("--enable-thread-sessions")
            if args.thread_session_counts:
                cmd.extend(["--thread-session-counts", args.thread_session_counts])
            if args.apply_runtime_actions:
                cmd.append("--apply-runtime-actions")
            if args.enable_lk_tracking:
                cmd.append("--enable-lk-tracking")
            if args.lk_force_refresh_on_failure:
                cmd.append("--lk-force-refresh-on-failure")
            if args.lk_max_failure_ratio is not None:
                cmd.extend(["--lk-max-failure-ratio", str(args.lk_max_failure_ratio)])
            if args.lk_min_valid_points is not None:
                cmd.extend(["--lk-min-valid-points", str(args.lk_min_valid_points)])
            if args.dashboard:
                cmd.extend(
                    [
                        "--host",
                        args.dashboard_host,
                        "--port",
                        str(args.dashboard_port),
                        "--jpeg-width",
                        str(args.dashboard_jpeg_width),
                        "--jpeg-quality",
                        str(args.dashboard_jpeg_quality),
                    ]
                )
                if args.dashboard_no_video_stream:
                    cmd.append("--no-video-stream")
            if args.thermal_state is not None:
                cmd.extend(["--thermal-state", args.thermal_state])
            if args.thermal_temp_c is not None:
                cmd.extend(["--thermal-temp-c", str(args.thermal_temp_c)])

            run_cmd(cmd)
            end_temp = read_cpu_temp_c()
            if end_temp is not None:
                print(f"End CPU temperature: {end_temp:.1f}C", flush=True)
            logs.append(output)

            if not args.skip_summary:
                run_cmd(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "summarize_baseline.py"),
                        "--input",
                        str(output),
                        "--output-dir",
                        str(args.result_dir),
                        "--label",
                        label,
                    ]
                )

            if not args.skip_plot:
                try:
                    run_cmd(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "plot_results.py"),
                            "--input",
                            str(output),
                            "--output-dir",
                            str(args.result_dir),
                        ]
                    )
                except subprocess.CalledProcessError as exc:
                    print(f"Plotting failed for {output}: {exc}", file=sys.stderr)
                    print("Install pandas/matplotlib or rerun with --skip-plot.", file=sys.stderr)

    print("Thermal experiment suite complete.")
    for log in logs:
        print(f"  {log}")


if __name__ == "__main__":
    main()
