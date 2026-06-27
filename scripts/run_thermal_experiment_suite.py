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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thermal baseline/adaptive experiment suite")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--duration-min", type=float, default=15.0)
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--run-id", default=datetime.now().strftime("thermal_%Y%m%d_%H%M%S"))
    parser.add_argument("--log-dir", type=Path, default=ROOT / "experiments" / "logs" / "thermal_suite")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "experiments" / "results" / "thermal_suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop-video", action="store_true")
    parser.add_argument("--cooldown-sec", type=float, default=0.0)
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


def run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    logs: list[Path] = []
    for index, strategy in enumerate(strategies, start=1):
        label = f"{args.run_id}_{index:02d}_{strategy}"
        output = args.log_dir / f"{label}.csv"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
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
        if args.thermal_state is not None:
            cmd.extend(["--thermal-state", args.thermal_state])
        if args.thermal_temp_c is not None:
            cmd.extend(["--thermal-temp-c", str(args.thermal_temp_c)])

        run_cmd(cmd)
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

        if args.cooldown_sec > 0 and index < len(strategies):
            print(f"Cooling down for {args.cooldown_sec:.1f}s...", flush=True)
            time.sleep(args.cooldown_sec)

    print("Thermal experiment suite complete.")
    for log in logs:
        print(f"  {log}")


if __name__ == "__main__":
    main()
