#!/usr/bin/env python3
"""Plot experiment log CSV metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot scene-runtime experiment logs")
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV log from experiments/logs/",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "results",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Install plotting deps: pip install matplotlib pandas")
        sys.exit(1)

    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Experiment: {stem}")

    if "latency_ms" in df.columns:
        axes[0, 0].plot(df["frame_id"], df["latency_ms"], linewidth=0.8)
        axes[0, 0].set_title("Inference latency (ms)")
        axes[0, 0].set_xlabel("frame_id")

    if "temp_c" in df.columns and df["temp_c"].notna().any():
        axes[0, 1].plot(df["frame_id"], df["temp_c"], color="tomato", linewidth=0.8)
    else:
        axes[0, 1].text(0.5, 0.5, "No temperature data", ha="center", va="center")
    axes[0, 1].set_title("CPU temperature (C)")

    if "workload" in df.columns:
        wl_map = {"light": 0, "medium": 1, "heavy": 2}
        axes[1, 0].plot(
            df["frame_id"],
            df["workload"].map(wl_map),
            drawstyle="steps-post",
            linewidth=0.8,
        )
        axes[1, 0].set_yticks([0, 1, 2])
        axes[1, 0].set_yticklabels(["light", "medium", "heavy"])
    axes[1, 0].set_title("Scene workload")

    if "fps" in df.columns:
        axes[1, 1].plot(df["frame_id"], df["fps"], color="green", linewidth=0.8)
    axes[1, 1].set_title("FPS")

    out = args.output_dir / f"{stem}_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
