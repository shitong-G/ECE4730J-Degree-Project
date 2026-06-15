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
    p.add_argument(
        "--max-frame-id",
        type=int,
        default=None,
        help="Only plot rows with frame_id < this value (e.g. 260 keeps frames 0-259)",
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
    if args.max_frame_id is not None:
        df = df[df["frame_id"] < args.max_frame_id].copy()
        if df.empty:
            print(f"No rows with frame_id < {args.max_frame_id}")
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    if args.max_frame_id is not None:
        stem = f"{stem}_frames0-{args.max_frame_id - 1}"

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

    if "freq_mhz_avg" in df.columns and df["freq_mhz_avg"].notna().any():
        fig_freq, ax_freq = plt.subplots(figsize=(12, 4))
        ax_freq.plot(df["frame_id"], df["freq_mhz_avg"], color="steelblue", linewidth=0.8)
        ax_freq.set_title(f"CPU frequency (MHz)")
        ax_freq.set_xlabel("frame_id")
        ax_freq.set_ylabel("MHz")
        ax_freq.grid(True, alpha=0.3)
        freq_out = args.output_dir / f"{stem}_cpu_freq.png"
        fig_freq.tight_layout()
        fig_freq.savefig(freq_out, dpi=120)
        plt.close(fig_freq)
        print(f"Saved plot: {freq_out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
