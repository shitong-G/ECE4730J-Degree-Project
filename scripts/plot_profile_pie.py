#!/usr/bin/env python3
"""Generate pie charts from per-frame profiling CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INFER_COLUMNS = {
    "preprocess": "preprocess_ms",
    "build_feed": "build_feed_ms",
    "onnx_run": "onnx_run_ms",
    "postprocess": "postprocess_ms",
}

FRAME_COLUMNS = {
    "scene": "scene_ms",
    "device": "device_ms",
    "runtime_state": "runtime_state_ms",
    "decision": "decision_ms",
    "inference": "infer_total_ms",
    "summary": "summary_ms",
    "main_log_write": "main_log_write_ms",
}


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )


def clean_positive(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


def make_pie(values: dict[str, float], title: str, output_path: Path) -> None:
    values = {k: float(v) for k, v in values.items() if float(v) > 0}

    if not values:
        raise ValueError(f"No positive timing values for chart: {title}")

    labels = list(values.keys())
    sizes = list(values.values())

    def autopct_fmt(pct: float) -> str:
        if pct < 1.0:
            return ""
        return f"{pct:.1f}%"

    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct=autopct_fmt,
        startangle=90,
        counterclock=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_inference_pie(df: pd.DataFrame, output_path: Path) -> None:
    required = list(INFER_COLUMNS.values()) + ["infer_total_ms"]
    require_columns(df, required)

    # Only use frames where actual inference was executed.
    if "did_infer" in df.columns:
        infer_df = df[df["did_infer"].astype(str).str.lower().isin(["true", "1", "yes"])]
    else:
        infer_df = df[df["infer_total_ms"] > 0]

    if infer_df.empty:
        raise ValueError("No inference rows found. Check did_infer or infer_total_ms.")

    values: dict[str, float] = {}

    for label, col in INFER_COLUMNS.items():
        values[label] = clean_positive(infer_df[col]).mean()

    infer_total_mean = clean_positive(infer_df["infer_total_ms"]).mean()
    measured_sum = sum(values.values())
    overhead = infer_total_mean - measured_sum

    if overhead > 0:
        values["other_infer_overhead"] = overhead

    make_pie(
        values,
        title="RT-DETR Inference Time Breakdown",
        output_path=output_path,
    )

    print("\nInference average timing:")
    for k, v in values.items():
        print(f"{k:24s}: {v:.3f} ms")
    print(f"{'infer_total_mean':24s}: {infer_total_mean:.3f} ms")


def plot_frame_pie(df: pd.DataFrame, output_path: Path) -> None:
    required = list(FRAME_COLUMNS.values()) + ["frame_total_ms"]
    require_columns(df, required)

    values: dict[str, float] = {}

    for label, col in FRAME_COLUMNS.items():
        values[label] = clean_positive(df[col]).mean()

    frame_total_mean = clean_positive(df["frame_total_ms"]).mean()
    measured_sum = sum(values.values())
    overhead = frame_total_mean - measured_sum

    if overhead > 0:
        values["other_frame_overhead"] = overhead

    make_pie(
        values,
        title="Runtime Per-Frame Time Breakdown",
        output_path=output_path,
    )

    print("\nFrame average timing:")
    for k, v in values.items():
        print(f"{k:24s}: {v:.3f} ms")
    print(f"{'frame_total_mean':24s}: {frame_total_mean:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to *_profile.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same directory as CSV.",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv
    out_dir: Path = args.out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    stem = csv_path.stem
    inference_output = out_dir / f"{stem}_inference_pie.png"
    frame_output = out_dir / f"{stem}_frame_pie.png"

    plot_inference_pie(df, inference_output)
    plot_frame_pie(df, frame_output)


if __name__ == "__main__":
    main()
