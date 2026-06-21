#!/usr/bin/env python3
"""Plot per-inference runtime, model, and device metrics into one JPG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot all available parameters for frames with full inference"
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Main experiment CSV log from experiments/logs/",
    )
    p.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional profile CSV. Defaults to <input_stem>_profile.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JPG path. Defaults to experiments/results/<input_stem>_inference_details.jpg",
    )
    p.add_argument(
        "--max-inferences",
        type=int,
        default=None,
        help="Plot at most this many inference events from the start of the log",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Rolling window in inference rows for smoother numeric trend lines; use 1 for raw values",
    )
    return p.parse_args()


def _require_plotting_deps():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Install plotting deps: pip install matplotlib pandas")
        sys.exit(1)
    return plt, pd


def _load_logs(input_path: Path, profile_path: Path | None, pd):
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    main = pd.read_csv(input_path)
    if "frame_id" not in main.columns:
        print("Input log must contain frame_id")
        sys.exit(1)

    if profile_path is None:
        profile_path = input_path.with_name(f"{input_path.stem}_profile.csv")

    if profile_path.exists():
        profile = pd.read_csv(profile_path)
        profile_cols = [
            col
            for col in profile.columns
            if col == "frame_id" or col not in {"timestamp", "strategy"}
        ]
        return main.merge(
            profile[profile_cols],
            on="frame_id",
            how="left",
            suffixes=("", "_profile"),
        )
    return main


def _full_inference_rows(df):
    if "did_infer" in df.columns:
        mask = df["did_infer"].astype(str).str.lower().isin({"true", "1"})
        return df[mask].copy()
    if "latency_ms" in df.columns:
        return df[df["latency_ms"].fillna(0) > 0].copy()
    return df.copy()


def _elapsed_minutes(df, pd):
    """Return elapsed experiment time in minutes, falling back to frame_id."""
    if "timestamp" in df.columns:
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce")
        if timestamps.notna().any():
            first = timestamps[timestamps.notna()].iloc[0]
            return (timestamps - first) / 60.0
    return pd.to_numeric(df["frame_id"], errors="coerce")


def _num_series(df, col, pd):
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce")
    if not values.notna().any():
        return None
    return values


def _smooth(values, window: int):
    if window <= 1:
        return values
    return values.rolling(window=window, min_periods=1).mean()


def _effective_inference_fps(df, pd):
    if "effective_inference_fps" in df.columns:
        return _num_series(df, "effective_inference_fps", pd)
    fps_col = "loop_fps" if "loop_fps" in df.columns else "fps"
    if fps_col not in df.columns or "inference_interval" not in df.columns:
        return None
    interval = pd.to_numeric(df["inference_interval"], errors="coerce").clip(lower=1)
    return pd.to_numeric(df[fps_col], errors="coerce") / interval


def _category_codes(df, col, pd):
    if col not in df.columns:
        return None, None, None
    values = df[col].fillna("none").astype(str)
    if values.empty:
        return None, None, None
    cats = pd.Categorical(values)
    return cats.codes, list(cats.categories), values


def _plot_missing(ax, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_lines(ax, x, df, cols, pd, title, ylabel=None) -> None:
    plotted = False
    smooth_window = int(df.attrs.get("smooth_window", 1))
    for col, label in cols:
        if col == "__effective_inference_fps__":
            values = _effective_inference_fps(df, pd)
        else:
            values = _num_series(df, col, pd)
        if values is None:
            continue
        ax.plot(
            x,
            _smooth(values, smooth_window),
            linewidth=0.9,
            marker=".",
            markersize=2.5,
            label=label,
        )
        plotted = True
    if not plotted:
        _plot_missing(ax, title)
        return
    ax.set_title(title)
    ax.set_xlabel("elapsed time (min)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _plot_steps(ax, x, df, cols, pd, title, ylabel=None) -> None:
    plotted = False
    for col, label in cols:
        values = _num_series(df, col, pd)
        if values is None:
            continue
        ax.step(x, values, where="post", linewidth=1.0, label=label)
        plotted = True
    if not plotted:
        _plot_missing(ax, title)
        return
    ax.set_title(title)
    ax.set_xlabel("elapsed time (min)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _plot_categories(ax, x, df, cols, pd, title) -> None:
    offset = 0
    ticks: list[int] = []
    labels: list[str] = []
    plotted = False
    for col, prefix in cols:
        codes, categories, _ = _category_codes(df, col, pd)
        if codes is None or categories is None:
            continue
        y = codes + offset
        ax.step(x, y, where="post", linewidth=1.0, label=prefix)
        ticks.extend([offset + i for i in range(len(categories))])
        labels.extend([f"{prefix}:{cat}" for cat in categories])
        offset += len(categories) + 1
        plotted = True
    if not plotted:
        _plot_missing(ax, title)
        return
    ax.set_title(title)
    ax.set_xlabel("elapsed time (min)")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(True, alpha=0.25)


def _add_run_summary(fig, df, input_path: Path) -> None:
    strategy = df["strategy"].iloc[0] if "strategy" in df.columns and not df.empty else "unknown"
    n = len(df)
    frame_min = int(df["frame_id"].min()) if n else 0
    frame_max = int(df["frame_id"].max()) if n else 0
    fig.suptitle(
        f"Per-Inference Runtime Details: {input_path.stem} | "
        f"strategy={strategy} | inferences={n} | frames={frame_min}-{frame_max}",
        fontsize=14,
    )


def main() -> None:
    args = parse_args()
    plt, pd = _require_plotting_deps()

    df = _load_logs(args.input, args.profile, pd)
    if "loop_fps" not in df.columns and "fps" in df.columns:
        df["loop_fps"] = df["fps"]
    df["_elapsed_min"] = _elapsed_minutes(df, pd)
    df = _full_inference_rows(df)
    if df.empty:
        print("No full inference rows found. Expected did_infer=True or latency_ms > 0.")
        sys.exit(1)

    df = df.sort_values("frame_id").reset_index(drop=True)
    if args.max_inferences is not None:
        df = df.head(args.max_inferences).copy()
    df.attrs["smooth_window"] = args.smooth_window

    x = pd.to_numeric(df["_elapsed_min"], errors="coerce")

    output = args.output
    if output is None:
        output = ROOT / "experiments" / "results" / f"{args.input.stem}_inference_details.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.ravel()
    _add_run_summary(fig, df, args.input)

    _plot_lines(
        axes[0],
        x,
        df,
        [("latency_ms", "logged latency"), ("infer_total_ms", "profile infer total")],
        pd,
        "Inference Latency",
        "ms",
    )
    _plot_lines(
        axes[1],
        x,
        df,
        [
            ("preprocess_ms", "preprocess"),
            ("build_feed_ms", "build feed"),
            ("onnx_run_ms", "onnx run"),
            ("postprocess_ms", "postprocess"),
        ],
        pd,
        "Inference Timing Breakdown",
        "ms",
    )
    _plot_lines(
        axes[2],
        x,
        df,
        [
            ("frame_total_ms", "frame total"),
            ("scene_ms", "scene"),
            ("device_ms", "device"),
            ("runtime_state_ms", "runtime state"),
            ("decision_ms", "decision"),
            ("summary_ms", "summary"),
            ("main_log_write_ms", "log write"),
        ],
        pd,
        "Runtime Loop Timing",
        "ms",
    )
    _plot_lines(
        axes[3],
        x,
        df,
        [
            ("__effective_inference_fps__", "effective inference FPS"),
        ],
        pd,
        "Effective Inference FPS",
        "fps",
    )
    _plot_lines(
        axes[4],
        x,
        df,
        [
            ("temp_c", "temperature"),
            ("power_w", "power"),
        ],
        pd,
        "Device Thermal/Power State",
    )
    _plot_lines(
        axes[5],
        x,
        df,
        [
            ("arm_clock_mhz", "arm clock"),
            ("freq_mhz_avg", "avg cpu freq"),
        ],
        pd,
        "Device Frequency State",
        "MHz",
    )
    _plot_steps(
        axes[6],
        x,
        df,
        [
            ("input_resolution", "input resolution"),
            ("query_budget", "query budget"),
        ],
        pd,
        "Model Workload Knobs",
    )
    _plot_steps(
        axes[7],
        x,
        df,
        [
            ("inference_interval", "inference interval"),
            ("cpu_threads", "cpu threads"),
            ("decoder_layers", "decoder layers"),
        ],
        pd,
        "Runtime Scheduling Knobs",
    )
    _plot_lines(
        axes[8],
        x,
        df,
        [
            ("detection_count", "detections"),
            ("confidence_mean", "mean confidence"),
        ],
        pd,
        "Detection Output Summary",
    )
    _plot_categories(
        axes[9],
        x,
        df,
        [
            ("workload", "workload"),
            ("thermal_state", "thermal"),
            ("action_mode", "action"),
            ("governor", "governor"),
        ],
        pd,
        "Categorical Runtime State",
    )
    _plot_lines(
        axes[10],
        x,
        df,
        [("infer_outer_ms", "infer outer"), ("onnx_run_ms", "onnx run")],
        pd,
        "Inference Call Overhead",
        "ms",
    )
    _plot_steps(
        axes[11],
        x,
        df,
        [("frame_id", "frame id")],
        pd,
        "Full Inference Frames",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output, dpi=140, format="jpg", pil_kwargs={"quality": 92})
    plt.close(fig)
    print(f"Saved per-inference plot: {output}")


if __name__ == "__main__":
    main()
