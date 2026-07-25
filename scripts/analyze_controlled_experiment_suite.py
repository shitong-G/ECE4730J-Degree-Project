#!/usr/bin/env python3
"""Summarise and plot a controlled strategy-suite experiment directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def numeric(series, pd):
    return pd.to_numeric(series, errors="coerce")


def inference_mask(frame, pd):
    if "did_infer" not in frame.columns:
        return frame.index == frame.index
    return frame["did_infer"].astype(str).str.lower().isin({"true", "1"})


def percentile(series, q: float, pd) -> float:
    values = numeric(series, pd).dropna()
    return float(values.quantile(q)) if not values.empty else float("nan")


def summarise_run(run_dir: Path, run_meta: dict, pd) -> tuple[dict, object, object]:
    runtime = pd.read_csv(run_dir / "runtime.csv")
    profile_path = run_dir / "runtime_profile.csv"
    profile = pd.read_csv(profile_path) if profile_path.exists() else pd.DataFrame()
    infer = runtime[inference_mask(runtime, pd)].copy()
    profile_infer = profile[inference_mask(profile, pd)].copy() if not profile.empty else profile
    timestamp = numeric(runtime.get("timestamp", pd.Series(dtype=float)), pd)
    duration_min = (timestamp.max() - timestamp.min()) / 60.0 if timestamp.notna().any() else float("nan")
    resolution_counts = (
        infer.get("resolved_input_resolution", infer.get("input_resolution", pd.Series(dtype=float)))
        .value_counts(dropna=True).sort_index()
    )
    interval_counts = runtime.get("inference_interval", pd.Series(dtype=float)).value_counts(dropna=True).sort_index()
    row = {
        "order": int(run_dir.name.split("_", 1)[0]),
        "strategy": run_meta["strategy"],
        "model_condition": run_meta.get("model_condition"),
        "duration_min": duration_min,
        "frames": len(runtime),
        "inference_frames": len(infer),
        "inference_ratio_pct": 100.0 * len(infer) / max(1, len(runtime)),
        "latency_p50_ms": percentile(infer.get("latency_ms", pd.Series(dtype=float)), .50, pd),
        "latency_p95_ms": percentile(infer.get("latency_ms", pd.Series(dtype=float)), .95, pd),
        "latency_mean_ms": numeric(infer.get("latency_ms", pd.Series(dtype=float)), pd).mean(),
        "onnx_p50_ms": percentile(profile_infer.get("onnx_run_ms", pd.Series(dtype=float)), .50, pd),
        "onnx_p95_ms": percentile(profile_infer.get("onnx_run_ms", pd.Series(dtype=float)), .95, pd),
        "infer_total_p95_ms": percentile(profile_infer.get("infer_total_ms", pd.Series(dtype=float)), .95, pd),
        "loop_fps_mean": numeric(runtime.get("loop_fps", runtime.get("fps", pd.Series(dtype=float))), pd).mean(),
        "effective_fps_mean": numeric(runtime.get("effective_inference_fps", pd.Series(dtype=float)), pd).mean(),
        "actual_detector_fps_mean": numeric(runtime.get("actual_inference_fps", pd.Series(dtype=float)), pd).mean(),
        "temp_start_c": numeric(runtime.get("temp_c", pd.Series(dtype=float)), pd).iloc[0] if "temp_c" in runtime and runtime["temp_c"].notna().any() else float("nan"),
        "temp_mean_c": numeric(runtime.get("temp_c", pd.Series(dtype=float)), pd).mean(),
        "temp_max_c": numeric(runtime.get("temp_c", pd.Series(dtype=float)), pd).max(),
        "temp_end_c": numeric(runtime.get("temp_c", pd.Series(dtype=float)), pd).dropna().iloc[-1] if "temp_c" in runtime and runtime["temp_c"].notna().any() else float("nan"),
        "throttled_frame_pct": 100.0 * runtime.get("currently_throttled", pd.Series(False, index=runtime.index)).astype(str).str.lower().isin({"true", "1"}).mean(),
        "resolution_distribution": ";".join(f"{int(k)}:{v}" for k, v in resolution_counts.items()),
        "interval_distribution": ";".join(f"{int(k)}:{v}" for k, v in interval_counts.items()),
    }
    return row, runtime, profile


def elapsed_minutes(frame, pd):
    values = numeric(frame["timestamp"], pd)
    return (values - values.iloc[0]) / 60.0


def plot_summary(summary, output_dir: Path, plt, np):
    labels = summary["strategy"].tolist()
    x = np.arange(len(summary))
    colors = ["#4c78a8"] + ["#59a14f"] * (len(summary) - 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].bar(x, summary["latency_p50_ms"], color=colors, label="P50")
    axes[0, 0].bar(x, summary["latency_p95_ms"] - summary["latency_p50_ms"], bottom=summary["latency_p50_ms"], color="#e15759", label="P95-P50")
    axes[0, 0].set_title("Full-inference latency")
    axes[0, 0].set_ylabel("ms")
    axes[0, 0].legend()
    axes[0, 1].bar(x - .18, summary["onnx_p50_ms"], .36, label="ONNX P50", color="#76b7b2")
    axes[0, 1].bar(x + .18, summary["onnx_p95_ms"], .36, label="ONNX P95", color="#f28e2b")
    axes[0, 1].set_title("ONNX Runtime execution")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].legend()
    axes[1, 0].bar(x - .18, summary["loop_fps_mean"], .36, label="process FPS", color="#4e79a7")
    axes[1, 0].bar(x + .18, summary["effective_fps_mean"], .36, label="effective detector FPS", color="#59a14f")
    axes[1, 0].set_title("Throughput")
    axes[1, 0].set_ylabel("FPS")
    axes[1, 0].legend()
    axes[1, 1].bar(x, summary["temp_max_c"], color=colors)
    axes[1, 1].axhline(80, color="#f28e2b", linestyle="--", linewidth=1, label="80°C")
    axes[1, 1].set_title("Maximum CPU temperature")
    axes[1, 1].set_ylabel("°C")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    fig.savefig(output_dir / "strategy_comparison.png", dpi=160)
    plt.close(fig)


def plot_timelines(runs, output_dir: Path, plt, pd):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    for name, runtime, profile in runs:
        x = elapsed_minutes(runtime, pd)
        axes[0].plot(x, numeric(runtime["temp_c"], pd), linewidth=.9, label=name)
        infer = runtime[inference_mask(runtime, pd)]
        if not infer.empty:
            axes[1].plot(elapsed_minutes(infer, pd), numeric(infer["latency_ms"], pd), linewidth=.7, label=name)
        axes[2].plot(x, numeric(runtime["effective_inference_fps"], pd), linewidth=.8, label=name)
    axes[0].axhline(80, color="#f28e2b", linestyle="--", linewidth=1)
    axes[0].set_ylabel("CPU temp (°C)")
    axes[1].set_ylabel("inference latency (ms)")
    axes[2].set_ylabel("effective FPS")
    axes[2].set_xlabel("formal experiment time (min)")
    for axis in axes:
        axis.legend(ncol=2, fontsize=7, loc="best")
    fig.savefig(output_dir / "all_strategy_timelines.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    manifest = json.loads((suite_dir / "manifest.json").read_text(encoding="utf-8"))
    output_dir = args.output_dir or suite_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install pandas and matplotlib to analyse the suite.") from exc

    rows, runs = [], []
    for run_meta in manifest["runs"]:
        order = next(path for path in suite_dir.iterdir() if path.is_dir() and path.name.endswith("_" + run_meta["strategy"]))
        row, runtime, profile = summarise_run(order, run_meta, pd)
        rows.append(row)
        runs.append((run_meta["strategy"], runtime, profile))
    summary = pd.DataFrame(rows).sort_values("order")
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    plot_summary(summary, output_dir, plt, np)
    plot_timelines(runs, output_dir, plt, pd)

    baseline = summary.iloc[0]
    lines = ["# Controlled suite analysis", "", "## Key metrics", "", "See `strategy_summary.csv` for the complete numeric table.", "", "## Comparability note", "", "The baseline is native FP32 RT-DETR; every subsequent strategy uses the pruned INT8 model family. Model change and policy change are therefore jointly confounded for baseline-versus-policy comparisons.", "", f"Baseline P50 latency: {baseline['latency_p50_ms']:.2f} ms; P95 latency: {baseline['latency_p95_ms']:.2f} ms."]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved analysis: {output_dir}")


if __name__ == "__main__":
    main()
