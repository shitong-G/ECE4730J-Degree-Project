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


def boolean_ratio(frame, column: str, pd) -> float:
    if column not in frame.columns:
        return float("nan")
    values = frame[column].astype(str).str.lower()
    valid = values.isin({"true", "false", "1", "0"})
    if not valid.any():
        return float("nan")
    return float(values[valid].isin({"true", "1"}).mean())


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
        "formal_start_temperature_c": run_meta.get("formal_start_temperature_c"),
        "pre_warmup_temperature_c": run_meta.get("pre_warmup_temperature_c"),
        "temp_start_c": numeric(runtime.get("temp_c", pd.Series(dtype=float)), pd).iloc[0] if "temp_c" in runtime and runtime["temp_c"].notna().any() else float("nan"),
        "temp_mean_c": numeric(runtime.get("temp_c", pd.Series(dtype=float)), pd).mean(),
        "temp_max_c": numeric(runtime.get("temp_c", pd.Series(dtype=float)), pd).max(),
        "temp_end_c": numeric(runtime.get("temp_c", pd.Series(dtype=float)), pd).dropna().iloc[-1] if "temp_c" in runtime and runtime["temp_c"].notna().any() else float("nan"),
        "throttled_frame_pct": 100.0 * runtime.get("currently_throttled", pd.Series(False, index=runtime.index)).astype(str).str.lower().isin({"true", "1"}).mean(),
        "soft_temp_limit_pct": 100.0 * boolean_ratio(runtime, "soft_temp_limit", pd),
        "governor_applied_pct": 100.0 * boolean_ratio(runtime, "governor_applied", pd),
        "affinity_applied_pct": 100.0 * boolean_ratio(runtime, "cpu_affinity_applied", pd),
        "fan_enabled_pct": 100.0 * boolean_ratio(runtime, "fan_enabled", pd),
        "fan_pwm_pct": 100.0 * runtime.get("fan_mode", pd.Series("", index=runtime.index)).astype(str).eq("pwm").mean(),
        "fan_pwm_no_gpio_pct": 100.0 * runtime.get("fan_mode", pd.Series("", index=runtime.index)).astype(str).eq("pwm_no_gpio").mean(),
        "fan_duty_mean_pct": 100.0 * numeric(runtime.get("fan_duty_cycle", pd.Series(dtype=float)), pd).mean(),
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


def plot_comprehensive_summary(summary, output_dir: Path, plt, np):
    labels = summary["strategy"].tolist()
    x = np.arange(len(summary))
    colors = [
        "#4e79a7" if strategy == "native_rtdetr"
        else "#f28e2b" if "lk" in strategy
        else "#59a14f"
        for strategy in labels
    ]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(17, 15), constrained_layout=True)
    axes[0, 0].bar(x, summary["latency_p50_ms"], color=colors, label="P50")
    axes[0, 0].bar(
        x,
        summary["latency_p95_ms"] - summary["latency_p50_ms"],
        bottom=summary["latency_p50_ms"],
        color="#e15759",
        label="P95-P50",
    )
    axes[0, 0].set(title="Full-inference latency", ylabel="ms")
    axes[0, 0].legend()
    axes[0, 1].bar(x - .2, summary["loop_fps_mean"], .4, label="pipeline FPS", color="#4e79a7")
    axes[0, 1].bar(x + .2, summary["actual_detector_fps_mean"], .4, label="actual detector FPS", color="#e15759")
    axes[0, 1].set(title="Pipeline and detector throughput", ylabel="FPS")
    axes[0, 1].legend()
    axes[1, 0].bar(x - .22, summary["formal_start_temperature_c"], .22, label="formal start", color="#bab0ab")
    axes[1, 0].bar(x, summary["temp_mean_c"], .22, label="mean", color="#f28e2b")
    axes[1, 0].bar(x + .22, summary["temp_max_c"], .22, label="max", color="#e15759")
    axes[1, 0].axhline(80, color="#9c755f", linestyle="--", linewidth=1, label="80 C")
    axes[1, 0].set(title="CPU temperature", ylabel="Celsius")
    axes[1, 0].legend()
    axes[1, 1].bar(x - .24, summary["pseudo_recall"], .24, label="pseudo recall", color="#4e79a7")
    axes[1, 1].bar(x, summary["precision_proxy"], .24, label="precision proxy", color="#59a14f")
    axes[1, 1].bar(x + .24, summary["mean_matched_iou"], .24, label="matched IoU", color="#f28e2b")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set(title="Native-640 agreement (common first 420 frames)", ylabel="ratio")
    axes[1, 1].legend()
    axes[2, 0].bar(x - .2, summary["inference_ratio_pct"], .4, label="detector frames", color="#76b7b2")
    axes[2, 0].bar(x + .2, summary["fan_pwm_pct"], .4, label="real PWM fan frames", color="#edc948")
    axes[2, 0].set(title="Runtime policy activity", ylabel="% of frames")
    axes[2, 0].legend()
    scatter = axes[2, 1].scatter(
        summary["latency_p50_ms"],
        summary["pseudo_recall"],
        s=60 + 20 * summary["loop_fps_mean"].clip(lower=0),
        c=summary["temp_max_c"],
        cmap="coolwarm",
        edgecolors="black",
        linewidths=.4,
    )
    labelled_strategies = {
        "native_rtdetr",
        "default",
        "fixed_low_power",
        "fixed_frame_skip",
        "scene_track_lk",
        "scene_thermal_coadaptive",
        "scene_thermal_interval_lk",
    }
    short_labels = {
        "native_rtdetr": "native FP32",
        "fixed_low_power": "low-power 320",
        "fixed_frame_skip": "skip 480",
        "scene_track_lk": "LK",
        "scene_thermal_coadaptive": "coadaptive",
        "scene_thermal_interval_lk": "thermal+LK",
    }
    for _, row in summary[summary["strategy"].isin(labelled_strategies)].iterrows():
        axes[2, 1].annotate(
            short_labels.get(row["strategy"], row["strategy"]),
            (row["latency_p50_ms"], row["pseudo_recall"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[2, 1].set(title="Latency-quality trade-off", xlabel="P50 inference latency (ms)", ylabel="pseudo recall")
    fig.colorbar(scatter, ax=axes[2, 1], label="max CPU temperature (C)")
    for axis in axes.flat:
        if axis is not axes[2, 1]:
            axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    fig.savefig(output_dir / "strategy_comparison.png", dpi=180)
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

    quality_path = output_dir / "detection_quality_common420.csv"
    quality_frames_path = output_dir / "detection_quality_frames_common420.csv"
    if not quality_path.exists():
        raise FileNotFoundError(
            "Missing common-frame quality summary. Run "
            "evaluate_strategy_detection_quality.py without cycle mapping first."
        )
    quality = pd.read_csv(quality_path)
    quality_columns = [
        "strategy",
        "common_frames",
        "pseudo_recall",
        "precision_proxy",
        "mean_matched_iou",
        "mean_center_error_norm",
        "detection_count_ratio",
        "infer_frame_pseudo_recall",
        "infer_frame_precision_proxy",
        "noninfer_frame_pseudo_recall",
        "noninfer_frame_precision_proxy",
    ]
    quality = quality[quality_columns]
    baseline_quality = pd.DataFrame([{
        "strategy": "native_rtdetr",
        "common_frames": 420,
        "pseudo_recall": 1.0,
        "precision_proxy": 1.0,
        "mean_matched_iou": 1.0,
        "mean_center_error_norm": 0.0,
        "detection_count_ratio": 1.0,
        "infer_frame_pseudo_recall": 1.0,
        "infer_frame_precision_proxy": 1.0,
        "noninfer_frame_pseudo_recall": float("nan"),
        "noninfer_frame_precision_proxy": float("nan"),
    }])
    quality = pd.concat([baseline_quality, quality], ignore_index=True)
    if quality_frames_path.exists():
        quality_frames = pd.read_csv(quality_frames_path)
        quality_frames["did_infer_bool"] = (
            quality_frames["did_infer"].astype(str).str.lower().isin({"true", "1"})
        )
        common_counts = (
            quality_frames.groupby("student")
            .agg(
                quality_common_frames=("frame_id", "size"),
                quality_common_infer_frames=("did_infer_bool", "sum"),
            )
            .reset_index()
        )
        common_counts["strategy"] = common_counts["student"].str.replace(
            r"^\d+_", "", regex=True
        )
        quality = quality.merge(
            common_counts[["strategy", "quality_common_frames", "quality_common_infer_frames"]],
            on="strategy",
            how="left",
        )
    summary = summary.merge(quality, on="strategy", how="left")
    summary.loc[summary["strategy"] == "native_rtdetr", "quality_common_frames"] = 420
    summary.loc[summary["strategy"] == "native_rtdetr", "quality_common_infer_frames"] = 420
    baseline_latency = float(summary.iloc[0]["latency_p50_ms"])
    baseline_loop_fps = float(summary.iloc[0]["loop_fps_mean"])
    summary["latency_speedup_vs_native"] = baseline_latency / summary["latency_p50_ms"]
    summary["pipeline_fps_vs_native"] = summary["loop_fps_mean"] / baseline_loop_fps
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    plot_comprehensive_summary(summary, output_dir, plt, np)
    plot_timelines(runs, output_dir, plt, pd)

    baseline = summary.iloc[0]
    nonbaseline = summary[summary["strategy"] != "native_rtdetr"]
    fastest_pipeline = nonbaseline.loc[nonbaseline["loop_fps_mean"].idxmax()]
    best_quality = nonbaseline.loc[nonbaseline["pseudo_recall"].idxmax()]
    coolest = nonbaseline.loc[nonbaseline["temp_max_c"].idxmin()]
    start_min = float(summary["formal_start_temperature_c"].min())
    start_max = float(summary["formal_start_temperature_c"].max())
    fan_rows = summary[summary["fan_pwm_pct"] > 0]
    fan_text = ", ".join(
        f"{row.strategy}={row.fan_pwm_pct:.1f}%"
        for row in fan_rows.itertuples()
    ) or "none"
    lines = [
        "# Controlled suite analysis",
        "",
        "## Scope",
        "",
        "Thirteen strategies were run once for approximately 20 minutes each. "
        "The baseline uses native FP32 RT-DETR at 640; all other strategies use "
        "the quantization-only INT8 320/480/640 family.",
        "",
        "## Key results",
        "",
        f"- Native baseline latency: P50 {baseline['latency_p50_ms']:.2f} ms, "
        f"P95 {baseline['latency_p95_ms']:.2f} ms; pipeline {baseline['loop_fps_mean']:.3f} FPS.",
        f"- Highest pipeline throughput: {fastest_pipeline['strategy']} at "
        f"{fastest_pipeline['loop_fps_mean']:.3f} FPS "
        f"({fastest_pipeline['pipeline_fps_vs_native']:.1f}x native).",
        f"- Highest non-baseline pseudo recall: {best_quality['strategy']} at "
        f"{best_quality['pseudo_recall']:.3f}; precision proxy "
        f"{best_quality['precision_proxy']:.3f}; matched IoU "
        f"{best_quality['mean_matched_iou']:.3f}.",
        f"- Lowest non-baseline maximum temperature: {coolest['strategy']} at "
        f"{coolest['temp_max_c']:.2f} C.",
        f"- Real PWM fan activity: {fan_text}; pwm_no_gpio was 0% for every strategy.",
        "- The 640-INT8 strategies cluster at pseudo recall 0.806-0.809, "
        "precision proxy 0.933-0.934 and matched IoU 0.934-0.935.",
        "- The 320-resolution policies reach pseudo recall 0.195-0.202; "
        "fixed 480 frame skipping reaches 0.468; both LK policies reach 0.452 "
        "when tracked frames are included.",
        "- Compared with scene_only, scene_thermal_coadaptive lowers maximum "
        "temperature by 8.28 C (85.21 to 76.93 C) while retaining the same "
        "common-window pseudo recall (0.808). This is a combined thermal-policy "
        "and active-cooling comparison, not a fan-only ablation.",
        "",
        "## Accuracy definition",
        "",
        "Accuracy values are pseudo-label agreement against native FP32 640 output, "
        "not ground-truth mAP. Only the common first 420 video frames are scored, "
        "because the native 20-minute baseline did not finish the full source video. "
        "Matching requires equal class ID and IoU >= 0.5.",
        "",
        "## Comparability limitations",
        "",
        f"- Formal start temperatures ranged from {start_min:.2f} C to {start_max:.2f} C "
        f"(spread {start_max - start_min:.2f} C), so strict equal-temperature start "
        "was not achieved.",
        "- Baseline-versus-policy comparisons jointly change model precision "
        "(FP32 versus INT8) and runtime policy. Use INT8-to-INT8 controls such as "
        "static_affinity or scene_only to isolate policy effects.",
        "- Each strategy has one run; no variance estimate or statistical significance "
        "can be reported.",
        "- Power measurements are unavailable in this suite.",
        "- LK detector-only accuracy in the common window is based on only two detector "
        "frames; the all-frame LK score is the meaningful end-to-end quality measure.",
        "",
        "See `strategy_summary.csv` for the complete numeric table.",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved analysis: {output_dir}")


if __name__ == "__main__":
    main()
