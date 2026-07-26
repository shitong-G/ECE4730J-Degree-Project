#!/usr/bin/env python3
"""Create defense tables and the seven required plots from a defense suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def boolean(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def values(
    rows: Iterable[dict[str, str]], column: str, *, positive: bool = False
) -> list[float]:
    output = [item for row in rows if (item := number(row.get(column))) is not None]
    return [item for item in output if item > 0] if positive else output


def average(items: list[float]) -> float | None:
    return mean(items) if items else None


def percentile(items: list[float], q: float) -> float | None:
    if not items:
        return None
    ordered = sorted(items)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def ratio_true(rows: list[dict[str, str]], column: str) -> float | None:
    items = [item for row in rows if (item := boolean(row.get(column))) is not None]
    return sum(items) / len(items) if items else None


def relative_change(early: float | None, late: float | None) -> float | None:
    if early is None or late is None or early == 0:
        return None
    return 100.0 * (late - early) / early


def window_means(rows: list[dict[str, str]], column: str, *, positive: bool = False) -> tuple[float | None, float | None]:
    size = max(1, len(rows) // 5)
    return (
        average(values(rows[:size], column, positive=positive)),
        average(values(rows[-size:], column, positive=positive)),
    )


def final_counter(
    rows: list[dict[str, str]], column: str, fallback: int
) -> float:
    if rows and (item := number(rows[-1].get(column))) is not None:
        return item
    return float(fallback)


def summarize_run(
    run_dir: Path,
    run_meta: dict[str, Any],
    quality: dict[str, str] | None,
) -> dict[str, Any]:
    rows = read_csv(run_dir / "runtime.csv")
    frame_count = len(rows)
    did_infer = [boolean(row.get("did_infer")) is True for row in rows]
    roi = [boolean(row.get("roi_refresh_applied")) is True for row in rows]
    detector_count = final_counter(rows, "detector_invocation_count", sum(did_infer))
    full_count = final_counter(
        rows,
        "full_detector_invocation_count",
        sum(infer and not is_roi for infer, is_roi in zip(did_infer, roi)),
    )
    roi_count = final_counter(rows, "roi_detector_invocation_count", sum(roi))
    detector_call_resolutions = [
        int(item)
        for row in rows
        if (item := number(row.get("detector_call_resolution"))) is not None
    ]
    latency = values(rows, "latency_ms", positive=True)
    temperatures = values(rows, "temp_c")
    early_latency, late_latency = window_means(rows, "latency_ms", positive=True)
    early_fps, late_fps = window_means(rows, "loop_fps")
    early_freq, late_freq = window_means(rows, "arm_clock_mhz")
    tracking_rows = [
        row for row in rows if str(row.get("tracking_mode", "")).lower() == "track"
    ]
    summary: dict[str, Any] = {
        "run": run_dir.name,
        "key": run_meta.get("key", run_dir.name),
        "title": run_meta.get("title", run_dir.name),
        "groups": ",".join(run_meta.get("groups") or []),
        "strategy": run_meta.get("strategy"),
        "model_kind": run_meta.get("model_kind"),
        "fixed_interval": run_meta.get("fixed_interval"),
        "fan_control": run_meta.get("fan_control"),
        "frames": frame_count,
        "detector_invocation_count": detector_count,
        "detector_invocation_ratio": detector_count / frame_count if frame_count else None,
        "full_detector_invocation_count": full_count,
        "full_detector_invocation_ratio": full_count / frame_count if frame_count else None,
        "roi_detector_invocation_count": roi_count,
        "roi_detector_invocation_ratio": roi_count / frame_count if frame_count else None,
        "detector_calls_320": detector_call_resolutions.count(320),
        "detector_calls_480": detector_call_resolutions.count(480),
        "detector_calls_640": detector_call_resolutions.count(640),
        "loop_fps_mean": average(values(rows, "loop_fps")),
        "actual_inference_fps_mean": average(values(rows, "actual_inference_fps")),
        "latency_ms_mean_detector_frames": average(latency),
        "latency_ms_p95_detector_frames": percentile(latency, 0.95),
        "temp_c_mean": average(temperatures),
        "temp_c_max": max(temperatures) if temperatures else None,
        "arm_clock_mhz_mean": average(values(rows, "arm_clock_mhz")),
        "power_w_mean": average(values(rows, "power_w")),
        "tracking_frame_ratio": len(tracking_rows) / frame_count if frame_count else None,
        "tracking_failure_ratio_mean": average(values(tracking_rows, "tracking_failure_ratio")),
        "tracking_mean_quality": average(values(tracking_rows, "tracking_mean_quality")),
        "fan_active_ratio": sum(
            (number(row.get("fan_duty_cycle")) or 0) > 0 for row in rows
        ) / frame_count if frame_count else None,
        "fan_duty_mean": average(values(rows, "fan_duty_cycle")),
        "currently_throttled_ratio": ratio_true(rows, "currently_throttled"),
        "soft_temp_limit_ratio": ratio_true(rows, "soft_temp_limit"),
        "early_latency_ms": early_latency,
        "late_latency_ms": late_latency,
        "latency_drift_pct": relative_change(early_latency, late_latency),
        "early_loop_fps": early_fps,
        "late_loop_fps": late_fps,
        "loop_fps_drift_pct": relative_change(early_fps, late_fps),
        "early_arm_clock_mhz": early_freq,
        "late_arm_clock_mhz": late_freq,
        "arm_clock_drift_pct": relative_change(early_freq, late_freq),
    }
    for column in (
        "pseudo_recall",
        "precision_proxy",
        "mean_matched_iou",
        "mean_center_error_norm",
        "detection_count_ratio",
        "infer_frame_pseudo_recall",
        "noninfer_frame_pseudo_recall",
        "noninfer_frame_precision_proxy",
        "lost_object_frame_ratio",
    ):
        summary[column] = number(quality.get(column)) if quality else None
    return summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def short_label(row: dict[str, Any]) -> str:
    return str(row["key"]).replace("int8_", "").replace("proposed_", "prop_")


def sampled_xy(
    rows: list[dict[str, str]],
    column: str,
    *,
    positive: bool = False,
    maximum: int = 1800,
) -> tuple[list[float], list[float]]:
    if not rows:
        return [], []
    timestamps = values(rows, "timestamp")
    origin = timestamps[0] if timestamps else 0.0
    points: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        y = number(row.get(column))
        timestamp = number(row.get("timestamp"))
        if y is None or timestamp is None or (positive and y <= 0):
            continue
        points.append(((timestamp - origin) / 60.0, y))
    stride = max(1, math.ceil(len(points) / maximum))
    selected = points[::stride]
    return [point[0] for point in selected], [point[1] for point in selected]


def annotate(ax: Any, xs: list[float], ys: list[float], labels: list[str]) -> None:
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)


def make_plots(
    run_data: list[tuple[dict[str, Any], list[dict[str, str]]]],
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    labels = [short_label(row) for row in summaries]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for summary, rows in run_data:
        x, y = sampled_xy(rows, "temp_c")
        axes[0].plot(x, y, linewidth=1.2, label=short_label(summary))
        x, y = sampled_xy(rows, "arm_clock_mhz")
        axes[1].plot(x, y, linewidth=1.0, label=short_label(summary))
    axes[0].set(title="Temperature vs time", xlabel="Formal-run time (min)", ylabel="CPU temperature (°C)")
    axes[1].set(title="ARM clock vs time", xlabel="Formal-run time (min)", ylabel="Clock (MHz)")
    axes[0].legend(ncol=2, fontsize=7)
    axes[1].legend(ncol=2, fontsize=7)
    fig.savefig(output_dir / "01_temperature_vs_time.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for summary, rows in run_data:
        x, y = sampled_xy(rows, "latency_ms", positive=True)
        ax.plot(x, y, ".", markersize=2, alpha=0.65, label=short_label(summary))
    ax.set(title="Detector-frame latency vs time", xlabel="Formal-run time (min)", ylabel="RT-DETR latency (ms)")
    ax.legend(ncol=2, fontsize=7)
    fig.savefig(output_dir / "02_latency_vs_time.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for summary, rows in run_data:
        x, y = sampled_xy(rows, "loop_fps")
        ax.plot(x, y, linewidth=1.0, label=short_label(summary))
    ax.set(title="End-to-end pipeline FPS vs time", xlabel="Formal-run time (min)", ylabel="Pipeline FPS")
    ax.legend(ncol=2, fontsize=7)
    fig.savefig(output_dir / "03_fps_vs_time.png")
    plt.close(fig)

    positions = list(range(len(summaries)))
    full = [100 * float(row.get("full_detector_invocation_ratio") or 0) for row in summaries]
    roi = [100 * float(row.get("roi_detector_invocation_ratio") or 0) for row in summaries]
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.bar(positions, full, label="Full-frame detector")
    ax.bar(positions, roi, bottom=full, label="ROI detector")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set(title="Detector invocation ratio", ylabel="Invocations / processed frames (%)")
    ax.legend()
    fig.savefig(output_dir / "04_detector_invocation_ratio.png")
    plt.close(fig)

    quality_rows = [row for row in summaries if row.get("pseudo_recall") is not None]
    xs = [float(row.get("latency_ms_mean_detector_frames") or 0) for row in quality_rows]
    ys = [float(row.get("pseudo_recall") or 0) for row in quality_rows]
    colors = [float(row.get("loop_fps_mean") or 0) for row in quality_rows]
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", s=75)
    annotate(ax, xs, ys, [short_label(row) for row in quality_rows])
    ax.set(title="Latency-accuracy trade-off", xlabel="Mean detector-frame latency (ms)", ylabel="Pseudo recall vs native FP32")
    fig.colorbar(scatter, ax=ax, label="Pipeline FPS")
    fig.savefig(output_dir / "05_latency_accuracy_tradeoff.png")
    plt.close(fig)

    xs = [float(row.get("temp_c_max") or 0) for row in summaries]
    ys = [float(row.get("loop_fps_mean") or 0) for row in summaries]
    colors = [float(row.get("detector_invocation_ratio") or 0) for row in summaries]
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    scatter = ax.scatter(xs, ys, c=colors, cmap="plasma_r", s=80)
    annotate(ax, xs, ys, labels)
    ax.set(title="Temperature-performance trade-off", xlabel="Maximum CPU temperature (°C)", ylabel="Mean pipeline FPS")
    fig.colorbar(scatter, ax=ax, label="Detector invocation ratio")
    fig.savefig(output_dir / "06_temperature_performance_tradeoff.png")
    plt.close(fig)

    fps = [float(row.get("loop_fps_mean") or 0) for row in summaries]
    fps_scale = max(fps) or 1.0
    normalized_fps = [value / fps_scale for value in fps]
    recall = [float(row.get("pseudo_recall") or 0) for row in summaries]
    detector_saving = [1 - float(row.get("detector_invocation_ratio") or 0) for row in summaries]
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.bar([p - width for p in positions], normalized_fps, width, label="Normalized pipeline FPS")
    ax.bar(positions, recall, width, label="Pseudo recall")
    ax.bar([p + width for p in positions], detector_saving, width, label="Detector saving ratio")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set(title="Defense ablation summary", ylabel="Normalized score / ratio")
    ax.legend()
    fig.savefig(output_dir / "07_ablation_bar_chart.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing suite manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = (args.output_dir or suite_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_rows = read_csv(output_dir / "quality_summary.csv")
    quality_by_run = {row.get("student", ""): row for row in quality_rows}
    summaries: list[dict[str, Any]] = []
    run_data: list[tuple[dict[str, Any], list[dict[str, str]]]] = []
    for run_meta in manifest.get("runs") or []:
        order = int(run_meta["order"])
        run_dir = suite_dir / f"{order:02d}_{run_meta['key']}"
        summary = summarize_run(run_dir, run_meta, quality_by_run.get(run_dir.name))
        summaries.append(summary)
        run_data.append((summary, read_csv(run_dir / "runtime.csv")))

    if not summaries:
        raise RuntimeError("No completed runs are listed in manifest.json")
    write_rows(output_dir / "defense_summary.csv", summaries)
    make_plots(run_data, summaries, output_dir)
    print(f"Saved defense summary: {output_dir / 'defense_summary.csv'}")
    for index in range(1, 8):
        matches = list(output_dir.glob(f"{index:02d}_*.png"))
        if matches:
            print(f"Saved plot: {matches[0]}")


if __name__ == "__main__":
    main()
