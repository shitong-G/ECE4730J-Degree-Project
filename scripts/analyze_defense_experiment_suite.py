#!/usr/bin/env python3
"""Create defense tables, core plots, and query-budget diagnostics."""

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
    inference_rows = [
        row for row in rows if boolean(row.get("did_infer")) is True
    ]
    applied_query_budgets = values(inference_rows, "query_budget_applied")
    query_modes = sorted(
        {
            row.get("query_budget_mode", "")
            for row in inference_rows
            if row.get("query_budget_mode")
        }
    )
    query_sources = sorted(
        {
            row.get("query_budget_source", "")
            for row in inference_rows
            if row.get("query_budget_source")
        }
    )
    query_states = sorted(
        {
            row.get("query_budget_temperature_state", "")
            for row in inference_rows
            if row.get("query_budget_temperature_state")
        }
    )
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
        "onnx_run_ms_mean": average(values(inference_rows, "onnx_run_ms", positive=True)),
        "temp_c_mean": average(temperatures),
        "temp_c_start": temperatures[0] if temperatures else None,
        "temp_c_max": max(temperatures) if temperatures else None,
        "temp_c_increase": (
            max(temperatures) - temperatures[0] if temperatures else None
        ),
        "arm_clock_mhz_mean": average(values(rows, "arm_clock_mhz")),
        "power_w_mean": average(values(rows, "power_w")),
        "tracking_frame_ratio": len(tracking_rows) / frame_count if frame_count else None,
        "tracking_failure_ratio_mean": average(values(tracking_rows, "tracking_failure_ratio")),
        "tracking_mean_quality": average(values(tracking_rows, "tracking_mean_quality")),
        "query_budget_applied_mean": average(applied_query_budgets),
        "query_budget_applied_min": min(applied_query_budgets) if applied_query_budgets else None,
        "query_budget_applied_max": max(applied_query_budgets) if applied_query_budgets else None,
        "query_budget_modes": ",".join(query_modes),
        "query_budget_sources": ",".join(query_sources),
        "query_budget_temperature_states": ",".join(query_states),
        "query_output_count_mean": average(
            values(inference_rows, "query_output_count", positive=True)
        ),
        "query_budget_ratio_mean": average(
            values(inference_rows, "query_budget_ratio", positive=True)
        ),
        "graph_query_budget_ratio": (
            sum(
                row.get("query_budget_mode") == "graph_input"
                for row in inference_rows
            )
            / len(inference_rows)
            if inference_rows
            else None
        ),
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

    query_rows = [
        row
        for row in summaries
        if row.get("query_budget_applied_mean") is not None
        and float(row.get("graph_query_budget_ratio") or 0) > 0
    ]
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    if query_rows:
        query_x = [
            float(row["query_budget_applied_mean"]) for row in query_rows
        ]
        query_latency = [
            float(row.get("latency_ms_mean_detector_frames") or 0)
            for row in query_rows
        ]
        query_recall = [
            float(row.get("pseudo_recall") or 0) for row in query_rows
        ]
        ax.scatter(query_x, query_latency, color="#4e79a7", s=75)
        annotate(ax, query_x, query_latency, [short_label(row) for row in query_rows])
        quality_ax = ax.twinx()
        quality_ax.scatter(
            query_x,
            query_recall,
            color="#e15759",
            marker="s",
            s=65,
            label="Pseudo recall",
        )
        quality_ax.set_ylabel("Pseudo recall vs native FP32")
        quality_ax.set_ylim(0, 1.05)
    else:
        ax.text(
            0.5,
            0.5,
            "No graph-query conditions in this suite",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set(
        title="Graph query-budget trade-off",
        xlabel="Mean graph-applied query budget",
        ylabel="Mean detector-frame latency (ms)",
    )
    fig.savefig(output_dir / "08_query_budget_tradeoff.png")
    plt.close(fig)

    # A timeline is intentionally generated from the adaptive condition, when
    # available.  It makes the runtime allocation observable rather than only
    # reporting one aggregate mean in the summary table.
    timeline_candidates = [
        (summary, rows)
        for summary, rows in run_data
        if any(
            str(row.get("query_budget_source", "")) == "temperature"
            and number(row.get("query_budget_applied")) is not None
            for row in rows
        )
    ]
    if not timeline_candidates:
        timeline_candidates = [
            (summary, rows)
            for summary, rows in run_data
            if any(number(row.get("query_budget_applied")) is not None for row in rows)
        ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    if timeline_candidates:
        summary, rows = timeline_candidates[0]
        timestamps = values(rows, "timestamp")
        origin = timestamps[0] if timestamps else 0.0
        points = []
        for row in rows:
            timestamp = number(row.get("timestamp"))
            budget = number(row.get("query_budget_applied"))
            if timestamp is None or budget is None:
                continue
            points.append(
                (
                    (timestamp - origin) / 60.0,
                    budget,
                    number(row.get("temp_c")),
                    str(row.get("query_budget_temperature_state") or "unknown"),
                    boolean(row.get("did_infer")) is True,
                )
            )
        if points:
            x = [item[0] for item in points]
            budget = [item[1] for item in points]
            temp = [item[2] for item in points if item[2] is not None]
            temp_x = [item[0] for item in points if item[2] is not None]
            invoked = [item[4] for item in points]
            axes[0].step(x, budget, where="post", linewidth=1.3, label="Applied Q")
            axes[0].scatter(
                [a for a, active in zip(x, invoked) if active],
                [b for b, active in zip(budget, invoked) if active],
                s=8,
                label="Detector invocation",
            )
            axes[0].set_ylabel("Query budget")
            axes[0].set_title(f"Query-budget timeline: {short_label(summary)}")
            axes[0].legend(fontsize=8)
            if temp:
                twin = axes[1].twinx()
                twin.plot(temp_x, temp, color="#e15759", linewidth=1.0, label="Temperature")
                twin.set_ylabel("Temperature (°C)")
            states = {"normal": 0, "warm": 1, "hot": 2, "critical": 3, "unknown": -1}
            state_y = [states.get(item[3], -1) for item in points]
            axes[1].step(x, state_y, where="post", color="#59a14f", linewidth=1.1)
            axes[1].set_yticks([0, 1, 2, 3], ["normal", "warm", "hot", "critical"])
            axes[1].set_ylabel("Thermal state")
        else:
            axes[0].text(0.5, 0.5, "No applied query-budget rows", ha="center", transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, "No graph-query condition in this suite", ha="center", transform=axes[0].transAxes)
    axes[1].set_xlabel("Formal-run time (min)")
    fig.savefig(output_dir / "query_budget_timeline.png")
    plt.close(fig)


def write_query_budget_summary(
    summaries: list[dict[str, Any]], output_dir: Path
) -> None:
    """Write the defense-facing query-budget comparison table."""
    rows: list[dict[str, Any]] = []
    for item in summaries:
        if item.get("query_budget_applied_mean") is None:
            continue
        key = str(item.get("key", ""))
        if "qthermal" in key or "temperature" in str(item.get("query_budget_sources", "")):
            budget_label = "Adaptive (64/48/40/32)"
        elif "q64" in key:
            budget_label = "Q=64"
        elif "q300" in key or item.get("query_budget_applied_mean") == 300:
            budget_label = "Q=300"
        else:
            budget_label = f"Q={item['query_budget_applied_mean']:.0f}"
        rows.append(
            {
                "Method": item.get("title", key),
                "Query budget": budget_label,
                "Average latency": item.get("latency_ms_mean_detector_frames"),
                "Average FPS": item.get("loop_fps_mean"),
                "Temperature increase": item.get("temp_c_increase"),
                "Pseudo recall": item.get("pseudo_recall"),
                "IoU": item.get("mean_matched_iou"),
                "Detector invocation ratio": item.get("detector_invocation_ratio"),
                "Query budget ratio": item.get("query_budget_ratio_mean"),
                "Query mode": item.get("query_budget_modes"),
                "Query execution cost": item.get("onnx_run_ms_mean"),
            }
        )
    path = output_dir / "query_budget_summary.csv"
    if rows:
        write_rows(path, rows)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "Method",
                    "Query budget",
                    "Average latency",
                    "Average FPS",
                    "Temperature increase",
                    "Pseudo recall",
                    "IoU",
                    "Detector invocation ratio",
                    "Query budget ratio",
                    "Query mode",
                    "Query execution cost",
                ]
            )


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
    write_query_budget_summary(summaries, output_dir)
    make_plots(run_data, summaries, output_dir)
    print(f"Saved defense summary: {output_dir / 'defense_summary.csv'}")
    for index in range(1, 9):
        matches = list(output_dir.glob(f"{index:02d}_*.png"))
        if matches:
            print(f"Saved plot: {matches[0]}")
    for name in ("query_budget_timeline.png", "query_budget_summary.csv"):
        path = output_dir / name
        if path.exists():
            print(f"Saved query output: {path}")


if __name__ == "__main__":
    main()
