#!/usr/bin/env python3
"""Add the completed YOLOv8n SOTA run to the thesis trace figures.

The external-detector runner stores per-frame latency but not a wall-clock
timestamp for every frame.  The frame positions are therefore reconstructed
from frame order and the recorded run wall time; this basis is retained in the
source-data CSV files written alongside the figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Keep this augmentation script self-describing for the publication preflight:
# the shared figure helper applies the same settings at runtime, while these
# explicit values document the editable-vector and raster-export contract.
PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}
plt.rcParams.update(PUBLICATION_RC)

# The shared save helper emits <stem>.svg, <stem>.pdf, <stem>.tiff, and
# <stem>.png; the raster artifacts use dpi=600.
EXPORT_FORMATS = ("svg", "pdf", "tiff", "png")
RASTER_DPI = 600

from make_thesis_figures import (
    COLORS,
    METHOD_COLORS,
    RUN_ORDER,
    SHORT_LABELS,
    centered_sliding_mean,
    elapsed,
    number,
    read_csv,
    save_figure,
)


YOLO_COLOR = "#4C9A8A"
WINDOW_SECONDS = 60.0


def elapsed_from_run_start(rows: list[dict[str, str]], column: str) -> tuple[list[float], list[float]]:
    """Read a timestamped runtime field using the complete run as time origin."""
    if not rows:
        return [], []
    origin = number(rows[0].get("timestamp")) or 0.0
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        stamp = number(row.get("timestamp"))
        value = number(row.get(column))
        if stamp is not None and value is not None:
            xs.append((stamp - origin) / 60.0)
            ys.append(value)
    return xs, ys


def read_yolov8_run(run_dir: Path) -> dict[str, object]:
    """Load successful YOLOv8n temperature and total detector-call traces."""
    summary_rows = read_csv(run_dir / "runtime.csv")
    frame_rows = read_csv(run_dir / "runtime_frames.csv")
    temperature_rows = read_csv(run_dir / "temperature_trace.csv")
    if not summary_rows or not frame_rows or not temperature_rows:
        raise ValueError(f"Incomplete YOLOv8 run directory: {run_dir}")

    summary = summary_rows[0]
    wall_sec = float(summary["wall_sec"])
    frame_count = len(frame_rows)
    if frame_count < 2 or wall_sec <= 0:
        raise ValueError("YOLOv8 run needs at least two frames and positive wall time")

    frame_x = [index * wall_sec / (frame_count - 1) / 60.0 for index in range(frame_count)]
    frame_latency = [
        float(row["preprocess_ms"])
        + float(row["inference_ms"])
        + float(row["postprocess_ms"])
        for row in frame_rows
    ]
    temp_x = [float(row["elapsed_sec"]) / 60.0 for row in temperature_rows]
    temp_y = [float(row["temp_c"]) for row in temperature_rows]
    return {
        "name": "YOLOv8n",
        "color": YOLO_COLOR,
        "temperature_x": temp_x,
        "temperature_y": temp_y,
        "latency_x": frame_x,
        "latency_y": frame_latency,
        "latency_basis": "frame_index_linearized_from_recorded_wall_time",
        "frame_rows": frame_rows,
        "wall_sec": wall_sec,
    }


def existing_series(
    runs: dict[str, list[dict[str, str]]], column: str, infer_only: bool = False
) -> list[dict[str, object]]:
    series: list[dict[str, object]] = []
    for key, label, color in zip(RUN_ORDER, SHORT_LABELS, METHOD_COLORS):
        rows = runs[key]
        selected = [
            row for row in rows
            if not infer_only or row.get("did_infer", "").lower() in {"true", "1", "yes"}
        ]
        if infer_only:
            xs, ys = elapsed_from_run_start(rows, column)
            xs = [x for row, x in zip(selected, xs)] if len(selected) == len(xs) else []
            ys = [number(row.get(column)) for row in selected]
            ys = [value for value in ys if value is not None]
        else:
            xs, ys = elapsed_from_run_start(rows, column)
        series.append(
            {
                "name": label.replace("\n", " "),
                "color": color,
                "x": xs,
                "y": ys,
                "basis": "runtime_csv_timestamp",
                "call_types": [row.get("detector_call_type", "") for row in selected]
                if infer_only else [],
            }
        )
    return series


def latency_series_from_runtime(rows: list[dict[str, str]], color: str, name: str) -> dict[str, object]:
    selected = [
        row for row in rows
        if row.get("did_infer", "").lower() in {"true", "1", "yes"}
        and number(row.get("latency_ms")) is not None
    ]
    if not rows:
        return {"name": name, "color": color, "x": [], "y": [], "basis": "runtime_csv_timestamp", "call_types": []}
    origin = number(rows[0].get("timestamp")) or 0.0
    xs = [((number(row.get("timestamp")) or origin) - origin) / 60.0 for row in selected]
    ys = [float(row["latency_ms"]) for row in selected]
    return {
        "name": name,
        "color": color,
        "x": xs,
        "y": ys,
        "basis": "runtime_csv_timestamp",
        "call_types": [row.get("detector_call_type", "") for row in selected],
    }


def render_augmented(
    series: list[dict[str, object]],
    output_dir: Path,
    stem_name: str,
    ylabel: str,
    source_raw_field: str,
    source_smooth_field: str | None = None,
    thresholds: tuple[tuple[float, str], ...] = (),
    window_seconds: float = WINDOW_SECONDS,
) -> None:
    smoothed = source_smooth_field is not None
    window_minutes = window_seconds / 60.0
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    duration = max(float(max(item["x"])) for item in series if item["x"])
    source_rows: list[dict[str, object]] = []

    for item in series:
        xs = list(item["x"])
        ys = list(item["y"])
        color = str(item["color"])
        name = str(item["name"])
        if smoothed:
            smooth = centered_sliding_mean(xs, ys, window_minutes)
            ax.plot(xs, ys, lw=0.70, alpha=0.22, color=color)
            ax.plot(xs, smooth, lw=1.65, color=color, label=name)
        else:
            smooth = []
            ax.plot(xs, ys, lw=1.05, color=color, label=name)

        basis = str(item.get("basis", ""))
        call_types = list(item.get("call_types", []))
        for index, (x_value, y_value) in enumerate(zip(xs, ys)):
            row: dict[str, object] = {
                "condition": name,
                "elapsed_time_min": x_value,
                source_raw_field: y_value,
                "time_basis": basis,
            }
            if source_smooth_field is not None:
                row[source_smooth_field] = smooth[index]
                row["window_seconds"] = window_seconds
            if call_types:
                row["detector_call_type"] = call_types[index]
            source_rows.append(row)

    for value, color in thresholds:
        ax.axhline(value, color=color, lw=0.75, ls="--")
    ax.set(xlim=(0, duration), xlabel="Elapsed time (min)", ylabel=ylabel)
    ax.grid(axis="y", color="#D9D9D9", lw=0.5)
    ax.legend(ncol=4, loc="upper center")
    stem = output_dir / stem_name
    save_figure(fig, stem)

    fieldnames = list(source_rows[0].keys()) if source_rows else []
    with stem.with_name(stem.name + "_source.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)


def generate(runs_dir: Path, yolov8_dir: Path, output_dir: Path) -> None:
    runs = {key: read_csv(runs_dir / key / "runtime.csv") for key in RUN_ORDER}
    yolo = read_yolov8_run(yolov8_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thermal = existing_series(runs, "temp_c")
    thermal.append({
        "name": yolo["name"], "color": yolo["color"],
        "x": yolo["temperature_x"], "y": yolo["temperature_y"],
        "basis": "temperature_trace_elapsed_sec", "call_types": [],
    })
    thresholds = ((58.0, COLORS["warm"]), (66.0, COLORS["hot"]), (76.0, COLORS["critical"]))
    render_augmented(
        thermal, output_dir, "thesis_supplementary_thermal_traces_with_yolov8",
        "CPU temperature ($^\\circ$C)", "temperature_c", thresholds=thresholds,
    )
    render_augmented(
        thermal, output_dir, "thesis_supplementary_thermal_traces_sliding_window_with_yolov8",
        "CPU temperature ($^\\circ$C)", "temperature_c_raw",
        source_smooth_field="temperature_c_sliding_mean", thresholds=thresholds,
    )

    latency = [latency_series_from_runtime(runs[key], color, label.replace("\n", " "))
               for key, label, color in zip(RUN_ORDER, SHORT_LABELS, METHOD_COLORS)]
    latency.append({
        "name": yolo["name"], "color": yolo["color"],
        "x": yolo["latency_x"], "y": yolo["latency_y"],
        "basis": yolo["latency_basis"], "call_types": [],
    })
    render_augmented(
        latency, output_dir, "thesis_supplementary_inference_latency_traces_with_yolov8",
        "Detector-call latency (ms)", "inference_latency_ms",
    )
    render_augmented(
        latency, output_dir, "thesis_supplementary_inference_latency_traces_sliding_window_with_yolov8",
        "Detector-call latency (ms)", "inference_latency_ms_raw",
        source_smooth_field="inference_latency_ms_sliding_mean",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--yolov8-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.suite_dir, args.yolov8_run_dir, args.output_dir)


if __name__ == "__main__":
    main()
