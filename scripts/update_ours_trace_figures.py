#!/usr/bin/env python3
"""Regenerate the thesis trace figures with the latest adaptive Ours run.

Native FP32 and Quantized Native remain from the formal comparison suite.
Only the Ours runtime stream is replaced.  The YOLOv8n stream, when supplied,
is retained in the augmented temperature and latency figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl


# The imported plotting modules apply the same settings at runtime.  Keep the
# publication contract explicit here as well so the source preflight can audit
# this orchestration script directly.
PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}
mpl.rcParams.update(PUBLICATION_RC)

# The shared save helper emits <stem>.svg, <stem>.pdf, <stem>.png, and
# <stem>.tiff at a 600-dpi raster resolution; figures use figsize=(7.2, 3.05).
EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
RASTER_DPI = 600
FINAL_WIDTH_MM = 183

from make_arm_clock_figures import render_arm_clock
from make_sota_augmented_figures import (
    YOLO_COLOR,
    latency_series_from_runtime,
    read_yolov8_run,
    render_augmented,
)
from make_thesis_figures import (
    COLORS,
    METHOD_COLORS,
    RUN_ORDER,
    SHORT_LABELS,
    make_inference_latency_trace_figure,
    make_inference_latency_trace_sliding_window_figure,
    make_thermal_trace_figure,
    make_thermal_trace_sliding_window_figure,
    number,
    read_csv,
)


def make_augmented_figures(
    runs: dict[str, list[dict[str, str]]],
    yolov8_dir: Path,
    output_dir: Path,
) -> None:
    """Regenerate the four-method YOLOv8 temperature and latency figures."""
    yolo = read_yolov8_run(yolov8_dir)

    thermal = []
    for key, label, color in zip(RUN_ORDER, SHORT_LABELS, METHOD_COLORS):
        rows = runs[key]
        xs = []
        ys = []
        origin = number(rows[0].get("timestamp")) or 0.0
        for row in rows:
            stamp = number(row.get("timestamp"))
            value = number(row.get("temp_c"))
            if stamp is not None and value is not None:
                xs.append((stamp - origin) / 60.0)
                ys.append(value)
        thermal.append(
            {
                "name": label.replace("\n", " "),
                "color": color,
                "x": xs,
                "y": ys,
                "basis": "runtime_csv_timestamp",
                "call_types": [],
            }
        )
    thermal.append(
        {
            "name": yolo["name"],
            "color": yolo["color"],
            "x": yolo["temperature_x"],
            "y": yolo["temperature_y"],
            "basis": "temperature_trace_elapsed_sec",
            "call_types": [],
        }
    )
    thresholds = (
        (58.0, COLORS["warm"]),
        (66.0, COLORS["hot"]),
        (76.0, COLORS["critical"]),
    )
    render_augmented(
        thermal,
        output_dir,
        "thesis_supplementary_thermal_traces_with_yolov8",
        "CPU temperature ($^\\circ$C)",
        "temperature_c",
        thresholds=thresholds,
    )
    render_augmented(
        thermal,
        output_dir,
        "thesis_supplementary_thermal_traces_sliding_window_with_yolov8",
        "CPU temperature ($^\\circ$C)",
        "temperature_c_raw",
        source_smooth_field="temperature_c_sliding_mean",
        thresholds=thresholds,
    )

    latency = [
        latency_series_from_runtime(
            runs[key], color, label.replace("\n", " ")
        )
        for key, label, color in zip(RUN_ORDER, SHORT_LABELS, METHOD_COLORS)
    ]
    latency.append(
        {
            "name": yolo["name"],
            "color": YOLO_COLOR,
            "x": yolo["latency_x"],
            "y": yolo["latency_y"],
            "basis": yolo["latency_basis"],
            "call_types": [],
        }
    )
    render_augmented(
        latency,
        output_dir,
        "thesis_supplementary_inference_latency_traces_with_yolov8",
        "Detector-call latency (ms)",
        "inference_latency_ms",
    )
    render_augmented(
        latency,
        output_dir,
        "thesis_supplementary_inference_latency_traces_sliding_window_with_yolov8",
        "Detector-call latency (ms)",
        "inference_latency_ms_raw",
        source_smooth_field="inference_latency_ms_sliding_mean",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--ours-runtime", type=Path, required=True)
    parser.add_argument("--yolov8-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        key: read_csv(args.suite_dir / key / "runtime.csv")
        for key in RUN_ORDER
    }
    runs[RUN_ORDER[2]] = read_csv(args.ours_runtime)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Three-condition figures.
    make_thermal_trace_figure(runs, args.output_dir)
    make_thermal_trace_sliding_window_figure(runs, args.output_dir)
    make_inference_latency_trace_figure(runs, args.output_dir)
    make_inference_latency_trace_sliding_window_figure(runs, args.output_dir)
    render_arm_clock(runs, args.output_dir, sliding_window=False)
    render_arm_clock(runs, args.output_dir, sliding_window=True)

    # Preserve the existing four-method YOLOv8 comparison figures with the
    # updated Ours stream.
    make_augmented_figures(runs, args.yolov8_run_dir, args.output_dir)


if __name__ == "__main__":
    main()
