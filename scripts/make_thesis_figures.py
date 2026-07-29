#!/usr/bin/env python3
"""Generate figures for the three-condition thesis comparison.

Sustained runtime and thermal results are read from one complete run of Native
FP32, Quantized Native and Ours.  Detection quality may come from a separate
manually annotated sequence, so its metrics file is passed explicitly.  The
figures are descriptive and deliberately omit confidence intervals and
hypothesis-test annotations.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean, median
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_ORDER = [
    "r01_01_fp32_native",
    "r01_02_int8_dynamic_q300_native",
    "r01_06_proposed_software",
]
SHORT_LABELS = [
    "Native\nFP32",
    "Quantized\nNative",
    "Ours",
]
COLORS = {
    "baseline": "#484878",
    "quantized": "#7884B4",
    "proposed": "#B85C78",
    "neutral": "#767676",
    "warm": "#C17C00",
    "hot": "#B4471A",
    "critical": "#8B1E1E",
}
METHOD_COLORS = [
    COLORS["baseline"],
    COLORS["quantized"],
    COLORS["proposed"],
]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    try:
        parsed = float(value or "")
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def truth(value: str | None) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def values(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    output: list[float] = []
    for row in rows:
        value = number(row.get(column))
        if value is not None:
            output.append(value)
    return output


def percentile(data: list[float], fraction: float) -> float:
    ordered = sorted(data)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def elapsed(rows: list[dict[str, str]], column: str) -> tuple[list[float], list[float]]:
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


def run_duration_minutes(rows: list[dict[str, str]]) -> float:
    stamps = values(rows, "timestamp")
    return (stamps[-1] - stamps[0]) / 60.0


def wall_clock_fps(rows: list[dict[str, str]]) -> float:
    stamps = values(rows, "timestamp")
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        raise ValueError("A run needs at least two increasing timestamps")
    return (len(rows) - 1) / (stamps[-1] - stamps[0])


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.17,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
    )


def make_motivation_figure(
    runs: dict[str, list[dict[str, str]]], output_dir: Path
) -> None:
    """Show why cold-start latency alone is not a sufficient deployment metric."""
    fig, (ax_temp, ax_latency) = plt.subplots(
        1, 2, figsize=(7.2, 2.35), gridspec_kw={"wspace": 0.30}
    )
    selected = RUN_ORDER[:2]
    selected_labels = ["Native FP32", "Quantized Native"]
    selected_colors = [COLORS["baseline"], COLORS["quantized"]]
    duration = max(run_duration_minutes(runs[key]) for key in selected)

    for key, label, color in zip(selected, selected_labels, selected_colors):
        xs, ys = elapsed(runs[key], "temp_c")
        ax_temp.plot(xs, ys, color=color, lw=1.25, label=label)
        infer_rows = [row for row in runs[key] if truth(row.get("did_infer"))]
        lx, ly = elapsed(infer_rows, "latency_ms")
        ax_latency.plot(lx, ly, color=color, lw=1.0, alpha=0.9, label=label)

    for value, color in [
        (58, COLORS["warm"]),
        (66, COLORS["hot"]),
        (76, COLORS["critical"]),
    ]:
        ax_temp.axhline(value, color=color, lw=0.75, ls="--")

    ax_temp.set(
        xlim=(0, duration),
        xlabel="Elapsed time (min)",
        ylabel="CPU temperature ($^\\circ$C)",
    )
    ax_latency.set(
        xlim=(0, duration),
        xlabel="Elapsed time (min)",
        ylabel="Detector latency (ms)",
    )
    for ax in (ax_temp, ax_latency):
        ax.grid(axis="y", color="#D9D9D9", lw=0.5)
        ax.legend(loc="best")
    add_panel_label(ax_temp, "a")
    add_panel_label(ax_latency, "b")
    save_figure(fig, output_dir / "thesis_motivation")


def summarize_run(rows: list[dict[str, str]]) -> dict[str, float]:
    infer = [row for row in rows if truth(row.get("did_infer"))]
    full = [row for row in infer if row.get("detector_call_type") == "full"]
    roi = [row for row in infer if row.get("detector_call_type") == "roi"]
    temperatures = values(rows, "temp_c")
    infer_latency = values(infer, "latency_ms")
    return {
        "mean_temp": fmean(temperatures),
        "max_temp": max(temperatures),
        "detector_rate": 100.0 * len(infer) / len(rows),
        "full_rate": 100.0 * len(full) / len(rows),
        "roi_rate": 100.0 * len(roi) / len(rows),
        "loop_fps": wall_clock_fps(rows),
        "median_infer_ms": median(infer_latency),
        "p95_infer_ms": percentile(infer_latency, 0.95),
    }


def make_runtime_figure(
    runs: dict[str, list[dict[str, str]]], output_dir: Path
) -> None:
    summaries = [summarize_run(runs[key]) for key in RUN_ORDER]
    x = np.arange(len(RUN_ORDER))
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 4.35), gridspec_kw={"hspace": 0.43, "wspace": 0.30}
    )
    ax_temp, ax_calls, ax_latency, ax_fps = axes.flat

    mean_temp = np.asarray([item["mean_temp"] for item in summaries])
    max_temp = np.asarray([item["max_temp"] for item in summaries])
    ax_temp.bar(x - 0.18, mean_temp, 0.36, color=METHOD_COLORS, alpha=0.90, label="Mean")
    ax_temp.bar(
        x + 0.18,
        max_temp,
        0.36,
        facecolor="white",
        edgecolor=METHOD_COLORS,
        linewidth=1.0,
        label="Maximum",
    )
    ax_temp.axhline(76, color=COLORS["critical"], lw=0.75, ls="--")
    ax_temp.set(ylabel="CPU temperature ($^\\circ$C)")
    ax_temp.legend(loc="lower left", ncol=2)

    full_rate = np.asarray([item["full_rate"] for item in summaries])
    roi_rate = np.asarray([item["roi_rate"] for item in summaries])
    ax_calls.bar(x, full_rate, color=METHOD_COLORS, label="Full-frame")
    ax_calls.bar(
        x,
        roi_rate,
        bottom=full_rate,
        color=METHOD_COLORS,
        alpha=0.42,
        hatch="///",
        label="ROI",
    )
    ax_calls.set(ylabel="Detector invocation rate (%)")
    ax_calls.legend(loc="upper right")

    med_latency = np.asarray([item["median_infer_ms"] for item in summaries])
    p95_latency = np.asarray([item["p95_infer_ms"] for item in summaries])
    ax_latency.bar(x, med_latency, color=METHOD_COLORS, label="Median")
    ax_latency.vlines(x, med_latency, p95_latency, color="#333333", lw=1.0)
    ax_latency.scatter(x, p95_latency, s=11, color="#333333", zorder=3, label="95th percentile")
    ax_latency.set(ylabel="Detector-call latency (ms)")
    ax_latency.legend(loc="upper right")

    loop_fps = np.asarray([item["loop_fps"] for item in summaries])
    ax_fps.bar(x, loop_fps, color=METHOD_COLORS)
    ax_fps.set(ylabel="Wall-clock rate (frames s$^{-1}$)")

    for label, ax in zip("abcd", axes.flat):
        ax.set_xticks(x, SHORT_LABELS)
        ax.grid(axis="y", color="#D9D9D9", lw=0.5)
        add_panel_label(ax, label)
    save_figure(fig, output_dir / "thesis_runtime_evaluation")


def make_quality_figure(manual_metrics: Path, output_dir: Path) -> None:
    rows = read_csv(manual_metrics)
    by_key = {row["run"]: row for row in rows}
    ordered = [by_key[key] for key in RUN_ORDER]
    x = np.arange(len(ordered))
    width = 0.24
    precision = 100 * np.asarray([float(row["precision"]) for row in ordered])
    recall = 100 * np.asarray([float(row["recall"]) for row in ordered])
    f1 = 100 * np.asarray([float(row["f1"]) for row in ordered])
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.bar(x - width, precision, width, color="#7884B4", label="Precision")
    ax.bar(x, recall, width, color="#D99A52", label="Recall")
    ax.bar(x + width, f1, width, color="#6AA27A", label="F1")
    ax.set(ylim=(0, 105), ylabel="Detection score (%)")
    ax.set_xticks(x, SHORT_LABELS)
    ax.grid(axis="y", color="#D9D9D9", lw=0.5)
    ax.legend(ncol=3, loc="upper right")
    save_figure(fig, output_dir / "thesis_quality_evaluation")


def make_thermal_trace_figure(
    runs: dict[str, list[dict[str, str]]], output_dir: Path
) -> None:
    """Expose full within-run thermal trajectories instead of only aggregates."""
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    duration = max(run_duration_minutes(runs[key]) for key in RUN_ORDER)
    for key, label, color in zip(RUN_ORDER, SHORT_LABELS, METHOD_COLORS):
        xs, ys = elapsed(runs[key], "temp_c")
        ax.plot(xs, ys, lw=1.05, color=color, label=label.replace("\n", " "))
    for value, color in [
        (58, COLORS["warm"]),
        (66, COLORS["hot"]),
        (76, COLORS["critical"]),
    ]:
        ax.axhline(value, color=color, lw=0.75, ls="--")
    ax.set(
        xlim=(0, duration),
        xlabel="Elapsed time (min)",
        ylabel="CPU temperature ($^\\circ$C)",
    )
    ax.grid(axis="y", color="#D9D9D9", lw=0.5)
    ax.legend(ncol=3, loc="upper center")
    save_figure(fig, output_dir / "thesis_supplementary_thermal_traces")


def make_controller_coverage_figure(
    formal_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    """Show which controller actions were exercised inside the formal Ours run."""
    fig, ax_temp = plt.subplots(1, 1, figsize=(3.6, 2.35))
    tx, temperatures = elapsed(formal_rows, "temp_c")
    ax_temp.plot(tx, temperatures, color=COLORS["proposed"], lw=1.15)
    for value, color in [
        (58, COLORS["warm"]),
        (66, COLORS["hot"]),
        (76, COLORS["critical"]),
    ]:
        ax_temp.axhline(value, color=color, lw=0.70, ls="--")
    ax_temp.set(
        xlabel="Elapsed time (min)",
        ylabel="CPU temperature ($^\\circ$C)",
    )
    ax_temp.grid(axis="y", color="#D9D9D9", lw=0.5)
    add_panel_label(ax_temp, "a")
    save_figure(fig, output_dir / "thesis_controller_coverage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-metrics", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path)
    args = parser.parse_args()

    runs = {
        key: read_csv(args.suite_dir / key / "runtime.csv")
        for key in RUN_ORDER
    }
    make_motivation_figure(runs, args.output_dir)
    make_runtime_figure(runs, args.output_dir)
    make_quality_figure(args.manual_metrics, args.output_dir)
    make_thermal_trace_figure(runs, args.output_dir)
    if args.activation_dir is not None:
        make_controller_coverage_figure(
            runs["r01_06_proposed_software"],
            args.output_dir,
        )


if __name__ == "__main__":
    main()
