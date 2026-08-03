#!/usr/bin/env python3
"""Generate ARM-clock traces for the formal three-condition experiment suite.

The plotted measurement is the runtime logger's ``arm_clock_mhz`` field.  The
raw trace is retained in full, including short thermal-throttling excursions;
the companion figure overlays a centered one-minute time-based mean.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from make_thesis_figures import (
    METHOD_COLORS,
    RUN_ORDER,
    SHORT_LABELS,
    centered_sliding_mean,
    elapsed,
    read_csv,
    run_duration_minutes,
    save_figure,
)


# Explicitly document the publication contract used by the shared figure
# helper: editable vector text and a 600-dpi submission raster.
PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 8.0,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}
mpl.rcParams.update(PUBLICATION_RC)

# save_figure emits <stem>.svg, <stem>.pdf, <stem>.png, and <stem>.tiff;
# the TIFF export uses dpi=600.
EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
RASTER_DPI = 600
WINDOW_SECONDS = 60.0


def render_arm_clock(
    runs: dict[str, list[dict[str, str]]],
    output_dir: Path,
    sliding_window: bool,
    window_seconds: float = WINDOW_SECONDS,
) -> None:
    window_minutes = window_seconds / 60.0
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    duration = max(run_duration_minutes(runs[key]) for key in RUN_ORDER)
    source_rows: list[dict[str, str | float]] = []

    for key, label, color in zip(RUN_ORDER, SHORT_LABELS, METHOD_COLORS):
        xs, ys = elapsed(runs[key], "arm_clock_mhz")
        display_label = label.replace("\n", " ")
        if sliding_window:
            smoothed = centered_sliding_mean(xs, ys, window_minutes)
            ax.plot(xs, ys, lw=0.70, alpha=0.22, color=color)
            ax.plot(xs, smoothed, lw=1.65, color=color, label=display_label)
            source_rows.extend(
                {
                    "condition": display_label,
                    "elapsed_time_min": elapsed_min,
                    "arm_clock_mhz_raw": raw,
                    "arm_clock_mhz_sliding_mean": smooth,
                    "window_seconds": window_seconds,
                }
                for elapsed_min, raw, smooth in zip(xs, ys, smoothed)
            )
        else:
            ax.plot(xs, ys, lw=1.05, color=color, label=display_label)
            source_rows.extend(
                {
                    "condition": display_label,
                    "elapsed_time_min": elapsed_min,
                    "arm_clock_mhz": value,
                }
                for elapsed_min, value in zip(xs, ys)
            )

    ax.set(
        xlim=(0, duration),
        xlabel="Elapsed time (min)",
        ylabel="ARM clock (MHz)",
    )
    ax.grid(axis="y", color="#D9D9D9", lw=0.5)
    ax.legend(ncol=3, loc="upper center")

    stem_name = (
        "thesis_supplementary_arm_clock_traces_sliding_window"
        if sliding_window
        else "thesis_supplementary_arm_clock_traces"
    )
    stem = output_dir / stem_name
    save_figure(fig, stem)

    if sliding_window:
        fieldnames = [
            "condition",
            "elapsed_time_min",
            "arm_clock_mhz_raw",
            "arm_clock_mhz_sliding_mean",
            "window_seconds",
        ]
    else:
        fieldnames = ["condition", "elapsed_time_min", "arm_clock_mhz"]
    with stem.with_name(stem.name + "_source.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        key: read_csv(args.suite_dir / key / "runtime.csv")
        for key in RUN_ORDER
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_arm_clock(runs, args.output_dir, sliding_window=False)
    render_arm_clock(runs, args.output_dir, sliding_window=True)


if __name__ == "__main__":
    main()
