#!/usr/bin/env python3
"""Generate effective per-frame latency and FPS traces for the thesis.

Effective latency is defined per recorded frame: detector frames use the
detector call latency, while tracking frames use the LK tracking time.  This
keeps the trace aligned with the actual frame-processing path rather than
reporting detector calls alone.
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
    number,
    read_csv,
    run_duration_minutes,
    save_figure,
    truth,
)


# Publication-facing settings are explicit here so that this standalone
# figure generator remains auditable even when the shared plotting helper is
# changed later.  ``save_figure`` emits SVG/PDF/PNG/TIFF; TIFF is 600 dpi.
PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}
mpl.rcParams.update(PUBLICATION_RC)

EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
EXPORT_SUFFIXES = (".svg", ".pdf", ".png", ".tiff")
RASTER_DPI = 600
FINAL_WIDTH_MM = 183


def _effective_trace(rows: list[dict[str, str]]) -> tuple[list[float], list[float], list[str], list[int]]:
    """Return elapsed time, effective latency, event type and frame id."""
    if not rows:
        return [], [], [], []
    origin = number(rows[0].get("timestamp"))
    if origin is None:
        raise ValueError("Runtime log has no valid initial timestamp")

    xs: list[float] = []
    ys: list[float] = []
    event_types: list[str] = []
    frame_ids: list[int] = []
    for index, row in enumerate(rows):
        stamp = number(row.get("timestamp"))
        if stamp is None:
            continue
        detector = truth(row.get("did_infer"))
        detector_latency = number(row.get("latency_ms"))
        tracking_latency = number(row.get("tracking_ms"))
        if detector and detector_latency is not None:
            latency = detector_latency
            event_type = "detector"
        elif tracking_latency is not None:
            latency = tracking_latency
            event_type = "lk_tracking"
        else:
            continue
        xs.append((stamp - origin) / 60.0)
        ys.append(latency)
        event_types.append(event_type)
        frame_ids.append(index)
    return xs, ys, event_types, frame_ids


def _fps_trace(rows: list[dict[str, str]]) -> tuple[list[float], list[float], list[int]]:
    """Return elapsed time, overall loop FPS and frame id for each row."""
    if not rows:
        return [], [], []
    origin = number(rows[0].get("timestamp"))
    if origin is None:
        raise ValueError("Runtime log has no valid initial timestamp")
    xs: list[float] = []
    ys: list[float] = []
    frame_ids: list[int] = []
    for index, row in enumerate(rows):
        stamp = number(row.get("timestamp"))
        fps = number(row.get("loop_fps"))
        if stamp is None or fps is None:
            continue
        xs.append((stamp - origin) / 60.0)
        ys.append(fps)
        frame_ids.append(index)
    return xs, ys, frame_ids


def _write_trace_csv(
    path: Path,
    traces: dict[str, tuple[list[float], list[float], list[str], list[int]]],
    sliding: bool,
    metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if metric == "effective_latency":
            if sliding:
                writer.writerow(
                    [
                        "condition",
                        "elapsed_time_min",
                        "effective_latency_ms_raw",
                        "effective_latency_ms_sliding_mean",
                        "event_type",
                        "frame_id",
                        "window_seconds",
                    ]
                )
            else:
                writer.writerow(
                    ["condition", "elapsed_time_min", "effective_latency_ms", "event_type", "frame_id"]
                )
        else:
            if sliding:
                writer.writerow(
                    [
                        "condition",
                        "elapsed_time_min",
                        "loop_fps_raw",
                        "loop_fps_sliding_mean",
                        "frame_id",
                        "window_seconds",
                    ]
                )
            else:
                writer.writerow(["condition", "elapsed_time_min", "loop_fps", "frame_id"])

        for run_key in RUN_ORDER:
            if run_key not in traces:
                continue
            xs, ys, event_types, frame_ids = traces[run_key]
            condition = SHORT_LABELS[RUN_ORDER.index(run_key)].replace("\n", " ")
            smooth = centered_sliding_mean(xs, ys, window_minutes=1.0) if sliding else []
            for index, (x_value, y_value, frame_id) in enumerate(zip(xs, ys, frame_ids)):
                if metric == "effective_latency":
                    event_type = event_types[index]
                    if sliding:
                        writer.writerow(
                            [condition, f"{x_value:.9f}", f"{y_value:.9f}", f"{smooth[index]:.9f}", event_type, frame_id, 60]
                        )
                    else:
                        writer.writerow([condition, f"{x_value:.9f}", f"{y_value:.9f}", event_type, frame_id])
                elif sliding:
                    writer.writerow(
                        [condition, f"{x_value:.9f}", f"{y_value:.9f}", f"{smooth[index]:.9f}", frame_id, 60]
                    )
                else:
                    writer.writerow([condition, f"{x_value:.9f}", f"{y_value:.9f}", frame_id])


def _render(
    traces: dict[str, tuple[list[float], list[float], list[str], list[int]]],
    output_dir: Path,
    metric: str,
    sliding: bool,
) -> Path:
    is_latency = metric == "effective_latency"
    ylabel = "Effective per-frame latency (ms)" if is_latency else "Loop FPS (frames s$^{-1}$)"
    stem_name = {
        ("effective_latency", False): "thesis_supplementary_effective_per_frame_latency_traces",
        ("effective_latency", True): "thesis_supplementary_effective_per_frame_latency_traces_sliding_window",
        ("fps", False): "thesis_supplementary_loop_fps_traces",
        ("fps", True): "thesis_supplementary_loop_fps_traces_sliding_window",
    }[(metric, sliding)]

    duration = max(
        run_duration_minutes(rows)
        for run_key, rows in traces["_rows"].items()
        if run_key in RUN_ORDER and rows
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    for index, run_key in enumerate(RUN_ORDER):
        xs, ys, event_types, frame_ids = traces[run_key]
        if not xs:
            continue
        color = METHOD_COLORS[index]
        label = SHORT_LABELS[index].replace("\n", " ")
        if sliding:
            ax.plot(xs, ys, color=color, linewidth=0.70, alpha=0.22, zorder=1)
            smooth = centered_sliding_mean(xs, ys, window_minutes=1.0)
            ax.plot(xs, smooth, color=color, linewidth=1.65, label=label, zorder=2)
        else:
            ax.plot(xs, ys, color=color, linewidth=1.05, label=label, zorder=2)

    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, duration)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, handlelength=2.5, columnspacing=1.6)
    fig.tight_layout(pad=0.6)
    output_stem = output_dir / stem_name
    save_figure(fig, output_stem)
    return output_stem


def _load_runs(paths: dict[str, Path], metric: str):
    traces: dict[str, tuple[list[float], list[float], list[str], list[int]]] = {}
    rows_by_run: dict[str, list[dict[str, str]]] = {}
    for run_key in RUN_ORDER:
        rows = read_csv(paths[run_key])
        rows_by_run[run_key] = rows
        if metric == "effective_latency":
            traces[run_key] = _effective_trace(rows)
        else:
            xs, ys, frame_ids = _fps_trace(rows)
            traces[run_key] = (xs, ys, ["" for _ in ys], frame_ids)
    traces["_rows"] = rows_by_run  # type: ignore[assignment]
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-runtime", type=Path, required=True)
    parser.add_argument("--quantized-runtime", type=Path, required=True)
    parser.add_argument("--ours-runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("thesis/figures"))
    args = parser.parse_args()

    paths = {
        RUN_ORDER[0]: args.native_runtime,
        RUN_ORDER[1]: args.quantized_runtime,
        RUN_ORDER[2]: args.ours_runtime,
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key} runtime log: {path}")

    for metric in ("effective_latency", "fps"):
        traces = _load_runs(paths, metric)
        _render(traces, args.output_dir, metric, sliding=False)
        _render(traces, args.output_dir, metric, sliding=True)
        source_stem = "effective_per_frame_latency" if metric == "effective_latency" else "loop_fps"
        _write_trace_csv(
            args.output_dir / f"source_{source_stem}_traces.csv",
            traces,
            sliding=False,
            metric=metric,
        )
        _write_trace_csv(
            args.output_dir / f"source_{source_stem}_traces_sliding_window.csv",
            traces,
            sliding=True,
            metric=metric,
        )


if __name__ == "__main__":
    main()
