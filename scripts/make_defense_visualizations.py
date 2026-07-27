"""Create presentation and paper visualizations from a completed defense log.

Performance/thermal metrics use the complete formal runtime CSVs. Accuracy is
read from ``analysis/quality_summary.csv`` and is explicitly labelled as the
common-frame comparison against the native teacher.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


LABELS = {
    "native_fp32": "Native FP32",
    "int8_every_frame": "INT8",
    "int8_fixed_skip_2": "INT8 skip-2",
    "int8_fixed_skip_5": "INT8 skip-5",
    "int8_fixed_skip_10": "INT8 skip-10",
    "int8_periodic_lk_5": "Periodic LK-5",
    "int8_event_lk": "Event LK",
    "int8_event_lk_roi": "LK+ROI Q300",
    "int8_event_lk_roi_q64": "LK+ROI Q64",
    "int8_event_lk_roi_qthermal": "LK+ROI thermal-Q",
    "proposed_software": "Proposed software",
}
ORDER = list(LABELS)
KEY_COLORS = {
    "native_fp32": "#4e79a7",
    "int8_every_frame": "#59a14f",
    "int8_fixed_skip_2": "#76b7b2",
    "int8_fixed_skip_5": "#86bc86",
    "int8_fixed_skip_10": "#8cd17d",
    "int8_periodic_lk_5": "#f28e2b",
    "int8_event_lk": "#e15759",
    "int8_event_lk_roi": "#b07aa1",
    "int8_event_lk_roi_q64": "#edc948",
    "int8_event_lk_roi_qthermal": "#ff9da7",
    "proposed_software": "#59a14f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def save(fig: Any, path: Path, *, paper: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300 if paper else 180, bbox_inches="tight")
    if paper:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def labels_for(rows: list[dict[str, str]]) -> list[str]:
    return [LABELS.get(row["key"], row["key"]) for row in rows]


def make_presentation(rows: list[dict[str, str]], quality: dict[str, dict[str, str]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selected_keys = [
        "native_fp32", "int8_every_frame", "int8_fixed_skip_5",
        "int8_periodic_lk_5", "int8_event_lk", "int8_event_lk_roi",
        "int8_event_lk_roi_q64", "int8_event_lk_roi_qthermal",
        "proposed_software",
    ]
    data = [row for row in rows if row["key"] in selected_keys]
    x = np.arange(len(data))
    names = labels_for(data)
    fps = np.array([f(row, "loop_fps_mean", 0) or 0 for row in data])
    native_fps = fps[0] or 1.0
    recall = np.array([f(quality.get(row["run"], {}), "pseudo_recall", 0) or 0 for row in data])
    temp = np.array([f(row, "temp_c_max", 0) or 0 for row in data])
    detector = np.array([f(row, "detector_invocation_ratio", 0) or 0 for row in data])
    colors = [KEY_COLORS.get(row["key"], "#777777") for row in data]

    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.8), constrained_layout=True)
    axes[0].bar(x, fps / native_fps, color=colors)
    axes[0].axhline(1, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("Pipeline FPS / native FPS")
    axes[0].set_title("Runtime efficiency")
    axes[0].set_ylim(0, max(1.1, float((fps / native_fps).max() * 1.15)))
    axes[1].bar(x, recall, color=colors)
    axes[1].set_ylabel("Pseudo recall vs native teacher")
    axes[1].set_title("Detection/tracking agreement")
    axes[1].set_ylim(0, 1.08)
    for ax in axes:
        ax.set_xticks(x, names, rotation=42, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("RT-DETR defense experiment: efficiency–quality trade-off", fontsize=16)
    fig.text(0.5, 0.01, "Accuracy uses the common 431-frame native-teacher comparison; higher is better.", ha="center", fontsize=9)
    save(fig, out / "ppt_main_efficiency_quality.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(13.33, 6.8), constrained_layout=True)
    ax2 = ax1.twinx()
    width = 0.38
    ax1.bar(x - width / 2, temp, width, color="#e15759", alpha=0.85, label="Maximum temperature (°C)")
    ax2.bar(x + width / 2, 100 * detector, width, color="#4e79a7", alpha=0.85, label="Detector invocation ratio (%)")
    ax1.set_ylabel("Maximum CPU temperature (°C)")
    ax2.set_ylabel("Detector invocation ratio (%)")
    ax1.set_xticks(x, names, rotation=42, ha="right", fontsize=8)
    ax1.set_title("Thermal load and detector workload")
    ax1.grid(axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, out / "ppt_thermal_and_detector_workload.png")
    plt.close(fig)

    query_keys = ["int8_event_lk_roi", "int8_event_lk_roi_q64", "int8_event_lk_roi_qthermal", "proposed_software"]
    qdata = [row for row in rows if row["key"] in query_keys]
    qx = np.arange(len(qdata))
    qmean = np.array([f(row, "query_budget_applied_mean", 300) or 300 for row in qdata])
    qlat = np.array([f(row, "latency_ms_mean_detector_frames", 0) or 0 for row in qdata])
    qrecall = np.array([f(quality.get(row["run"], {}), "pseudo_recall", 0) or 0 for row in qdata])
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.8), constrained_layout=True)
    axes[0].bar(qx, qmean, color=[KEY_COLORS[row["key"]] for row in qdata])
    axes[0].set_ylabel("Mean applied query budget")
    axes[0].set_title("Runtime query allocation")
    axes[1].scatter(qlat, qrecall, s=130, c=[KEY_COLORS[row["key"]] for row in qdata])
    for x0, y0, row in zip(qlat, qrecall, qdata):
        axes[1].annotate(LABELS[row["key"]], (x0, y0), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axes[1].set_xlabel("Mean detector-frame latency (ms)")
    axes[1].set_ylabel("Pseudo recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Query budget trade-off")
    save(fig, out / "ppt_query_budget_tradeoff.png")
    plt.close(fig)


def make_paper(rows: list[dict[str, str]], quality: dict[str, dict[str, str]], frames: list[dict[str, str]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = labels_for(rows)
    x = np.arange(len(rows))
    colors = [KEY_COLORS.get(row["key"], "#777777") for row in rows]
    fps = np.array([f(row, "loop_fps_mean", 0) or 0 for row in rows])
    latency = np.array([f(row, "latency_ms_mean_detector_frames", 0) or 0 for row in rows])
    temp = np.array([f(row, "temp_c_max", 0) or 0 for row in rows])
    inv = np.array([f(row, "detector_invocation_ratio", 0) or 0 for row in rows])
    recall = np.array([f(quality.get(row["run"], {}), "pseudo_recall", 0) or 0 for row in rows])
    iou = np.array([f(quality.get(row["run"], {}), "mean_matched_iou", 0) or 0 for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(x, fps, color=colors)
    axes[0, 0].set_ylabel("Pipeline FPS")
    axes[0, 0].set_title("(a) End-to-end throughput")
    axes[0, 1].bar(x, latency, color=colors)
    axes[0, 1].set_ylabel("Detector latency (ms)")
    axes[0, 1].set_title("(b) Detector-frame latency")
    axes[1, 0].bar(x, recall, color=colors, label="Pseudo recall")
    axes[1, 0].plot(x, iou, "ko--", markersize=3, label="Matched IoU")
    axes[1, 0].set_ylim(0, 1.08)
    axes[1, 0].set_ylabel("Agreement with native teacher")
    axes[1, 0].set_title("(c) Detection/tracking quality")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(x, temp, color="#e15759")
    axes[1, 1].set_ylabel("Maximum CPU temperature (°C)")
    axes[1, 1].set_title("(d) Thermal envelope")
    for ax in axes.flat:
        ax.set_xticks(x, names, rotation=55, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Defense experiment summary (20-minute formal runs)")
    save(fig, out / "paper_main_metrics.png", paper=True)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].scatter(latency, recall, c=temp, cmap="inferno", s=70, edgecolor="black", linewidth=0.3)
    for x0, y0, name in zip(latency, recall, names):
        axes[0].annotate(name, (x0, y0), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("Mean detector-frame latency (ms)")
    axes[0].set_ylabel("Pseudo recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Latency–quality Pareto view")
    axes[1].scatter(inv * 100, fps, c=temp, cmap="inferno", s=70, edgecolor="black", linewidth=0.3)
    for x0, y0, name in zip(inv * 100, fps, names):
        axes[1].annotate(name, (x0, y0), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[1].set_xlabel("Detector invocation ratio (%)")
    axes[1].set_ylabel("Pipeline FPS")
    axes[1].set_title("Invocation reduction–throughput view")
    save(fig, out / "paper_pareto_tradeoffs.png", paper=True)
    plt.close(fig)

    temporal_keys = ["int8_event_lk", "int8_event_lk_roi", "int8_event_lk_roi_q64", "int8_event_lk_roi_qthermal", "proposed_software"]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for key in temporal_keys:
        label = LABELS[key]
        subset = [row for row in frames if row.get("student") == next((r["run"] for r in rows if r["key"] == key), "")]
        if not subset:
            continue
        bins = []
        for start in range(0, 431, 50):
            values = [f(row, "pseudo_recall", None) for row in subset if start <= (f(row, "frame_id", -1) or -1) < start + 50]
            values = [value for value in values if value is not None]
            if values:
                bins.append((start + 25, sum(values) / len(values)))
        if bins:
            ax.plot([item[0] for item in bins], [item[1] for item in bins], marker="o", linewidth=1.4, label=label, color=KEY_COLORS[key])
    ax.set_xlabel("Student frame ID (common teacher window)")
    ax.set_ylabel("Per-frame pseudo recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Temporal quality degradation under tracking")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, out / "paper_temporal_quality_decay.png", paper=True)
    plt.close(fig)

    query_keys = ["int8_event_lk_roi", "int8_event_lk_roi_q64", "int8_event_lk_roi_qthermal", "proposed_software"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False, constrained_layout=True)
    for key in query_keys:
        row = next((r for r in rows if r["key"] == key), None)
        if row is None:
            continue
        # ``out`` is ``<suite>/analysis/visualizations/paper``.
        suite_root = out.parents[2]
        csv_path = suite_root / row["run"] / "runtime.csv"
        if not csv_path.exists():
            continue
        runtime = read_csv(csv_path)
        points = [(idx, f(item, "query_budget_applied"), f(item, "temp_c")) for idx, item in enumerate(runtime) if f(item, "query_budget_applied") is not None]
        if not points:
            continue
        axes[0].step([p[0] for p in points], [p[1] for p in points], where="post", label=LABELS[key], color=KEY_COLORS[key])
        axes[1].plot([p[0] for p in points], [p[2] for p in points], ".", markersize=2, label=LABELS[key], color=KEY_COLORS[key])
    axes[0].set_ylabel("Applied query budget")
    axes[0].set_title("Runtime query allocation")
    axes[1].set_xlabel("Runtime frame index")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_title("Temperature at query allocation points")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.25)
    save(fig, out / "paper_query_allocation_timeline.png", paper=True)
    plt.close(fig)

    # Temperature traces for the most interpretable baseline/proposed subset.
    trace_keys = ["native_fp32", "int8_every_frame", "int8_periodic_lk_5", "int8_event_lk", "int8_event_lk_roi_qthermal", "proposed_software"]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for key in trace_keys:
        row = next((r for r in rows if r["key"] == key), None)
        if row is None:
            continue
        suite_root = out.parents[2]
        runtime = read_csv(suite_root / row["run"] / "runtime.csv")
        if not runtime:
            continue
        t0 = f(runtime[0], "timestamp", 0) or 0
        points = [(f(item, "timestamp", 0) - t0, f(item, "temp_c")) for item in runtime if f(item, "temp_c") is not None]
        if points:
            ax.plot([p[0] / 60 for p in points], [p[1] for p in points], linewidth=1.0, label=LABELS[key], color=KEY_COLORS[key])
    ax.set_xlabel("Formal runtime (min)")
    ax.set_ylabel("CPU temperature (°C)")
    ax.set_title("Thermal trajectories over 20-minute runs")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, out / "paper_temperature_trajectories.png", paper=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite = args.suite_dir.resolve()
    analysis = suite / "analysis"
    output = (args.output_dir or analysis / "visualizations").resolve()
    summary_rows = read_csv(analysis / "defense_summary.csv")
    summary_rows = [row for row in summary_rows if row.get("key") in LABELS]
    quality_rows = read_csv(analysis / "quality_summary.csv")
    quality = {row.get("student", ""): row for row in quality_rows}
    quality_frames = read_csv(analysis / "quality_frames.csv")
    presentation = output / "presentation"
    paper = output / "paper"
    make_presentation(summary_rows, quality, presentation)
    make_paper(summary_rows, quality, quality_frames, paper)
    notes = output / "README.md"
    notes.write_text(
        "# Defense visualizations\n\n"
        "- `presentation/`: simplified 16:9 figures for defense slides.\n"
        "- `paper/`: high-resolution PNG and PDF figures for the thesis.\n\n"
        "Performance and temperature use complete 20-minute runtime logs.\n"
        "Accuracy is pseudo-label agreement with the native FP32 teacher on the "
        "common 431 teacher frames; the 952-frame video cycle was not fully "
        "covered by the 20-minute native run.\n",
        encoding="utf-8",
    )
    print(f"Saved presentation figures: {presentation}")
    print(f"Saved paper figures: {paper}")


if __name__ == "__main__":
    main()
