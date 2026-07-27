"""Re-evaluate teacher agreement at several confidence thresholds.

The runtime logs already contain detections retained at the original 0.5
threshold.  This script can therefore evaluate any threshold >= 0.5 without
rerunning inference.  The same threshold is applied to teacher and students.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import evaluate_strategy_detection_quality as quality_eval


LABELS = {
    "01_native_fp32": "Native FP32",
    "02_int8_every_frame": "INT8 every",
    "03_int8_fixed_skip_2": "INT8 skip-2",
    "04_int8_fixed_skip_5": "INT8 skip-5",
    "05_int8_fixed_skip_10": "INT8 skip-10",
    "06_int8_periodic_lk_5": "Periodic LK-5",
    "07_int8_event_lk": "Event LK",
    "08_int8_event_lk_roi": "LK+ROI Q300",
    "09_int8_event_lk_roi_q64": "LK+ROI Q64",
    "10_int8_event_lk_roi_qthermal": "LK+ROI thermal-Q",
    "11_proposed_software": "Proposed software",
}
COLORS = {
    "01_native_fp32": "#4e79a7",
    "02_int8_every_frame": "#59a14f",
    "03_int8_fixed_skip_2": "#76b7b2",
    "04_int8_fixed_skip_5": "#86bc86",
    "05_int8_fixed_skip_10": "#8cd17d",
    "06_int8_periodic_lk_5": "#f28e2b",
    "07_int8_event_lk": "#e15759",
    "08_int8_event_lk_roi": "#b07aa1",
    "09_int8_event_lk_roi_q64": "#edc948",
    "10_int8_event_lk_roi_qthermal": "#ff9da7",
    "11_proposed_software": "#2f8f46",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--threshold-start", type=float, default=0.5)
    parser.add_argument("--threshold-stop", type=float, default=0.8)
    parser.add_argument("--threshold-step", type=float, default=0.1)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--threshold-step must be positive")
    if start < 0.5:
        raise ValueError("Logs were filtered at 0.5; thresholds below 0.5 cannot be reconstructed")
    values: list[float] = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 10))
        value += step
    return values


def filter_records(
    records: dict[int, dict[str, Any]], threshold: float
) -> dict[int, dict[str, Any]]:
    filtered: dict[int, dict[str, Any]] = {}
    for frame_id, row in records.items():
        new_row = dict(row)
        new_row["detections"] = [
            detection
            for detection in (row.get("detections") or [])
            if float(detection.get("score", 0.0)) >= threshold
        ]
        filtered[frame_id] = new_row
    return filtered


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def add_composite_metrics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        recall = number(row, "pseudo_recall")
        precision = number(row, "precision_proxy")
        f1 = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
        row["agreement_f1"] = f1
        row["quality_score"] = f1 * number(row, "mean_matched_iou")


def mark_pareto(
    rows: list[dict[str, Any]], detector_latency: dict[str, float]
) -> None:
    native_latency = detector_latency["01_native_fp32"]
    for row in rows:
        latency = detector_latency.get(str(row["student"]), 0.0)
        row["detector_latency_ms"] = latency
        row["relative_speed"] = native_latency / latency if latency > 0 else 0.0
    for candidate in rows:
        candidate["pareto"] = not any(
            other is not candidate
            and number(other, "relative_speed") >= number(candidate, "relative_speed")
            and number(other, "quality_score") >= number(candidate, "quality_score")
            and (
                number(other, "relative_speed") > number(candidate, "relative_speed")
                or number(other, "quality_score") > number(candidate, "quality_score")
            )
            for other in rows
        )


def plot_pareto(rows: list[dict[str, Any]], path: Path, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frontier = sorted(
        (row for row in rows if row["pareto"]),
        key=lambda row: number(row, "relative_speed"),
    )
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for row in rows:
        proposed = row["student"] == "11_proposed_software"
        ax.scatter(
            number(row, "relative_speed"),
            number(row, "quality_score"),
            s=140 if proposed else 75,
            marker="*" if proposed else "o",
            color=COLORS[str(row["student"])],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            LABELS.get(str(row["student"]), str(row["student"])),
            (number(row, "relative_speed"), number(row, "quality_score")),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.plot(
        [number(row, "relative_speed") for row in frontier],
        [number(row, "quality_score") for row in frontier],
        "k--",
        linewidth=1.2,
        label="Pareto frontier",
    )
    ax.set_xlabel("Relative inference speed — higher is better")
    ax.set_ylabel("Composite teacher-agreement quality Q — higher is better")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"Speed–quality Pareto frontier, confidence ≥ {threshold:.1f}")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.text(
        0.01,
        0.01,
        "Q = matched IoU × F1(pseudo recall, precision proxy)",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("pseudo_recall", "Pseudo recall"),
        ("precision_proxy", "Precision proxy"),
        ("mean_matched_iou", "Mean matched IoU"),
        ("quality_score", "Composite quality Q"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for student in LABELS:
        subset = sorted(
            (row for row in rows if row["student"] == student),
            key=lambda row: number(row, "confidence_threshold"),
        )
        if not subset:
            continue
        for ax, (metric, title) in zip(axes.flat, metrics):
            ax.plot(
                [number(row, "confidence_threshold") for row in subset],
                [number(row, metric) for row in subset],
                marker="o",
                linewidth=2.2 if student == "11_proposed_software" else 1.0,
                color=COLORS[student],
                label=LABELS[student],
            )
            ax.set_title(title)
            ax.set_xlabel("Confidence threshold")
            ax.set_ylim(0.0, 1.03)
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("Confidence-threshold sensitivity (common 431-frame teacher window)")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    suite = args.suite_dir.resolve()
    analysis = suite / "analysis"
    output = (args.output_dir or analysis / "confidence_sweep").resolve()
    thresholds = threshold_values(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )

    run_names = [name for name in LABELS if (suite / name / "runtime_detections.jsonl").exists()]
    if "01_native_fp32" not in run_names:
        raise FileNotFoundError("Native teacher detection log is missing")
    raw_records = {
        name: quality_eval._load_jsonl(suite / name / "runtime_detections.jsonl")
        for name in run_names
    }
    runtime_rows = {
        name: quality_eval._load_csv(suite / name / "runtime.csv")
        for name in run_names
    }
    defense_rows = read_csv(analysis / "defense_summary.csv")
    detector_latency = {
        row["run"]: number(row, "latency_ms_mean_detector_frames")
        for row in defense_rows
        if row.get("run") in run_names
    }

    combined: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_name = f"threshold_{threshold:.1f}".replace(".", "p")
        threshold_dir = output / threshold_name
        teacher = filter_records(raw_records["01_native_fp32"], threshold)
        summaries: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        for run_name in run_names:
            student = filter_records(raw_records[run_name], threshold)
            summary, frames = quality_eval._compare_one(
                teacher=teacher,
                student=student,
                student_path=suite / run_name / "runtime_detections.jsonl",
                csv_rows=runtime_rows[run_name],
                iou_threshold=args.iou_threshold,
                teacher_cycle_frames=0,
            )
            summary["confidence_threshold"] = threshold
            for frame in frames:
                frame["confidence_threshold"] = threshold
            summaries.append(summary)
            frame_rows.extend(frames)
        add_composite_metrics(summaries)
        mark_pareto(summaries, detector_latency)
        combined.extend(summaries)
        write_csv(threshold_dir / "quality_summary.csv", summaries)
        write_csv(threshold_dir / "quality_frames.csv", frame_rows)
        quality_eval._plot_summary(
            summaries,
            threshold_dir / "quality_overview.png",
            label_source="student",
        )
        plot_pareto(
            summaries,
            threshold_dir / "speed_quality_pareto.png",
            threshold,
        )
        print(f"Saved threshold {threshold:.1f}: {threshold_dir}")

    write_csv(output / "confidence_sweep_summary.csv", combined)
    plot_sensitivity(combined, output / "confidence_sensitivity.png")
    print(f"Saved sweep summary: {output}")


if __name__ == "__main__":
    main()
