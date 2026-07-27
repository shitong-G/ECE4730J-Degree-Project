"""Compute and plot a latency--teacher-agreement trade-off.

This is not ground-truth accuracy.  The quality score is a conservative
teacher-agreement composite:

    F1_agreement = 2 * recall * precision_proxy / (recall + precision_proxy)
    Q = F1_agreement * mean_matched_IoU

Lower detector latency and higher Q are better.  A point is Pareto-optimal
when no other run is both faster (or equal) and higher quality (or equal).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LABELS = {
    "native_fp32": "Native FP32",
    "int8_every_frame": "INT8 every",
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
COLORS = {
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
    "proposed_software": "#2f8f46",
}


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute(summary: list[dict[str, str]], quality: list[dict[str, str]]) -> list[dict[str, object]]:
    quality_by_run = {row.get("student", ""): row for row in quality}
    rows: list[dict[str, object]] = []
    for row in summary:
        key = row.get("key", "")
        if key not in LABELS or row.get("run") not in quality_by_run:
            continue
        qrow = quality_by_run[row["run"]]
        recall = number(qrow, "pseudo_recall")
        precision = number(qrow, "precision_proxy")
        iou = number(qrow, "mean_matched_iou")
        f1 = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
        rows.append(
            {
                "run": row["run"],
                "key": key,
                "label": LABELS[key],
                "latency_ms": number(row, "latency_ms_mean_detector_frames"),
                "fps": number(row, "loop_fps_mean"),
                "recall": recall,
                "precision_proxy": precision,
                "matched_iou": iou,
                "agreement_f1": f1,
                "quality_score": f1 * iou,
                "max_temp_c": number(row, "temp_c_max"),
                "detector_ratio": number(row, "detector_invocation_ratio"),
            }
        )
    if not rows:
        return rows
    native_latency = next((float(r["latency_ms"]) for r in rows if r["key"] == "native_fp32"), None)
    if native_latency:
        for row in rows:
            row["relative_speed"] = native_latency / float(row["latency_ms"])
            row["quality_adjusted_speedup"] = float(row["quality_score"]) * native_latency / float(row["latency_ms"])
    else:
        for row in rows:
            row["relative_speed"] = 0.0
            row["quality_adjusted_speedup"] = 0.0
    # Sort by increasing latency. A point is dominated if a faster/equal point
    # has at least as much quality and one of the two is strictly better.
    for candidate in rows:
        candidate["pareto"] = not any(
            other is not candidate
            and float(other["latency_ms"]) <= float(candidate["latency_ms"])
            and float(other["quality_score"]) >= float(candidate["quality_score"])
            and (
                float(other["latency_ms"]) < float(candidate["latency_ms"])
                or float(other["quality_score"]) > float(candidate["quality_score"])
            )
            for other in rows
        )
    return sorted(rows, key=lambda item: float(item["latency_ms"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    suite = args.suite_dir.resolve()
    analysis = suite / "analysis"
    output = (args.output_dir or analysis / "visualizations" / "tradeoff").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = compute(read(analysis / "defense_summary.csv"), read(analysis / "quality_summary.csv"))
    if not rows:
        raise RuntimeError("No matching summary and quality rows found")

    with (output / "latency_accuracy_tradeoff.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["run", "key", "label", "latency_ms", "relative_speed", "fps", "recall", "precision_proxy", "matched_iou", "agreement_f1", "quality_score", "quality_adjusted_speedup", "max_temp_c", "detector_ratio", "pareto"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frontier = sorted((row for row in rows if row["pareto"]), key=lambda row: float(row["relative_speed"]))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for row in rows:
        ax.scatter(float(row["relative_speed"]), float(row["quality_score"]), s=85, color=COLORS.get(str(row["key"]), "#777"), edgecolor="black", linewidth=0.35, zorder=3)
        ax.annotate(str(row["label"]), (float(row["relative_speed"]), float(row["quality_score"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.plot([float(row["relative_speed"]) for row in frontier], [float(row["quality_score"]) for row in frontier], color="#222", linestyle="--", linewidth=1.2, label="Pareto frontier", zorder=2)
    ax.set_xlabel("Relative inference speed (native latency / strategy latency) — higher is better")
    ax.set_ylabel("Composite teacher-agreement quality Q — higher is better")
    ax.set_title("Speed–accuracy trade-off (common native-teacher window)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.text(0.01, 0.01, "Q = matched IoU × F1(recall, precision proxy); not ground-truth mAP", transform=ax.transAxes, fontsize=8, va="bottom")
    fig.savefig(output / "latency_accuracy_pareto_paper.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "latency_accuracy_pareto_paper.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.33, 6.8), constrained_layout=True)
    axes[0].scatter([float(row["relative_speed"]) for row in rows], [float(row["quality_score"]) for row in rows], s=130, c=[COLORS.get(str(row["key"]), "#777") for row in rows], edgecolor="black", linewidth=0.4)
    axes[0].plot([float(row["relative_speed"]) for row in frontier], [float(row["quality_score"]) for row in frontier], "k--", linewidth=1.2, label="Pareto frontier")
    for row in rows:
        axes[0].annotate(str(row["label"]), (float(row["relative_speed"]), float(row["quality_score"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Relative inference speed, higher is better")
    axes[0].set_ylabel("Composite quality Q, higher is better")
    axes[0].set_title("Quality–speed frontier")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].bar([str(row["label"]) for row in rows], [float(row["quality_adjusted_speedup"]) for row in rows], color=[COLORS.get(str(row["key"]), "#777") for row in rows])
    axes[1].set_ylabel("Quality-adjusted speedup U = Q × native latency / latency")
    axes[1].set_title("Single-number trade-off utility")
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Defense experiment trade-off summary", fontsize=16)
    fig.text(0.5, 0.01, "Accuracy is teacher agreement on the common 431-frame window, not human ground truth.", ha="center", fontsize=9)
    fig.savefig(output / "latency_accuracy_tradeoff_presentation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trade-off outputs: {output}")
    print("Pareto frontier:", ", ".join(str(row["label"]) for row in frontier))


if __name__ == "__main__":
    main()
