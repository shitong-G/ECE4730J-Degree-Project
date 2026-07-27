"""Analyze a defense suite with an explicit quality floor before latency.

The quality values are agreement with the native FP32 teacher on the common
window, not human-annotated mAP.  A run is *acceptable* only when all quality
constraints are met.  Latency/FPS is ranked only inside that feasible set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LABELS = {
    "native_fp32": "Native FP32",
    "int8_every_frame": "INT8 every-frame",
    "int8_fixed_skip_2": "INT8 skip-2",
    "int8_fixed_skip_5": "INT8 skip-5",
    "int8_fixed_skip_10": "INT8 skip-10",
    "int8_periodic_lk_5": "Periodic LK-5",
    "int8_event_lk": "Event LK",
    "int8_event_lk_roi": "LK+ROI Q300",
    "int8_event_lk_roi_q64": "LK+ROI Q64",
    "int8_event_lk_roi_qthermal": "LK+ROI thermal-Q",
    "proposed_software": "Proposed controller",
}
COLORS = {
    "native_fp32": "#4e79a7", "int8_event_lk": "#e15759",
    "int8_event_lk_roi": "#b07aa1", "int8_event_lk_roi_q64": "#edc948",
    "int8_event_lk_roi_qthermal": "#ff9da7", "proposed_software": "#2f8f46",
}


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def build(summary: list[dict[str, str]], quality: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, object]]:
    by_run = {r.get("student", ""): r for r in quality}
    rows: list[dict[str, object]] = []
    for s in summary:
        key, run = s.get("key", ""), s.get("run", "")
        if key not in LABELS or run not in by_run:
            continue
        q = by_run[run]
        recall, precision, iou, count = (f(q, k) for k in ("pseudo_recall", "precision_proxy", "mean_matched_iou", "detection_count_ratio"))
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        reasons = []
        checks = [("recall", recall, args.min_recall, ">="), ("precision_proxy", precision, args.min_precision, ">="), ("matched_iou", iou, args.min_iou, ">="), ("count_ratio", count, args.min_count_ratio, ">=")]
        for name, value, floor, _ in checks:
            if value < floor:
                reasons.append(f"{name}<{floor:g}")
        if count > args.max_count_ratio:
            reasons.append(f"count_ratio>{args.max_count_ratio:g}")
        rows.append({
            "run": run, "key": key, "label": LABELS[key],
            "latency_ms": f(s, "latency_ms_mean_detector_frames"),
            "fps": f(s, "loop_fps_mean"), "recall": recall,
            "precision_proxy": precision, "matched_iou": iou,
            "count_ratio": count, "agreement_f1": f1, "quality_score": f1 * iou,
            "detector_ratio": f(s, "detector_invocation_ratio"), "max_temp_c": f(s, "temp_c_max"),
            "acceptable": not reasons, "rejection_reason": ";".join(reasons) or "PASS",
        })
    native = next((r for r in rows if r["key"] == "native_fp32"), None)
    base = float(native["latency_ms"]) if native and float(native["latency_ms"]) > 0 else 0.0
    for r in rows:
        r["speedup_vs_native"] = base / float(r["latency_ms"]) if base and float(r["latency_ms"]) else 0.0
    feasible = [r for r in rows if r["acceptable"]]
    for r in rows:
        r["latency_rank_feasible"] = (sorted(feasible, key=lambda x: float(x["latency_ms"])).index(r) + 1) if r in feasible else ""
    return sorted(rows, key=lambda x: (not bool(x["acceptable"]), float(x["latency_ms"])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--min-recall", type=float, default=0.50)
    p.add_argument("--min-precision", type=float, default=0.65)
    p.add_argument("--min-iou", type=float, default=0.75)
    p.add_argument("--min-count-ratio", type=float, default=0.70)
    p.add_argument("--max-count-ratio", type=float, default=1.20)
    p.add_argument("--quality-summary", type=Path, default=None)
    args = p.parse_args()
    suite = args.suite_dir.resolve()
    analysis = suite / "analysis"
    out = (args.output_dir or analysis / "quality_constrained").resolve()
    out.mkdir(parents=True, exist_ok=True)
    quality_path = (args.quality_summary or analysis / "quality_summary.csv").resolve()
    rows = build(read(analysis / "defense_summary.csv"), read(quality_path), args)
    if not rows:
        raise RuntimeError("No matching rows found")
    fields = list(rows[0].keys())
    with (out / "quality_constrained_summary.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    passed = [r for r in rows if r["acceptable"]]
    rejected = [r for r in rows if not r["acceptable"]]
    # Presentation figure: the decision is visually explicit.
    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    for r in rows:
        ax.scatter(float(r["speedup_vs_native"]), float(r["quality_score"]), s=120 if r["acceptable"] else 85,
                   color=COLORS.get(str(r["key"]), "#888"), marker="o" if r["acceptable"] else "x",
                   edgecolor="black" if r["acceptable"] else None, linewidth=0.4, alpha=1.0 if r["acceptable"] else .65)
        ax.annotate(str(r["label"]), (float(r["speedup_vs_native"]), float(r["quality_score"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Relative speedup vs native (higher is better)")
    ax.set_ylabel("Teacher-agreement quality Q (higher is better)")
    ax.set_title("Quality-constrained latency trade-off (PASS = filled circle)")
    ax.set_ylim(bottom=0); ax.grid(alpha=.25)
    ax.text(.01, .01, f"PASS requires recall≥{args.min_recall:.2f}, precision≥{args.min_precision:.2f}, IoU≥{args.min_iou:.2f}, count ratio {args.min_count_ratio:.2f}–{args.max_count_ratio:.2f}; Q is not mAP", transform=ax.transAxes, fontsize=8, va="bottom")
    fig.savefig(out / "quality_constrained_tradeoff_presentation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Paper figure: metrics and latency are separate so no hidden weighting is used.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    labels = [str(r["label"]) for r in rows]
    colors = [COLORS.get(str(r["key"]), "#888") for r in rows]
    for ax, key, title, floor in [(axes[0,0], "recall", "Pseudo recall", args.min_recall), (axes[0,1], "precision_proxy", "Precision proxy", args.min_precision), (axes[1,0], "matched_iou", "Matched IoU", args.min_iou), (axes[1,1], "latency_ms", "Detector latency (ms)", None)]:
        vals = [float(r[key]) for r in rows]
        ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=.3)
        if floor is not None: ax.axhline(floor, color="#b22222", ls="--", lw=1, label=f"floor {floor:.2f}")
        ax.set_title(title); ax.tick_params(axis="x", rotation=55); ax.grid(axis="y", alpha=.2)
        if key != "latency_ms": ax.set_ylim(0, 1.08)
        if floor is not None: ax.legend(fontsize=8)
    fig.suptitle("Defense suite: quality floor first, latency second", fontsize=15)
    fig.text(.5, .01, "Only PASS runs are eligible for latency ranking; agreement uses the common native-teacher window.", ha="center", fontsize=9)
    fig.savefig(out / "quality_constrained_metrics_paper.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "quality_constrained_metrics_paper.pdf", bbox_inches="tight")
    plt.close(fig)
    # Eligible latency ranking only.
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar([str(r["label"]) for r in passed], [float(r["latency_ms"]) for r in passed], color=[COLORS.get(str(r["key"]), "#888") for r in passed])
    ax.set_ylabel("Mean detector latency (ms), lower is better"); ax.set_title("Latency ranking inside the acceptable-quality set")
    ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "acceptable_latency_ranking_presentation.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "acceptable_latency_ranking_paper.pdf", bbox_inches="tight")
    plt.close(fig)
    report = ["# Quality-constrained latency analysis", "", "Quality is teacher agreement, not human ground-truth accuracy.", "", "## Predeclared acceptance floor", f"- pseudo recall >= {args.min_recall:.2f}", f"- precision proxy >= {args.min_precision:.2f}", f"- matched IoU >= {args.min_iou:.2f}", f"- detection-count ratio in [{args.min_count_ratio:.2f}, {args.max_count_ratio:.2f}]", "", "## Decision"]
    for r in rows:
        report.append(f"- **{r['label']}**: {'PASS' if r['acceptable'] else 'REJECT'}; latency {float(r['latency_ms']):.1f} ms, FPS {float(r['fps']):.2f}; {r['rejection_reason']}")
    report += ["", "Latency is ranked only among PASS rows. A rejected row can be faster, but it does not satisfy the thesis quality requirement."]
    (out / "quality_constrained_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Saved quality-constrained analysis: {out}")
    print("PASS latency order:", ", ".join(str(r["label"]) for r in sorted(passed, key=lambda x: float(x["latency_ms"]))) or "none")


if __name__ == "__main__":
    main()
