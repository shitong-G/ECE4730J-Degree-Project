#!/usr/bin/env python3
"""Publication-style descriptive figures for final_thermal_20260728_132853.

The matrix has one 10-minute run per condition.  Figures deliberately retain
all rows and label pseudo-label agreement as non-ground-truth evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "r01_01_fp32_native": "FP32 native",
    "r01_02_int8_dynamic_q300_native": "INT8 Q300 native",
    "r01_03_int8_dynamic_q300_lk_roi": "INT8 Q300 + LK/ROI",
    "r01_04_int8_dynamic_q40_lk_roi": "INT8 Q40 + LK/ROI",
    "r01_05_int8_dynamic_qthermal_lk_roi": "INT8 thermal-Q + LK/ROI",
    "r01_06_proposed_software": "Proposed software",
}
COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#AF7AA1"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "legend.frameon": False,
})


def num(value: str | None) -> float | None:
    try:
        parsed = float(value or "")
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str | None) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def elapsed_series(rows: list[dict[str, str]], column: str, *, infer_only: bool = False) -> tuple[list[float], list[float]]:
    origin = num(rows[0].get("timestamp")) or 0.0
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if infer_only and not truth(row.get("did_infer")):
            continue
        stamp, value = num(row.get("timestamp")), num(row.get(column))
        if stamp is not None and value is not None:
            xs.append((stamp - origin) / 60.0)
            ys.append(value)
    return xs, ys


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(runs: dict[str, list[dict[str, str]]], column: str, stem: Path, title: str, ylabel: str, *, infer_only: bool = False, thresholds: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for color, (key, rows) in zip(COLORS, runs.items()):
        xs, ys = elapsed_series(rows, column, infer_only=infer_only)
        if infer_only:
            # RT-DETR calls in LK/ROI modes are event-triggered.  Mark calls
            # independently so a long tracking-only interval is not falsely
            # represented as a continuous latency trend.
            ax.scatter(xs, ys, color=color, s=13, alpha=0.82, label=LABELS[key])
        else:
            ax.plot(xs, ys, color=color, lw=1.1, label=LABELS[key])
    if thresholds:
        for value, text, color in [(58, "warm", "#C17C00"), (66, "hot", "#B4471A"), (76, "critical", "#8B1E1E")]:
            ax.axhline(value, color=color, ls="--", lw=0.75, alpha=0.8)
            ax.text(10.02, value, f" {text} {value}°C", color=color, va="bottom", fontsize=7)
    ax.set(xlim=(0, 10), xlabel="Elapsed formal-run time (min)", ylabel=ylabel, title=title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, fontsize=7, loc="best")
    save(fig, stem)


def rolling_mean_by_time(points: list[tuple[float, float]], window_sec: float = 30.0) -> tuple[list[float], list[float]]:
    """Trailing time-window mean, so methods with different loop FPS remain comparable."""
    active: deque[tuple[float, float]] = deque()
    total = 0.0; xs: list[float] = []; means: list[float] = []
    window_min = window_sec / 60.0
    for x, value in points:
        active.append((x, value)); total += value
        while active and active[0][0] < x - window_min:
            _, old = active.popleft(); total -= old
        xs.append(x); means.append(total / len(active))
    return xs, means


def per_frame_latency(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
    """Use zero for tracking-only frames; retain actual detector latency otherwise."""
    origin = num(rows[0].get("timestamp")) or 0.0
    points: list[tuple[float, float]] = []
    for row in rows:
        stamp = num(row.get("timestamp"))
        if stamp is None:
            continue
        latency = num(row.get("latency_ms")) if truth(row.get("did_infer")) else 0.0
        points.append(((stamp - origin) / 60.0, latency or 0.0))
    return points


def plot_raw_and_rolling(runs: dict[str, list[dict[str, str]]], stem: Path, title: str, ylabel: str, *, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for color, (key, rows) in zip(COLORS, runs.items()):
        if mode == "latency":
            points = per_frame_latency(rows)
        elif mode == "fps":
            points = list(zip(*elapsed_series(rows, "loop_fps")))
        else:
            raise ValueError(mode)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.scatter(xs, ys, color=color, s=8, alpha=0.18, linewidths=0)
        mx, my = rolling_mean_by_time(points, window_sec=30.0)
        ax.plot(mx, my, color=color, lw=1.45, label=LABELS[key])
    ax.set(xlim=(0, 10), xlabel="Elapsed formal-run time (min)", ylabel=ylabel, title=title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, fontsize=7, loc="best")
    save(fig, stem)


def plot_full_latency(profiles: dict[str, list[dict[str, str]]], stem: Path) -> None:
    """Plot serial end-to-end latency: source acquisition plus all runtime work."""
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for color, (key, rows) in zip(COLORS, profiles.items()):
        origin = num(rows[0].get("timestamp")) or 0.0
        points = [
            ((stamp - origin) / 60.0, value)
            for row in rows
            for stamp, value in [(num(row.get("timestamp")), num(row.get("serial_total_ms")))]
            if stamp is not None and value is not None
        ]
        xs = [point[0] for point in points]; ys = [point[1] for point in points]
        ax.scatter(xs, ys, color=color, s=8, alpha=0.18, linewidths=0)
        mx, my = rolling_mean_by_time(points, window_sec=30.0)
        ax.plot(mx, my, color=color, lw=1.45, label=LABELS[key])
    ax.set(xlim=(0, 10), xlabel="Elapsed formal-run time (min)", ylabel="End-to-end latency (ms)", title="Full per-frame latency (capture + runtime)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, fontsize=7, loc="best")
    save(fig, stem)


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def normalized(det: dict[str, Any], resolution: int) -> list[float]:
    return [float(x) / float(resolution) for x in det["bbox"]]


def match(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, list[float]]:
    ref_res = int(reference.get("resolved_input_resolution") or reference.get("input_resolution") or 640)
    cand_res = int(candidate.get("resolved_input_resolution") or candidate.get("input_resolution") or 640)
    pairs: list[tuple[float, int, int]] = []
    for ri, ref_det in enumerate(reference.get("detections") or []):
        for ci, cand_det in enumerate(candidate.get("detections") or []):
            if ref_det.get("class_id") != cand_det.get("class_id"):
                continue
            overlap = iou(normalized(ref_det, ref_res), normalized(cand_det, cand_res))
            if overlap >= 0.5:
                pairs.append((overlap, ri, ci))
    used_ref: set[int] = set(); used_cand: set[int] = set(); matched: list[float] = []
    for overlap, ri, ci in sorted(pairs, reverse=True):
        if ri not in used_ref and ci not in used_cand:
            used_ref.add(ri); used_cand.add(ci); matched.append(overlap)
    return len(matched), matched


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {int(item["frame_id"]): item for line in handle if line.strip() for item in [json.loads(line)]}


def quality(suite: Path, run_keys: list[str]) -> list[dict[str, float | str]]:
    teacher = load_jsonl(suite / "r01_01_fp32_native" / "runtime_detections.jsonl")
    output = []
    for key in run_keys:
        candidate = load_jsonl(suite / key / "runtime_detections.jsonl")
        common = sorted(set(teacher) & set(candidate))
        matched = total_candidate = 0; overlaps: list[float] = []
        for frame in common:
            count, scores = match(teacher[frame], candidate[frame])
            matched += count; overlaps.extend(scores); total_candidate += len(candidate[frame].get("detections") or [])
        output.append({"run": key, "common_frames": len(common), "precision_proxy": matched / total_candidate if total_candidate else 0.0, "mean_matched_iou": sum(overlaps) / len(overlaps) if overlaps else 0.0})
    return output


def plot_quality(rows: list[dict[str, float | str]], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    labels = [
        "FP32\nnative", "INT8 Q300\nnative", "INT8 Q300\n+ LK/ROI",
        "INT8 Q40\n+ LK/ROI", "INT8 thermal-Q\n+ LK/ROI", "Proposed\nsoftware",
    ]
    positions = list(range(len(rows))); width = 0.34
    precision = [100 * float(row["precision_proxy"]) for row in rows]
    matched_iou = [100 * float(row["mean_matched_iou"]) for row in rows]
    ax.bar([p-width/2 for p in positions], precision, width, label="Precision proxy", color="#4E79A7")
    ax.bar([p+width/2 for p in positions], matched_iou, width, label="Matched IoU", color="#59A14F")
    ax.set(ylim=(0, 105), ylabel="Agreement (%)", title="Relative agreement with FP32 reference")
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=0.22); ax.legend(loc="upper right")
    fig.subplots_adjust(bottom=0.25)
    fig.text(0.5, 0.025, "Pseudo-label agreement at IoU ≥ 0.5; it is not manually annotated accuracy.", ha="center", fontsize=7, color="#555555")
    save(fig, stem)


def write_summary(runs: dict[str, list[dict[str, str]]], quality_rows: list[dict[str, float | str]], output: Path) -> None:
    quality_by_key = {str(row["run"]): row for row in quality_rows}
    fields = ["run", "formal_start_temp_c", "mean_temp_c", "max_temp_c", "detector_calls", "detector_rate", "median_inference_latency_ms", "p95_inference_latency_ms", "active_protection_frames", "precision_proxy", "mean_matched_iou"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for key, rows in runs.items():
            temps = [num(row.get("temp_c")) for row in rows]; temps = [x for x in temps if x is not None]
            inferred = [row for row in rows if truth(row.get("did_infer"))]
            latency = [num(row.get("latency_ms")) for row in inferred]; latency = [x for x in latency if x is not None]
            q = quality_by_key[key]
            writer.writerow({"run":key, "formal_start_temp_c":rows[0].get("temp_c"), "mean_temp_c":f"{sum(temps)/len(temps):.3f}", "max_temp_c":f"{max(temps):.3f}", "detector_calls":len(inferred), "detector_rate":f"{len(inferred)/len(rows):.5f}", "median_inference_latency_ms":f"{median(latency):.3f}", "p95_inference_latency_ms":f"{sorted(latency)[round(.95*(len(latency)-1))]:.3f}", "active_protection_frames":sum(truth(row.get("under_voltage")) or truth(row.get("currently_throttled")) or truth(row.get("soft_temp_limit")) for row in rows), "precision_proxy":f"{float(q['precision_proxy']):.5f}", "mean_matched_iou":f"{float(q['mean_matched_iou']):.5f}"})


def write_coverage(rows: list[dict[str, str]], output: Path) -> None:
    columns = ["control_thermal_state", "inference_interval", "query_budget", "resolved_input_resolution"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "field", "value", "frames"]); writer.writeheader()
        for source, records in rows:
            for column in columns:
                for value, count in Counter(row.get(column, "") for row in records).items():
                    writer.writerow({"source":source, "field":column, "value":value, "frames":count})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    run_keys = list(LABELS)
    runs = {key: read_csv(args.suite_dir / key / "runtime.csv") for key in run_keys}
    profiles = {key: read_csv(args.suite_dir / key / "runtime_profile.csv") for key in run_keys}
    activation = read_csv(args.activation_dir / "runtime.csv")
    plot_timeseries(runs, "temp_c", args.output_dir / "temperature_timeseries", "CPU temperature: final thermal matrix", "CPU temperature (°C)", thresholds=True)
    plot_timeseries(runs, "arm_clock_mhz", args.output_dir / "arm_clock_timeseries", "ARM clock: final thermal matrix", "ARM clock (MHz)")
    plot_raw_and_rolling(runs, args.output_dir / "inference_latency_timeseries", "Per-frame inference latency (tracking-only frames = 0)", "Per-frame latency (ms)", mode="latency")
    plot_raw_and_rolling(runs, args.output_dir / "loop_fps_timeseries", "End-to-end loop FPS", "Loop FPS", mode="fps")
    plot_full_latency(profiles, args.output_dir / "full_latency_timeseries")
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); xs, ys = elapsed_series(activation, "temp_c")
    ax.plot(xs, ys, color="#AF7AA1", lw=1.3); ax.axhline(58, color="#C17C00", ls="--", lw=.75); ax.axhline(66, color="#B4471A", ls="--", lw=.75)
    ax.set(xlim=(0,10), xlabel="Elapsed logged-run time (min)", ylabel="CPU temperature (°C)", title="Supplementary proposed-controller activation attempt")
    ax.grid(axis="y", alpha=.22); fig.text(.5,.005,"Warm-up to 65°C occurred before logging; logged segment did not reach hot/critical state.",ha="center",fontsize=7,color="#555555"); save(fig,args.output_dir/"proposed_activation_trace")
    q = quality(args.suite_dir, run_keys); plot_quality(q, args.output_dir / "quality_pseudo_agreement")
    write_summary(runs, q, args.output_dir / "formal_matrix_summary.csv")
    with (args.output_dir / "quality_pseudo_agreement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(q[0])); writer.writeheader(); writer.writerows(q)
    write_coverage([("formal_proposed", runs["r01_06_proposed_software"]), ("activation_supplement", activation)], args.output_dir / "proposed_control_coverage.csv")


if __name__ == "__main__":
    main()
