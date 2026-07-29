#!/usr/bin/env python3
"""Create dependency-free SVG figures and summaries for final thermal runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import median


LABELS = {
    "r01_01_fp32_native": "FP32 / native",
    "r01_02_int8_dynamic_q300_native": "INT8 Q300 / native",
    "r01_03_int8_dynamic_q300_lk_roi": "INT8 Q300 / LK+ROI",
    "r01_04_int8_dynamic_q40_lk_roi": "INT8 Q40 / LK+ROI",
    "r01_05_int8_dynamic_qthermal_lk_roi": "INT8 thermal-Q / LK+ROI",
    "r01_06_proposed_software": "Proposed thermal / LK+ROI",
    "proposed_rerun": "Proposed thermal / LK+ROI (rerun)",
}
COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#af7aa1"]
WIDTH, HEIGHT, LEFT, RIGHT, TOP, BOTTOM = 1280, 720, 86, 42, 50, 104


def number(value: str | None) -> float | None:
    try:
        parsed = float(value or "")
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def line_series(rows: list[dict[str, str]], column: str, inferred_only: bool) -> list[tuple[float, float]]:
    origin = number(rows[0].get("timestamp")) or 0.0
    pairs = []
    for row in rows:
        if inferred_only and row.get("did_infer", "").lower() != "true":
            continue
        stamp, value = number(row.get("timestamp")), number(row.get(column))
        if stamp is not None and value is not None:
            pairs.append(((stamp - origin) / 60.0, value))
    return pairs


def percentile(values: list[float], p: float) -> float | None:
    return sorted(values)[round((len(values) - 1) * p)] if values else None


def svg_header(title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>text{font-family:Arial,sans-serif;fill:#1f2937}.tick{font-size:12px}.legend{font-size:13px}.title{font-size:22px;font-weight:bold}.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#cbd5e1;stroke-width:1;stroke-dasharray:4 4}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{LEFT}" y="30">{html.escape(title)}</text>',
    ]


def write_line_chart(
    runs: dict[str, list[dict[str, str]]], column: str, output: Path,
    title: str, ylabel: str, *, inferred_only: bool = False, thresholds: list[tuple[float, str, str]] | None = None,
) -> None:
    data = {key: line_series(rows, column, inferred_only) for key, rows in runs.items()}
    values = [y for points in data.values() for _, y in points]
    if not values:
        raise ValueError(f"No {column} values available")
    ymin, ymax = min(values), max(values)
    if thresholds:
        ymin, ymax = min(ymin, min(x[0] for x in thresholds)), max(ymax, max(x[0] for x in thresholds))
    span = max(1.0, ymax - ymin)
    ymin -= 0.06 * span
    ymax += 0.10 * span
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    sx = lambda x: LEFT + min(20.0, max(0.0, x)) / 20.0 * plot_w
    sy = lambda y: TOP + (ymax - y) / (ymax - ymin) * plot_h
    parts = svg_header(title)
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        y = sy(value)
        parts += [f'<line class="grid" x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH-RIGHT}" y2="{y:.1f}"/>', f'<text class="tick" x="{LEFT-10}" y="{y+4:.1f}" text-anchor="end">{value:.0f}</text>']
    for minute in range(0, 21, 5):
        x = sx(minute)
        parts += [f'<line class="grid" x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{HEIGHT-BOTTOM}"/>', f'<text class="tick" x="{x:.1f}" y="{HEIGHT-BOTTOM+22}" text-anchor="middle">{minute}</text>']
    parts += [f'<line class="axis" x1="{LEFT}" y1="{HEIGHT-BOTTOM}" x2="{WIDTH-RIGHT}" y2="{HEIGHT-BOTTOM}"/>', f'<line class="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT-BOTTOM}"/>']
    if thresholds:
        for value, label, color in thresholds:
            y = sy(value)
            parts += [f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH-RIGHT}" y2="{y:.1f}" stroke="{color}" stroke-width="1" stroke-dasharray="6 4"/>', f'<text class="tick" x="{WIDTH-RIGHT-2}" y="{y-4:.1f}" text-anchor="end" fill="{color}">{html.escape(label)}</text>']
    for index, (key, points) in enumerate(data.items()):
        stride = max(1, math.ceil(len(points) / 1400))
        point_text = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points[::stride])
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{COLORS[index]}" stroke-width="1.35" opacity="0.92"/>')
        ly = TOP + 18 * index
        parts += [f'<line x1="{WIDTH-280}" y1="{ly-4}" x2="{WIDTH-260}" y2="{ly-4}" stroke="{COLORS[index]}" stroke-width="3"/>', f'<text class="legend" x="{WIDTH-254}" y="{ly}">{html.escape(LABELS[key])}</text>']
    parts += [f'<text class="tick" x="{LEFT + plot_w/2}" y="{HEIGHT-32}" text-anchor="middle">Elapsed formal-run time (min)</text>', f'<text class="tick" transform="translate(20 {TOP+plot_h/2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>', "</svg>"]
    output.write_text("\n".join(parts), encoding="utf-8")


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a, area_b = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]), max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def match(teacher: list[dict], student: list[dict]) -> tuple[int, list[float]]:
    candidates = []
    for ti, td in enumerate(teacher):
        for si, sd in enumerate(student):
            if td.get("class_id") == sd.get("class_id"):
                score = iou(td["bbox"], sd["bbox"])
                if score >= 0.5:
                    candidates.append((score, ti, si))
    used_t, used_s, scores = set(), set(), []
    for score, ti, si in sorted(candidates, reverse=True):
        if ti not in used_t and si not in used_s:
            used_t.add(ti); used_s.add(si); scores.append(score)
    return len(scores), scores


def load_jsonl(path: Path) -> dict[int, dict]:
    return {int(row["frame_id"]): row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def write_quality_chart(teacher_path: Path, detection_paths: dict[str, Path], output_dir: Path) -> None:
    teacher = load_jsonl(teacher_path)
    rows = []
    for key, detection_path in detection_paths.items():
        student = load_jsonl(detection_path)
        common = sorted(set(teacher) & set(student))
        total_teacher = total_student = matched = 0
        ious: list[float] = []
        for frame_id in common:
            reference, candidate = teacher[frame_id].get("detections", []), student[frame_id].get("detections", [])
            count, scores = match(reference, candidate)
            total_teacher += len(reference); total_student += len(candidate); matched += count; ious.extend(scores)
        rows.append({"run": key, "common_frames": len(common), "precision_proxy": matched / total_student if total_student else 0.0, "mean_matched_iou": sum(ious)/len(ious) if ious else 0.0})
    with (output_dir / "quality_pseudo_agreement_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    parts = svg_header("Relative detection agreement with FP32 teacher (first common frames)")
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    metrics = [("precision_proxy", "Precision proxy", "#f28e2b"), ("mean_matched_iou", "Matched IoU", "#59a14f")]
    n, group_w = len(rows), plot_w / len(rows)
    for tick in range(0, 101, 20):
        y = TOP + (100-tick)/100*plot_h
        parts += [f'<line class="grid" x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH-RIGHT}" y2="{y:.1f}"/>', f'<text class="tick" x="{LEFT-10}" y="{y+4:.1f}" text-anchor="end">{tick}%</text>']
    for idx, row in enumerate(rows):
        base = LEFT + idx * group_w + group_w*0.18
        bar_w = group_w*0.18
        for mi, (field, _, color) in enumerate(metrics):
            value = float(row[field])*100; h = value/100*plot_h; x = base + mi*bar_w
            parts += [f'<rect x="{x:.1f}" y="{TOP+plot_h-h:.1f}" width="{bar_w-3:.1f}" height="{h:.1f}" fill="{color}"/>', f'<text class="tick" x="{x+(bar_w-3)/2:.1f}" y="{TOP+plot_h-h-5:.1f}" text-anchor="middle">{value:.1f}</text>']
        parts.append(f'<text class="tick" x="{LEFT+(idx+.5)*group_w:.1f}" y="{HEIGHT-BOTTOM+22}" text-anchor="middle">{html.escape(LABELS[row["run"]])}</text>')
    for mi, (_, label, color) in enumerate(metrics):
        x = LEFT + mi*180
        parts += [f'<rect x="{x}" y="{HEIGHT-48}" width="14" height="14" fill="{color}"/>', f'<text class="legend" x="{x+20}" y="{HEIGHT-36}">{label}</text>']
    parts += [f'<text class="tick" x="{LEFT + plot_w/2}" y="{HEIGHT-12}" text-anchor="middle">Not ground-truth accuracy: FP32 pseudo-label agreement at IoU ≥ 0.5</text>', "</svg>"]
    (output_dir / "quality_pseudo_agreement.svg").write_text("\n".join(parts), encoding="utf-8")


def write_runtime_summary(runs: dict[str, list[dict[str, str]]], output: Path) -> None:
    fields = ["run", "formal_start_temp_c", "mean_temp_c", "max_temp_c", "p95_temp_c", "median_arm_clock_mhz", "detector_calls", "detector_rate", "median_inference_latency_ms", "p95_inference_latency_ms"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for key, rows in runs.items():
            vals = lambda col, subset=rows: [x for x in (number(row.get(col)) for row in subset) if x is not None]
            inferred = [row for row in rows if row.get("did_infer", "").lower() == "true"]
            temps, clocks, latencies = vals("temp_c"), vals("arm_clock_mhz"), vals("latency_ms", inferred)
            writer.writerow({"run":key, "formal_start_temp_c":rows[0].get("temp_c"), "mean_temp_c":f"{sum(temps)/len(temps):.3f}", "max_temp_c":f"{max(temps):.3f}", "p95_temp_c":f"{percentile(temps,.95):.3f}", "median_arm_clock_mhz":f"{median(clocks):.3f}" if clocks else "", "detector_calls":len(inferred), "detector_rate":f"{len(inferred)/len(rows):.5f}", "median_inference_latency_ms":f"{median(latencies):.3f}", "p95_inference_latency_ms":f"{percentile(latencies,.95):.3f}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--proposed-rerun-dir", type=Path, default=None,
        help="Replace r01_06 in the generated figures with this standalone proposed rerun.",
    )
    args = parser.parse_args()
    output = args.output_dir or args.suite_dir / "visualizations"; output.mkdir(parents=True, exist_ok=True)
    runs = {p.parent.name: read_csv(p) for p in sorted(args.suite_dir.glob("r01_*/runtime.csv")) if p.parent.name in LABELS}
    detection_paths = {key: args.suite_dir / key / "runtime_detections.jsonl" for key in runs}
    if args.proposed_rerun_dir is not None:
        rerun_csv = args.proposed_rerun_dir / "runtime.csv"
        rerun_detections = args.proposed_rerun_dir / "runtime_detections.jsonl"
        if not rerun_csv.exists() or not rerun_detections.exists():
            raise FileNotFoundError("--proposed-rerun-dir requires runtime.csv and runtime_detections.jsonl")
        runs.pop("r01_06_proposed_software", None)
        detection_paths.pop("r01_06_proposed_software", None)
        runs["proposed_rerun"] = read_csv(rerun_csv)
        detection_paths["proposed_rerun"] = rerun_detections
    write_line_chart(runs, "temp_c", output/"temperature_timeseries.svg", "CPU temperature over formal run", "CPU temperature (°C)", thresholds=[(58,"warm 58°C","#f3a712"),(66,"hot 66°C","#e4572e"),(76,"critical 76°C","#b22222")])
    write_line_chart(runs, "arm_clock_mhz", output/"arm_clock_timeseries.svg", "CPU ARM clock over formal run", "ARM clock (MHz)")
    write_line_chart(runs, "latency_ms", output/"inference_latency_timeseries.svg", "RT-DETR latency (detector calls only)", "Detector latency (ms)", inferred_only=True)
    write_runtime_summary(runs, output/"runtime_summary.csv")
    write_quality_chart(
        args.suite_dir / "r01_01_fp32_native" / "runtime_detections.jsonl",
        detection_paths,
        output,
    )
    print(f"Saved visualizations to {output}")


if __name__ == "__main__":
    main()
