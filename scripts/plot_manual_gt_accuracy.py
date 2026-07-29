#!/usr/bin/env python3
"""Evaluate final-matrix detections against LabelImg Pascal-VOC annotations.

The annotations are drawn on the exact 640x640 detector canvas.  Each method
is evaluated on the same labelled frame IDs using class-aware, one-to-one IoU
matching.  The script writes source-data CSV files and a publication-ready
precision/recall/F1 comparison figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
LABEL_ALIASES = {"people": "person"}
RUNS = [
    ("r01_01_fp32_native", "FP32\nnative"),
    ("r01_02_int8_dynamic_q300_native", "INT8 Q300\nnative"),
    ("r01_03_int8_dynamic_q300_lk_roi", "INT8 Q300\nLK+ROI"),
    ("r01_04_int8_dynamic_q40_lk_roi", "INT8 Q40\nLK+ROI"),
    ("r01_05_int8_dynamic_qthermal_lk_roi", "INT8 Qthermal\nLK+ROI"),
    ("r01_06_proposed_software", "Proposed\nsoftware"),
]
COLORS = {"Precision": "#3B82B6", "Recall": "#E78B39", "F1": "#4F9D69"}


@dataclass(frozen=True)
class Box:
    label: str
    xyxy: tuple[float, float, float, float]
    score: float = 1.0


def canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(label.strip().lower(), label.strip().lower())


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def read_voc_annotations(directory: Path) -> dict[int, list[Box]]:
    annotations: dict[int, list[Box]] = {}
    for path in sorted(directory.glob("frame_*.xml")):
        frame_id = int(path.stem.split("_")[-1])
        root = ET.parse(path).getroot()
        boxes: list[Box] = []
        for obj in root.findall("object"):
            name = canonical_label(obj.findtext("name", default=""))
            node = obj.find("bndbox")
            if node is None:
                continue
            boxes.append(Box(name, tuple(float(node.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
        annotations[frame_id] = boxes
    if not annotations:
        raise RuntimeError(f"no LabelImg Pascal-VOC XML files found in {directory}")
    return annotations


def read_predictions(path: Path, frame_ids: set[int]) -> dict[int, list[Box]]:
    result: dict[int, list[Box]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        frame_id = int(row["frame_id"])
        if frame_id not in frame_ids:
            continue
        detections: list[Box] = []
        for det in row.get("detections", []):
            class_id = int(det["class_id"])
            label = COCO80[class_id] if 0 <= class_id < len(COCO80) else str(class_id)
            detections.append(Box(label, tuple(float(v) for v in det["bbox"]), float(det["score"])))
        result[frame_id] = detections
    missing = sorted(frame_ids - result.keys())
    if missing:
        raise RuntimeError(f"{path.parent.name}: no output rows for labelled frame IDs {missing}")
    return result


def score_frame(gt: list[Box], pred: list[Box], threshold: float) -> tuple[int, int, int, list[float]]:
    unmatched_gt = set(range(len(gt)))
    matched_ious: list[float] = []
    true_positive = 0
    for candidate in sorted(pred, key=lambda box: box.score, reverse=True):
        options = [(iou(candidate, gt[idx]), idx) for idx in unmatched_gt if gt[idx].label == candidate.label]
        if not options:
            continue
        best_iou, best_idx = max(options)
        if best_iou >= threshold:
            unmatched_gt.remove(best_idx)
            true_positive += 1
            matched_ious.append(best_iou)
    false_positive = len(pred) - true_positive
    false_negative = len(gt) - true_positive
    return true_positive, false_positive, false_negative, matched_ious


def metric_row(run: str, display_name: str, gt: dict[int, list[Box]], pred: dict[int, list[Box]], threshold: float) -> dict[str, object]:
    tp = fp = fn = 0
    ious: list[float] = []
    for frame_id, gt_boxes in gt.items():
        a, b, c, frame_ious = score_frame(gt_boxes, pred[frame_id], threshold)
        tp += a
        fp += b
        fn += c
        ious.extend(frame_ious)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "run": run, "method": display_name.replace("\n", " "), "frames": len(gt), "gt_boxes": sum(map(len, gt.values())),
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
        "mean_matched_iou": sum(ious) / len(ious) if ious else 0.0,
    }


def export_figure(rows: list[dict[str, object]], output: Path, threshold: float, title: str, subtitle: str | None) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8, "axes.linewidth": 0.8, "axes.spines.right": False, "axes.spines.top": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.1, 3.55), constrained_layout=True)
    x = list(range(len(rows)))
    width = 0.24
    for offset, metric in zip((-width, 0, width), ("Precision", "Recall", "F1")):
        values = [float(row[metric.lower()]) * 100 for row in rows]
        bars = ax.bar([value + offset for value in x], values, width=width, color=COLORS[metric], label=metric)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=6.4)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, [str(row["method"]).replace(" ", "\n", 1) for row in rows])
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    detail = subtitle or f"{rows[0]['frames']} annotated frames · IoU ≥ {threshold:.2f} · class-aware one-to-one matching"
    ax.text(0, 1.01, detail, transform=ax.transAxes, fontsize=7, color="#4D5560")
    ax.yaxis.grid(True, color="#D9DEE4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), handlelength=1.3, columnspacing=1.8)
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 300}), (".tiff", {"dpi": 600})):
        fig.savefig(output.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, default=Path("experiments/final_thermal_20260728_132853"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/annotations/sample3_50frames_640"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--prefix", default="manual_gt", help="Output filename prefix")
    parser.add_argument("--title", default="Manual-GT detection quality on matched frames")
    parser.add_argument("--subtitle", default=None)
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")
    output_dir = args.output_dir or args.suite_dir / "analysis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = read_voc_annotations(args.annotations_dir)
    rows: list[dict[str, object]] = []
    per_frame: list[dict[str, object]] = []
    for run, display_name in RUNS:
        predictions = read_predictions(args.suite_dir / run / "runtime_detections.jsonl", set(gt))
        rows.append(metric_row(run, display_name, gt, predictions, args.iou))
        for frame_id, gt_boxes in gt.items():
            tp, fp, fn, ious = score_frame(gt_boxes, predictions[frame_id], args.iou)
            per_frame.append({"run": run, "method": display_name.replace("\n", " "), "frame_id": frame_id, "gt_boxes": len(gt_boxes), "pred_boxes": len(predictions[frame_id]), "tp": tp, "fp": fp, "fn": fn, "mean_matched_iou": sum(ious) / len(ious) if ious else 0.0})
    with (output_dir / f"{args.prefix}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / f"{args.prefix}_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame[0]))
        writer.writeheader(); writer.writerows(per_frame)
    figure_path = output_dir / f"{args.prefix}_detection_quality"
    export_figure(rows, figure_path, args.iou, args.title, args.subtitle)
    print(json.dumps(rows, indent=2))
    print(f"figure: {figure_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
