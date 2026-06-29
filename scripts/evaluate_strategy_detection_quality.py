#!/usr/bin/env python3
"""Compare strategy detection logs against a native RT-DETR teacher."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare per-frame detection boxes across strategies")
    parser.add_argument("--teacher", type=Path, required=True, help="Native *_detections.jsonl")
    parser.add_argument("--students", nargs="+", type=Path, required=True, help="Student *_detections.jsonl files")
    parser.add_argument("--teacher-csv", type=Path, default=None, help="Optional native runtime CSV")
    parser.add_argument("--student-csvs", nargs="*", type=Path, default=[], help="Optional student runtime CSV files")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/strategy_detection_quality_summary.csv"),
    )
    parser.add_argument(
        "--matches-output",
        type=Path,
        default=Path("experiments/results/strategy_detection_quality_frames.csv"),
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["frame_id"])] = row
    return rows


def _load_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _to_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _series(rows: list[dict[str, str]], column: str, *, positive: bool = False) -> list[float]:
    values = [value for value in (_to_float(row.get(column)) for row in rows) if value is not None]
    return [value for value in values if value > 0.0] if positive else values


def _bool_ratio(rows: list[dict[str, str]], column: str) -> float | None:
    values = [value for value in (_to_bool(row.get(column)) for row in rows) if value is not None]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _norm_box(det: dict[str, Any], resolution: int) -> list[float]:
    box = [float(value) for value in det["bbox"]]
    scale = float(max(1, resolution))
    return [box[0] / scale, box[1] / scale, box[2] / scale, box[3] / scale]


def _center_error(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _match(
    teacher_row: dict[str, Any],
    student_row: dict[str, Any],
    iou_threshold: float,
) -> tuple[int, list[float], list[float]]:
    teacher_res = int(teacher_row.get("resolved_input_resolution") or teacher_row.get("input_resolution") or 640)
    student_res = int(student_row.get("resolved_input_resolution") or student_row.get("input_resolution") or 640)
    candidates: list[tuple[float, int, int]] = []
    teacher = teacher_row.get("detections") or []
    student = student_row.get("detections") or []
    for ti, td in enumerate(teacher):
        tbox = _norm_box(td, teacher_res)
        for si, sd in enumerate(student):
            if int(td.get("class_id", -1)) != int(sd.get("class_id", -2)):
                continue
            sbox = _norm_box(sd, student_res)
            iou = _iou(tbox, sbox)
            if iou >= iou_threshold:
                candidates.append((iou, ti, si))
    used_t: set[int] = set()
    used_s: set[int] = set()
    ious: list[float] = []
    center_errors: list[float] = []
    for iou, ti, si in sorted(candidates, reverse=True):
        if ti in used_t or si in used_s:
            continue
        used_t.add(ti)
        used_s.add(si)
        ious.append(iou)
        center_errors.append(
            _center_error(
                _norm_box(teacher[ti], teacher_res),
                _norm_box(student[si], student_res),
            )
        )
    return len(ious), ious, center_errors


def _csv_metrics(rows: list[dict[str, str]]) -> dict[str, float | None]:
    return {
        "latency_ms_mean": _mean(_series(rows, "latency_ms", positive=True)),
        "latency_ms_p95": _percentile(_series(rows, "latency_ms", positive=True), 0.95),
        "actual_inference_fps_mean": _mean(_series(rows, "actual_inference_fps")),
        "loop_fps_mean": _mean(_series(rows, "loop_fps")),
        "temp_c_mean": _mean(_series(rows, "temp_c")),
        "temp_c_max": max(_series(rows, "temp_c")) if _series(rows, "temp_c") else None,
        "power_w_mean": _mean(_series(rows, "power_w")),
        "soft_temp_limit_ratio": _bool_ratio(rows, "soft_temp_limit"),
        "currently_throttled_ratio": _bool_ratio(rows, "currently_throttled"),
        "arm_freq_capped_ratio": _bool_ratio(rows, "arm_freq_capped"),
    }


def _compare_one(
    *,
    teacher: dict[int, dict[str, Any]],
    student: dict[int, dict[str, Any]],
    student_path: Path,
    csv_rows: list[dict[str, str]],
    iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common_ids = sorted(set(teacher) & set(student))
    total_teacher = total_student = total_matches = 0
    infer_teacher = infer_student = infer_matches = 0
    noninfer_teacher = noninfer_student = noninfer_matches = 0
    ious: list[float] = []
    center_errors: list[float] = []
    frame_rows: list[dict[str, Any]] = []

    for frame_id in common_ids:
        trow = teacher[frame_id]
        srow = student[frame_id]
        matches, frame_ious, frame_center_errors = _match(trow, srow, iou_threshold)
        teacher_count = len(trow.get("detections") or [])
        student_count = len(srow.get("detections") or [])
        total_teacher += teacher_count
        total_student += student_count
        total_matches += matches
        ious.extend(frame_ious)
        center_errors.extend(frame_center_errors)
        did_infer = bool(srow.get("did_infer"))
        if did_infer:
            infer_teacher += teacher_count
            infer_student += student_count
            infer_matches += matches
        else:
            noninfer_teacher += teacher_count
            noninfer_student += student_count
            noninfer_matches += matches
        frame_rows.append(
            {
                "student": student_path.stem,
                "frame_id": frame_id,
                "did_infer": did_infer,
                "tracking_mode": srow.get("tracking_mode"),
                "teacher_count": teacher_count,
                "student_count": student_count,
                "matches": matches,
                "pseudo_recall": matches / teacher_count if teacher_count else 1.0,
                "precision_proxy": matches / student_count if student_count else (1.0 if not teacher_count else 0.0),
                "mean_iou": mean(frame_ious) if frame_ious else 0.0,
                "mean_center_error_norm": mean(frame_center_errors) if frame_center_errors else 0.0,
            }
        )

    summary: dict[str, Any] = {
        "student": student_path.stem,
        "strategy": next(iter(student.values())).get("strategy") if student else None,
        "common_frames": len(common_ids),
        "teacher_frames": len(teacher),
        "student_frames": len(student),
        "pseudo_recall": total_matches / total_teacher if total_teacher else 1.0,
        "precision_proxy": total_matches / total_student if total_student else 0.0,
        "mean_matched_iou": mean(ious) if ious else 0.0,
        "mean_center_error_norm": mean(center_errors) if center_errors else 0.0,
        "detection_count_ratio": total_student / total_teacher if total_teacher else 0.0,
        "infer_frame_pseudo_recall": infer_matches / infer_teacher if infer_teacher else 1.0,
        "infer_frame_precision_proxy": infer_matches / infer_student if infer_student else 0.0,
        "noninfer_frame_pseudo_recall": noninfer_matches / noninfer_teacher if noninfer_teacher else 1.0,
        "noninfer_frame_precision_proxy": noninfer_matches / noninfer_student if noninfer_student else 0.0,
    }
    summary.update(_csv_metrics(csv_rows))
    return summary, frame_rows


def main() -> None:
    args = parse_args()
    teacher = _load_jsonl(args.teacher)
    csv_by_stem = {path.stem: _load_csv(path) for path in args.student_csvs}
    summaries: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for student_path in args.students:
        student = _load_jsonl(student_path)
        csv_rows = csv_by_stem.get(student_path.stem.replace("_detections", ""))
        if csv_rows is None:
            csv_rows = csv_by_stem.get(student_path.stem, [])
        summary, rows = _compare_one(
            teacher=teacher,
            student=student,
            student_path=student_path,
            csv_rows=csv_rows,
            iou_threshold=args.iou_threshold,
        )
        summaries.append(summary)
        frame_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with args.matches_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)
    print(f"Saved summary: {args.output}")
    print(f"Saved per-frame details: {args.matches_output}")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
