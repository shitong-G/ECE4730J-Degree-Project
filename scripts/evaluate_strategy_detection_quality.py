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
    parser.add_argument(
        "--include-teacher-summary",
        action="store_true",
        help="Include the teacher as a quality=1 baseline row in the summary and plot.",
    )
    parser.add_argument("--student-csvs", nargs="*", type=Path, default=[], help="Optional student runtime CSV files")
    parser.add_argument(
        "--teacher-cycle-frames",
        type=int,
        default=0,
        help=(
            "Map each student frame ID to frame_id %% N in the teacher log. "
            "Use this for a deterministically looped video whose teacher log contains one full cycle."
        ),
    )
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
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional PNG overview of pseudo-label agreement for every strategy.",
    )
    parser.add_argument(
        "--label-source",
        choices=["strategy", "student"],
        default="strategy",
        help=(
            "Use the runtime strategy or the student log's parent directory for "
            "plot labels. The latter distinguishes ablations sharing one strategy."
        ),
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
    frame_count = len(rows)
    did_infer = [
        str(row.get("did_infer", "")).strip().lower() in {"1", "true", "yes"}
        for row in rows
    ]
    roi_refresh = [
        str(row.get("roi_refresh_applied", "")).strip().lower()
        in {"1", "true", "yes"}
        for row in rows
    ]
    fan_pwm = [str(row.get("fan_mode", "")).strip().lower() == "pwm" for row in rows]
    active_fan_duty = [
        value
        for row, active in zip(rows, fan_pwm)
        if active
        for value in [_to_float(row.get("fan_duty_cycle"))]
        if value is not None
    ]
    tracking_rows = [
        row for row in rows if str(row.get("tracking_mode", "")).lower() == "track"
    ]
    final_row = rows[-1] if rows else {}
    cumulative_detector_count = _to_float(final_row.get("detector_invocation_count"))
    cumulative_full_count = _to_float(
        final_row.get("full_detector_invocation_count")
    )
    cumulative_roi_count = _to_float(final_row.get("roi_detector_invocation_count"))
    return {
        "runtime_frames": frame_count,
        "detector_invocation_count": (
            cumulative_detector_count
            if cumulative_detector_count is not None
            else float(sum(did_infer))
        ),
        "detector_invocation_rate": (
            cumulative_detector_count / frame_count
            if frame_count and cumulative_detector_count is not None
            else sum(did_infer) / frame_count
            if frame_count
            else None
        ),
        "full_detector_invocation_count": (
            cumulative_full_count
            if cumulative_full_count is not None
            else float(sum(infer and not roi for infer, roi in zip(did_infer, roi_refresh)))
        ),
        "full_detector_invocation_rate": (
            cumulative_full_count / frame_count
            if frame_count and cumulative_full_count is not None
            else sum(infer and not roi for infer, roi in zip(did_infer, roi_refresh)) / frame_count
            if frame_count
            else None
        ),
        "roi_detector_invocation_count": (
            cumulative_roi_count
            if cumulative_roi_count is not None
            else float(sum(roi_refresh))
        ),
        "roi_detector_invocation_rate": (
            cumulative_roi_count / frame_count
            if frame_count and cumulative_roi_count is not None
            else sum(roi_refresh) / frame_count
            if frame_count
            else None
        ),
        "tracking_frame_ratio": len(tracking_rows) / frame_count if frame_count else None,
        "tracking_failure_ratio_mean": _mean(
            _series(tracking_rows, "tracking_failure_ratio")
        ),
        "tracking_mean_quality": _mean(
            _series(tracking_rows, "tracking_mean_quality")
        ),
        "tracking_failed_box_count_mean": _mean(
            _series(tracking_rows, "tracking_failed_box_count")
        ),
        "latency_ms_mean": _mean(_series(rows, "latency_ms", positive=True)),
        "latency_ms_p95": _percentile(_series(rows, "latency_ms", positive=True), 0.95),
        "actual_inference_fps_mean": _mean(_series(rows, "actual_inference_fps")),
        "loop_fps_mean": _mean(_series(rows, "loop_fps")),
        "temp_c_mean": _mean(_series(rows, "temp_c")),
        "temp_c_max": max(_series(rows, "temp_c")) if _series(rows, "temp_c") else None,
        "fan_pwm_ratio": sum(fan_pwm) / frame_count if frame_count else None,
        "fan_duty_mean_when_pwm": _mean(active_fan_duty),
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
    teacher_cycle_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if teacher_cycle_frames > 0:
        missing = [frame_id for frame_id in range(teacher_cycle_frames) if frame_id not in teacher]
        if missing:
            raise ValueError(
                "Teacher cycle is incomplete: missing teacher frame IDs "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        aligned_ids = [
            (frame_id, frame_id % teacher_cycle_frames)
            for frame_id in sorted(student)
        ]
    else:
        aligned_ids = [(frame_id, frame_id) for frame_id in sorted(set(teacher) & set(student))]
    total_teacher = total_student = total_matches = 0
    infer_teacher = infer_student = infer_matches = 0
    noninfer_teacher = noninfer_student = noninfer_matches = 0
    ious: list[float] = []
    center_errors: list[float] = []
    frame_rows: list[dict[str, Any]] = []

    for frame_id, teacher_frame_id in aligned_ids:
        trow = teacher[teacher_frame_id]
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
                "student": student_path.parent.name,
                "frame_id": frame_id,
                "teacher_frame_id": teacher_frame_id,
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
        "student": student_path.parent.name,
        "strategy": next(iter(student.values())).get("strategy") if student else None,
        "common_frames": len(aligned_ids),
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
        "lost_object_frame_ratio": (
            sum(
                1
                for row in frame_rows
                if row["teacher_count"] > 0 and row["student_count"] == 0
            )
            / sum(1 for row in frame_rows if row["teacher_count"] > 0)
            if any(row["teacher_count"] > 0 for row in frame_rows)
            else 0.0
        ),
    }
    summary.update(_csv_metrics(csv_rows))
    return summary, frame_rows


def _plot_summary(
    rows: list[dict[str, Any]],
    output_path: Path,
    label_source: str = "strategy",
) -> None:
    """Plot teacher-agreement metrics, not ground-truth accuracy or mAP."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        str(
            row["student"]
            if label_source == "student"
            else row["strategy"] or row["student"]
        )
        for row in rows
    ]
    positions = list(range(len(rows)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    width = 0.25
    metrics = [
        ("pseudo_recall", "All-frame pseudo recall"),
        ("precision_proxy", "All-frame precision proxy"),
        ("mean_matched_iou", "Mean matched IoU"),
    ]
    for metric_index, (metric, label) in enumerate(metrics):
        axes[0, 0].bar(
            [position + (metric_index - 1) * width for position in positions],
            [float(row[metric]) for row in rows],
            width=width,
            label=label,
        )
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_xticks(positions, labels, rotation=40, ha="right")
    axes[0, 0].set_ylabel("Agreement with native-640 teacher")
    axes[0, 0].set_title("End-to-end output quality")
    axes[0, 0].legend(fontsize=8)

    detector_metrics = [
        ("infer_frame_pseudo_recall", "Detector-frame pseudo recall"),
        ("infer_frame_precision_proxy", "Detector-frame precision proxy"),
    ]
    for metric_index, (metric, label) in enumerate(detector_metrics):
        axes[0, 1].bar(
            [position + (metric_index - 0.5) * 0.36 for position in positions],
            [float(row[metric]) for row in rows],
            width=0.36,
            label=label,
        )
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_xticks(positions, labels, rotation=40, ha="right")
    axes[0, 1].set_ylabel("Agreement with native-640 teacher")
    axes[0, 1].set_title("Detector-frame quality")
    axes[0, 1].legend(fontsize=8)

    pipeline_fps = [float(row.get("loop_fps_mean") or 0.0) for row in rows]
    invocation_pct = [
        100.0 * float(row.get("detector_invocation_rate") or 0.0)
        for row in rows
    ]
    axes[1, 0].bar(
        [position - 0.2 for position in positions],
        pipeline_fps,
        width=0.4,
        color="#4e79a7",
        label="Pipeline FPS",
    )
    throughput_right = axes[1, 0].twinx()
    throughput_right.bar(
        [position + 0.2 for position in positions],
        invocation_pct,
        width=0.4,
        color="#e15759",
        label="Detector invocation",
    )
    axes[1, 0].set_xticks(positions, labels, rotation=40, ha="right")
    axes[1, 0].set_ylabel("Pipeline FPS")
    throughput_right.set_ylabel("Detector invocation (%)")
    axes[1, 0].set_title("Throughput and detector usage")
    axes[1, 0].legend(loc="upper left", fontsize=8)
    throughput_right.legend(loc="upper right", fontsize=8)

    temp_mean = [float(row.get("temp_c_mean") or 0.0) for row in rows]
    temp_max = [float(row.get("temp_c_max") or 0.0) for row in rows]
    fan_pwm_pct = [
        100.0 * float(row.get("fan_pwm_ratio") or 0.0) for row in rows
    ]
    axes[1, 1].bar(
        [position - 0.2 for position in positions],
        temp_mean,
        width=0.4,
        color="#f28e2b",
        label="Mean temp",
    )
    axes[1, 1].bar(
        [position + 0.2 for position in positions],
        temp_max,
        width=0.4,
        color="#e15759",
        label="Max temp",
    )
    thermal_right = axes[1, 1].twinx()
    thermal_right.plot(
        positions,
        fan_pwm_pct,
        color="#59a14f",
        marker="o",
        linewidth=1.5,
        label="PWM fan frames",
    )
    axes[1, 1].set_xticks(positions, labels, rotation=40, ha="right")
    axes[1, 1].set_ylabel("CPU temperature (C)")
    thermal_right.set_ylabel("PWM fan frames (%)")
    axes[1, 1].set_title("Sustained thermal behavior")
    axes[1, 1].legend(loc="upper left", fontsize=8)
    thermal_right.legend(loc="upper right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    teacher = _load_jsonl(args.teacher)
    if args.student_csvs and len(args.student_csvs) != len(args.students):
        raise ValueError("--student-csvs must provide one CSV for every --students entry")
    summaries: list[dict[str, Any]] = []
    if args.include_teacher_summary:
        teacher_summary: dict[str, Any] = {
            "student": args.teacher.parent.name,
            "strategy": next(iter(teacher.values())).get("strategy") if teacher else None,
            "common_frames": len(teacher),
            "teacher_frames": len(teacher),
            "student_frames": len(teacher),
            "pseudo_recall": 1.0,
            "precision_proxy": 1.0,
            "mean_matched_iou": 1.0,
            "mean_center_error_norm": 0.0,
            "detection_count_ratio": 1.0,
            "infer_frame_pseudo_recall": 1.0,
            "infer_frame_precision_proxy": 1.0,
            "noninfer_frame_pseudo_recall": 1.0,
            "noninfer_frame_precision_proxy": 1.0,
            "lost_object_frame_ratio": 0.0,
        }
        teacher_summary.update(_csv_metrics(_load_csv(args.teacher_csv)))
        summaries.append(teacher_summary)
    frame_rows: list[dict[str, Any]] = []
    for index, student_path in enumerate(args.students):
        student = _load_jsonl(student_path)
        csv_rows = _load_csv(args.student_csvs[index]) if args.student_csvs else []
        summary, rows = _compare_one(
            teacher=teacher,
            student=student,
            student_path=student_path,
            csv_rows=csv_rows,
            iou_threshold=args.iou_threshold,
            teacher_cycle_frames=max(0, args.teacher_cycle_frames),
        )
        summaries.append(summary)
        frame_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = list(
        dict.fromkeys(key for row in summaries for key in row.keys())
    )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    if frame_rows:
        with args.matches_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)
    if args.plot_output is not None:
        _plot_summary(summaries, args.plot_output, args.label_source)
    print(f"Saved summary: {args.output}")
    print(f"Saved per-frame details: {args.matches_output}")
    if args.plot_output is not None:
        print(f"Saved plot: {args.plot_output}")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
