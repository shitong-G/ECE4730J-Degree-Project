#!/usr/bin/env python3
"""Evaluate temporal quality of frame skipping vs LK tracking.

The teacher runs RT-DETR on every processed frame. Student policies run RT-DETR
only on keyframes and produce detections on intermediate frames either by
reusing the last detector output or by updating boxes with sparse LK tracking.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.inference.onnx_engine import ONNXRTDETREngine
from scene_runtime.inference.postprocess import Detection
from scene_runtime.tracking import LKTrackingReport, SparseLKBoxTracker
from scene_runtime.utils.video import FrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal tracking quality evaluation")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx")
    parser.add_argument(
        "--model-template",
        default=None,
        help="Optional model template containing {resolution}",
    )
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--keyframe-interval", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--policies",
        default="skip_reuse,lk_track",
        help="Comma-separated policies: skip_reuse,lk_track",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "results" / "tracking_quality",
    )
    return parser.parse_args()


def _parse_policies(text: str) -> list[str]:
    policies = [item.strip() for item in text.split(",") if item.strip()]
    allowed = {"skip_reuse", "lk_track"}
    unknown = sorted(set(policies) - allowed)
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}")
    return policies


def _create_engine(args: argparse.Namespace) -> ONNXRTDETREngine:
    model_paths = None
    model_path = str(args.model)
    if args.model_template:
        model_path = args.model_template.format(resolution=args.resolution)
        model_paths = {args.resolution: model_path}
    engine = ONNXRTDETREngine(
        model_path=model_path,
        model_paths_by_resolution=model_paths,
        enable_thread_sessions=True,
        thread_session_counts=[args.threads],
    )
    engine.load()
    return engine


def _load_frames(video: Path, max_frames: int, stride: int) -> list[tuple[int, object]]:
    frames: list[tuple[int, object]] = []
    frame_cap = None if max_frames <= 0 else max_frames * max(1, stride)
    source = FrameSource(video, synthetic=False, max_frames=frame_cap, loop=False)
    try:
        for source_frame_id, frame in enumerate(source):
            if source_frame_id % max(1, stride) == 0:
                frames.append((source_frame_id, frame))
                if max_frames > 0 and len(frames) >= max_frames:
                    break
    finally:
        source.release()
    return frames


def _infer(
    engine: ONNXRTDETREngine,
    frame: object,
    action: RuntimeAction,
    score_threshold: float,
) -> tuple[list[Detection], float]:
    t0 = time.perf_counter()
    detections = [
        det for det in engine.infer(frame, action) if det.score >= score_threshold
    ]
    return detections, (time.perf_counter() - t0) * 1000.0


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _center_error(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    resolution: int,
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / max(1.0, float(resolution))


def _match(
    teacher: list[Detection],
    student: list[Detection],
    iou_threshold: float,
) -> list[tuple[int, int, float, float]]:
    candidates: list[tuple[float, int, int]] = []
    for ti, td in enumerate(teacher):
        for si, sd in enumerate(student):
            if td.class_id != sd.class_id:
                continue
            iou = _iou(td.bbox, sd.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, ti, si))
    matches: list[tuple[int, int, float, float]] = []
    used_teacher: set[int] = set()
    used_student: set[int] = set()
    for iou, ti, si in sorted(candidates, reverse=True):
        if ti in used_teacher or si in used_student:
            continue
        used_teacher.add(ti)
        used_student.add(si)
        matches.append((ti, si, iou, 0.0))
    return matches


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _evaluate_policy(
    *,
    policy: str,
    frames: list[tuple[int, object]],
    teacher_detections: dict[int, list[Detection]],
    engine: ONNXRTDETREngine,
    action: RuntimeAction,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    last_detections: list[Detection] = []
    tracker = SparseLKBoxTracker() if policy == "lk_track" else None
    detector_calls = 0
    detector_latencies: list[float] = []
    total_latencies: list[float] = []
    keyframe_ious: list[float] = []
    skip_ious: list[float] = []
    keyframe_center_errors: list[float] = []
    skip_center_errors: list[float] = []
    frame_rows: list[dict[str, object]] = []
    total_teacher = total_student = total_matches = 0
    skip_teacher = skip_student = skip_matches = 0
    key_teacher = key_student = key_matches = 0

    for processed_id, (source_frame_id, frame) in enumerate(frames):
        is_keyframe = processed_id % max(1, args.keyframe_interval) == 0
        tracking_report = LKTrackingReport()
        t0 = time.perf_counter()
        detector_ms = 0.0
        if is_keyframe:
            last_detections, detector_ms = _infer(
                engine,
                frame,
                action,
                args.score_threshold,
            )
            detector_calls += 1
            detector_latencies.append(detector_ms)
            if tracker is not None:
                tracking_report = tracker.reset(
                    frame,
                    last_detections,
                    engine.last_resolved_input_resolution or args.resolution,
                )
        elif tracker is not None:
            t_track = time.perf_counter()
            last_detections, tracking_report = tracker.update(frame)
            tracking_report.tracking_ms = (time.perf_counter() - t_track) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        total_latencies.append(total_ms)

        teacher = teacher_detections[source_frame_id]
        student = last_detections
        matches = _match(teacher, student, args.iou_threshold)
        ious = [iou for _, _, iou, _ in matches]
        center_errors = [
            _center_error(teacher[ti].bbox, student[si].bbox, args.resolution)
            for ti, si, _, _ in matches
        ]
        total_teacher += len(teacher)
        total_student += len(student)
        total_matches += len(matches)
        if is_keyframe:
            key_teacher += len(teacher)
            key_student += len(student)
            key_matches += len(matches)
            keyframe_ious.extend(ious)
            keyframe_center_errors.extend(center_errors)
        else:
            skip_teacher += len(teacher)
            skip_student += len(student)
            skip_matches += len(matches)
            skip_ious.extend(ious)
            skip_center_errors.extend(center_errors)

        frame_rows.append(
            {
                "policy": policy,
                "processed_frame": processed_id,
                "source_frame": source_frame_id,
                "is_keyframe": is_keyframe,
                "teacher_count": len(teacher),
                "student_count": len(student),
                "matches": len(matches),
                "pseudo_recall": len(matches) / len(teacher) if teacher else 1.0,
                "precision_proxy": len(matches) / len(student) if student else (1.0 if not teacher else 0.0),
                "mean_iou": _safe_mean(ious),
                "mean_center_error_norm": _safe_mean(center_errors),
                "total_latency_ms": total_ms,
                "detector_latency_ms": detector_ms,
                "tracking_ms": tracking_report.tracking_ms,
                "tracking_failure_ratio": tracking_report.failure_ratio,
                "tracking_mean_quality": tracking_report.mean_quality,
                "tracking_reason": tracking_report.reason,
            }
        )

    frames_count = max(1, len(frames))
    summary = {
        "policy": policy,
        "frames": len(frames),
        "keyframe_interval": args.keyframe_interval,
        "detector_calls": detector_calls,
        "detector_invocation_rate": detector_calls / frames_count,
        "mean_total_latency_ms": _safe_mean(total_latencies),
        "mean_detector_latency_ms": _safe_mean(detector_latencies),
        "pseudo_recall": total_matches / total_teacher if total_teacher else 1.0,
        "precision_proxy": total_matches / total_student if total_student else 0.0,
        "skip_frame_pseudo_recall": skip_matches / skip_teacher if skip_teacher else 1.0,
        "skip_frame_precision_proxy": skip_matches / skip_student if skip_student else 0.0,
        "keyframe_pseudo_recall": key_matches / key_teacher if key_teacher else 1.0,
        "keyframe_precision_proxy": key_matches / key_student if key_student else 0.0,
        "mean_keyframe_iou": _safe_mean(keyframe_ious),
        "mean_skip_frame_iou": _safe_mean(skip_ious),
        "mean_keyframe_center_error_norm": _safe_mean(keyframe_center_errors),
        "mean_skip_frame_center_error_norm": _safe_mean(skip_center_errors),
        "detection_count_ratio": total_student / total_teacher if total_teacher else 0.0,
        "skip_detection_count_ratio": skip_student / skip_teacher if skip_teacher else 0.0,
    }
    return summary, frame_rows


def main() -> None:
    args = parse_args()
    if args.keyframe_interval < 1:
        raise ValueError("--keyframe-interval must be >= 1")
    policies = _parse_policies(args.policies)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(args.video, args.max_frames, args.frame_stride)
    if not frames:
        raise RuntimeError(f"No frames loaded from {args.video}")

    action = RuntimeAction(
        mode="tracking_quality_eval",
        input_resolution=args.resolution,
        inference_interval=1,
        cpu_threads=args.threads,
    )

    teacher_engine = _create_engine(args)
    teacher_detections: dict[int, list[Detection]] = {}
    teacher_latencies: list[float] = []
    for source_frame_id, frame in frames:
        detections, latency_ms = _infer(
            teacher_engine,
            frame,
            action,
            args.score_threshold,
        )
        teacher_detections[source_frame_id] = detections
        teacher_latencies.append(latency_ms)
    del teacher_engine

    summary_rows: list[dict[str, object]] = [
        {
            "policy": "teacher_detect_every_frame",
            "frames": len(frames),
            "keyframe_interval": 1,
            "detector_calls": len(frames),
            "detector_invocation_rate": 1.0,
            "mean_total_latency_ms": _safe_mean(teacher_latencies),
            "mean_detector_latency_ms": _safe_mean(teacher_latencies),
            "pseudo_recall": 1.0,
            "precision_proxy": 1.0,
            "skip_frame_pseudo_recall": 1.0,
            "skip_frame_precision_proxy": 1.0,
            "keyframe_pseudo_recall": 1.0,
            "keyframe_precision_proxy": 1.0,
            "mean_keyframe_iou": 1.0,
            "mean_skip_frame_iou": 1.0,
            "mean_keyframe_center_error_norm": 0.0,
            "mean_skip_frame_center_error_norm": 0.0,
            "detection_count_ratio": 1.0,
            "skip_detection_count_ratio": 1.0,
        }
    ]
    frame_rows: list[dict[str, object]] = []

    for policy in policies:
        engine = _create_engine(args)
        summary, rows = _evaluate_policy(
            policy=policy,
            frames=frames,
            teacher_detections=teacher_detections,
            engine=engine,
            action=action,
            args=args,
        )
        del engine
        summary_rows.append(summary)
        frame_rows.extend(rows)

    summary_path = args.output_dir / "tracking_quality_summary.csv"
    frames_path = args.output_dir / "tracking_quality_frames.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with frames_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(frame_rows[0].keys()) if frame_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved per-frame details: {frames_path}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
