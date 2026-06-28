#!/usr/bin/env python3
"""Evaluate resolution-quality trade-off using 640 detections as pseudo labels."""

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
from scene_runtime.utils.video import FrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolution pseudo-label quality evaluation")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument(
        "--model-template",
        default="models/rtdetr_r18_lite_pi4_{resolution}.onnx",
        help="Model path template with {resolution}",
    )
    parser.add_argument("--teacher-resolution", type=int, default=640)
    parser.add_argument("--student-resolutions", default="480,320")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "results" / "quality",
    )
    return parser.parse_args()


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _match(
    teacher: list[Detection],
    student: list[Detection],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for ti, td in enumerate(teacher):
        for si, sd in enumerate(student):
            if td.class_id != sd.class_id:
                continue
            iou = _iou(td.bbox, sd.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, ti, si))

    matches: list[tuple[int, int, float]] = []
    used_t: set[int] = set()
    used_s: set[int] = set()
    for iou, ti, si in sorted(candidates, reverse=True):
        if ti in used_t or si in used_s:
            continue
        used_t.add(ti)
        used_s.add(si)
        matches.append((ti, si, iou))
    return matches


def _load_frames(video: Path, max_frames: int, stride: int) -> list:
    frames = []
    source = FrameSource(video, synthetic=False, max_frames=max_frames, loop=False)
    try:
        for frame_id, frame in enumerate(source):
            if frame_id % max(1, stride) == 0:
                frames.append((frame_id, frame))
    finally:
        source.release()
    return frames


def _create_engine(model_template: str, resolution: int, threads: int) -> ONNXRTDETREngine:
    engine = ONNXRTDETREngine(
        model_path=model_template.format(resolution=resolution),
        dry_run=False,
        enable_thread_sessions=True,
        thread_session_counts=[threads],
    )
    engine.load()
    return engine


def _infer_all(
    engine: ONNXRTDETREngine,
    frames: list,
    resolution: int,
    threads: int,
    score_threshold: float,
) -> tuple[dict[int, list[Detection]], dict[int, float]]:
    action = RuntimeAction(
        mode="quality_eval",
        input_resolution=resolution,
        inference_interval=1,
        cpu_threads=threads,
    )
    detections_by_frame: dict[int, list[Detection]] = {}
    latency_by_frame: dict[int, float] = {}
    for frame_id, frame in frames:
        t0 = time.perf_counter()
        detections = [
            det for det in engine.infer(frame, action) if det.score >= score_threshold
        ]
        latency_by_frame[frame_id] = (time.perf_counter() - t0) * 1000.0
        detections_by_frame[frame_id] = detections
    return detections_by_frame, latency_by_frame


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    args = parse_args()
    students = _parse_ints(args.student_resolutions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(args.video, args.max_frames, args.frame_stride)
    if not frames:
        raise RuntimeError(f"No frames loaded from {args.video}")

    teacher_engine = _create_engine(
        args.model_template,
        args.teacher_resolution,
        args.threads,
    )
    teacher_dets, teacher_lat = _infer_all(
        teacher_engine,
        frames,
        args.teacher_resolution,
        args.threads,
        args.score_threshold,
    )
    del teacher_engine

    summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    teacher_counts = [len(teacher_dets[fid]) for fid, _ in frames]
    teacher_latency = list(teacher_lat.values())
    summary_rows.append(
        {
            "resolution": args.teacher_resolution,
            "role": "teacher",
            "frames": len(frames),
            "mean_latency_ms": _safe_mean(teacher_latency),
            "mean_detection_count": _safe_mean([float(v) for v in teacher_counts]),
            "pseudo_recall": 1.0,
            "precision_proxy": 1.0,
            "mean_matched_iou": 1.0,
            "mean_confidence_drop": 0.0,
            "detection_count_ratio": 1.0,
        }
    )

    for resolution in students:
        student_engine = _create_engine(args.model_template, resolution, args.threads)
        student_dets, student_lat = _infer_all(
            student_engine,
            frames,
            resolution,
            args.threads,
            args.score_threshold,
        )
        del student_engine

        total_teacher = 0
        total_student = 0
        total_matches = 0
        ious: list[float] = []
        conf_drops: list[float] = []

        for frame_id, _ in frames:
            teacher = teacher_dets[frame_id]
            student = student_dets[frame_id]
            matches = _match(teacher, student, args.iou_threshold)
            total_teacher += len(teacher)
            total_student += len(student)
            total_matches += len(matches)
            for ti, si, iou in matches:
                ious.append(iou)
                conf_drops.append(teacher[ti].score - student[si].score)
            match_rows.append(
                {
                    "frame_id": frame_id,
                    "student_resolution": resolution,
                    "teacher_count": len(teacher),
                    "student_count": len(student),
                    "matches": len(matches),
                    "pseudo_recall": len(matches) / len(teacher) if teacher else 1.0,
                    "precision_proxy": len(matches) / len(student) if student else (1.0 if not teacher else 0.0),
                    "mean_iou": _safe_mean([iou for _, _, iou in matches]),
                }
            )

        summary_rows.append(
            {
                "resolution": resolution,
                "role": "student",
                "frames": len(frames),
                "mean_latency_ms": _safe_mean(list(student_lat.values())),
                "mean_detection_count": _safe_mean(
                    [float(len(student_dets[fid])) for fid, _ in frames]
                ),
                "pseudo_recall": total_matches / total_teacher if total_teacher else 1.0,
                "precision_proxy": total_matches / total_student if total_student else 0.0,
                "mean_matched_iou": _safe_mean(ious),
                "mean_confidence_drop": _safe_mean(conf_drops),
                "detection_count_ratio": (
                    total_student / total_teacher if total_teacher else 0.0
                ),
            }
        )

    summary_path = args.output_dir / "resolution_quality_summary.csv"
    matches_path = args.output_dir / "resolution_quality_matches.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with matches_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "frame_id",
            "student_resolution",
            "teacher_count",
            "student_count",
            "matches",
            "pseudo_recall",
            "precision_proxy",
            "mean_iou",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(match_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved matches: {matches_path}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
