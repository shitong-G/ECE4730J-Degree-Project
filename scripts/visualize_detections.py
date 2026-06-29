#!/usr/bin/env python3
"""Render RT-DETR ONNX detections on a video."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.inference.onnx_engine import ONNXRTDETREngine


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ONNX RT-DETR detections")
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx")
    parser.add_argument(
        "--model-template",
        default=None,
        help="Optional model template with {resolution}, e.g. models/rtdetr_r18_lite_pi4_{resolution}.onnx",
    )
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _label(class_id: int) -> str:
    if 0 <= class_id < len(COCO80):
        return COCO80[class_id]
    return str(class_id)


def _color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (50, 220, 120), (80, 170, 255), (240, 180, 70), (220, 90, 90),
        (180, 120, 255), (70, 220, 220), (230, 120, 200), (120, 220, 80),
    ]
    return palette[class_id % len(palette)]


def _draw_box(frame, detection, input_resolution: int, latency_ms: float) -> None:
    h, w = frame.shape[:2]
    sx = w / float(input_resolution)
    sy = h / float(input_resolution)
    x1, y1, x2, y2 = detection.bbox
    pt1 = (max(0, int(x1 * sx)), max(0, int(y1 * sy)))
    pt2 = (min(w - 1, int(x2 * sx)), min(h - 1, int(y2 * sy)))
    color = _color(detection.class_id)
    cv2.rectangle(frame, pt1, pt2, color, 2)
    text = f"{_label(detection.class_id)} {detection.score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(0, pt1[1] - th - 6)
    cv2.rectangle(frame, (pt1[0], y), (pt1[0] + tw + 6, y + th + 6), color, -1)
    cv2.putText(
        frame,
        text,
        (pt1[0] + 3, y + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (10, 10, 10),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{latency_ms:.0f} ms",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = (
            ROOT
            / "experiments"
            / "visualizations"
            / f"{args.video.stem}_detections_{args.resolution}.mp4"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    model_paths = None
    if args.model_template:
        model_paths = {
            args.resolution: args.model_template.format(resolution=args.resolution)
        }

    engine = ONNXRTDETREngine(
        model_path=str(args.model),
        model_paths_by_resolution=model_paths,
        enable_thread_sessions=True,
        thread_session_counts=[args.threads],
    )
    engine.load()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, args.frame_stride),
        (width, height),
    )

    action = RuntimeAction(
        mode="visualize",
        input_resolution=args.resolution,
        inference_interval=1,
        cpu_threads=args.threads,
        governor=None,
    )
    frame_id = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % max(1, args.frame_stride) != 0:
                frame_id += 1
                continue
            t0 = time.perf_counter()
            detections = [
                det
                for det in engine.infer(frame, action)
                if det.score >= args.score_threshold
            ]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resolved = engine.last_resolved_input_resolution or args.resolution
            for detection in detections:
                _draw_box(frame, detection, resolved, latency_ms)
            cv2.putText(
                frame,
                f"res={resolved} det={len(detections)}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            written += 1
            if args.max_frames and written >= args.max_frames:
                break
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    print(f"Saved visualization: {output}")
    print(f"Frames written: {written}")


if __name__ == "__main__":
    main()
