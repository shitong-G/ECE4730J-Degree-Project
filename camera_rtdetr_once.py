#!/usr/bin/env python3
"""Capture one IMX219 frame, run RT-DETR ONNX inference, and save a boxed JPEG."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.inference.onnx_engine import ONNXRTDETREngine

from capture_imx219_frame import capture_and_convert


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


def _default_model_path() -> Path:
    preferred = ROOT / "models" / "rtdetr_r18_lite_pi4_640.onnx"
    fallback = ROOT / "models" / "rtdetr_r18_lite_pi4.onnx"
    return preferred if preferred.exists() else fallback


def _draw_box(frame, detection, input_resolution: int) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--media-device", default="/dev/media3")
    parser.add_argument("--video-device", default="/dev/video0")
    parser.add_argument("--sensor-entity", default="imx219 10-0010")
    parser.add_argument("--sensor-format", default="SRGGB10_1X10")
    parser.add_argument("--pixel-format", default="RG10")
    parser.add_argument("--width", type=int, default=1640)
    parser.add_argument("--height", type=int, default=1232)
    parser.add_argument("--stride-pixels", type=int, default=1648)
    parser.add_argument("--stream-mmap", type=int, default=3)
    parser.add_argument("--stream-count", type=int, default=1)
    parser.add_argument("--raw-output", default="camera_latest.raw")
    parser.add_argument("--jpg-output", default="camera_latest.jpg")
    parser.add_argument("--skip-capture", action="store_true", help="Reuse an existing raw file")

    parser.add_argument("--black-level", type=float, default=64.0)
    parser.add_argument("--r-gain", type=float, default=1.2)
    parser.add_argument("--g-gain", type=float, default=0.8)
    parser.add_argument("--b-gain", type=float, default=1.2)
    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--saturation", type=float, default=1.8)
    parser.add_argument("--sharpen", type=float, default=0.2)
    parser.add_argument(
        "--resize-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize lite ISP output back to the original raw resolution.",
    )

    parser.add_argument("--model", type=Path, default=_default_model_path())
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="camera_rtdetr_result.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    jpg_path = capture_and_convert(args)
    frame = cv2.imread(str(jpg_path))
    if frame is None:
        raise RuntimeError(f"Failed to read converted camera image: {jpg_path}")

    engine = ONNXRTDETREngine(
        model_path=str(args.model),
        dry_run=args.dry_run,
        enable_thread_sessions=True,
        thread_session_counts=[args.threads],
    )
    engine.load()

    action = RuntimeAction(
        mode="camera_once",
        input_resolution=args.resolution,
        inference_interval=1,
        cpu_threads=args.threads,
        governor=None,
    )

    t0 = time.perf_counter()
    detections = [
        det for det in engine.infer(frame, action) if det.score >= args.score_threshold
    ]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resolved = engine.last_resolved_input_resolution or args.resolution

    annotated = frame.copy()
    for detection in detections:
        _draw_box(annotated, detection, resolved)
    cv2.putText(
        annotated,
        f"res={resolved} det={len(detections)} {latency_ms:.0f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise RuntimeError(f"Failed to save RT-DETR result: {output}")

    print(f"Saved raw frame: {args.raw_output}")
    print(f"Saved camera image: {args.jpg_output}")
    print(f"Saved RT-DETR result: {output}")
    print(f"Detections: {len(detections)}, latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
