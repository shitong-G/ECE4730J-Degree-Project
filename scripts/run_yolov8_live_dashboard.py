#!/usr/bin/env python3
"""Run YOLOv8 on a looping video and expose the shared live dashboard."""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "baselines" / "yolov8n_640.onnx",
    )
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample3.mp4")
    parser.add_argument("--loop-video", action="store_true")
    parser.add_argument("--duration-min", type=float, default=20.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jpeg-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _best_effort_ip() -> str | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def _fps(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def _detections_from_result(result, *, frame_width: int, frame_height: int, input_size: int):
    from scene_runtime.inference.postprocess import Detection

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    sx = input_size / max(1.0, float(frame_width))
    sy = input_size / max(1.0, float(frame_height))
    detections: list[Detection] = []
    for box, score, class_id in zip(xyxy, scores, classes):
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        detections.append(
            Detection(
                class_id=int(class_id),
                score=float(score),
                bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
            )
        )
    return detections


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.duration_min <= 0:
        raise ValueError("--duration-min must be positive")

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.threads)))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, args.threads)))

    import cv2
    from ultralytics import YOLO
    from scene_runtime.dashboard import LiveDashboardServer, LiveDashboardState

    try:
        import torch

        torch.set_num_threads(max(1, args.threads))
    except (ImportError, RuntimeError):
        pass

    model = YOLO(str(args.model))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    state = LiveDashboardState(
        jpeg_width=args.jpeg_width,
        jpeg_quality=args.jpeg_quality,
        score_threshold=args.score_threshold,
        show_stream=True,
    )
    server = LiveDashboardServer(state, host=args.host, port=args.port)
    server.start()
    ip = _best_effort_ip()
    print("YOLOv8 live dashboard started.")
    print(f"  local/bind: http://{args.host}:{args.port}")
    print(f"  LAN URL:    http://{ip}:{args.port}" if ip else "  LAN URL:    unavailable")
    print(f"  model:      {args.model}")
    print(f"  duration:   {args.duration_min:.1f} min")
    print("Press Ctrl+C to stop.")

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "experiments" / "logs" / f"yolov8n_live_{stamp}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    output_file = output.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        output_file,
        fieldnames=[
            "timestamp",
            "frame_id",
            "latency_ms",
            "loop_fps",
            "detection_count",
            "confidence_mean",
        ],
    )
    writer.writeheader()

    history: deque[float] = deque(maxlen=60)
    frame_id = 0
    deadline = time.perf_counter() + args.duration_min * 60.0
    try:
        while time.perf_counter() < deadline:
            ok, frame = capture.read()
            if not ok:
                if not args.loop_video:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok:
                    break

            frame_start = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - frame_start) * 1000.0
            result = results[0] if results else None
            detections = (
                _detections_from_result(
                    result,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                    input_size=args.imgsz,
                )
                if result is not None
                else []
            )
            frame_id += 1
            now = time.perf_counter()
            history.append(now)
            loop_fps = _fps(history)
            confidence_mean = (
                sum(item.score for item in detections) / len(detections)
                if detections
                else 0.0
            )
            payload = {
                "timestamp": time.time(),
                "updated_at": time.time(),
                "frame_id": frame_id,
                "strategy": "yolov8n_external",
                "workload": "external_detector",
                "action_mode": "full_detector",
                "did_infer": True,
                "tracking_mode": "disabled",
                "tracking_reason": "external_detector",
                "tracking_ms": 0.0,
                "latency_ms": latency_ms,
                "loop_fps": loop_fps,
                "fps": loop_fps,
                "effective_inference_fps": loop_fps,
                "actual_inference_fps": loop_fps,
                "input_resolution": args.imgsz,
                "resolved_input_resolution": args.imgsz,
                "inference_interval": 1,
                "cpu_threads": args.threads,
                "detection_count": len(detections),
                "confidence_mean": confidence_mean,
                "temp_c": None,
            }
            state.publish(payload, frame, detections, args.imgsz)
            writer.writerow(
                {
                    "timestamp": payload["timestamp"],
                    "frame_id": frame_id,
                    "latency_ms": latency_ms,
                    "loop_fps": loop_fps,
                    "detection_count": len(detections),
                    "confidence_mean": confidence_mean,
                }
            )
            output_file.flush()
    except KeyboardInterrupt:
        print("\nStopping YOLOv8 dashboard.")
    finally:
        output_file.close()
        capture.release()
        server.stop()
        print(f"Log: {output}")


if __name__ == "__main__":
    main()
