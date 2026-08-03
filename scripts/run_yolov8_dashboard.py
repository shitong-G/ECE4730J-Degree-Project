#!/usr/bin/env python3
"""Run YOLOv8 inference and expose the results through the live dashboard."""

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.dashboard import LiveDashboardServer, LiveDashboardState
from scene_runtime.device.temperature import read_temperature_c
from scene_runtime.inference.postprocess import Detection
from scene_runtime.utils.video import FrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "baselines" / "yolov8n_640.onnx")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--camera", choices=["csi", "imx219-raw"], default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--duration-min", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--loop-video", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--history", type=int, default=600)
    parser.add_argument("--jpeg-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--no-video-stream", action="store_true")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-framerate", type=int, default=30)
    return parser.parse_args()


def best_effort_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return None


def make_detections(result, confidence: float) -> list[Detection]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy()
    detections: list[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        score_value = float(score)
        if score_value < confidence:
            continue
        detections.append(
            Detection(
                class_id=int(label),
                score=score_value,
                bbox=tuple(float(value) for value in box),
            )
        )
    return detections


def main() -> None:
    args = parse_args()
    if args.video is None and args.camera is None:
        raise ValueError("Provide --video or --camera")
    if args.video is not None and not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if args.duration_min <= 0 and args.max_frames <= 0 and args.video is not None and not args.loop_video:
        print("YOLOv8 dashboard will stop at the end of the video.")

    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.threads)))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, args.threads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, args.threads)))
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    state = LiveDashboardState(
        max_history=args.history,
        jpeg_width=args.jpeg_width,
        jpeg_quality=args.jpeg_quality,
        score_threshold=args.confidence,
        show_stream=not args.no_video_stream,
    )
    server = LiveDashboardServer(state, host=args.host, port=args.port)
    server.start()
    source = FrameSource(
        args.video,
        loop=args.loop_video,
        camera_backend=args.camera,
        camera_size=(args.camera_width, args.camera_height),
        camera_framerate=args.camera_framerate,
        frame_source_mode="serial",
    )

    output_handle = None
    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            output_handle,
            fieldnames=[
                "frame", "elapsed_sec", "temperature_c", "detection_count",
                "latency_ms", "inference_ms", "preprocess_ms", "postprocess_ms",
                "loop_fps", "inference_fps", "input_resolution", "model",
            ],
        )
        writer.writeheader()

    ip = best_effort_ip()
    print("YOLOv8 dashboard started.")
    print(f"  Dashboard: http://{ip or args.host}:{args.port}")
    print(f"  Model:     {args.model}")
    print(f"  Input:     {args.video or args.camera}")
    print("Press Ctrl+C to stop.", flush=True)

    started = time.perf_counter()
    frame_id = 0
    try:
        for frame in source:
            now = time.perf_counter()
            elapsed = now - started
            if args.duration_min > 0 and elapsed >= args.duration_min * 60.0:
                break
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
            frame_id += 1

            infer_started = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.confidence,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - infer_started) * 1000.0
            result = results[0]
            detections = make_detections(result, args.confidence)
            speed = result.speed or {}
            temperature = read_temperature_c()
            total_elapsed = time.perf_counter() - started
            profile = source.last_profile
            loop_fps = frame_id / total_elapsed if total_elapsed > 0 else 0.0
            inference_ms = float(speed.get("inference", latency_ms))
            inference_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
            payload = {
                "host": ip or socket.gethostname(),
                "strategy": "yolov8n",
                "model": str(args.model),
                "action_mode": "yolov8_detect",
                "frame_id": frame_id,
                "did_infer": True,
                "inference_interval": 1,
                "input_resolution": args.imgsz,
                "resolved_input_resolution": args.imgsz,
                "cpu_threads": args.threads,
                "detection_count": len(detections),
                "tracking_mode": "disabled",
                "tracking_reason": "YOLOv8 standalone detector",
                "latency_ms": latency_ms,
                "onnx_run_ms": inference_ms,
                "inference_ms": inference_ms,
                "tracking_ms": 0.0,
                "capture_ms": profile.get("capture_ms", 0.0),
                "loop_fps": loop_fps,
                "fps": loop_fps,
                "actual_inference_fps": inference_fps,
                "effective_inference_fps": inference_fps,
                "temperature_c": temperature,
                "elapsed_sec": total_elapsed,
            }
            state.publish(payload, frame, detections, args.imgsz)
            if writer is not None:
                writer.writerow({
                    "frame": frame_id,
                    "elapsed_sec": total_elapsed,
                    "temperature_c": temperature,
                    "detection_count": len(detections),
                    "latency_ms": latency_ms,
                    "inference_ms": inference_ms,
                    "preprocess_ms": speed.get("preprocess", ""),
                    "postprocess_ms": speed.get("postprocess", ""),
                    "loop_fps": loop_fps,
                    "inference_fps": inference_fps,
                    "input_resolution": args.imgsz,
                    "model": str(args.model),
                })
                output_handle.flush()
    except KeyboardInterrupt:
        print("\nStopping YOLOv8 dashboard.")
    finally:
        source.release()
        if output_handle is not None:
            output_handle.close()
        server.stop()


if __name__ == "__main__":
    main()
