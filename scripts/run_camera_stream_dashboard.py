#!/usr/bin/env python3
"""Stream camera frames to the live dashboard without inference or tracking.

This is deliberately separate from ``run_live_dashboard.py`` so a camera
latency test never loads ONNX Runtime, a model, the runtime controller, or the
LK tracker.  Its only per-frame work is camera capture, RGB-to-BGR conversion
(performed by the CSI camera path), JPEG encoding, and MJPEG transmission.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.dashboard import LiveDashboardServer, LiveDashboardState
from scene_runtime.utils.video import FrameSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        choices=["csi", "imx219-raw"],
        default="csi",
        help="Camera backend. csi (Picamera2) is the recommended low-latency path.",
    )
    parser.add_argument("--width", type=int, default=640, help="CSI capture width.")
    parser.add_argument("--height", type=int, default=480, help="CSI capture height.")
    parser.add_argument("--framerate", type=int, default=30, help="Requested CSI frame rate.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jpeg-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--history", type=int, default=600)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Stop after this many seconds; 0 runs until Ctrl+C.",
    )

    # Keep the raw IMX219 path available for comparing its capture pipeline.
    parser.add_argument("--imx219-media-device", default="/dev/media3")
    parser.add_argument("--imx219-video-device", default="/dev/video0")
    parser.add_argument("--imx219-sensor-entity", default="imx219 10-0010")
    parser.add_argument("--imx219-width", type=int, default=1640)
    parser.add_argument("--imx219-height", type=int, default=1232)
    parser.add_argument("--imx219-stride-pixels", type=int, default=1648)
    parser.add_argument("--imx219-frame-width", type=int, default=640)
    parser.add_argument("--imx219-frame-height", type=int, default=480)
    parser.add_argument("--imx219-runtime-mode", choices=["lite-isp", "gray"], default="lite-isp")
    parser.add_argument("--imx219-raw-output", type=Path, default=Path("camera_latest.raw"))
    parser.add_argument("--imx219-jpg-output", type=Path, default=Path("camera_latest.jpg"))
    parser.add_argument("--imx219-capture-interval-sec", type=float, default=0.0)
    parser.add_argument(
        "--imx219-save-latest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a local JPEG too. Disabled by default to avoid extra disk I/O.",
    )
    return parser.parse_args()


def best_effort_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return None


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.framerate <= 0:
        raise ValueError("--width, --height, and --framerate must be positive")
    if args.duration_sec < 0:
        raise ValueError("--duration-sec cannot be negative")

    state = LiveDashboardState(
        max_history=args.history,
        jpeg_width=args.jpeg_width,
        jpeg_quality=args.jpeg_quality,
        show_stream=True,
    )
    server = LiveDashboardServer(state, host=args.host, port=args.port)
    source = FrameSource(
        None,
        camera_backend=args.camera,
        camera_size=(args.width, args.height),
        camera_framerate=args.framerate,
        imx219_media_device=args.imx219_media_device,
        imx219_video_device=args.imx219_video_device,
        imx219_sensor_entity=args.imx219_sensor_entity,
        imx219_width=args.imx219_width,
        imx219_height=args.imx219_height,
        imx219_stride_pixels=args.imx219_stride_pixels,
        imx219_frame_width=args.imx219_frame_width,
        imx219_frame_height=args.imx219_frame_height,
        imx219_raw_output=args.imx219_raw_output,
        imx219_jpg_output=args.imx219_jpg_output,
        imx219_save_latest=args.imx219_save_latest,
        imx219_capture_interval_sec=args.imx219_capture_interval_sec,
        imx219_runtime_mode=args.imx219_runtime_mode,
        frame_source_mode="serial",
    )

    server.start()
    ip = best_effort_ip()
    print("Camera-only dashboard started (inference and tracking are disabled).")
    print(f"  Dashboard: http://{ip or args.host}:{args.port}")
    print(f"  Camera:    {args.camera}")
    print("Press Ctrl+C to stop.")

    started = time.perf_counter()
    frames = 0
    try:
        for frame in source:
            now = time.perf_counter()
            frames += 1
            elapsed = now - started
            profile = source.last_profile
            payload = {
                "host": ip or socket.gethostname(),
                "strategy": "camera_only",
                "action_mode": "stream",
                "frame_id": frames,
                "did_infer": False,
                "inference_interval": 0,
                "input_resolution": int(frame.shape[1]),
                "resolved_input_resolution": int(frame.shape[1]),
                "cpu_threads": 0,
                "detection_count": 0,
                "tracking_mode": "disabled",
                "tracking_reason": "camera-only stream",
                "latency_ms": 0.0,
                "onnx_run_ms": 0.0,
                "tracking_ms": 0.0,
                "capture_ms": profile.get("capture_ms", 0.0),
                "isp_ms": profile.get("isp_ms", 0.0),
                "source_total_ms": profile.get("source_total_ms", 0.0),
                "source_resize_ms": profile.get("source_resize_ms", 0.0),
                "source_runtime_resize_ms": profile.get("source_runtime_resize_ms", 0.0),
                "source_consumer_wait_ms": profile.get("source_consumer_wait_ms", 0.0),
                "source_frame_age_ms": profile.get("source_frame_age_ms", 0.0),
                "source_dropped_frames": profile.get("source_dropped_frames", 0.0),
                "source_error_count": profile.get("source_error_count", 0.0),
                "serial_total_ms": profile.get("source_total_ms", 0.0),
                "loop_fps": frames / elapsed if elapsed > 0 else 0.0,
                "fps": frames / elapsed if elapsed > 0 else 0.0,
                "actual_inference_fps": 0.0,
                "effective_inference_fps": 0.0,
            }
            # Empty detections means encode_annotated_jpeg only produces the JPEG stream;
            # it never runs RT-DETR, post-processing, or tracking.
            state.publish(payload, frame, [], int(frame.shape[1]))
            if args.duration_sec and elapsed >= args.duration_sec:
                break
    except KeyboardInterrupt:
        print("\nStopping camera-only dashboard.")
    finally:
        source.release()
        server.stop()


if __name__ == "__main__":
    main()
