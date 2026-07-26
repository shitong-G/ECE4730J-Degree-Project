#!/usr/bin/env python3
"""Run a timed experiment with selectable strategy and logging."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.runtime.config import load_config
from scene_runtime.runtime.loop import RuntimeLoop
from scene_runtime.utils.video import FrameSource

from summarize_stage_latency import print_stage_latency_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scene-runtime experiment runner")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument(
        "--strategy",
        default="scene_thermal_coadaptive",
        choices=[
            "native_rtdetr",
            "default",
            "static_affinity",
            "fixed_low_power",
            "fixed_frame_skip",
            "thermal_only",
            "thermal_balanced",
            "thermal_interval_first",
            "scene_only",
            "scene_track_lk",
            "scene_thermal_coadaptive",
            "scene_thermal_interval_first",
            "scene_thermal_interval_lk",
        ],
    )
    p.add_argument("--video", type=Path, default=None, help="Video path or omit for synthetic")
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Override inference.model_path and disable configured per-resolution "
            "mappings, ensuring this exact ONNX model is used."
        ),
    )
    p.add_argument(
        "--model-paths-by-resolution",
        default=None,
        help=(
            "Override the resolution-to-ONNX mapping as "
            "'320=path320.onnx,480=path480.onnx,640=path640.onnx'. "
            "Use this for a matched family of pruned/quantized models."
        ),
    )
    p.add_argument(
        "--camera",
        choices=["csi"],
        default=None,
        help="Use Raspberry Pi CSI camera via Picamera2",
    )
    p.add_argument(
        "--loop-video",
        action="store_true",
        help="Loop the input video until duration-min is reached",
    )
    p.add_argument(
        "--frame-width",
        type=int,
        default=0,
        help="Resize runtime input frames to this width before scene analysis, LK tracking, and RT-DETR. Use with --frame-height.",
    )
    p.add_argument(
        "--frame-height",
        type=int,
        default=0,
        help="Resize runtime input frames to this height before scene analysis, LK tracking, and RT-DETR. Use with --frame-width.",
    )
    p.add_argument("--dry-run", action="store_true", help="Simulate inference without ONNX model")
    p.add_argument("--duration-min", type=float, default=15.0)
    p.add_argument(
        "--warmup-until-temp-c",
        type=float,
        default=None,
        help=(
            "Before opening experiment logs, repeatedly run real RT-DETR inference "
            "until this CPU temperature is reached."
        ),
    )
    p.add_argument(
        "--warmup-max-sec",
        type=float,
        default=900.0,
        help="Fail the temperature-driven RT-DETR warmup after this many seconds.",
    )
    p.add_argument("--output", type=Path, default=None, help="CSV log output path")
    p.add_argument(
        "--log-detections",
        action="store_true",
        help="Write per-frame detection boxes to <output>_detections.jsonl",
    )
    p.add_argument("--detection-output", type=Path, default=None)
    p.add_argument(
        "--thermal-state",
        choices=["normal", "warm", "hot", "critical", "unknown"],
        default=None,
        help="Override detected thermal state for policy testing on non-Pi machines",
    )
    p.add_argument(
        "--thermal-temp-c",
        type=float,
        default=None,
        help="Override detected temperature in Celsius for thermal policy testing",
    )
    p.add_argument(
        "--enable-thread-sessions",
        action="store_true",
        help="Preload ONNX Runtime sessions for configured cpu thread counts",
    )
    p.add_argument(
        "--thread-session-counts",
        default=None,
        help="Comma-separated thread counts, e.g. 1,2,3,4",
    )
    p.add_argument(
        "--apply-runtime-actions",
        action="store_true",
        help="Best-effort apply governor and CPU affinity from RuntimeAction",
    )
    p.add_argument(
        "--enable-lk-tracking",
        action="store_true",
        help="Use Lucas-Kanade tracking to update boxes on skipped detector frames",
    )
    p.add_argument(
        "--lk-force-refresh-on-failure",
        action="store_true",
        help="Run RT-DETR immediately when LK tracking quality degrades",
    )
    p.add_argument(
        "--disable-roi-refresh",
        action="store_true",
        help=(
            "Disable ROI detector refresh while leaving LK/event-triggered full-frame "
            "refresh enabled. Intended for the ROI ablation."
        ),
    )
    p.add_argument("--lk-max-failure-ratio", type=float, default=None)
    p.add_argument("--lk-min-valid-points", type=int, default=None)
    p.add_argument(
        "--fan-control",
        choices=["config", "enabled", "disabled"],
        default="config",
        help="Use configured fan behavior, force threshold/PWM fan on, or force it off.",
    )
    p.add_argument("--fan-on-temp-c", type=float, default=None)
    p.add_argument("--fan-off-temp-c", type=float, default=None)
    p.add_argument("--fan-full-temp-c", type=float, default=None)
    p.add_argument("--fan-min-duty-cycle", type=float, default=None)
    p.add_argument("--fan-max-duty-cycle", type=float, default=None)
    p.add_argument(
        "--fan-temperature-only",
        action="store_true",
        help="Trigger the formal-run fan only from temperature thresholds.",
    )
    p.add_argument(
        "--no-stage-summary",
        action="store_true",
        help="Do not print post-experiment overall/stage latency summary",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (args.frame_width > 0) != (args.frame_height > 0):
        raise ValueError("--frame-width and --frame-height must be set together")

    config = load_config(args.config, args.strategy)
    if args.model is not None and args.model_paths_by_resolution is not None:
        raise ValueError("--model and --model-paths-by-resolution are mutually exclusive")
    if args.model is not None:
        if not args.model.exists():
            raise FileNotFoundError(f"ONNX model does not exist: {args.model}")
        inference_cfg = config.setdefault("inference", {})
        inference_cfg["model_path"] = str(args.model)
        inference_cfg.pop("model_paths_by_resolution", None)
    if args.model_paths_by_resolution is not None:
        model_map: dict[int, str] = {}
        for item in args.model_paths_by_resolution.split(","):
            resolution_text, separator, path_text = item.strip().partition("=")
            if not separator or not resolution_text or not path_text:
                raise ValueError(
                    "--model-paths-by-resolution entries must be RESOLUTION=PATH"
                )
            resolution = int(resolution_text)
            model_path = Path(path_text)
            if resolution <= 0 or not model_path.exists():
                raise FileNotFoundError(
                    f"Invalid model mapping {item!r}; model must exist and resolution must be positive"
                )
            model_map[resolution] = str(model_path)
        if not model_map:
            raise ValueError("--model-paths-by-resolution cannot be empty")
        inference_cfg = config.setdefault("inference", {})
        inference_cfg["model_paths_by_resolution"] = model_map
        default_resolution = int(config.get("runtime", {}).get("default_input_resolution", 640))
        inference_cfg["model_path"] = model_map.get(
            default_resolution,
            model_map[max(model_map)],
        )
    if args.warmup_until_temp_c is not None:
        if args.warmup_max_sec <= 0:
            raise ValueError("--warmup-max-sec must be positive")
        runtime_cfg = config.setdefault("runtime", {})
        runtime_cfg["temperature_warmup_target_c"] = float(args.warmup_until_temp_c)
        runtime_cfg["temperature_warmup_max_sec"] = float(args.warmup_max_sec)
    if args.thermal_state is not None:
        config.setdefault("thermal", {})["override_state"] = args.thermal_state
    if args.thermal_temp_c is not None:
        config.setdefault("thermal", {})["override_temp_c"] = args.thermal_temp_c
    if args.enable_thread_sessions:
        config.setdefault("inference", {})["enable_thread_sessions"] = True
    if args.thread_session_counts:
        counts = [int(item.strip()) for item in args.thread_session_counts.split(",") if item.strip()]
        config.setdefault("inference", {})["thread_session_counts"] = counts
    if args.apply_runtime_actions:
        config.setdefault("os_control", {})["apply_runtime_actions"] = True
    if args.enable_lk_tracking:
        config.setdefault("tracking", {})["enable_lk_tracking"] = True
    if args.lk_force_refresh_on_failure:
        config.setdefault("tracking", {})["lk_force_refresh_on_failure"] = True
    if args.disable_roi_refresh:
        config.setdefault("tracking", {})["roi_refresh_enabled"] = False
    if args.lk_max_failure_ratio is not None:
        config.setdefault("tracking", {})["lk_max_failure_ratio"] = args.lk_max_failure_ratio
    if args.lk_min_valid_points is not None:
        config.setdefault("tracking", {})["lk_min_valid_points"] = args.lk_min_valid_points
    fan_cfg = config.setdefault("fan", {})
    if args.fan_control == "enabled":
        fan_cfg["enabled"] = True
        fan_cfg["enabled_strategies"] = ["*"]
    elif args.fan_control == "disabled":
        fan_cfg["enabled"] = False
    if args.fan_temperature_only:
        fan_cfg["temperature_only"] = True
    fan_overrides = {
        "on_temp_c": args.fan_on_temp_c,
        "off_temp_c": args.fan_off_temp_c,
        "full_temp_c": args.fan_full_temp_c,
        "min_duty_cycle": args.fan_min_duty_cycle,
        "max_duty_cycle": args.fan_max_duty_cycle,
    }
    for key, value in fan_overrides.items():
        if value is not None:
            fan_cfg[key] = value
    fan_off = float(fan_cfg.get("off_temp_c", 0.0))
    fan_on = float(fan_cfg.get("on_temp_c", fan_off))
    fan_full = float(fan_cfg.get("full_temp_c", fan_on))
    fan_min = float(fan_cfg.get("min_duty_cycle", 0.0))
    fan_max = float(fan_cfg.get("max_duty_cycle", 1.0))
    if not fan_off <= fan_on <= fan_full:
        raise ValueError("Fan thresholds must satisfy off_temp_c <= on_temp_c <= full_temp_c")
    if not 0.0 <= fan_min <= fan_max <= 1.0:
        raise ValueError("Fan duty cycles must satisfy 0 <= min <= max <= 1")

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            ROOT / "experiments" / "logs" / f"{args.strategy}_{stamp}.csv"
        )

    duration_sec = args.duration_min * 60.0
    max_frames = int(duration_sec * 10) if args.dry_run else None
    camera_cfg = config.get("camera", {})
    use_camera = args.camera or (None if args.video else camera_cfg.get("backend"))
    if use_camera == "video":
        use_camera = None
    if args.dry_run and args.camera is None:
        use_camera = None
    frame_size = (
        (int(args.frame_width), int(args.frame_height))
        if args.frame_width > 0 and args.frame_height > 0
        else None
    )

    source = FrameSource(
        args.video,
        synthetic=(args.video is None and use_camera is None),
        max_frames=max_frames,
        loop=args.loop_video,
        frame_size=frame_size,
        camera_backend=use_camera,
        camera_size=(
            int(camera_cfg.get("csi_width", 640)),
            int(camera_cfg.get("csi_height", 480)),
        ),
        camera_framerate=int(camera_cfg.get("csi_framerate", 30)),
    )

    loop = RuntimeLoop(
        config,
        source,
        dry_run=args.dry_run,
        duration_sec=duration_sec,
        log_path=args.output,
        detection_log_path=(
            args.detection_output
            if args.detection_output is not None
            else args.output.with_name(args.output.stem + "_detections.jsonl")
            if args.log_detections
            else None
        ),
    )

    log_path = loop.run()
    print(f"Experiment finished.")
    print(f"  strategy: {args.strategy}")
    print(f"  dry_run:  {args.dry_run}")
    print(f"  log:      {log_path}")
    print(f"  profile:  {log_path.with_name(log_path.stem + '_profile.csv')}")
    if not args.no_stage_summary:
        print("\nPost-experiment stage latency summary")
        print("=====================================")
        print_stage_latency_summary(log_path)


if __name__ == "__main__":
    main()
