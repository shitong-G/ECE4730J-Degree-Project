#!/usr/bin/env python3
"""Run an experiment and expose a live browser dashboard."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_runtime.dashboard import LiveDashboardServer, LiveDashboardState
from scene_runtime.device.fan import PwmFanController
from scene_runtime.runtime.config import load_config
from scene_runtime.runtime.loop import RuntimeLoop
from scene_runtime.utils.video import FrameSource

from summarize_stage_latency import print_stage_latency_summary


class BackgroundFrameDrain:
    """Run a frame source in the background and discard frames."""

    def __init__(self, source: FrameSource, label: str, *, progress_interval: int = 20) -> None:
        self._source = source
        self._label = label
        self._progress_interval = max(0, int(progress_interval))
        self._stop = False
        self._thread: threading.Thread | None = None
        self._count = 0
        self._error: BaseException | None = None
        self._error_count = 0
        self._last_profile: dict[str, float] = {}
        self._iterator = None
        self._lock = threading.Lock()
        self._condition = threading.Condition()
        self._pending_capture = False
        self._capture_active = False
        self._skipped_triggers = 0
        self._trigger_count = 0
        self._last_capture_time = 0.0
        self._last_reported_count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def last_profile(self) -> dict[str, float]:
        return dict(self._last_profile)

    @property
    def skipped_triggers(self) -> int:
        return self._skipped_triggers

    def snapshot(self) -> dict[str, float | bool]:
        """Return a lightweight, thread-safe diagnostic snapshot."""
        with self._condition:
            pending = self._pending_capture
            active = self._capture_active
            skipped = self._skipped_triggers
            triggers = self._trigger_count
        with self._lock:
            count = self._count
            errors = self._error_count
            profile = dict(self._last_profile)
            last_capture_time = self._last_capture_time
        age_ms = 0.0
        if last_capture_time > 0:
            age_ms = max(0.0, (time.perf_counter() - last_capture_time) * 1000.0)
        return {
            "bg_camera_active": active,
            "bg_camera_pending": pending,
            "bg_camera_count": float(count),
            "bg_camera_triggers": float(triggers),
            "bg_camera_skipped": float(skipped),
            "bg_camera_errors": float(errors),
            "bg_camera_last_source_ms": float(profile.get("source_total_ms", 0.0)),
            "bg_camera_last_capture_ms": float(profile.get("capture_ms", 0.0)),
            "bg_camera_last_isp_ms": float(profile.get("isp_ms", 0.0)),
            "bg_camera_last_age_ms": age_ms,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self._label}-background-drain",
            daemon=True,
        )
        self._thread.start()

    def start_on_demand(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_on_demand,
            name=f"{self._label}-on-demand-drain",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        with self._condition:
            self._condition.notify_all()
        self._source.release()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def trigger_capture(self) -> bool:
        """Request one asynchronous capture if no request is already pending."""
        with self._condition:
            if self._stop:
                return False
            self._trigger_count += 1
            if self._pending_capture or self._capture_active:
                self._skipped_triggers += 1
                return False
            self._pending_capture = True
            self._condition.notify_all()
            return True

    def capture_once(self) -> bool:
        """Capture and discard one frame on demand."""
        count_after_capture: int | None = None
        with self._lock:
            if self._stop:
                return False
            try:
                if self._iterator is None:
                    self._iterator = iter(self._source)
                next(self._iterator)
                self._count += 1
                count_after_capture = self._count
                self._last_profile = self._source.last_profile
                self._last_capture_time = time.perf_counter()
            except StopIteration:
                self._iterator = None
                return False
            except BaseException as exc:
                self._record_recoverable_error(exc)
                return False
        self._print_progress_if_needed(count_after_capture)
        return True

    def _print_progress_if_needed(self, count: int | None) -> None:
        if count is None or self._progress_interval <= 0:
            return
        if count % self._progress_interval != 0 or count == self._last_reported_count:
            return
        with self._condition:
            skipped = self._skipped_triggers
            triggers = self._trigger_count
        self._last_reported_count = count
        print(
            f"Background camera progress: captured={count}, "
            f"skipped={skipped}, triggers={triggers}",
            flush=True,
        )

    def _run(self) -> None:
        while not self._stop:
            if not self.capture_once():
                time.sleep(0.25)
        self._source.release()

    def _record_recoverable_error(self, exc: BaseException) -> None:
        self._error = exc
        self._error_count += 1
        self._iterator = None
        self._source.release()

    def _run_on_demand(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._stop and not self._pending_capture:
                        self._condition.wait(timeout=0.2)
                    if self._stop:
                        break
                    self._pending_capture = False
                    self._capture_active = True
                try:
                    self.capture_once()
                finally:
                    with self._condition:
                        self._capture_active = False
        finally:
            self._source.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scene-runtime with live web dashboard")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "raspberry_pi4.yaml")
    parser.add_argument(
        "--strategy",
        default="thermal_balanced",
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
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "sample.mp4")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Override inference.model_path, for example an INT8 ONNX model. "
            "This also disables configured per-resolution model mappings so the "
            "selected file is always the model that is loaded."
        ),
    )
    parser.add_argument(
        "--model-paths-by-resolution",
        default=None,
        help=(
            "Override the resolution-to-ONNX mapping as "
            "'320=path320.onnx,480=path480.onnx,640=path640.onnx'."
        ),
    )
    parser.add_argument(
        "--camera",
        choices=["csi", "imx219-raw"],
        default=None,
        help="Use a camera instead of --video. csi uses Picamera2; imx219-raw uses media-ctl/v4l2-ctl RG10 capture.",
    )
    parser.add_argument(
        "--background-camera",
        choices=["imx219-raw"],
        default=None,
        help="Run a camera in the background while --video remains the runtime input.",
    )
    parser.add_argument(
        "--background-camera-trigger",
        choices=["continuous", "post-tracking", "during-tracking"],
        default="during-tracking",
        help="continuous captures as before; during-tracking starts one async capture when LK tracking starts; post-tracking captures after each LK tracking frame.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=0,
        help="Resize runtime input frames to this width before scene analysis, LK tracking, and RT-DETR. Use with --frame-height.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=0,
        help="Resize runtime input frames to this height before scene analysis, LK tracking, and RT-DETR. Use with --frame-width.",
    )
    parser.add_argument("--imx219-media-device", default="/dev/media3")
    parser.add_argument("--imx219-video-device", default="/dev/video0")
    parser.add_argument("--imx219-sensor-entity", default="imx219 10-0010")
    parser.add_argument("--imx219-width", type=int, default=1640)
    parser.add_argument("--imx219-height", type=int, default=1232)
    parser.add_argument("--imx219-stride-pixels", type=int, default=1648)
    parser.add_argument(
        "--imx219-frame-width",
        type=int,
        default=640,
        help="Resize the lite-ISP BGR frame to this width before runtime inference/tracking.",
    )
    parser.add_argument(
        "--imx219-frame-height",
        type=int,
        default=480,
        help="Resize the lite-ISP BGR frame to this height before runtime inference/tracking.",
    )
    parser.add_argument("--imx219-raw-output", type=Path, default=Path("camera_latest.raw"))
    parser.add_argument("--imx219-jpg-output", type=Path, default=Path("camera_latest.jpg"))
    parser.add_argument(
        "--imx219-save-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continuously overwrite camera_latest.raw/camera_latest.jpg for inspection.",
    )
    parser.add_argument(
        "--imx219-capture-interval-sec",
        type=float,
        default=0.0,
        help="Minimum interval between IMX219 raw captures. Default 0 means capture as fast as the source allows.",
    )
    parser.add_argument(
        "--imx219-runtime-mode",
        choices=["lite-isp", "gray"],
        default="lite-isp",
        help="Use lite-isp for color-tuned frames, or gray to skip color tuning for faster inference/tracking input.",
    )
    parser.add_argument(
        "--frame-source-mode",
        choices=["serial", "latest-thread"],
        default="latest-thread",
        help="Use latest-thread for producer-consumer camera capture, or serial for old blocking capture.",
    )
    parser.add_argument("--loop-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--duration-min", type=float, default=15.0)
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=1,
        help="Run the same experiment this many times sequentially, printing one report per run.",
    )
    parser.add_argument(
        "--repeat-cooldown-sec",
        type=float,
        default=0.0,
        help="Optional cooldown delay between repeated runs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-detections", action="store_true")
    parser.add_argument("--detection-output", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--history", type=int, default=600, help="Number of dashboard samples to keep")
    parser.add_argument("--jpeg-width", type=int, default=960, help="Live stream width; lower reduces network load")
    parser.add_argument("--jpeg-quality", type=int, default=78)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--no-video-stream", action="store_true")
    parser.add_argument(
        "--thermal-state",
        choices=["normal", "warm", "hot", "critical", "unknown"],
        default=None,
    )
    parser.add_argument("--thermal-temp-c", type=float, default=None)
    parser.add_argument("--enable-thread-sessions", action="store_true")
    parser.add_argument("--thread-session-counts", default=None)
    parser.add_argument("--apply-runtime-actions", action="store_true")
    parser.add_argument(
        "--fan-control",
        choices=["config", "enabled", "disabled"],
        default="config",
        help="Use configured fan behavior, force fan on, or force it off.",
    )
    parser.add_argument("--enable-lk-tracking", action="store_true")
    parser.add_argument("--lk-force-refresh-on-failure", action="store_true")
    parser.add_argument("--lk-max-failure-ratio", type=float, default=None)
    parser.add_argument("--lk-min-valid-points", type=int, default=None)
    parser.add_argument(
        "--query-budget-mode",
        choices=["auto", "strict", "postprocess", "disabled"],
        default=None,
        help="Select dynamic ONNX query-budget handling; strict requires a dynamic model.",
    )
    parser.add_argument("--query-budget-override", type=int, default=None)
    parser.add_argument("--query-budget-input-name", default=None)
    parser.add_argument("--max-query-budget", type=int, default=None)
    parser.add_argument(
        "--temperature-query-budget",
        action="store_true",
        help="Select graph query budget from the raw CPU thermal state.",
    )
    parser.add_argument("--query-budget-normal", type=int, default=None)
    parser.add_argument("--query-budget-warm", type=int, default=None)
    parser.add_argument("--query-budget-hot", type=int, default=None)
    parser.add_argument("--query-budget-critical", type=int, default=None)
    parser.add_argument("--query-budget-hysteresis-c", type=float, default=None)
    parser.add_argument(
        "--no-stage-summary",
        action="store_true",
        help="Do not print post-experiment overall/stage latency summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.camera is not None and args.background_camera is not None:
        raise ValueError("--camera and --background-camera are mutually exclusive")
    if (args.frame_width > 0) != (args.frame_height > 0):
        raise ValueError("--frame-width and --frame-height must be set together")
    if args.repeat_runs < 1:
        raise ValueError("--repeat-runs must be at least 1")

    config = load_config(args.config, args.strategy)
    if args.model is not None and args.model_paths_by_resolution is not None:
        raise ValueError("--model and --model-paths-by-resolution are mutually exclusive")
    if args.model is not None:
        if not args.model.exists():
            raise FileNotFoundError(f"ONNX model does not exist: {args.model}")
        inference_cfg = config.setdefault("inference", {})
        inference_cfg["model_path"] = str(args.model)
        # raspberry_pi4.yaml maps 640 to the original FP32 model.  Leaving that
        # mapping in place would silently defeat --model for the common 640 case.
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
    if args.thermal_state is not None:
        config.setdefault("thermal", {})["override_state"] = args.thermal_state
    if args.thermal_temp_c is not None:
        config.setdefault("thermal", {})["override_temp_c"] = args.thermal_temp_c
    if args.enable_thread_sessions:
        config.setdefault("inference", {})["enable_thread_sessions"] = True
    if args.thread_session_counts:
        config.setdefault("inference", {})["thread_session_counts"] = [
            int(item.strip()) for item in args.thread_session_counts.split(",") if item.strip()
        ]
    if args.apply_runtime_actions:
        config.setdefault("os_control", {})["apply_runtime_actions"] = True
    fan_cfg = config.setdefault("fan", {})
    if args.fan_control == "enabled":
        fan_cfg["enabled"] = True
        fan_cfg["enabled_strategies"] = ["*"]
    elif args.fan_control == "disabled":
        fan_cfg["enabled"] = False
    if args.enable_lk_tracking:
        config.setdefault("tracking", {})["enable_lk_tracking"] = True
    if args.lk_force_refresh_on_failure:
        config.setdefault("tracking", {})["lk_force_refresh_on_failure"] = True
    if args.lk_max_failure_ratio is not None:
        config.setdefault("tracking", {})["lk_max_failure_ratio"] = args.lk_max_failure_ratio
    if args.lk_min_valid_points is not None:
        config.setdefault("tracking", {})["lk_min_valid_points"] = args.lk_min_valid_points
    if args.query_budget_override is not None:
        if args.query_budget_override < 1:
            raise ValueError("--query-budget-override must be positive")
        config.setdefault("runtime", {})["query_budget_override"] = args.query_budget_override
    if args.max_query_budget is not None:
        if args.max_query_budget < 1:
            raise ValueError("--max-query-budget must be positive")
        config.setdefault("inference", {})["max_query_budget"] = args.max_query_budget
    if args.query_budget_mode is not None:
        config.setdefault("inference", {})["query_budget_mode"] = args.query_budget_mode
    if args.query_budget_input_name is not None:
        config.setdefault("inference", {})["query_budget_input_name"] = args.query_budget_input_name
    query_control = config.setdefault("query_budget_control", {})
    if args.temperature_query_budget:
        query_control["enabled"] = True
    query_budget_overrides = {
        "normal_budget": args.query_budget_normal,
        "warm_budget": args.query_budget_warm,
        "hot_budget": args.query_budget_hot,
        "critical_budget": args.query_budget_critical,
        "hysteresis_c": args.query_budget_hysteresis_c,
    }
    for key, value in query_budget_overrides.items():
        if value is None:
            continue
        if key.endswith("_budget") and value < 1:
            raise ValueError(f"--query-budget-{key[:-7]} must be positive")
        if key == "hysteresis_c" and value < 0:
            raise ValueError("--query-budget-hysteresis-c cannot be negative")
        query_control[key] = value

    base_output = args.output

    state = LiveDashboardState(
        max_history=args.history,
        jpeg_quality=args.jpeg_quality,
        jpeg_width=args.jpeg_width,
        score_threshold=args.score_threshold,
        show_stream=not args.no_video_stream,
    )
    server = LiveDashboardServer(state, host=args.host, port=args.port)
    server.start()

    ip = _best_effort_ip()
    bind_url = f"http://{args.host}:{args.port}"
    lan_url = f"http://{ip}:{args.port}" if ip else bind_url
    print("Live dashboard started.")
    print(f"  local/bind: {bind_url}")
    print(f"  LAN URL:    {lan_url}")
    print(f"  repeat:     {args.repeat_runs} run(s)")
    if args.model is not None:
        print(f"  model:      {args.model} (override; per-resolution mappings disabled)")
    if args.model_paths_by_resolution is not None:
        print(f"  model family: {args.model_paths_by_resolution}")
    if args.query_budget_mode is not None:
        print(f"  query budget: {args.query_budget_mode}")
    if args.temperature_query_budget:
        print("  query policy: temperature-adaptive")
    if args.fan_control != "config":
        print(f"  fan control: {args.fan_control}")
    if args.camera == "imx219-raw":
        interval = float(args.imx219_capture_interval_sec)
        capture_mode = (
            "continuous/as-fast-as-possible"
            if interval <= 0
            else f"throttled, min interval {interval:.3f}s"
        )
        print(f"  camera:     imx219-raw ({capture_mode})")
        print(f"  source:     {args.frame_source_mode}")
        print(f"  camera mode:{args.imx219_runtime_mode}")
        print("  inference:  decided by runtime strategy / LK tracking, not by camera capture")
    if args.background_camera:
        print(f"  runtime input:      {args.video}")
        print(f"  background camera:  {args.background_camera} (frames discarded)")
        print(f"  camera trigger:     {args.background_camera_trigger}")
    if args.frame_width > 0 and args.frame_height > 0:
        print(f"  runtime resize:     {args.frame_width}x{args.frame_height}")
    print("Press Ctrl+C to stop.")

    duration_sec = args.duration_min * 60.0
    camera_cfg = config.get("camera", {})
    use_camera = args.camera
    frame_size = (
        (int(args.frame_width), int(args.frame_height))
        if args.frame_width > 0 and args.frame_height > 0
        else None
    )

    def output_path_for_run(run_index: int) -> Path:
        if base_output is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_run{run_index:02d}" if args.repeat_runs > 1 else ""
            return ROOT / "experiments" / "logs" / f"{args.strategy}_live_{stamp}{suffix}.csv"
        if args.repeat_runs == 1:
            return base_output
        return base_output.with_name(
            f"{base_output.stem}_run{run_index:02d}{base_output.suffix}"
        )

    def detection_output_for_run(output_path: Path, run_index: int) -> Path | None:
        if not args.log_detections:
            return None
        if args.detection_output is None:
            return output_path.with_name(output_path.stem + "_detections.jsonl")
        if args.repeat_runs == 1:
            return args.detection_output
        return args.detection_output.with_name(
            f"{args.detection_output.stem}_run{run_index:02d}{args.detection_output.suffix}"
        )

    def make_source() -> FrameSource:
        return FrameSource(
            None if use_camera else args.video,
            synthetic=(args.video is None and use_camera is None),
            max_frames=int(duration_sec * 10) if args.dry_run else None,
            loop=args.loop_video,
            frame_size=frame_size,
            camera_backend=use_camera,
            camera_size=(
                int(camera_cfg.get("csi_width", 640)),
                int(camera_cfg.get("csi_height", 480)),
            ),
            camera_framerate=int(camera_cfg.get("csi_framerate", 30)),
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
            frame_source_mode=args.frame_source_mode,
        )

    def make_background_drain() -> BackgroundFrameDrain | None:
        if args.background_camera != "imx219-raw":
            return None
        background_source = FrameSource(
            None,
            synthetic=False,
            camera_backend="imx219-raw",
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
            imx219_capture_interval_sec=(
                0.0
                if args.background_camera_trigger != "continuous"
                else args.imx219_capture_interval_sec
            ),
            imx219_runtime_mode=args.imx219_runtime_mode,
            frame_source_mode="serial",
        )
        drain = BackgroundFrameDrain(background_source, "imx219", progress_interval=20)
        if args.background_camera_trigger == "continuous":
            drain.start()
            print("Background camera started continuously; frames are not sent to inference.")
        elif args.background_camera_trigger == "during-tracking":
            drain.start_on_demand()
            print("Background camera armed for async capture during LK tracking.")
        else:
            print("Background camera armed for post-tracking capture; frames are not sent to inference.")
        return drain

    def background_summary_lines(background_drain: BackgroundFrameDrain | None) -> list[str]:
        if background_drain is None:
            return []
        background_drain.stop()
        lines = []
        profile = background_drain.last_profile
        lines.append(f"Background camera frames captured: {background_drain.count}")
        if profile:
            lines.append(
                "Background camera last timings: "
                f"capture={profile.get('capture_ms', 0.0):.1f} ms, "
                f"isp={profile.get('isp_ms', 0.0):.1f} ms, "
                f"source={profile.get('source_total_ms', 0.0):.1f} ms"
            )
        if background_drain.error is not None:
            lines.append(
                f"Background camera last recoverable error: {background_drain.error}"
            )
        if background_drain.error_count:
            lines.append(
                f"Background camera recoverable errors: {background_drain.error_count}"
            )
        if background_drain.skipped_triggers:
            lines.append(
                f"Background camera skipped triggers: {background_drain.skipped_triggers}"
            )
        return lines

    background_drain = None
    shared_fan = PwmFanController(config)
    completed_runs: list[dict[str, Any]] = []
    try:
        for run_index in range(1, args.repeat_runs + 1):
            output_path = output_path_for_run(run_index)
            profile_output = output_path.with_name(output_path.stem + "_profile.csv")
            print("\nExperiment run")
            print("==============")
            print(f"Run:        {run_index}/{args.repeat_runs}")
            print(f"Log:        {output_path}")
            print(f"Profile:    {profile_output}")

            source = make_source()
            background_drain = make_background_drain()

            def on_tracking_start(_payload: dict) -> None:
                if (
                    background_drain is not None
                    and args.background_camera_trigger == "during-tracking"
                ):
                    background_drain.trigger_capture()

            def diagnostics_snapshot() -> dict[str, float | bool]:
                if background_drain is None:
                    return {}
                return background_drain.snapshot()

            def publish_live(
                payload: dict,
                frame,
                detections,
                resolved_input_resolution,
            ) -> None:
                state.publish(payload, frame, detections, resolved_input_resolution)
                if (
                    background_drain is not None
                    and args.background_camera_trigger == "post-tracking"
                    and not bool(payload.get("did_infer"))
                    and payload.get("tracking_mode") == "track"
                ):
                    background_drain.capture_once()

            loop = RuntimeLoop(
                config,
                source,
                dry_run=args.dry_run,
                duration_sec=duration_sec,
                log_path=output_path,
                detection_log_path=detection_output_for_run(output_path, run_index),
                live_callback=publish_live,
                tracking_start_callback=on_tracking_start,
                diagnostics_callback=diagnostics_snapshot,
                fan_controller=shared_fan,
            )

            try:
                log_path = loop.run()
                print(f"Experiment finished. Log: {log_path}")
                print(f"Profile log: {profile_output}")
                completed_runs.append(
                    {
                        "run_index": run_index,
                        "log_path": log_path,
                        "profile_path": profile_output,
                        "background_lines": [],
                    }
                )
            finally:
                background_lines = background_summary_lines(background_drain)
                if completed_runs and completed_runs[-1].get("run_index") == run_index:
                    completed_runs[-1]["background_lines"] = background_lines
                elif background_lines:
                    completed_runs.append(
                        {
                            "run_index": run_index,
                            "log_path": output_path,
                            "profile_path": profile_output,
                            "background_lines": background_lines,
                            "incomplete": True,
                        }
                    )
                background_drain = None

            if run_index < args.repeat_runs and args.repeat_cooldown_sec > 0:
                print(
                    f"Cooling down for {args.repeat_cooldown_sec:.1f} seconds "
                    f"before run {run_index + 1}/{args.repeat_runs}."
                )
                time.sleep(args.repeat_cooldown_sec)
    finally:
        if completed_runs and not args.no_stage_summary:
            print("\nRepeated experiment reports")
            print("===========================")
            for result in completed_runs:
                run_index = int(result["run_index"])
                log_path = Path(result["log_path"])
                profile_path = Path(result["profile_path"])
                print("\nPost-experiment stage latency summary")
                print("=====================================")
                print(f"Run: {run_index}/{args.repeat_runs}")
                if result.get("incomplete"):
                    print("Warning: this run did not finish cleanly; summary may be incomplete.")
                print_stage_latency_summary(log_path, profile_path=profile_path)
                for line in result.get("background_lines", []):
                    print(line)
        if background_drain is not None:
            background_drain.stop()
        shared_fan.close()
        server.stop()


def _best_effort_ip() -> str | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


if __name__ == "__main__":
    main()
