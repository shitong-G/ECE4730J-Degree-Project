"""Main runtime loop orchestrating scene, device, controller, and inference."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.controller.query_budget import ThermalQueryBudgetController
from scene_runtime.device.action_applier import AppliedRuntimeState, RuntimeActionApplier
from scene_runtime.controller.runtime_controller import RuntimeDecisionController
from scene_runtime.device.fan import FanState, PwmFanController
from scene_runtime.device.state_monitor import DeviceStateMonitor
from scene_runtime.inference.onnx_engine import ONNXRTDETREngine
from scene_runtime.inference.postprocess import Detection, detections_summary
from scene_runtime.runtime.detection_logger import DetectionLogger
from scene_runtime.runtime.logger import LogRecord, RuntimeLogger
from scene_runtime.runtime.metrics import MetricsTracker
from scene_runtime.scene.detection_history import DetectionHistory
from scene_runtime.scene.workload_estimator import SceneWorkloadEstimator
from scene_runtime.tracking import LKTrackingReport, ResidualMotionGate, SparseLKBoxTracker
from scene_runtime.utils.timing import Timer
from scene_runtime.utils.video import FrameSource

from scene_runtime.runtime.profile_logger import ProfileLogger, ProfileRecord

class RuntimeLoop:
    """
    Per-frame embedded runtime pipeline (backbone).

    Matches the thesis figure's **Scene-Thermal Co-Adaptation** control plane on
    Raspberry Pi; RT-DETR backbone/encoder run inside ``ONNXRTDETREngine`` at Step 6.

    Per-frame workflow
    ------------------
    1. Capture current frame
    2. Extract lightweight scene workload features
    3. Read Raspberry Pi device state (SoC temp feedback path)
    4. Classify runtime state (scene workload × thermal)
    5. Select runtime action (layer router schedule + query budget + edge knobs)
    6. Run RT-DETR inference or skip/update per ``inference_interval``
    7. Log performance and update history for the next decision

    BACKBONE gaps (see README): adaptive policies, dynamic decoder/query in ONNX,
    applying governor/affinity/threads to the OS.
    """

    def __init__(
        self,
        config: dict[str, Any],
        frame_source: FrameSource,
        *,
        dry_run: bool = False,
        duration_sec: float | None = None,
        log_path: Path | None = None,
        detection_log_path: Path | None = None,
        live_callback: Callable[[dict[str, Any], np.ndarray, list[Detection], int | None], None] | None = None,
        tracking_start_callback: Callable[[dict[str, Any]], None] | None = None,
        diagnostics_callback: Callable[[], dict[str, Any]] | None = None,
        fan_controller: PwmFanController | None = None,
    ) -> None:
        self._config = config
        self._source = frame_source
        self._dry_run = dry_run
        self._duration_sec = duration_sec
        strategy = config.get("project", {}).get("strategy", "default")

        self._scene = SceneWorkloadEstimator(config)
        self._device = DeviceStateMonitor()
        self._controller = RuntimeDecisionController(config)
        self._fan = fan_controller or PwmFanController(config)
        self._owns_fan = fan_controller is None
        self._history = DetectionHistory()
        os_control_cfg = config.get("os_control", {})
        self._action_applier = RuntimeActionApplier(
            enabled=bool(os_control_cfg.get("apply_runtime_actions", False))
        )

        runtime_cfg = config.get("runtime", {})
        infer_cfg = config.get("inference", {})
        self._engine = ONNXRTDETREngine(
            model_path=infer_cfg.get("model_path"),
            model_paths_by_resolution=infer_cfg.get("model_paths_by_resolution"),
            dry_run=dry_run,
            dry_run_latency_ms=float(runtime_cfg.get("dry_run_latency_ms", 45.0)),
            providers=infer_cfg.get("onnx_providers"),
            enable_thread_sessions=bool(infer_cfg.get("enable_thread_sessions", False)),
            thread_session_counts=infer_cfg.get("thread_session_counts"),
            warmup_runs=int(infer_cfg.get("warmup_runs", 0)),
            warmup_resolutions=infer_cfg.get("warmup_resolutions"),
            warmup_threads=infer_cfg.get("warmup_threads"),
            inter_op_num_threads=int(infer_cfg.get("inter_op_num_threads", 1)),
            execution_mode=str(infer_cfg.get("execution_mode", "sequential")),
            graph_optimization_level=str(
                infer_cfg.get("graph_optimization_level", "all")
            ),
            enable_cpu_mem_arena=bool(infer_cfg.get("enable_cpu_mem_arena", True)),
            enable_mem_pattern=bool(infer_cfg.get("enable_mem_pattern", True)),
            log_severity_level=int(infer_cfg.get("log_severity_level", 3)),
            query_budget_mode=str(infer_cfg.get("query_budget_mode", "auto")),
            query_budget_input_name=str(
                infer_cfg.get("query_budget_input_name", "query_budget")
            ),
            max_query_budget=int(infer_cfg.get("max_query_budget", 300)),
        )

        log_cfg = config.get("logging", {})
        default_log = Path(log_cfg.get("output_dir", "experiments/logs")) / f"run_{strategy}.csv"
        self._log_path = log_path or default_log
        self._logger = RuntimeLogger(self._log_path, fmt=log_cfg.get("format", "csv"))
        self._metrics = MetricsTracker(
            window=int(runtime_cfg.get("metrics_window_frames", 120))
        )
        query_override = runtime_cfg.get("query_budget_override")
        self._query_budget_override = (
            int(query_override) if query_override is not None else None
        )
        self._thermal_query_budget = ThermalQueryBudgetController(config)
        self._query_budget_source = "action"
        self._query_budget_temperature_state: str | None = None
        self._pre_run_warmup_enabled = bool(
            runtime_cfg.get("pre_run_warmup_enabled", False)
        )
        self._pre_run_warmup_settle_sec = float(
            runtime_cfg.get("pre_run_warmup_settle_sec", 0.0)
        )
        self._pre_run_warmup_full_runs = int(
            runtime_cfg.get("pre_run_warmup_full_runs", 0)
        )
        self._pre_run_warmup_roi_runs = int(
            runtime_cfg.get("pre_run_warmup_roi_runs", 0)
        )
        self._pre_run_warmup_resolution = int(
            runtime_cfg.get("pre_run_warmup_resolution", 640)
        )
        self._pre_run_warmup_threads = int(
            runtime_cfg.get("pre_run_warmup_threads", 4)
        )
        self._pre_run_warmup_roi_resolution = int(
            runtime_cfg.get("pre_run_warmup_roi_resolution", 320)
        )
        target = runtime_cfg.get("temperature_warmup_target_c")
        self._temperature_warmup_target_c = (
            float(target) if target is not None else None
        )
        self._temperature_warmup_max_sec = max(
            0.0, float(runtime_cfg.get("temperature_warmup_max_sec", 0.0))
        )
        self._strategy = strategy
        tracking_cfg = config.get("tracking", {})
        self._lk_tracking_enabled = bool(tracking_cfg.get("enable_lk_tracking", False))
        self._lk_force_refresh = bool(
            tracking_cfg.get("lk_force_refresh_on_failure", False)
        )
        self._scene_event_triggered_tracking = bool(
            tracking_cfg.get("scene_event_triggered", False)
        )
        self._safety_refresh_frames = int(
            tracking_cfg.get("safety_refresh_frames", 300)
        )
        self._safety_refresh_defer_when_healthy = bool(
            tracking_cfg.get("safety_refresh_defer_when_healthy", True)
        )
        default_hard_limit = (
            self._safety_refresh_frames * 3
            if self._safety_refresh_frames > 0
            else 0
        )
        self._safety_refresh_hard_limit_frames = int(
            tracking_cfg.get("safety_refresh_hard_limit_frames", default_hard_limit)
        )
        self._safety_refresh_healthy_max_failure_ratio = float(
            tracking_cfg.get("safety_refresh_healthy_max_failure_ratio", 0.05)
        )
        self._safety_refresh_healthy_min_quality = float(
            tracking_cfg.get("safety_refresh_healthy_min_quality", 0.75)
        )
        self._roi_refresh_enabled = bool(tracking_cfg.get("roi_refresh_enabled", True))
        self._roi_refresh_resolution = int(tracking_cfg.get("roi_refresh_resolution", 320))
        self._roi_refresh_expand_ratio = float(
            tracking_cfg.get("roi_refresh_expand_ratio", 1.8)
        )
        self._roi_refresh_max_area_ratio = float(
            tracking_cfg.get("roi_refresh_max_area_ratio", 0.25)
        )
        self._roi_refresh_max_failure_ratio = float(
            tracking_cfg.get("roi_refresh_max_failure_ratio", 0.60)
        )
        self._roi_refresh_min_survivors = int(
            tracking_cfg.get("roi_refresh_min_survivors", 1)
        )
        self._roi_refresh_lk_quality_enabled = bool(
            tracking_cfg.get("roi_refresh_lk_quality_enabled", False)
        )
        self._roi_refresh_lk_max_failed_boxes = int(
            tracking_cfg.get("roi_refresh_lk_max_failed_boxes", 1)
        )
        self._roi_refresh_lk_max_failure_ratio = float(
            tracking_cfg.get("roi_refresh_lk_max_failure_ratio", 0.40)
        )
        self._roi_refresh_lk_max_area_ratio = float(
            tracking_cfg.get("roi_refresh_lk_max_area_ratio", 0.18)
        )
        self._roi_slow_fuse_enabled = bool(
            tracking_cfg.get("roi_slow_fuse_enabled", True)
        )
        self._roi_slow_fuse_threshold_ms = float(
            tracking_cfg.get("roi_slow_fuse_threshold_ms", 2500.0)
        )
        self._roi_slow_fuse_consecutive_limit = int(
            tracking_cfg.get("roi_slow_fuse_consecutive_limit", 1)
        )
        self._roi_slow_fuse_cooldown_frames = int(
            tracking_cfg.get("roi_slow_fuse_cooldown_frames", 180)
        )
        self._roi_slow_fuse_until_frame = -1
        self._roi_slow_fuse_consecutive_count = 0
        self._lk_quality_confirm_enabled = bool(
            tracking_cfg.get("lk_quality_confirm_enabled", True)
        )
        self._lk_quality_confirm_frames = int(
            tracking_cfg.get("lk_quality_confirm_frames", 1)
        )
        self._lk_quality_confirm_max_failure_ratio = float(
            tracking_cfg.get("lk_quality_confirm_max_failure_ratio", 0.60)
        )
        self._lk_quality_confirm_min_survivors = int(
            tracking_cfg.get("lk_quality_confirm_min_survivors", 1)
        )
        self._lk_quality_confirm_count = 0
        self._lk_quality_confirm_total_deferred = 0
        self._last_detector_frame = -10**9
        self._last_full_detector_frame = -10**9
        self._lk_tracker = (
            SparseLKBoxTracker(
                max_corners=int(tracking_cfg.get("lk_max_corners", 40)),
                min_valid_points=int(tracking_cfg.get("lk_min_valid_points", 5)),
                min_survival_ratio=float(
                    tracking_cfg.get("lk_min_survival_ratio", 0.35)
                ),
                max_forward_backward_error=float(
                    tracking_cfg.get("lk_max_forward_backward_error", 1.5)
                ),
                max_failure_ratio=float(
                    tracking_cfg.get("lk_max_failure_ratio", 0.30)
                ),
                redetect_interval=int(tracking_cfg.get("lk_redetect_interval", 5)),
                redetect_min_points=int(tracking_cfg.get("lk_redetect_min_points", 8)),
                win_size=int(tracking_cfg.get("lk_win_size", 15)),
                max_level=int(tracking_cfg.get("lk_max_level", 2)),
                max_iterations=int(tracking_cfg.get("lk_max_iterations", 15)),
            )
            if self._lk_tracking_enabled
            else None
        )
        self._motion_gate = (
            ResidualMotionGate(
                gate_width=int(tracking_cfg.get("gate_width", 320)),
                pixel_threshold=int(tracking_cfg.get("motion_threshold", 24)),
                outside_ratio_threshold=float(
                    tracking_cfg.get("outside_ratio_threshold", 0.010)
                ),
                min_component_area=int(tracking_cfg.get("min_component_area", 120)),
                scene_change_ratio_threshold=float(
                    tracking_cfg.get("scene_change_ratio_threshold", 0.35)
                ),
                mask_expand_ratio=float(tracking_cfg.get("mask_expand_ratio", 0.28)),
                enable_camera_compensation=not bool(
                    tracking_cfg.get("disable_camera_compensation", False)
                ),
            )
            if self._scene_event_triggered_tracking and self._lk_tracking_enabled
            else None
        )

        self._frame_id = 0
        self._inference_counter = 0
        self._detector_invocation_count = 0
        self._full_detector_invocation_count = 0
        self._roi_detector_invocation_count = 0
        self._last_detections: list[Detection] = []
        self._prev_frame: np.ndarray | None = None
        self._current_action: RuntimeAction | None = None
        self._last_detection_resolution: int | None = None

        profile_log_path = self._log_path.with_name(
            self._log_path.stem + "_profile.csv"
        )
        self._profile_logger = ProfileLogger(profile_log_path)
        self._profile_log_path = profile_log_path
        self._detection_logger = DetectionLogger(detection_log_path)
        self._live_callback = live_callback
        self._tracking_start_callback = tracking_start_callback
        self._diagnostics_callback = diagnostics_callback

    def run(self) -> Path:
        """Execute the 7-step per-frame loop until duration or source ends."""
        self._engine.load()
        source_iter = iter(self._source)
        try:
            first_frame = next(source_iter)
        except StopIteration:
            return self._log_path

        self._run_pre_logging_warmup(first_frame)
        self._run_temperature_warmup(first_frame)

        self._logger.open()
        self._profile_logger.open()
        self._detection_logger.open()

        start = time.perf_counter()
        try:
            self._process_frame(first_frame)
            for frame in source_iter:
                if self._duration_sec and (time.perf_counter() - start) >= self._duration_sec:
                    break
                self._process_frame(frame)
        finally:
            if self._owns_fan:
                self._fan.close()
            close_device = getattr(self._device, "close", None)
            if close_device is not None:
                close_device()
            self._profile_logger.close()
            self._detection_logger.close()
            self._logger.close()
            self._source.release()

        return self._log_path

    def _run_pre_logging_warmup(self, frame: np.ndarray) -> None:
        """Warm governor and ONNX paths before frame 0 is timed or logged."""
        if not self._pre_run_warmup_enabled:
            return
        runtime_cfg = self._config.get("runtime", {})
        warmup_action = RuntimeAction(
            mode="pre_run_warmup",
            input_resolution=self._pre_run_warmup_resolution,
            inference_interval=1,
            cpu_threads=self._pre_run_warmup_threads,
            governor=str(runtime_cfg.get("warmup_governor", "performance")),
        )
        self._action_applier.apply(warmup_action)
        if self._pre_run_warmup_settle_sec > 0:
            time.sleep(self._pre_run_warmup_settle_sec)

        for _ in range(max(0, self._pre_run_warmup_full_runs)):
            self._engine.infer(frame, warmup_action)

        roi_runs = max(0, self._pre_run_warmup_roi_runs)
        if roi_runs <= 0:
            return
        roi_frame = self._center_crop_for_warmup(frame)
        roi_action = replace(
            warmup_action,
            input_resolution=self._pre_run_warmup_roi_resolution,
            mode="pre_run_roi_warmup",
        )
        for _ in range(roi_runs):
            self._engine.infer(roi_frame, roi_action)

    def _run_temperature_warmup(self, frame: np.ndarray) -> None:
        """Heat the active RT-DETR/ORT path until a controlled start temperature.

        This executes before either CSV logger is opened, so warmup frames never
        contaminate formal metrics.  It deliberately uses the current run's
        model mapping and ONNX Runtime configuration rather than synthetic CPU
        load, preserving the relevant thermal workload.
        """
        target = self._temperature_warmup_target_c
        if target is None:
            return
        if self._temperature_warmup_max_sec <= 0:
            raise ValueError("temperature_warmup_max_sec must be positive")

        runtime_cfg = self._config.get("runtime", {})
        action = RuntimeAction(
            mode="temperature_warmup",
            input_resolution=int(runtime_cfg.get("default_input_resolution", 640)),
            inference_interval=1,
            cpu_threads=int(runtime_cfg.get("default_cpu_threads", 4)),
            governor=str(runtime_cfg.get("warmup_governor", "performance")),
        )
        self._action_applier.apply(action)
        started = time.perf_counter()
        warmup_frames = 0
        while True:
            # Always execute at least one inference.  This warms the selected
            # model/session even when the cooling gate released at exactly the
            # target temperature.
            self._engine.infer(frame, action)
            warmup_frames += 1
            device_state = self._device.snapshot(self._config)
            temp_c = device_state.get("temp_c")
            try:
                temperature = float(temp_c)
            except (TypeError, ValueError):
                raise RuntimeError("CPU temperature is unavailable during RT-DETR warmup")
            if temperature >= target:
                print(
                    f"RT-DETR temperature warmup complete: {temperature:.2f}C "
                    f">= {target:.2f}C after {warmup_frames} inference frame(s).",
                    flush=True,
                )
                return
            if time.perf_counter() - started >= self._temperature_warmup_max_sec:
                raise TimeoutError(
                    f"RT-DETR temperature warmup did not reach {target:.2f}C "
                    f"within {self._temperature_warmup_max_sec:.0f}s; last={temperature:.2f}C"
                )

    @staticmethod
    def _center_crop_for_warmup(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        side = max(32, min(height, width) // 3)
        cx = width // 2
        cy = height // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(width, x1 + side)
        y2 = min(height, y1 + side)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else frame

    def _elapsed_ms(self, t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _process_frame(self, frame: np.ndarray) -> None:
        """One iteration of the 7-step runtime workflow, with profiling."""
        frame_t0 = time.perf_counter()

        self._metrics.mark_frame()

        # Step 2 — scene workload estimation
        t0 = time.perf_counter()
        scene_state = self._scene.update(frame, self._prev_frame, self._history)
        scene_ms = self._elapsed_ms(t0)

        # Step 3 — device state
        t0 = time.perf_counter()
        device_state = self._device.snapshot(self._config)
        device_ms = self._elapsed_ms(t0)

        # Step 4 — runtime state classification
        t0 = time.perf_counter()
        runtime_state = self._controller.classify_runtime_state(scene_state, device_state)
        runtime_state_ms = self._elapsed_ms(t0)

        # Step 5 — runtime action decision
        t0 = time.perf_counter()
        action = self._controller.decide(
            scene_state,
            device_state,
            self._metrics.snapshot(),
        )
        if self._query_budget_override is not None:
            action = replace(action, query_budget=self._query_budget_override)
            self._query_budget_source = "fixed"
            self._query_budget_temperature_state = None
        elif self._thermal_query_budget.enabled:
            budget, budget_state = self._thermal_query_budget.update(
                str(device_state.get("thermal_state") or "unknown"),
                device_state.get("temp_c"),
            )
            action = replace(action, query_budget=budget)
            self._query_budget_source = "temperature"
            self._query_budget_temperature_state = budget_state
        else:
            self._query_budget_source = "action"
            self._query_budget_temperature_state = None
        decision_ms = self._elapsed_ms(t0)

        self._current_action = action
        _ = runtime_state
        applied_state = self._action_applier.apply(action)
        fan_state = self._fan.update(device_state, action.mode)

        # Step 6 — inference or skip
        if self._scene_event_triggered_tracking and self._lk_tracker is not None:
            run_infer = self._should_run_event_detector(action)
        else:
            run_infer = (self._inference_counter % action.inference_interval) == 0

        latency_ms = 0.0
        infer_outer_ms = 0.0
        infer_profile = {
            "preprocess_ms": 0.0,
            "build_feed_ms": 0.0,
            "session_select_ms": 0.0,
            "onnx_run_ms": 0.0,
            "postprocess_ms": 0.0,
            "infer_total_ms": 0.0,
        }
        infer_diagnostics: dict[str, float] = self._empty_infer_diagnostics()

        tracking_report = LKTrackingReport()
        detector_call_type: str | None = None
        detector_call_resolution: int | None = None

        if run_infer:
            (
                self._last_detections,
                infer_outer_ms,
                infer_profile,
            ) = self._infer_with_diagnostics(frame, action, infer_diagnostics)
            self._detector_invocation_count += 1
            self._full_detector_invocation_count += 1
            detector_call_type = "full"
            detector_call_resolution = (
                self._engine.last_resolved_input_resolution
                or action.input_resolution
            )
            latency_ms = float(infer_profile.get("infer_total_ms", infer_outer_ms))

            self._metrics.record_latency(latency_ms)
            self._metrics.record_inference()
            self._last_detection_resolution = self._engine.last_resolved_input_resolution
            tracking_report = self._reset_lk_tracker(frame)
            self._last_detector_frame = self._frame_id
            self._last_full_detector_frame = self._frame_id
        elif self._lk_tracker is not None:
            previous_boxes = self._detections_to_frame_boxes(
                self._last_detections,
                self._prev_frame,
                self._last_detection_resolution,
            )
            if self._tracking_start_callback is not None:
                self._tracking_start_callback(
                    {
                        "frame_id": self._frame_id,
                        "strategy": self._strategy,
                        "tracking_mode": "track",
                        "did_infer": False,
                    }
                )
            t0 = time.perf_counter()
            tracked_detections, tracking_report = self._lk_tracker.update(frame)
            tracking_report.tracking_ms = self._elapsed_ms(t0)
            self._last_detections = tracked_detections
            if self._scene_event_triggered_tracking:
                self._apply_event_refresh_gate(
                    frame,
                    action,
                    tracking_report,
                    previous_boxes,
                )
                self._apply_lk_quality_confirmation(tracking_report)
            if tracking_report.should_refresh and (
                self._lk_force_refresh or self._scene_event_triggered_tracking
            ):
                if not self._can_run_detector_now(action):
                    tracking_report.reason = f"refresh_deferred_{tracking_report.reason}"
                    tracking_report.should_refresh = False
                    t0 = None
                else:
                    t0 = time.perf_counter()
                if t0 is not None:
                    refresh_reason = tracking_report.reason
                    roi_result = self._try_roi_refresh(
                        frame,
                        action,
                        tracking_report,
                        tracked_detections,
                        infer_diagnostics,
                    )
                    if roi_result is not None:
                        original_tracking_report = tracking_report
                        self._last_detections, infer_outer_ms, infer_profile = roi_result
                        self._detector_invocation_count += 1
                        self._roi_detector_invocation_count += 1
                        detector_call_type = "roi"
                        detector_call_resolution = (
                            self._engine.last_resolved_input_resolution
                            or self._roi_refresh_resolution
                        )
                        self._update_roi_slow_fuse(infer_profile)
                        latency_ms = float(infer_profile.get("infer_total_ms", infer_outer_ms))
                        self._metrics.record_latency(latency_ms)
                        self._metrics.record_inference()
                        run_infer = True
                        self._last_detector_frame = self._frame_id
                        tracking_report = self._reset_lk_tracker(
                            frame,
                            reason=f"roi_refresh_{refresh_reason}",
                            input_resolution=self._last_detection_resolution,
                        )
                        self._copy_roi_refresh_fields(
                            original_tracking_report,
                            tracking_report,
                        )
                    else:
                        original_tracking_report = tracking_report
                        (
                            self._last_detections,
                            infer_outer_ms,
                            infer_profile,
                        ) = self._infer_with_diagnostics(
                            frame,
                            action,
                            infer_diagnostics,
                        )
                        self._detector_invocation_count += 1
                        self._full_detector_invocation_count += 1
                        detector_call_type = "full"
                        detector_call_resolution = (
                            self._engine.last_resolved_input_resolution
                            or action.input_resolution
                        )
                        latency_ms = float(infer_profile.get("infer_total_ms", infer_outer_ms))
                        self._metrics.record_latency(latency_ms)
                        self._metrics.record_inference()
                        run_infer = True
                        self._last_detector_frame = self._frame_id
                        self._last_full_detector_frame = self._frame_id
                        self._last_detection_resolution = self._engine.last_resolved_input_resolution
                        tracking_report = self._reset_lk_tracker(
                            frame,
                            reason=f"forced_refresh_{refresh_reason}",
                        )
                        self._copy_roi_refresh_fields(
                            original_tracking_report,
                            tracking_report,
                        )

        # Detection summary
        t0 = time.perf_counter()
        summary = detections_summary(self._last_detections)
        summary_ms = self._elapsed_ms(t0)

        if run_infer:
            self._history.push(
                summary["detection_count"],
                [d.score for d in self._last_detections],
                latency_ms,
            )

        # Original main log
        t0 = time.perf_counter()
        self._write_log(
            scene_state,
            device_state,
            action,
            applied_state,
            summary,
            latency_ms,
            run_infer,
            tracking_report,
            fan_state,
            detector_call_type,
            detector_call_resolution,
        )
        main_log_write_ms = self._elapsed_ms(t0)

        frame_total_ms = self._elapsed_ms(frame_t0)
        source_profile = self._source.last_profile
        serial_total_ms = frame_total_ms + float(source_profile.get("source_total_ms", 0.0))

        self._profile_logger.write(
            ProfileRecord(
                timestamp=time.time(),
                frame_id=self._frame_id,
                strategy=self._strategy,
                did_infer=run_infer,

                serial_total_ms=serial_total_ms,
                source_total_ms=float(source_profile.get("source_total_ms", 0.0)),
                source_wait_ms=float(source_profile.get("source_wait_ms", 0.0)),
                capture_ms=float(source_profile.get("capture_ms", 0.0)),
                isp_ms=float(source_profile.get("isp_ms", 0.0)),
                source_resize_ms=float(source_profile.get("source_resize_ms", 0.0)),
                source_save_ms=float(source_profile.get("source_save_ms", 0.0)),
                source_runtime_resize_ms=float(source_profile.get("source_runtime_resize_ms", 0.0)),
                source_consumer_wait_ms=float(source_profile.get("source_consumer_wait_ms", 0.0)),
                source_frame_age_ms=float(source_profile.get("source_frame_age_ms", 0.0)),
                source_dropped_frames=float(source_profile.get("source_dropped_frames", 0.0)),
                source_error_count=float(source_profile.get("source_error_count", 0.0)),
                frame_total_ms=frame_total_ms,
                scene_ms=scene_ms,
                device_ms=device_ms,
                runtime_state_ms=runtime_state_ms,
                decision_ms=decision_ms,

                infer_outer_ms=infer_outer_ms,
                preprocess_ms=float(infer_profile.get("preprocess_ms", 0.0)),
                build_feed_ms=float(infer_profile.get("build_feed_ms", 0.0)),
                session_select_ms=float(infer_profile.get("session_select_ms", 0.0)),
                onnx_run_ms=float(infer_profile.get("onnx_run_ms", 0.0)),
                postprocess_ms=float(infer_profile.get("postprocess_ms", 0.0)),
                infer_total_ms=float(infer_profile.get("infer_total_ms", latency_ms)),

                diag_infer_start_load1=float(infer_diagnostics.get("diag_infer_start_load1", 0.0)),
                diag_infer_end_load1=float(infer_diagnostics.get("diag_infer_end_load1", 0.0)),
                diag_infer_start_mem_available_mb=float(infer_diagnostics.get("diag_infer_start_mem_available_mb", 0.0)),
                diag_infer_end_mem_available_mb=float(infer_diagnostics.get("diag_infer_end_mem_available_mb", 0.0)),
                diag_infer_start_process_threads=float(infer_diagnostics.get("diag_infer_start_process_threads", 0.0)),
                diag_infer_end_process_threads=float(infer_diagnostics.get("diag_infer_end_process_threads", 0.0)),
                diag_infer_start_bg_active=float(infer_diagnostics.get("diag_infer_start_bg_active", 0.0)),
                diag_infer_end_bg_active=float(infer_diagnostics.get("diag_infer_end_bg_active", 0.0)),
                diag_infer_start_bg_pending=float(infer_diagnostics.get("diag_infer_start_bg_pending", 0.0)),
                diag_infer_end_bg_pending=float(infer_diagnostics.get("diag_infer_end_bg_pending", 0.0)),
                diag_infer_start_bg_count=float(infer_diagnostics.get("diag_infer_start_bg_count", 0.0)),
                diag_infer_end_bg_count=float(infer_diagnostics.get("diag_infer_end_bg_count", 0.0)),
                diag_infer_start_bg_skipped=float(infer_diagnostics.get("diag_infer_start_bg_skipped", 0.0)),
                diag_infer_end_bg_skipped=float(infer_diagnostics.get("diag_infer_end_bg_skipped", 0.0)),
                diag_infer_start_bg_errors=float(infer_diagnostics.get("diag_infer_start_bg_errors", 0.0)),
                diag_infer_end_bg_errors=float(infer_diagnostics.get("diag_infer_end_bg_errors", 0.0)),
                diag_infer_start_bg_last_source_ms=float(infer_diagnostics.get("diag_infer_start_bg_last_source_ms", 0.0)),
                diag_infer_end_bg_last_source_ms=float(infer_diagnostics.get("diag_infer_end_bg_last_source_ms", 0.0)),
                diag_infer_bg_captures_delta=float(infer_diagnostics.get("diag_infer_bg_captures_delta", 0.0)),
                diag_infer_bg_skipped_delta=float(infer_diagnostics.get("diag_infer_bg_skipped_delta", 0.0)),
                diag_infer_bg_errors_delta=float(infer_diagnostics.get("diag_infer_bg_errors_delta", 0.0)),

                summary_ms=summary_ms,
                main_log_write_ms=main_log_write_ms,
            )
        )

        self._detection_logger.write(
            timestamp=time.time(),
            frame_id=self._frame_id,
            strategy=self._strategy,
            did_infer=run_infer,
            tracking_mode=tracking_report.mode,
            tracking_reason=tracking_report.reason,
            input_resolution=action.input_resolution,
            resolved_input_resolution=self._last_detection_resolution,
            query_budget_requested=(
                self._engine.last_requested_query_budget
                if run_infer
                else action.query_budget
            ),
            query_budget_applied=(
                self._engine.last_applied_query_budget if run_infer else None
            ),
            query_budget_mode=(
                self._engine.last_query_budget_mode if run_infer else "not_invoked"
            ),
            query_output_count=(
                self._engine.last_query_output_count if run_infer else None
            ),
            detections=self._last_detections,
        )

        if self._live_callback is not None:
            live_payload = self._build_live_payload(
                scene_state=scene_state,
                device_state=device_state,
                action=action,
                applied_state=applied_state,
                summary=summary,
                latency_ms=latency_ms,
                did_infer=run_infer,
                tracking_report=tracking_report,
                fan_state=fan_state,
                infer_profile=infer_profile,
                frame_total_ms=frame_total_ms,
                source_profile=source_profile,
                serial_total_ms=serial_total_ms,
                infer_diagnostics=infer_diagnostics,
            )
            self._live_callback(
                live_payload,
                frame,
                self._last_detections,
                self._last_detection_resolution,
            )

        self._prev_frame = frame.copy()
        self._inference_counter += 1
        self._frame_id += 1

    def _should_run_event_detector(self, action: RuntimeAction) -> bool:
        """Initial detector scheduling for event-triggered scene policies."""
        if self._last_detector_frame < 0:
            return True
        if not self._last_detections:
            return self._can_run_detector_now(action)
        return False

    def _can_run_detector_now(self, action: RuntimeAction) -> bool:
        """Honor thermal policy by treating action interval as a minimum gap."""
        min_gap = max(1, int(action.inference_interval))
        return (self._frame_id - self._last_detector_frame) >= min_gap

    def _try_roi_refresh(
        self,
        frame: np.ndarray,
        action: RuntimeAction,
        tracking_report: LKTrackingReport,
        tracked_detections: list[Detection],
        infer_diagnostics: dict[str, float] | None = None,
    ) -> tuple[list[Detection], float, dict[str, float]] | None:
        """Run one low-resolution ROI detector refresh when local evidence allows it."""
        if not self._roi_refresh_enabled:
            return None
        if self._last_detection_resolution is None:
            return None
        if tracking_report.reason == "lk_tracking_quality_degraded":
            if not self._roi_refresh_lk_quality_enabled:
                return None
            if not self._lk_quality_roi_allowed(tracking_report):
                return None
        elif tracking_report.reason != "unexplained_motion_outside_tracks":
            return None
        if (
            getattr(self, "_roi_slow_fuse_enabled", False)
            and getattr(self, "_frame_id", 0)
            < getattr(self, "_roi_slow_fuse_until_frame", -1)
        ):
            tracking_report.roi_refresh_candidate = True
            tracking_report.roi_refresh_reason = tracking_report.reason
            tracking_report.roi_refresh_reject_reason = "slow_fuse_active"
            return None

        boxes = tracking_report.refresh_boxes_frame or tracking_report.failed_boxes_frame
        roi = self._build_roi(frame, boxes)
        if roi is None:
            return None
        x1, y1, x2, y2 = roi
        frame_h, frame_w = frame.shape[:2]
        roi_area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(frame_w * frame_h))
        max_area_ratio = (
            self._roi_refresh_lk_max_area_ratio
            if tracking_report.reason == "lk_tracking_quality_degraded"
            else self._roi_refresh_max_area_ratio
        )
        tracking_report.roi_refresh_candidate = True
        tracking_report.roi_refresh_reason = tracking_report.reason
        tracking_report.roi_refresh_area_ratio = float(roi_area_ratio)
        tracking_report.roi_refresh_width_px = float(x2 - x1)
        tracking_report.roi_refresh_height_px = float(y2 - y1)
        tracking_report.roi_refresh_max_area_ratio = float(max_area_ratio)
        if roi_area_ratio > max_area_ratio:
            tracking_report.roi_refresh_reject_reason = "area_too_large"
            return None

        crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
        if crop.size == 0:
            tracking_report.roi_refresh_reject_reason = "empty_crop"
            return None

        tracking_report.roi_refresh_applied = True
        tracking_report.roi_refresh_reject_reason = None
        roi_action = replace(action, input_resolution=self._roi_refresh_resolution)
        roi_detections, infer_outer_ms, infer_profile = self._infer_with_diagnostics(
            crop,
            roi_action,
            infer_diagnostics,
        )
        roi_resolution = self._engine.last_resolved_input_resolution or self._roi_refresh_resolution
        mapped = self._map_roi_detections_to_full_resolution(
            roi_detections,
            roi=roi,
            roi_resolution=roi_resolution,
            frame_shape=frame.shape,
            output_resolution=self._last_detection_resolution,
        )
        merged = self._merge_detections(tracked_detections, mapped)
        return merged, infer_outer_ms, infer_profile

    def _infer_with_diagnostics(
        self,
        frame: np.ndarray,
        action: RuntimeAction,
        infer_diagnostics: dict[str, float] | None = None,
    ) -> tuple[list[Detection], float, dict[str, float]]:
        """Run one detector call while sampling lightweight diagnostic context."""
        diagnostics = infer_diagnostics if infer_diagnostics is not None else {}
        diagnostics.update(self._sample_infer_diagnostics("start"))
        t0 = time.perf_counter()
        detections = self._engine.infer(frame, action)
        infer_outer_ms = self._elapsed_ms(t0)
        diagnostics.update(self._sample_infer_diagnostics("end"))
        self._finalize_infer_diagnostics(diagnostics)
        return detections, infer_outer_ms, self._engine.last_profile

    @staticmethod
    def _empty_infer_diagnostics() -> dict[str, float]:
        keys = [
            "diag_infer_start_load1",
            "diag_infer_end_load1",
            "diag_infer_start_mem_available_mb",
            "diag_infer_end_mem_available_mb",
            "diag_infer_start_process_threads",
            "diag_infer_end_process_threads",
            "diag_infer_start_bg_active",
            "diag_infer_end_bg_active",
            "diag_infer_start_bg_pending",
            "diag_infer_end_bg_pending",
            "diag_infer_start_bg_count",
            "diag_infer_end_bg_count",
            "diag_infer_start_bg_skipped",
            "diag_infer_end_bg_skipped",
            "diag_infer_start_bg_errors",
            "diag_infer_end_bg_errors",
            "diag_infer_start_bg_last_source_ms",
            "diag_infer_end_bg_last_source_ms",
            "diag_infer_bg_captures_delta",
            "diag_infer_bg_skipped_delta",
            "diag_infer_bg_errors_delta",
        ]
        return {key: 0.0 for key in keys}

    def _sample_infer_diagnostics(self, phase: str) -> dict[str, float]:
        prefix = f"diag_infer_{phase}_"
        diagnostics = {
            f"{prefix}load1": self._load_average_1m(),
            f"{prefix}mem_available_mb": self._mem_available_mb(),
            f"{prefix}process_threads": self._process_thread_count(),
            f"{prefix}bg_active": 0.0,
            f"{prefix}bg_pending": 0.0,
            f"{prefix}bg_count": 0.0,
            f"{prefix}bg_skipped": 0.0,
            f"{prefix}bg_errors": 0.0,
            f"{prefix}bg_last_source_ms": 0.0,
        }
        diagnostics_callback = getattr(self, "_diagnostics_callback", None)
        if diagnostics_callback is None:
            return diagnostics
        try:
            extra = diagnostics_callback()
        except Exception:
            return diagnostics
        diagnostics[f"{prefix}bg_active"] = self._bool_float(extra.get("bg_camera_active"))
        diagnostics[f"{prefix}bg_pending"] = self._bool_float(extra.get("bg_camera_pending"))
        diagnostics[f"{prefix}bg_count"] = self._float_or_zero(extra.get("bg_camera_count"))
        diagnostics[f"{prefix}bg_skipped"] = self._float_or_zero(extra.get("bg_camera_skipped"))
        diagnostics[f"{prefix}bg_errors"] = self._float_or_zero(extra.get("bg_camera_errors"))
        diagnostics[f"{prefix}bg_last_source_ms"] = self._float_or_zero(
            extra.get("bg_camera_last_source_ms")
        )
        return diagnostics

    @staticmethod
    def _finalize_infer_diagnostics(diagnostics: dict[str, float]) -> None:
        diagnostics["diag_infer_bg_captures_delta"] = max(
            0.0,
            diagnostics.get("diag_infer_end_bg_count", 0.0)
            - diagnostics.get("diag_infer_start_bg_count", 0.0),
        )
        diagnostics["diag_infer_bg_skipped_delta"] = max(
            0.0,
            diagnostics.get("diag_infer_end_bg_skipped", 0.0)
            - diagnostics.get("diag_infer_start_bg_skipped", 0.0),
        )
        diagnostics["diag_infer_bg_errors_delta"] = max(
            0.0,
            diagnostics.get("diag_infer_end_bg_errors", 0.0)
            - diagnostics.get("diag_infer_start_bg_errors", 0.0),
        )

    @staticmethod
    def _load_average_1m() -> float:
        try:
            return float(os.getloadavg()[0])
        except (AttributeError, OSError):
            return 0.0

    @staticmethod
    def _mem_available_mb() -> float:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return 0.0
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            return 0.0
        return 0.0

    @staticmethod
    def _process_thread_count() -> float:
        task_dir = Path("/proc/self/task")
        try:
            if task_dir.exists():
                return float(sum(1 for _ in task_dir.iterdir()))
        except OSError:
            pass
        return 0.0

    @staticmethod
    def _bool_float(value: Any) -> float:
        if isinstance(value, str):
            return 1.0 if value.strip().lower() in {"1", "true", "yes"} else 0.0
        return 1.0 if bool(value) else 0.0

    @staticmethod
    def _float_or_zero(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _update_roi_slow_fuse(self, infer_profile: dict[str, float]) -> None:
        if not getattr(self, "_roi_slow_fuse_enabled", False):
            return
        latency_ms = float(
            infer_profile.get("onnx_run_ms")
            or infer_profile.get("infer_total_ms")
            or 0.0
        )
        if latency_ms <= getattr(self, "_roi_slow_fuse_threshold_ms", 2500.0):
            self._roi_slow_fuse_consecutive_count = 0
            return
        self._roi_slow_fuse_consecutive_count = (
            getattr(self, "_roi_slow_fuse_consecutive_count", 0) + 1
        )
        if self._roi_slow_fuse_consecutive_count < max(
            1,
            getattr(self, "_roi_slow_fuse_consecutive_limit", 1),
        ):
            return
        self._roi_slow_fuse_until_frame = max(
            getattr(self, "_roi_slow_fuse_until_frame", -1),
            getattr(self, "_frame_id", 0)
            + max(1, getattr(self, "_roi_slow_fuse_cooldown_frames", 180)),
        )
        self._roi_slow_fuse_consecutive_count = 0

    @staticmethod
    def _copy_roi_refresh_fields(
        source: LKTrackingReport,
        target: LKTrackingReport,
    ) -> None:
        target.roi_refresh_candidate = source.roi_refresh_candidate
        target.roi_refresh_applied = source.roi_refresh_applied
        target.roi_refresh_reason = source.roi_refresh_reason
        target.roi_refresh_reject_reason = source.roi_refresh_reject_reason
        target.roi_refresh_area_ratio = source.roi_refresh_area_ratio
        target.roi_refresh_width_px = source.roi_refresh_width_px
        target.roi_refresh_height_px = source.roi_refresh_height_px
        target.roi_refresh_max_area_ratio = source.roi_refresh_max_area_ratio
        target.lk_quality_confirm_count = source.lk_quality_confirm_count
        target.lk_quality_confirm_deferred = source.lk_quality_confirm_deferred
        target.lk_quality_confirm_total_deferred = (
            source.lk_quality_confirm_total_deferred
        )

    def _apply_lk_quality_confirmation(
        self,
        tracking_report: LKTrackingReport,
    ) -> None:
        """Defer soft LK-quality refreshes until degradation persists."""
        if tracking_report.reason != "lk_tracking_quality_degraded":
            self._lk_quality_confirm_count = 0
            tracking_report.lk_quality_confirm_count = 0
            tracking_report.lk_quality_confirm_total_deferred = (
                self._lk_quality_confirm_total_deferred
            )
            return

        if not tracking_report.should_refresh:
            tracking_report.lk_quality_confirm_count = self._lk_quality_confirm_count
            tracking_report.lk_quality_confirm_total_deferred = (
                self._lk_quality_confirm_total_deferred
            )
            return

        self._lk_quality_confirm_count += 1
        tracking_report.lk_quality_confirm_count = self._lk_quality_confirm_count
        tracking_report.lk_quality_confirm_total_deferred = (
            self._lk_quality_confirm_total_deferred
        )

        if not self._lk_quality_confirm_enabled:
            return
        if self._lk_quality_confirm_frames <= 0:
            return
        if tracking_report.track_count_after < self._lk_quality_confirm_min_survivors:
            return
        if (
            tracking_report.failure_ratio
            > self._lk_quality_confirm_max_failure_ratio
        ):
            return
        if self._lk_quality_confirm_count > self._lk_quality_confirm_frames:
            return

        tracking_report.should_refresh = False
        tracking_report.reason = "lk_quality_degraded_confirming"
        tracking_report.lk_quality_confirm_deferred = True
        self._lk_quality_confirm_total_deferred += 1
        tracking_report.lk_quality_confirm_total_deferred = (
            self._lk_quality_confirm_total_deferred
        )

    def _lk_quality_roi_allowed(self, tracking_report: LKTrackingReport) -> bool:
        """Allow ROI for LK degradation only when the failure is genuinely local."""
        if tracking_report.track_count_after < self._roi_refresh_min_survivors:
            return False
        if tracking_report.failure_ratio > self._roi_refresh_lk_max_failure_ratio:
            return False
        failed_count = len(tracking_report.failed_boxes_frame)
        if failed_count == 0:
            return False
        if failed_count > self._roi_refresh_lk_max_failed_boxes:
            return False
        return True

    def _build_roi(
        self,
        frame: np.ndarray,
        boxes: list[np.ndarray],
    ) -> tuple[float, float, float, float] | None:
        valid = [np.asarray(box, dtype=np.float32) for box in boxes if box is not None]
        if not valid:
            return None
        height, width = frame.shape[:2]
        stacked = np.vstack(valid)
        x1 = float(np.min(stacked[:, 0]))
        y1 = float(np.min(stacked[:, 1]))
        x2 = float(np.max(stacked[:, 2]))
        y2 = float(np.max(stacked[:, 3]))
        if x2 <= x1 or y2 <= y1:
            return None
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        half_w = 0.5 * (x2 - x1) * self._roi_refresh_expand_ratio
        half_h = 0.5 * (y2 - y1) * self._roi_refresh_expand_ratio
        side = max(half_w * 2.0, half_h * 2.0, 32.0)
        half = 0.5 * side
        rx1 = max(0.0, cx - half)
        ry1 = max(0.0, cy - half)
        rx2 = min(float(width), cx + half)
        ry2 = min(float(height), cy + half)
        if rx2 - rx1 < 8 or ry2 - ry1 < 8:
            return None
        return rx1, ry1, rx2, ry2

    @staticmethod
    def _map_roi_detections_to_full_resolution(
        detections: list[Detection],
        *,
        roi: tuple[float, float, float, float],
        roi_resolution: int,
        frame_shape: tuple[int, ...],
        output_resolution: int,
    ) -> list[Detection]:
        x1, y1, x2, y2 = roi
        roi_w = max(1.0, x2 - x1)
        roi_h = max(1.0, y2 - y1)
        frame_h, frame_w = frame_shape[:2]
        mapped: list[Detection] = []
        for detection in detections:
            bx1, by1, bx2, by2 = detection.bbox
            fx1 = x1 + bx1 / float(roi_resolution) * roi_w
            fy1 = y1 + by1 / float(roi_resolution) * roi_h
            fx2 = x1 + bx2 / float(roi_resolution) * roi_w
            fy2 = y1 + by2 / float(roi_resolution) * roi_h
            mapped.append(
                Detection(
                    class_id=detection.class_id,
                    score=detection.score,
                    bbox=(
                        float(fx1 / max(1, frame_w) * output_resolution),
                        float(fy1 / max(1, frame_h) * output_resolution),
                        float(fx2 / max(1, frame_w) * output_resolution),
                        float(fy2 / max(1, frame_h) * output_resolution),
                    ),
                )
            )
        return mapped

    @staticmethod
    def _merge_detections(
        base: list[Detection],
        updates: list[Detection],
        *,
        iou_threshold: float = 0.50,
    ) -> list[Detection]:
        candidates = sorted(
            [*base, *updates],
            key=lambda detection: detection.score,
            reverse=True,
        )
        kept: list[Detection] = []
        for detection in candidates:
            if all(_detection_iou(detection, existing) < iou_threshold for existing in kept):
                kept.append(detection)
        return kept

    def _apply_event_refresh_gate(
        self,
        frame: np.ndarray,
        action: RuntimeAction,
        tracking_report: LKTrackingReport,
        previous_boxes: list[np.ndarray],
    ) -> None:
        """Update tracking_report when event-triggered scene logic wants RT-DETR."""
        if tracking_report.should_refresh:
            tracking_report.reason = "lk_tracking_quality_degraded"
            tracking_report.refresh_boxes_frame = list(tracking_report.failed_boxes_frame)
            return

        if self._motion_gate is not None:
            current_boxes = self._detections_to_frame_boxes(
                self._last_detections,
                frame,
                self._last_detection_resolution,
            )
            gate_report = self._motion_gate.analyze(
                self._prev_frame,
                frame,
                previous_boxes,
                current_boxes,
            )
            if gate_report.should_refresh:
                tracking_report.should_refresh = True
                tracking_report.reason = gate_report.reason
                tracking_report.refresh_boxes_frame = list(gate_report.roi_boxes_frame)
                return

        if (
            self._safety_refresh_frames > 0
            and self._frame_id - self._last_full_detector_frame >= self._safety_refresh_frames
        ):
            frames_since_full = self._frame_id - self._last_full_detector_frame
            hard_limit_reached = (
                self._safety_refresh_hard_limit_frames > 0
                and frames_since_full >= self._safety_refresh_hard_limit_frames
            )
            if (
                self._safety_refresh_defer_when_healthy
                and not hard_limit_reached
                and self._tracking_report_can_defer_safety_refresh(tracking_report)
            ):
                tracking_report.reason = "track_healthy_safety_refresh_deferred"
                return
            tracking_report.should_refresh = True
            tracking_report.reason = (
                "long_interval_safety_refresh_hard_limit"
                if hard_limit_reached
                else "long_interval_safety_refresh"
            )
            return

        if action.inference_interval > 1:
            tracking_report.reason = "track_healthy_thermal_min_gap"

    def _tracking_report_can_defer_safety_refresh(
        self,
        tracking_report: LKTrackingReport,
    ) -> bool:
        if tracking_report.should_refresh:
            return False
        if tracking_report.mode != "track":
            return False
        if tracking_report.track_count_after <= 0:
            return False
        if tracking_report.failure_ratio > self._safety_refresh_healthy_max_failure_ratio:
            return False
        if tracking_report.mean_quality < self._safety_refresh_healthy_min_quality:
            return False
        return True

    def _detections_to_frame_boxes(
        self,
        detections: list[Detection],
        frame: np.ndarray | None,
        input_resolution: int | None,
    ) -> list[np.ndarray]:
        """Convert detector-space boxes to original frame coordinates."""
        if frame is None or not detections:
            return []
        height, width = frame.shape[:2]
        resolution = float(input_resolution or max(height, width))
        sx = width / resolution
        sy = height / resolution
        boxes: list[np.ndarray] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            boxes.append(
                np.asarray([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)
            )
        return boxes

    def _reset_lk_tracker(
        self,
        frame: np.ndarray,
        *,
        reason: str = "detector_frame",
        input_resolution: int | None = None,
    ) -> LKTrackingReport:
        if self._lk_tracker is None:
            return LKTrackingReport()
        report = self._lk_tracker.reset(
            frame,
            self._last_detections,
            input_resolution or self._last_detection_resolution,
        )
        self._lk_quality_confirm_count = 0
        report.reason = reason
        return report

    def _build_live_payload(
        self,
        *,
        scene_state: dict[str, Any],
        device_state: dict[str, Any],
        action: RuntimeAction,
        applied_state: AppliedRuntimeState,
        summary: dict[str, Any],
        latency_ms: float,
        did_infer: bool,
        tracking_report: LKTrackingReport,
        fan_state: FanState,
        infer_profile: dict[str, float],
        frame_total_ms: float,
        source_profile: dict[str, float],
        serial_total_ms: float,
        infer_diagnostics: dict[str, float],
    ) -> dict[str, Any]:
        loop_fps = self._metrics.fps
        throttling = device_state.get("throttling") or {}
        return {
            "timestamp": time.time(),
            "frame_id": self._frame_id,
            "strategy": self._strategy,
            "workload": scene_state.get("workload", "medium"),
            "thermal_state": self._controller.last_control_thermal_state,
            "raw_thermal_state": self._controller.last_raw_thermal_state,
            "control_thermal_state": self._controller.last_control_thermal_state,
            "action_mode": action.mode,
            "decision_reason": self._controller.last_decision_reason,
            "thermal_pressure_level": self._controller.last_thermal_pressure_level,
            "temp_slope_c_per_min": self._controller.last_temp_slope_c_per_min,
            "temp_c": device_state.get("temp_c"),
            "freq_mhz_avg": device_state.get("freq_mhz_avg"),
            "arm_clock_mhz": device_state.get("arm_clock_mhz"),
            "arm_clock_stale": device_state.get("arm_clock_stale"),
            "firmware_poll_ms": device_state.get("firmware_poll_ms"),
            "power_w": device_state.get("power_w"),
            "throttling_raw": throttling.get("raw"),
            "throttling_stale": device_state.get("throttling_stale"),
            "under_voltage": throttling.get("under_voltage"),
            "arm_freq_capped": throttling.get("arm_freq_capped"),
            "currently_throttled": throttling.get("currently_throttled"),
            "soft_temp_limit": throttling.get("soft_temp_limit"),
            "under_voltage_occurred": throttling.get("under_voltage_occurred"),
            "arm_freq_capped_occurred": throttling.get("arm_freq_capped_occurred"),
            "throttled_occurred": throttling.get("throttled_occurred"),
            "soft_temp_limit_occurred": throttling.get("soft_temp_limit_occurred"),
            "did_infer": did_infer,
            "tracking_mode": tracking_report.mode,
            "tracking_reason": tracking_report.reason,
            "tracking_ms": tracking_report.tracking_ms,
            "tracking_track_count_before": tracking_report.track_count_before,
            "tracking_track_count_after": tracking_report.track_count_after,
            "tracking_failed_box_count": len(tracking_report.failed_boxes_frame),
            "tracking_failure_ratio": tracking_report.failure_ratio,
            "tracking_mean_quality": tracking_report.mean_quality,
            "tracking_should_refresh": tracking_report.should_refresh,
            "lk_quality_confirm_count": tracking_report.lk_quality_confirm_count,
            "lk_quality_confirm_deferred": tracking_report.lk_quality_confirm_deferred,
            "lk_quality_confirm_total_deferred": (
                tracking_report.lk_quality_confirm_total_deferred
            ),
            "roi_refresh_candidate": tracking_report.roi_refresh_candidate,
            "roi_refresh_applied": tracking_report.roi_refresh_applied,
            "roi_refresh_reason": tracking_report.roi_refresh_reason,
            "roi_refresh_reject_reason": tracking_report.roi_refresh_reject_reason,
            "roi_refresh_area_ratio": tracking_report.roi_refresh_area_ratio,
            "roi_refresh_width_px": tracking_report.roi_refresh_width_px,
            "roi_refresh_height_px": tracking_report.roi_refresh_height_px,
            "roi_refresh_max_area_ratio": tracking_report.roi_refresh_max_area_ratio,
            "fan_enabled": fan_state.enabled,
            "fan_duty_cycle": fan_state.duty_cycle,
            "fan_mode": fan_state.mode,
            "latency_ms": latency_ms,
            "loop_fps": loop_fps,
            "fps": loop_fps,
            "effective_inference_fps": loop_fps / max(action.inference_interval, 1),
            "actual_inference_fps": self._metrics.inference_fps,
            "input_resolution": action.input_resolution,
            "resolved_input_resolution": self._last_detection_resolution,
            "inference_interval": action.inference_interval,
            "cpu_threads": action.cpu_threads,
            "governor": action.governor,
            "requested_governor": applied_state.requested_governor,
            "applied_governor": applied_state.applied_governor,
            "governor_applied": applied_state.governor_applied,
            "requested_cpu_affinity": applied_state.requested_cpu_affinity,
            "applied_cpu_affinity": applied_state.applied_cpu_affinity,
            "cpu_affinity_applied": applied_state.cpu_affinity_applied,
            "decoder_layers": action.decoder_layers,
            "query_budget": action.query_budget,
            "query_budget_requested": (
                self._engine.last_requested_query_budget if did_infer else action.query_budget
            ),
            "query_budget_applied": (
                self._engine.last_applied_query_budget if did_infer else None
            ),
            "query_budget_mode": (
                self._engine.last_query_budget_mode if did_infer else "not_invoked"
            ),
            "query_budget_source": self._query_budget_source,
            "query_budget_temperature_state": self._query_budget_temperature_state,
            "query_output_count": (
                self._engine.last_query_output_count if did_infer else None
            ),
            "query_budget_ratio": (
                self._engine.last_applied_query_budget / self._engine.max_query_budget
                if did_infer
                and self._engine.last_applied_query_budget is not None
                and self._engine.max_query_budget > 0
                else None
            ),
            "detection_count": summary["detection_count"],
            "confidence_mean": summary["confidence_mean"],
            "serial_total_ms": serial_total_ms,
            "source_total_ms": float(source_profile.get("source_total_ms", 0.0)),
            "source_wait_ms": float(source_profile.get("source_wait_ms", 0.0)),
            "capture_ms": float(source_profile.get("capture_ms", 0.0)),
            "isp_ms": float(source_profile.get("isp_ms", 0.0)),
            "source_resize_ms": float(source_profile.get("source_resize_ms", 0.0)),
            "source_save_ms": float(source_profile.get("source_save_ms", 0.0)),
            "source_runtime_resize_ms": float(source_profile.get("source_runtime_resize_ms", 0.0)),
            "source_consumer_wait_ms": float(source_profile.get("source_consumer_wait_ms", 0.0)),
            "source_frame_age_ms": float(source_profile.get("source_frame_age_ms", 0.0)),
            "source_dropped_frames": float(source_profile.get("source_dropped_frames", 0.0)),
            "source_error_count": float(source_profile.get("source_error_count", 0.0)),
            "frame_total_ms": frame_total_ms,
            "preprocess_ms": float(infer_profile.get("preprocess_ms", 0.0)),
            "build_feed_ms": float(infer_profile.get("build_feed_ms", 0.0)),
            "session_select_ms": float(infer_profile.get("session_select_ms", 0.0)),
            "onnx_run_ms": float(infer_profile.get("onnx_run_ms", 0.0)),
            "postprocess_ms": float(infer_profile.get("postprocess_ms", 0.0)),
            "infer_total_ms": float(infer_profile.get("infer_total_ms", latency_ms)),
            "diag_infer_start_load1": float(infer_diagnostics.get("diag_infer_start_load1", 0.0)),
            "diag_infer_end_load1": float(infer_diagnostics.get("diag_infer_end_load1", 0.0)),
            "diag_infer_start_mem_available_mb": float(infer_diagnostics.get("diag_infer_start_mem_available_mb", 0.0)),
            "diag_infer_end_mem_available_mb": float(infer_diagnostics.get("diag_infer_end_mem_available_mb", 0.0)),
            "diag_infer_start_process_threads": float(infer_diagnostics.get("diag_infer_start_process_threads", 0.0)),
            "diag_infer_end_process_threads": float(infer_diagnostics.get("diag_infer_end_process_threads", 0.0)),
            "diag_infer_start_bg_active": float(infer_diagnostics.get("diag_infer_start_bg_active", 0.0)),
            "diag_infer_end_bg_active": float(infer_diagnostics.get("diag_infer_end_bg_active", 0.0)),
            "diag_infer_start_bg_pending": float(infer_diagnostics.get("diag_infer_start_bg_pending", 0.0)),
            "diag_infer_end_bg_pending": float(infer_diagnostics.get("diag_infer_end_bg_pending", 0.0)),
            "diag_infer_start_bg_count": float(infer_diagnostics.get("diag_infer_start_bg_count", 0.0)),
            "diag_infer_end_bg_count": float(infer_diagnostics.get("diag_infer_end_bg_count", 0.0)),
            "diag_infer_start_bg_skipped": float(infer_diagnostics.get("diag_infer_start_bg_skipped", 0.0)),
            "diag_infer_end_bg_skipped": float(infer_diagnostics.get("diag_infer_end_bg_skipped", 0.0)),
            "diag_infer_start_bg_errors": float(infer_diagnostics.get("diag_infer_start_bg_errors", 0.0)),
            "diag_infer_end_bg_errors": float(infer_diagnostics.get("diag_infer_end_bg_errors", 0.0)),
            "diag_infer_start_bg_last_source_ms": float(infer_diagnostics.get("diag_infer_start_bg_last_source_ms", 0.0)),
            "diag_infer_end_bg_last_source_ms": float(infer_diagnostics.get("diag_infer_end_bg_last_source_ms", 0.0)),
            "diag_infer_bg_captures_delta": float(infer_diagnostics.get("diag_infer_bg_captures_delta", 0.0)),
            "diag_infer_bg_skipped_delta": float(infer_diagnostics.get("diag_infer_bg_skipped_delta", 0.0)),
            "diag_infer_bg_errors_delta": float(infer_diagnostics.get("diag_infer_bg_errors_delta", 0.0)),
        }

    def _write_log(
        self,
        scene_state: dict[str, Any],
        device_state: dict[str, Any],
        action: RuntimeAction,
        applied_state: AppliedRuntimeState,
        summary: dict[str, Any],
        latency_ms: float,
        did_infer: bool,
        tracking_report: LKTrackingReport,
        fan_state: FanState,
        detector_call_type: str | None,
        detector_call_resolution: int | None,
    ) -> None:
        loop_fps = self._metrics.fps
        effective_inference_fps = loop_fps / max(action.inference_interval, 1)
        actual_inference_fps = self._metrics.inference_fps
        throttling = device_state.get("throttling") or {}
        raw_thermal_state = self._controller.last_raw_thermal_state
        control_thermal_state = self._controller.last_control_thermal_state
        processed_frames = self._frame_id + 1
        record = LogRecord(
            timestamp=time.time(),
            frame_id=self._frame_id,
            strategy=self._strategy,
            workload=scene_state.get("workload", "medium"),
            thermal_state=control_thermal_state,
            raw_thermal_state=raw_thermal_state,
            control_thermal_state=control_thermal_state,
            action_mode=action.mode,
            decision_reason=self._controller.last_decision_reason,
            thermal_pressure_level=self._controller.last_thermal_pressure_level,
            temp_slope_c_per_min=self._controller.last_temp_slope_c_per_min,
            temp_c=device_state.get("temp_c"),
            freq_mhz_avg=device_state.get("freq_mhz_avg"),
            arm_clock_mhz=device_state.get("arm_clock_mhz"),
            arm_clock_stale=device_state.get("arm_clock_stale"),
            firmware_poll_ms=float(device_state.get("firmware_poll_ms") or 0.0),
            power_w=device_state.get("power_w"),
            throttling_raw=throttling.get("raw"),
            throttling_stale=device_state.get("throttling_stale"),
            under_voltage=throttling.get("under_voltage"),
            arm_freq_capped=throttling.get("arm_freq_capped"),
            currently_throttled=throttling.get("currently_throttled"),
            soft_temp_limit=throttling.get("soft_temp_limit"),
            under_voltage_occurred=throttling.get("under_voltage_occurred"),
            arm_freq_capped_occurred=throttling.get("arm_freq_capped_occurred"),
            throttled_occurred=throttling.get("throttled_occurred"),
            soft_temp_limit_occurred=throttling.get("soft_temp_limit_occurred"),
            did_infer=did_infer,
            detector_invocation_count=self._detector_invocation_count,
            detector_invocation_ratio=(
                self._detector_invocation_count / processed_frames
            ),
            full_detector_invocation_count=self._full_detector_invocation_count,
            full_detector_invocation_ratio=(
                self._full_detector_invocation_count / processed_frames
            ),
            roi_detector_invocation_count=self._roi_detector_invocation_count,
            roi_detector_invocation_ratio=(
                self._roi_detector_invocation_count / processed_frames
            ),
            detector_call_type=detector_call_type,
            detector_call_resolution=detector_call_resolution,
            tracking_mode=tracking_report.mode,
            tracking_reason=tracking_report.reason,
            tracking_ms=tracking_report.tracking_ms,
            tracking_track_count_before=tracking_report.track_count_before,
            tracking_track_count_after=tracking_report.track_count_after,
            tracking_failed_box_count=len(tracking_report.failed_boxes_frame),
            tracking_failure_ratio=tracking_report.failure_ratio,
            tracking_mean_quality=tracking_report.mean_quality,
            tracking_should_refresh=tracking_report.should_refresh,
            lk_quality_confirm_count=tracking_report.lk_quality_confirm_count,
            lk_quality_confirm_deferred=tracking_report.lk_quality_confirm_deferred,
            lk_quality_confirm_total_deferred=(
                tracking_report.lk_quality_confirm_total_deferred
            ),
            roi_refresh_candidate=tracking_report.roi_refresh_candidate,
            roi_refresh_applied=tracking_report.roi_refresh_applied,
            roi_refresh_reason=tracking_report.roi_refresh_reason,
            roi_refresh_reject_reason=tracking_report.roi_refresh_reject_reason,
            roi_refresh_area_ratio=tracking_report.roi_refresh_area_ratio,
            roi_refresh_width_px=tracking_report.roi_refresh_width_px,
            roi_refresh_height_px=tracking_report.roi_refresh_height_px,
            roi_refresh_max_area_ratio=tracking_report.roi_refresh_max_area_ratio,
            latency_ms=latency_ms,
            fps=loop_fps,
            loop_fps=loop_fps,
            effective_inference_fps=effective_inference_fps,
            actual_inference_fps=actual_inference_fps,
            input_resolution=action.input_resolution,
            resolved_input_resolution=self._last_detection_resolution,
            inference_interval=action.inference_interval,
            cpu_threads=action.cpu_threads,
            governor=action.governor,
            requested_governor=applied_state.requested_governor,
            applied_governor=applied_state.applied_governor,
            governor_applied=applied_state.governor_applied,
            governor_apply_error=applied_state.governor_apply_error,
            requested_cpu_affinity=applied_state.requested_cpu_affinity,
            applied_cpu_affinity=applied_state.applied_cpu_affinity,
            cpu_affinity_applied=applied_state.cpu_affinity_applied,
            cpu_affinity_apply_error=applied_state.cpu_affinity_apply_error,
            decoder_layers=action.decoder_layers,
            query_budget=action.query_budget,
            query_budget_requested=(
                self._engine.last_requested_query_budget if did_infer else action.query_budget
            ),
            query_budget_applied=(
                self._engine.last_applied_query_budget if did_infer else None
            ),
            query_budget_mode=(
                self._engine.last_query_budget_mode if did_infer else "not_invoked"
            ),
            query_budget_supported=(
                self._engine.last_query_budget_supported if did_infer else None
            ),
            query_budget_source=self._query_budget_source,
            query_budget_temperature_state=self._query_budget_temperature_state,
            query_output_count=(
                self._engine.last_query_output_count if did_infer else None
            ),
            query_budget_ratio=(
                (
                    self._engine.last_applied_query_budget
                    / self._engine.max_query_budget
                )
                if did_infer
                and self._engine.last_applied_query_budget is not None
                and self._engine.max_query_budget > 0
                else None
            ),
            onnx_run_ms=(
                float(self._engine.last_profile.get("onnx_run_ms", 0.0))
                if did_infer
                else None
            ),
            fan_enabled=fan_state.enabled,
            fan_duty_cycle=fan_state.duty_cycle,
            fan_mode=fan_state.mode,
            detection_count=summary["detection_count"],
            confidence_mean=summary["confidence_mean"],
        )
        self._logger.write(record)


def _detection_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)
