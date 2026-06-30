"""Main runtime loop orchestrating scene, device, controller, and inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.device.action_applier import AppliedRuntimeState, RuntimeActionApplier
from scene_runtime.controller.runtime_controller import RuntimeDecisionController
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
    ) -> None:
        self._config = config
        self._source = frame_source
        self._dry_run = dry_run
        self._duration_sec = duration_sec
        strategy = config.get("project", {}).get("strategy", "default")

        self._scene = SceneWorkloadEstimator(config)
        self._device = DeviceStateMonitor()
        self._controller = RuntimeDecisionController(config)
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
        )

        log_cfg = config.get("logging", {})
        default_log = Path(log_cfg.get("output_dir", "experiments/logs")) / f"run_{strategy}.csv"
        self._log_path = log_path or default_log
        self._logger = RuntimeLogger(self._log_path, fmt=log_cfg.get("format", "csv"))
        self._metrics = MetricsTracker(
            window=int(runtime_cfg.get("metrics_window_frames", 120))
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
        self._last_detector_frame = -10**9
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
        self._last_detections: list[Detection] = []
        self._prev_frame: np.ndarray | None = None
        self._current_action: RuntimeAction | None = None

        profile_log_path = self._log_path.with_name(
            self._log_path.stem + "_profile.csv"
        )
        self._profile_logger = ProfileLogger(profile_log_path)
        self._profile_log_path = profile_log_path
        self._detection_logger = DetectionLogger(detection_log_path)
        self._live_callback = live_callback

    def run(self) -> Path:
        """Execute the 7-step per-frame loop until duration or source ends."""
        self._engine.load()
        self._logger.open()
        self._profile_logger.open()
        self._detection_logger.open()

        start = time.perf_counter()
        try:
            for frame in self._source:
                if self._duration_sec and (time.perf_counter() - start) >= self._duration_sec:
                    break
                self._process_frame(frame)
        finally:
            self._profile_logger.close()
            self._detection_logger.close()
            self._logger.close()
            self._source.release()

        return self._log_path

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
        decision_ms = self._elapsed_ms(t0)

        self._current_action = action
        _ = runtime_state
        applied_state = self._action_applier.apply(action)

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
            "onnx_run_ms": 0.0,
            "postprocess_ms": 0.0,
            "infer_total_ms": 0.0,
        }

        tracking_report = LKTrackingReport()

        if run_infer:
            t0 = time.perf_counter()
            self._last_detections = self._engine.infer(frame, action)
            infer_outer_ms = self._elapsed_ms(t0)

            infer_profile = self._engine.last_profile
            latency_ms = float(infer_profile.get("infer_total_ms", infer_outer_ms))

            self._metrics.record_latency(latency_ms)
            self._metrics.record_inference()
            tracking_report = self._reset_lk_tracker(frame)
            self._last_detector_frame = self._frame_id
        elif self._lk_tracker is not None:
            previous_boxes = self._detections_to_frame_boxes(
                self._last_detections,
                self._prev_frame,
                self._engine.last_resolved_input_resolution,
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
                    self._last_detections = self._engine.infer(frame, action)
                    infer_outer_ms = self._elapsed_ms(t0)
                    infer_profile = self._engine.last_profile
                    latency_ms = float(infer_profile.get("infer_total_ms", infer_outer_ms))
                    self._metrics.record_latency(latency_ms)
                    self._metrics.record_inference()
                    run_infer = True
                    self._last_detector_frame = self._frame_id
                    tracking_report = self._reset_lk_tracker(
                        frame,
                        reason=f"forced_refresh_{tracking_report.reason}",
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
        )
        main_log_write_ms = self._elapsed_ms(t0)

        frame_total_ms = self._elapsed_ms(frame_t0)

        self._profile_logger.write(
            ProfileRecord(
                timestamp=time.time(),
                frame_id=self._frame_id,
                strategy=self._strategy,
                did_infer=run_infer,

                frame_total_ms=frame_total_ms,
                scene_ms=scene_ms,
                device_ms=device_ms,
                runtime_state_ms=runtime_state_ms,
                decision_ms=decision_ms,

                infer_outer_ms=infer_outer_ms,
                preprocess_ms=float(infer_profile.get("preprocess_ms", 0.0)),
                build_feed_ms=float(infer_profile.get("build_feed_ms", 0.0)),
                onnx_run_ms=float(infer_profile.get("onnx_run_ms", 0.0)),
                postprocess_ms=float(infer_profile.get("postprocess_ms", 0.0)),
                infer_total_ms=float(infer_profile.get("infer_total_ms", latency_ms)),

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
            resolved_input_resolution=self._engine.last_resolved_input_resolution,
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
                infer_profile=infer_profile,
                frame_total_ms=frame_total_ms,
            )
            self._live_callback(
                live_payload,
                frame,
                self._last_detections,
                self._engine.last_resolved_input_resolution,
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
            return

        if self._motion_gate is not None:
            current_boxes = self._detections_to_frame_boxes(
                self._last_detections,
                frame,
                self._engine.last_resolved_input_resolution,
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
                return

        if (
            self._safety_refresh_frames > 0
            and self._frame_id - self._last_detector_frame >= self._safety_refresh_frames
        ):
            tracking_report.should_refresh = True
            tracking_report.reason = "long_interval_safety_refresh"
            return

        if action.inference_interval > 1:
            tracking_report.reason = "track_healthy_thermal_min_gap"

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
    ) -> LKTrackingReport:
        if self._lk_tracker is None:
            return LKTrackingReport()
        report = self._lk_tracker.reset(
            frame,
            self._last_detections,
            self._engine.last_resolved_input_resolution,
        )
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
        infer_profile: dict[str, float],
        frame_total_ms: float,
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
            "power_w": device_state.get("power_w"),
            "throttling_raw": throttling.get("raw"),
            "under_voltage": throttling.get("under_voltage"),
            "arm_freq_capped": throttling.get("arm_freq_capped"),
            "currently_throttled": throttling.get("currently_throttled"),
            "soft_temp_limit": throttling.get("soft_temp_limit"),
            "did_infer": did_infer,
            "tracking_mode": tracking_report.mode,
            "tracking_reason": tracking_report.reason,
            "tracking_ms": tracking_report.tracking_ms,
            "tracking_failure_ratio": tracking_report.failure_ratio,
            "tracking_mean_quality": tracking_report.mean_quality,
            "tracking_should_refresh": tracking_report.should_refresh,
            "latency_ms": latency_ms,
            "loop_fps": loop_fps,
            "fps": loop_fps,
            "effective_inference_fps": loop_fps / max(action.inference_interval, 1),
            "actual_inference_fps": self._metrics.inference_fps,
            "input_resolution": action.input_resolution,
            "resolved_input_resolution": self._engine.last_resolved_input_resolution,
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
            "detection_count": summary["detection_count"],
            "confidence_mean": summary["confidence_mean"],
            "frame_total_ms": frame_total_ms,
            "preprocess_ms": float(infer_profile.get("preprocess_ms", 0.0)),
            "build_feed_ms": float(infer_profile.get("build_feed_ms", 0.0)),
            "onnx_run_ms": float(infer_profile.get("onnx_run_ms", 0.0)),
            "postprocess_ms": float(infer_profile.get("postprocess_ms", 0.0)),
            "infer_total_ms": float(infer_profile.get("infer_total_ms", latency_ms)),
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
    ) -> None:
        loop_fps = self._metrics.fps
        effective_inference_fps = loop_fps / max(action.inference_interval, 1)
        actual_inference_fps = self._metrics.inference_fps
        throttling = device_state.get("throttling") or {}
        raw_thermal_state = self._controller.last_raw_thermal_state
        control_thermal_state = self._controller.last_control_thermal_state
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
            power_w=device_state.get("power_w"),
            throttling_raw=throttling.get("raw"),
            under_voltage=throttling.get("under_voltage"),
            arm_freq_capped=throttling.get("arm_freq_capped"),
            currently_throttled=throttling.get("currently_throttled"),
            soft_temp_limit=throttling.get("soft_temp_limit"),
            did_infer=did_infer,
            tracking_mode=tracking_report.mode,
            tracking_reason=tracking_report.reason,
            tracking_ms=tracking_report.tracking_ms,
            tracking_failure_ratio=tracking_report.failure_ratio,
            tracking_mean_quality=tracking_report.mean_quality,
            tracking_should_refresh=tracking_report.should_refresh,
            latency_ms=latency_ms,
            fps=loop_fps,
            loop_fps=loop_fps,
            effective_inference_fps=effective_inference_fps,
            actual_inference_fps=actual_inference_fps,
            input_resolution=action.input_resolution,
            resolved_input_resolution=self._engine.last_resolved_input_resolution,
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
            detection_count=summary["detection_count"],
            confidence_mean=summary["confidence_mean"],
        )
        self._logger.write(record)
