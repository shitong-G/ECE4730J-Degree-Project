"""Main runtime loop orchestrating scene, device, controller, and inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.controller.runtime_controller import RuntimeDecisionController
from scene_runtime.device.state_monitor import DeviceStateMonitor
from scene_runtime.inference.onnx_engine import ONNXRTDETREngine
from scene_runtime.inference.postprocess import Detection, detections_summary
from scene_runtime.runtime.logger import LogRecord, RuntimeLogger
from scene_runtime.runtime.metrics import MetricsTracker
from scene_runtime.scene.detection_history import DetectionHistory
from scene_runtime.scene.workload_estimator import SceneWorkloadEstimator
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

        runtime_cfg = config.get("runtime", {})
        infer_cfg = config.get("inference", {})
        self._engine = ONNXRTDETREngine(
            model_path=infer_cfg.get("model_path"),
            dry_run=dry_run,
            dry_run_latency_ms=float(runtime_cfg.get("dry_run_latency_ms", 45.0)),
            providers=infer_cfg.get("onnx_providers"),
        )

        log_cfg = config.get("logging", {})
        default_log = Path(log_cfg.get("output_dir", "experiments/logs")) / f"run_{strategy}.csv"
        self._log_path = log_path or default_log
        self._logger = RuntimeLogger(self._log_path, fmt=log_cfg.get("format", "csv"))
        self._metrics = MetricsTracker()
        self._strategy = strategy

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

    def run(self) -> Path:
        """Execute the 7-step per-frame loop until duration or source ends."""
        self._engine.load()
        self._logger.open()
        self._profile_logger.open()

        start = time.perf_counter()
        try:
            for frame in self._source:
                if self._duration_sec and (time.perf_counter() - start) >= self._duration_sec:
                    break
                self._process_frame(frame)
        finally:
            self._profile_logger.close()
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

        # Step 6 — inference or skip
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

        if run_infer:
            t0 = time.perf_counter()
            self._last_detections = self._engine.infer(frame, action)
            infer_outer_ms = self._elapsed_ms(t0)

            infer_profile = self._engine.last_profile
            latency_ms = float(infer_profile.get("infer_total_ms", infer_outer_ms))

            self._metrics.record_latency(latency_ms)

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
        self._write_log(scene_state, device_state, action, summary, latency_ms)
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

        self._prev_frame = frame.copy()
        self._inference_counter += 1
        self._frame_id += 1

    def _write_log(
        self,
        scene_state: dict[str, Any],
        device_state: dict[str, Any],
        action: RuntimeAction,
        summary: dict[str, Any],
        latency_ms: float,
    ) -> None:
        record = LogRecord(
            timestamp=time.time(),
            frame_id=self._frame_id,
            strategy=self._strategy,
            workload=scene_state.get("workload", "medium"),
            temp_c=device_state.get("temp_c"),
            freq_mhz_avg=device_state.get("freq_mhz_avg"),
            arm_clock_mhz=device_state.get("arm_clock_mhz"),
            power_w=device_state.get("power_w"),
            latency_ms=latency_ms,
            fps=self._metrics.fps,
            input_resolution=action.input_resolution,
            inference_interval=action.inference_interval,
            cpu_threads=action.cpu_threads,
            governor=action.governor,
            decoder_layers=action.decoder_layers,
            query_budget=action.query_budget,
            detection_count=summary["detection_count"],
            confidence_mean=summary["confidence_mean"],
        )
        self._logger.write(record)
