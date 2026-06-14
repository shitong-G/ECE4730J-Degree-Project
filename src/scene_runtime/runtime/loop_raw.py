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

    def run(self) -> Path:
        """Execute the 7-step per-frame loop until duration or source ends."""
        self._engine.load()
        self._logger.open()
        start = time.perf_counter()

        try:
            for frame in self._source:
                if self._duration_sec and (time.perf_counter() - start) >= self._duration_sec:
                    break

                self._process_frame(frame)
        finally:
            self._logger.close()
            self._source.release()

        return self._log_path

    def _process_frame(self, frame: np.ndarray) -> None:
        """One iteration of the 7-step runtime workflow."""
        self._metrics.mark_frame()

        # Step 1 — capture (frame supplied by FrameSource / camera / video)
        # Step 2 — lightweight scene features + workload label (stub classifier)
        scene_state = self._scene.update(frame, self._prev_frame, self._history)

        # Step 3 — SoC temp, freq, throttling (feeds back next frame via re-read)
        device_state = self._device.snapshot(self._config)

        # Step 4 — fuse scene × thermal runtime state
        runtime_state = self._controller.classify_runtime_state(scene_state, device_state)

        # Step 5 — Layer Router & Schedule → RuntimeAction (query_budget, decoder_layers, …)
        action = self._controller.decide(
            scene_state,
            device_state,
            self._metrics.snapshot(),
        )
        self._current_action = action
        _ = runtime_state  # logged indirectly via scene_state + device_state columns

        # Step 6 — infer or skip frame; reuse last detections when skipping
        run_infer = (self._inference_counter % action.inference_interval) == 0
        latency_ms = 0.0

        if run_infer:
            with Timer() as t:
                self._last_detections = self._engine.infer(frame, action)
            latency_ms = t.elapsed_ms
            self._metrics.record_latency(latency_ms)
            summary = detections_summary(self._last_detections)
            self._history.push(
                summary["detection_count"],
                [d.score for d in self._last_detections],
                latency_ms,
            )
        else:
            summary = detections_summary(self._last_detections)

        # Step 7 — log + update detection history / metrics for next decision
        self._write_log(scene_state, device_state, action, summary, latency_ms)
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
