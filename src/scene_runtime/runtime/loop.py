"""Main runtime loop orchestrating scene, device, controller, and inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

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
    Orchestrates frame capture, scene/device updates, control, and inference.

    BACKBONE: logs actions but does not apply governor, affinity, or ONNX thread
    changes to the OS (see README TODO — Members 2, 3, 4).

    Skips inference according to ``inference_interval`` and reuses last detections.
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
        self._current_action = None

    def run(self) -> Path:
        """Execute loop until duration or source exhausted. Returns log path."""
        self._engine.load()
        self._logger.open()
        start = time.perf_counter()

        try:
            for frame in self._source:
                elapsed = time.perf_counter() - start
                if self._duration_sec and elapsed >= self._duration_sec:
                    break

                self._metrics.mark_frame()
                scene_state = self._scene.update(
                    frame, self._prev_frame, self._history
                )
                device_state = self._device.snapshot(self._config)
                action = self._controller.decide(
                    scene_state,
                    device_state,
                    self._metrics.snapshot(),
                )
                self._current_action = action

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

                self._inference_counter += 1
                self._write_log(scene_state, device_state, action, summary, latency_ms)
                self._prev_frame = frame.copy()
                self._frame_id += 1
        finally:
            self._logger.close()
            self._source.release()

        return self._log_path

    def _write_log(
        self,
        scene_state: dict[str, Any],
        device_state: dict[str, Any],
        action: Any,
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
