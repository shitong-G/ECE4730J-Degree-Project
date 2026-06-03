"""ONNX Runtime RT-DETR inference engine with dry-run fallback."""

from __future__ import annotations

import random
import time
from typing import Any

import cv2
import numpy as np

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.inference.postprocess import Detection, postprocess_rtdetr_outputs
from scene_runtime.inference.rtdetr_engine import BaseInferenceEngine


class ONNXRTDETREngine(BaseInferenceEngine):
    """
    RT-DETR inference via ONNX Runtime.

    Supports ``dry_run=True`` to simulate latency and fake detections without a model.
    """

    def __init__(
        self,
        model_path: str | None = None,
        dry_run: bool = False,
        dry_run_latency_ms: float = 45.0,
        providers: list[str] | None = None,
    ) -> None:
        self._model_path = model_path
        self._dry_run = dry_run
        self._dry_run_latency_ms = dry_run_latency_ms
        self._providers = providers or ["CPUExecutionProvider"]
        self._session: Any = None

    def load(self) -> None:
        """Load ONNX session or no-op in dry-run mode."""
        if self._dry_run:
            return
        if not self._model_path:
            raise FileNotFoundError(
                "model_path required for non-dry-run inference. "
                "Export ONNX via scripts/export_model_onnx.sh"
            )
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for real inference. "
                "Install with: pip install onnxruntime"
            ) from e

        # TODO: set session options for cpu_threads from RuntimeAction
        self._session = ort.InferenceSession(
            self._model_path,
            providers=self._providers,
        )
        # TODO: cache input/output names and expected shapes from RT-DETR export

    def preprocess(self, frame: np.ndarray, input_resolution: int) -> np.ndarray:
        """BGR resize, RGB, normalize — adjust to match RT-DETR export."""
        resized = cv2.resize(frame, (input_resolution, input_resolution))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        # NCHW
        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0)

    def postprocess(self, raw_outputs: list[np.ndarray]) -> list[Detection]:
        return postprocess_rtdetr_outputs(raw_outputs)

    def infer(self, frame: np.ndarray, config: RuntimeAction) -> list[Detection]:
        """Run inference or dry-run simulation."""
        blob = self.preprocess(frame, config.input_resolution)
        if self._dry_run:
            time.sleep(self._dry_run_latency_ms / 1000.0)
            return self._fake_detections(config)
        if self._session is None:
            raise RuntimeError("Engine not loaded. Call load() first.")
        # TODO: map blob to correct input name; apply decoder_layers / query_budget if ONNX supports
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})
        return self.postprocess(list(outputs))

    def _fake_detections(self, config: RuntimeAction) -> list[Detection]:
        """Generate plausible fake detections for dry-run experiments."""
        n = random.randint(0, 5)
        h = config.input_resolution
        w = config.input_resolution
        dets: list[Detection] = []
        for _ in range(n):
            x1 = random.uniform(0, w * 0.7)
            y1 = random.uniform(0, h * 0.7)
            x2 = x1 + random.uniform(20, w * 0.25)
            y2 = y1 + random.uniform(20, h * 0.25)
            dets.append(
                Detection(
                    class_id=random.randint(0, 79),
                    score=random.uniform(0.4, 0.95),
                    bbox=(x1, y1, min(x2, w), min(y2, h)),
                )
            )
        return dets
