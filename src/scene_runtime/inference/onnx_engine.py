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
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._fixed_input_size: int | None = None

        self._last_profile: dict[str, float] = {
            "preprocess_ms": 0.0,
            "build_feed_ms": 0.0,
            "onnx_run_ms": 0.0,
            "postprocess_ms": 0.0,
            "infer_total_ms": 0.0,
        }

    @property
    def last_profile(self) -> dict[str, float]:
        """Return timing profile from the most recent infer() call."""
        return dict(self._last_profile)
        
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
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._fixed_input_size = self._read_fixed_input_size()

    def _read_fixed_input_size(self) -> int | None:
        """Return fixed H=W when the ONNX image input has static spatial dims."""
        if self._session is None:
            return None
        shape = self._session.get_inputs()[0].shape
        if len(shape) != 4:
            return None
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int) and height == width:
            return height
        return None

    def _resolve_input_resolution(self, requested: int) -> int:
        """Use the exported ONNX spatial size when the model fixes H and W."""
        if self._fixed_input_size is not None:
            return self._fixed_input_size
        return requested

    def _build_feeds(self, blob: np.ndarray, input_resolution: int) -> dict[str, np.ndarray]:
        """Map preprocessed tensors to RT-DETR ONNX inputs."""
        feeds: dict[str, np.ndarray] = {}
        orig_sizes = np.array(
            [[input_resolution, input_resolution]],
            dtype=np.int64,
        )
        for name in self._input_names:
            if name == "images":
                feeds[name] = blob
            elif name == "orig_target_sizes":
                feeds[name] = orig_sizes
            else:
                raise ValueError(f"Unsupported ONNX input: {name}")
        return feeds

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
        """Run inference or dry-run simulation, with module-level timing."""
        profile = {
            "preprocess_ms": 0.0,
            "build_feed_ms": 0.0,
            "onnx_run_ms": 0.0,
            "postprocess_ms": 0.0,
            "infer_total_ms": 0.0,
        }
    
        total_t0 = time.perf_counter()
    
        if self._dry_run:
            time.sleep(self._dry_run_latency_ms / 1000.0)
            detections = self._fake_detections(config)
            profile["infer_total_ms"] = (time.perf_counter() - total_t0) * 1000.0
            self._last_profile = profile
            return detections
    
        if self._session is None:
            raise RuntimeError("Engine not loaded. Call load() first.")
    
        input_resolution = self._resolve_input_resolution(config.input_resolution)
    
        t0 = time.perf_counter()
        blob = self.preprocess(frame, input_resolution)
        profile["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0
    
        t0 = time.perf_counter()
        feeds = self._build_feeds(blob, input_resolution)
        profile["build_feed_ms"] = (time.perf_counter() - t0) * 1000.0
    
        t0 = time.perf_counter()
        outputs = self._session.run(self._output_names, feeds)
        profile["onnx_run_ms"] = (time.perf_counter() - t0) * 1000.0
    
        t0 = time.perf_counter()
        detections = self.postprocess(list(outputs))
        profile["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0
    
        profile["infer_total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        self._last_profile = profile
    
        return detections

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
