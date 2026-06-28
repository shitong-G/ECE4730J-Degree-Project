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
        model_paths_by_resolution: dict[int | str, str] | None = None,
        dry_run: bool = False,
        dry_run_latency_ms: float = 45.0,
        providers: list[str] | None = None,
        enable_thread_sessions: bool = False,
        thread_session_counts: list[int] | None = None,
    ) -> None:
        self._model_path = model_path
        self._model_paths_by_resolution = {
            int(resolution): path
            for resolution, path in (model_paths_by_resolution or {}).items()
        }
        self._dry_run = dry_run
        self._dry_run_latency_ms = dry_run_latency_ms
        self._providers = providers or ["CPUExecutionProvider"]
        self._ort: Any = None
        self._session: Any = None
        self._sessions_by_threads: dict[int, Any] = {}
        self._sessions_by_resolution_threads: dict[int, dict[int, Any]] = {}
        self._enable_thread_sessions = enable_thread_sessions
        self._thread_session_counts = sorted(set(thread_session_counts or []))
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._input_names_by_resolution: dict[int, list[str]] = {}
        self._output_names_by_resolution: dict[int, list[str]] = {}
        self._fixed_input_size: int | None = None
        self._fixed_input_sizes_by_resolution: dict[int, int | None] = {}
        self._last_requested_input_resolution: int | None = None
        self._last_resolved_input_resolution: int | None = None

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

    @property
    def last_requested_input_resolution(self) -> int | None:
        """Return the latest action-requested image resolution."""
        return self._last_requested_input_resolution

    @property
    def last_resolved_input_resolution(self) -> int | None:
        """Return the latest actual image resolution fed to ONNX."""
        return self._last_resolved_input_resolution
        
    def load(self) -> None:
        """Load ONNX session or no-op in dry-run mode."""
        if self._dry_run:
            return
        if not self._model_path and not self._model_paths_by_resolution:
            raise FileNotFoundError(
                "model_path or model_paths_by_resolution required for non-dry-run inference. "
                "Export ONNX via scripts/export_model_onnx.sh"
            )
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for real inference. "
                "Install with: pip install onnxruntime"
            ) from e
        self._ort = ort

        if self._model_paths_by_resolution:
            first_resolution = sorted(self._model_paths_by_resolution)[0]
            first_thread = (self._thread_session_counts or [0])[0]
            self._session = self._get_resolution_session(first_resolution, first_thread)
            return

        if self._enable_thread_sessions:
            counts = self._thread_session_counts or [1, 2, 3, 4]
            for threads in counts:
                self._sessions_by_threads[int(threads)] = self._create_session(
                    ort,
                    self._model_path,
                    int(threads),
                )
            self._session = self._sessions_by_threads[counts[0]]
        else:
            self._session = self._create_session(ort, self._model_path, None)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._fixed_input_size = self._read_fixed_input_size(self._session)

    def _get_resolution_session(self, resolution: int, threads: int) -> Any:
        """Create or return a cached session for one resolution/thread pair."""
        if self._ort is None:
            raise RuntimeError("ONNX Runtime is not loaded")
        resolution = int(resolution)
        thread_key = int(threads) if self._enable_thread_sessions else 0
        sessions = self._sessions_by_resolution_threads.setdefault(resolution, {})
        if thread_key not in sessions:
            model_path = self._model_paths_by_resolution[resolution]
            sessions[thread_key] = self._create_session(
                self._ort,
                model_path,
                thread_key if thread_key else None,
            )
        session = sessions[thread_key]
        if resolution not in self._input_names_by_resolution:
            self._input_names_by_resolution[resolution] = [
                i.name for i in session.get_inputs()
            ]
            self._output_names_by_resolution[resolution] = [
                o.name for o in session.get_outputs()
            ]
            self._fixed_input_sizes_by_resolution[resolution] = (
                self._read_fixed_input_size(session)
            )
        self._input_names = self._input_names_by_resolution[resolution]
        self._output_names = self._output_names_by_resolution[resolution]
        self._fixed_input_size = self._fixed_input_sizes_by_resolution[resolution]
        return session

    def _create_session(self, ort: Any, model_path: str, cpu_threads: int | None) -> Any:
        """Create one ONNX Runtime session, optionally pinning intra-op threads."""
        if cpu_threads is None:
            return ort.InferenceSession(
                model_path,
                providers=self._providers,
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(cpu_threads)
        options.inter_op_num_threads = 1
        return ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=self._providers,
        )

    def _select_resolution(self, requested_resolution: int) -> int | None:
        """Select the nearest configured model resolution."""
        if not self._model_paths_by_resolution:
            return None
        requested = int(requested_resolution)
        if requested in self._model_paths_by_resolution:
            return requested
        return min(
            self._model_paths_by_resolution,
            key=lambda value: (
                abs(value - requested),
                value > requested,
                value,
            ),
        )

    def _select_session(
        self,
        requested_threads: int,
        requested_resolution: int | None = None,
    ) -> tuple[Any, int | None]:
        """Return the pre-created session nearest to the requested thread count."""
        selected_resolution = (
            self._select_resolution(requested_resolution)
            if requested_resolution is not None
            else None
        )
        if selected_resolution is not None:
            if self._enable_thread_sessions:
                requested = int(requested_threads)
                counts = self._thread_session_counts or [1, 2, 3, 4]
                thread_key = (
                    requested
                    if requested in counts
                    else min(counts, key=lambda value: abs(value - requested))
                )
            else:
                thread_key = 0
            return (
                self._get_resolution_session(selected_resolution, thread_key),
                selected_resolution,
            )
        else:
            sessions = self._sessions_by_threads
        if not sessions:
            return self._session, selected_resolution
        requested = int(requested_threads)
        if requested in sessions:
            return sessions[requested], selected_resolution
        closest = min(sessions, key=lambda value: abs(value - requested))
        return sessions[closest], selected_resolution

    def _read_fixed_input_size(self, session: Any | None) -> int | None:
        """Return fixed H=W when the ONNX image input has static spatial dims."""
        if session is None:
            return None
        shape = session.get_inputs()[0].shape
        if len(shape) != 4:
            return None
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int) and height == width:
            return height
        return None

    def _resolve_input_resolution(self, requested: int) -> int:
        """Use the exported ONNX spatial size when the model fixes H and W."""
        selected_resolution = self._select_resolution(int(requested))
        if selected_resolution is not None:
            return (
                self._fixed_input_sizes_by_resolution.get(selected_resolution)
                or selected_resolution
            )
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
    
        self._last_requested_input_resolution = int(config.input_resolution)
        input_resolution = self._resolve_input_resolution(config.input_resolution)
        self._last_resolved_input_resolution = int(input_resolution)
    
        t0 = time.perf_counter()
        blob = self.preprocess(frame, input_resolution)
        profile["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0
    
        t0 = time.perf_counter()
        feeds = self._build_feeds(blob, input_resolution)
        profile["build_feed_ms"] = (time.perf_counter() - t0) * 1000.0
    
        t0 = time.perf_counter()
        session, selected_resolution = self._select_session(
            config.cpu_threads,
            input_resolution,
        )
        if selected_resolution is not None:
            self._input_names = self._input_names_by_resolution[selected_resolution]
            self._output_names = self._output_names_by_resolution[selected_resolution]
        outputs = session.run(self._output_names, feeds)
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
