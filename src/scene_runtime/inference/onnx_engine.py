"""ONNX Runtime RT-DETR inference engine with dry-run fallback."""

from __future__ import annotations

import random
import time
import logging
from typing import Any

import cv2
import numpy as np

from scene_runtime.controller.actions import RuntimeAction
from scene_runtime.inference.postprocess import Detection, postprocess_rtdetr_outputs
from scene_runtime.inference.rtdetr_engine import BaseInferenceEngine


LOGGER = logging.getLogger(__name__)


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
        warmup_runs: int = 0,
        warmup_resolutions: list[int] | None = None,
        warmup_threads: list[int] | None = None,
        inter_op_num_threads: int = 1,
        execution_mode: str = "sequential",
        graph_optimization_level: str = "all",
        enable_cpu_mem_arena: bool = True,
        enable_mem_pattern: bool = True,
        log_severity_level: int = 3,
        query_budget_mode: str = "auto",
        query_budget_input_name: str = "query_budget",
        max_query_budget: int = 300,
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
        self._warmup_runs = max(0, int(warmup_runs))
        self._warmup_resolutions = (
            [int(value) for value in warmup_resolutions]
            if warmup_resolutions is not None
            else None
        )
        self._warmup_threads = (
            [int(value) for value in warmup_threads]
            if warmup_threads is not None
            else None
        )
        self._inter_op_num_threads = int(inter_op_num_threads)
        self._execution_mode = str(execution_mode)
        self._graph_optimization_level = str(graph_optimization_level)
        self._enable_cpu_mem_arena = bool(enable_cpu_mem_arena)
        self._enable_mem_pattern = bool(enable_mem_pattern)
        self._log_severity_level = int(log_severity_level)
        self._query_budget_mode = str(query_budget_mode).lower()
        if self._query_budget_mode not in {
            "auto",
            "strict",
            "postprocess",
            "disabled",
        }:
            raise ValueError(
                "query_budget_mode must be auto, strict, postprocess, or disabled"
            )
        self._query_budget_input_name = str(query_budget_input_name)
        self._max_query_budget = max(1, int(max_query_budget))
        self._last_requested_query_budget: int | None = None
        self._last_applied_query_budget: int | None = None
        self._last_query_budget_mode = "not_invoked"
        self._last_query_budget_supported = False
        self._last_query_output_count: int | None = None
        self._query_mode_warning_emitted = False
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
            "session_select_ms": 0.0,
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

    @property
    def last_requested_query_budget(self) -> int | None:
        return self._last_requested_query_budget

    @property
    def last_applied_query_budget(self) -> int | None:
        return self._last_applied_query_budget

    @property
    def last_query_budget_mode(self) -> str:
        return self._last_query_budget_mode

    @property
    def last_query_budget_supported(self) -> bool:
        return self._last_query_budget_supported

    @property
    def last_query_output_count(self) -> int | None:
        return self._last_query_output_count

    @property
    def max_query_budget(self) -> int:
        """The full exported RT-DETR query budget used for ratios."""
        return self._max_query_budget
        
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
            first_resolution = self._initial_resolution()
            first_thread = (self._thread_session_counts or [0])[0]
            self._session = self._get_resolution_session(first_resolution, first_thread)
            self._warmup()
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
        self._warmup()

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

    def _initial_resolution(self) -> int:
        """Pick the first session to load without forcing unused low-res models."""
        if not self._model_paths_by_resolution:
            raise RuntimeError("No resolution-specific model paths configured")
        if self._warmup_resolutions:
            for resolution in self._warmup_resolutions:
                if int(resolution) in self._model_paths_by_resolution:
                    return int(resolution)
        if self._model_path:
            for resolution, path in self._model_paths_by_resolution.items():
                if path == self._model_path:
                    return int(resolution)
        return max(self._model_paths_by_resolution)

    def _create_session(self, ort: Any, model_path: str, cpu_threads: int | None) -> Any:
        """Create one ONNX Runtime session, optionally pinning intra-op threads."""
        options = self._create_session_options(ort, cpu_threads)
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=self._providers,
        )
        self._validate_query_graph(session, model_path)
        return session

    def _validate_query_graph(self, session: Any, model_path: str) -> None:
        """Reject strict runs unless both TopK nodes were converted.

        The converter writes these node names into ONNX metadata.  Checking the
        metadata at session creation catches an accidentally static or
        postprocess-only model before a formal experiment starts, without
        requiring the optional ``onnx`` Python package at runtime.
        """
        if self._query_budget_mode != "strict":
            return
        input_names = {item.name for item in session.get_inputs()}
        if self._query_budget_input_name not in input_names:
            raise RuntimeError(
                "Strict query-budget run rejected: model "
                f"{model_path!r} has no {self._query_budget_input_name!r} input."
            )
        metadata = {}
        try:
            metadata = dict(session.get_modelmeta().custom_metadata_map or {})
        except Exception:  # pragma: no cover - provider-specific metadata API
            metadata = {}
        node_text = metadata.get("dynamic_query_budget_nodes", "")
        required = ("/model/decoder/TopK", "/postprocessor/TopK")
        if metadata.get("dynamic_query_budget") != "true" or not all(
            name in node_text for name in required
        ):
            raise RuntimeError(
                "Strict query-budget run rejected: dynamic model metadata does "
                "not prove that decoder TopK and final prediction TopK both use "
                f"{self._query_budget_input_name!r}. Recreate it with "
                "tools/make_dynamic_query_onnx.py."
            )

    def _create_session_options(self, ort: Any, cpu_threads: int | None) -> Any:
        options = ort.SessionOptions()
        if cpu_threads is not None:
            options.intra_op_num_threads = int(cpu_threads)
        if self._inter_op_num_threads > 0:
            options.inter_op_num_threads = self._inter_op_num_threads
        options.enable_cpu_mem_arena = self._enable_cpu_mem_arena
        options.enable_mem_pattern = self._enable_mem_pattern
        options.log_severity_level = self._log_severity_level
        if self._execution_mode.lower() == "sequential" and hasattr(ort, "ExecutionMode"):
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        elif self._execution_mode.lower() == "parallel" and hasattr(ort, "ExecutionMode"):
            options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        graph_level = self._graph_optimization_level.lower()
        if hasattr(ort, "GraphOptimizationLevel"):
            if graph_level in {"all", "enable_all"}:
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            elif graph_level in {"extended", "enable_extended"}:
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            elif graph_level in {"basic", "enable_basic"}:
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            elif graph_level in {"disable", "disabled", "none"}:
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return options

    def _warmup(self) -> None:
        """Create selected sessions and run dummy inference before timing starts."""
        if self._dry_run or self._warmup_runs <= 0:
            return
        resolutions = self._warmup_resolutions
        if resolutions is None:
            if self._model_paths_by_resolution:
                resolutions = sorted(self._model_paths_by_resolution)
            elif self._fixed_input_size is not None:
                resolutions = [self._fixed_input_size]
            else:
                resolutions = [640]
        threads = self._warmup_threads
        if threads is None:
            threads = self._thread_session_counts if self._enable_thread_sessions else [0]
        if not threads:
            threads = [0]

        saved_profile = dict(self._last_profile)
        saved_requested = self._last_requested_input_resolution
        saved_resolved = self._last_resolved_input_resolution
        for resolution in resolutions:
            if self._model_paths_by_resolution and int(resolution) not in self._model_paths_by_resolution:
                continue
            frame = np.zeros((int(resolution), int(resolution), 3), dtype=np.uint8)
            for threads_count in threads:
                action = RuntimeAction(
                    mode="onnx_warmup",
                    input_resolution=int(resolution),
                    inference_interval=1,
                    cpu_threads=int(threads_count) if int(threads_count) > 0 else 1,
                )
                for _ in range(self._warmup_runs):
                    self.infer(frame, action)
        self._last_profile = saved_profile
        self._last_requested_input_resolution = saved_requested
        self._last_resolved_input_resolution = saved_resolved

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

    def _build_feeds(
        self,
        blob: np.ndarray,
        input_resolution: int,
        graph_query_budget: int | None,
    ) -> dict[str, np.ndarray]:
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
            elif name == self._query_budget_input_name:
                if graph_query_budget is None:
                    raise RuntimeError(
                        f"ONNX requires {self._query_budget_input_name!r}, but no "
                        "graph query budget was resolved"
                    )
                feeds[name] = np.asarray([graph_query_budget], dtype=np.int64)
            else:
                raise ValueError(f"Unsupported ONNX input: {name}")
        return feeds

    def _resolve_query_budget(
        self,
        action: RuntimeAction,
    ) -> tuple[int, int | None, str, bool]:
        requested = (
            self._max_query_budget
            if action.query_budget is None
            else int(action.query_budget)
        )
        requested = min(max(1, requested), self._max_query_budget)
        supported = self._query_budget_input_name in self._input_names
        if supported:
            applied = (
                self._max_query_budget
                if self._query_budget_mode == "disabled"
                else requested
            )
            mode = (
                "graph_input_fixed_max"
                if self._query_budget_mode == "disabled"
                else "graph_input"
            )
            return requested, applied, mode, True
        if self._query_budget_mode == "strict":
            raise RuntimeError(
                "Dynamic query budget requested in strict mode, but the ONNX "
                f"model has no {self._query_budget_input_name!r} input. Convert "
                "it with tools/make_dynamic_query_onnx.py."
            )
        if self._query_budget_mode == "postprocess":
            return requested, requested, "postprocess_only", False
        return requested, None, "unsupported_inactive", False

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
            "session_select_ms": 0.0,
            "onnx_run_ms": 0.0,
            "postprocess_ms": 0.0,
            "infer_total_ms": 0.0,
        }
    
        total_t0 = time.perf_counter()

        requested_budget = (
            self._max_query_budget
            if config.query_budget is None
            else min(max(1, int(config.query_budget)), self._max_query_budget)
        )
        if self._dry_run:
            time.sleep(self._dry_run_latency_ms / 1000.0)
            detections = self._fake_detections(config)
            self._last_requested_query_budget = requested_budget
            self._last_applied_query_budget = requested_budget
            self._last_query_budget_mode = "dry_run"
            self._last_query_budget_supported = True
            self._last_query_output_count = requested_budget
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
        session, selected_resolution = self._select_session(
            config.cpu_threads,
            input_resolution,
        )
        if selected_resolution is not None:
            self._input_names = self._input_names_by_resolution[selected_resolution]
            self._output_names = self._output_names_by_resolution[selected_resolution]
        profile["session_select_ms"] = (time.perf_counter() - t0) * 1000.0

        requested_budget, applied_budget, budget_mode, budget_supported = (
            self._resolve_query_budget(config)
        )
        if self._query_budget_mode == "strict" and budget_mode != "graph_input":
            raise RuntimeError(
                "Strict query-budget run requires query_budget_mode=graph_input; "
                f"got {budget_mode!r}. Postprocess-only truncation is not "
                "inference acceleration."
            )
        if (
            budget_mode == "postprocess_only"
            and not self._query_mode_warning_emitted
        ):
            LOGGER.warning(
                "Query budget is postprocess_only; this does not reduce decoder "
                "computation and must not be reported as query acceleration."
            )
            self._query_mode_warning_emitted = True
        if (
            budget_mode == "unsupported_inactive"
            and requested_budget < self._max_query_budget
            and not self._query_mode_warning_emitted
        ):
            LOGGER.warning(
                "Requested query budget %d on a static/unsupported ONNX model; "
                "decoder still runs at full budget %d.",
                requested_budget,
                self._max_query_budget,
            )
            self._query_mode_warning_emitted = True
        self._last_requested_query_budget = requested_budget
        self._last_applied_query_budget = applied_budget
        self._last_query_budget_mode = budget_mode
        self._last_query_budget_supported = budget_supported

        t0 = time.perf_counter()
        feeds = self._build_feeds(
            blob,
            input_resolution,
            applied_budget if budget_supported else None,
        )
        profile["build_feed_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        outputs = session.run(self._output_names, feeds)
        self._last_query_output_count = (
            int(np.asarray(outputs[2]).shape[-1])
            if len(outputs) >= 3 and np.asarray(outputs[2]).ndim >= 1
            else None
        )
        profile["onnx_run_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        detections = self.postprocess(list(outputs))
        if budget_mode == "postprocess_only" and applied_budget is not None:
            detections = detections[:applied_budget]
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
