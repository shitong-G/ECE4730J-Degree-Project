"""Per-frame module-level profiling logger."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_COLUMNS = [
    "timestamp",
    "frame_id",
    "strategy",
    "did_infer",

    "serial_total_ms",
    "source_total_ms",
    "source_wait_ms",
    "capture_ms",
    "isp_ms",
    "source_resize_ms",
    "source_save_ms",
    "source_runtime_resize_ms",
    "source_consumer_wait_ms",
    "source_frame_age_ms",
    "source_dropped_frames",
    "source_error_count",
    "frame_total_ms",
    "scene_ms",
    "device_ms",
    "runtime_state_ms",
    "decision_ms",
    "action_apply_ms",
    "fan_update_ms",
    "tracker_reset_ms",

    "infer_outer_ms",
    "preprocess_ms",
    "build_feed_ms",
    "session_select_ms",
    "onnx_run_ms",
    "postprocess_ms",
    "infer_total_ms",

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

    "summary_ms",
    "main_log_write_ms",
]


@dataclass
class ProfileRecord:
    timestamp: float
    frame_id: int
    strategy: str
    did_infer: bool

    serial_total_ms: float
    source_total_ms: float
    source_wait_ms: float
    capture_ms: float
    isp_ms: float
    source_resize_ms: float
    source_save_ms: float
    source_runtime_resize_ms: float
    source_consumer_wait_ms: float
    source_frame_age_ms: float
    source_dropped_frames: float
    source_error_count: float
    frame_total_ms: float
    scene_ms: float
    device_ms: float
    runtime_state_ms: float
    decision_ms: float
    action_apply_ms: float
    fan_update_ms: float
    tracker_reset_ms: float

    infer_outer_ms: float
    preprocess_ms: float
    build_feed_ms: float
    session_select_ms: float
    onnx_run_ms: float
    postprocess_ms: float
    infer_total_ms: float

    diag_infer_start_load1: float
    diag_infer_end_load1: float
    diag_infer_start_mem_available_mb: float
    diag_infer_end_mem_available_mb: float
    diag_infer_start_process_threads: float
    diag_infer_end_process_threads: float
    diag_infer_start_bg_active: float
    diag_infer_end_bg_active: float
    diag_infer_start_bg_pending: float
    diag_infer_end_bg_pending: float
    diag_infer_start_bg_count: float
    diag_infer_end_bg_count: float
    diag_infer_start_bg_skipped: float
    diag_infer_end_bg_skipped: float
    diag_infer_start_bg_errors: float
    diag_infer_end_bg_errors: float
    diag_infer_start_bg_last_source_ms: float
    diag_infer_end_bg_last_source_ms: float
    diag_infer_bg_captures_delta: float
    diag_infer_bg_skipped_delta: float
    diag_infer_bg_errors_delta: float

    summary_ms: float
    main_log_write_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {col: getattr(self, col) for col in PROFILE_COLUMNS}


class ProfileLogger:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer: csv.DictWriter | None = None

    def open(self) -> None:
        write_header = not self._path.exists() or self._path.stat().st_size == 0
        self._file = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=PROFILE_COLUMNS)
        if write_header:
            self._writer.writeheader()

    def write(self, record: ProfileRecord) -> None:
        if self._writer is None:
            raise RuntimeError("ProfileLogger not opened")
        self._writer.writerow(record.to_dict())
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
        self._file = None
        self._writer = None
