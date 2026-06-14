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

    "frame_total_ms",
    "scene_ms",
    "device_ms",
    "runtime_state_ms",
    "decision_ms",

    "infer_outer_ms",
    "preprocess_ms",
    "build_feed_ms",
    "onnx_run_ms",
    "postprocess_ms",
    "infer_total_ms",

    "summary_ms",
    "main_log_write_ms",
]


@dataclass
class ProfileRecord:
    timestamp: float
    frame_id: int
    strategy: str
    did_infer: bool

    frame_total_ms: float
    scene_ms: float
    device_ms: float
    runtime_state_ms: float
    decision_ms: float

    infer_outer_ms: float
    preprocess_ms: float
    build_feed_ms: float
    onnx_run_ms: float
    postprocess_ms: float
    infer_total_ms: float

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
