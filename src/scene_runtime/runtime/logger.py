"""CSV/JSONL experiment logger with fixed schema."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG_COLUMNS = [
    "timestamp",
    "frame_id",
    "strategy",
    "workload",
    "temp_c",
    "freq_mhz_avg",
    "arm_clock_mhz",
    "power_w",
    "latency_ms",
    "fps",
    "input_resolution",
    "inference_interval",
    "cpu_threads",
    "governor",
    "decoder_layers",
    "query_budget",
    "detection_count",
    "confidence_mean",
]


@dataclass
class LogRecord:
    """One per-frame log row."""

    timestamp: float
    frame_id: int
    strategy: str
    workload: str
    temp_c: float | None
    freq_mhz_avg: float | None
    arm_clock_mhz: float | None
    power_w: float | None
    latency_ms: float
    fps: float
    input_resolution: int
    inference_interval: int
    cpu_threads: int
    governor: str | None
    decoder_layers: int | None
    query_budget: int | None
    detection_count: int
    confidence_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {col: getattr(self, col) for col in LOG_COLUMNS}

    def validate(self) -> None:
        """Ensure all required columns are present."""
        d = self.to_dict()
        missing = [c for c in LOG_COLUMNS if c not in d]
        if missing:
            raise ValueError(f"Log record missing columns: {missing}")


class RuntimeLogger:
    """Append-only CSV or JSONL logger."""

    def __init__(self, path: Path, fmt: str = "csv") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fmt = fmt.lower()
        self._file = None
        self._csv_writer: csv.DictWriter | None = None
        self._header_written = False

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8", newline="")
        if self._fmt == "csv":
            write_header = self._path.stat().st_size == 0
            self._csv_writer = csv.DictWriter(self._file, fieldnames=LOG_COLUMNS)
            if write_header:
                self._csv_writer.writeheader()
                self._header_written = True

    def write(self, record: LogRecord) -> None:
        record.validate()
        row = record.to_dict()
        if self._fmt == "jsonl":
            self._file.write(json.dumps(row) + "\n")
        else:
            if self._csv_writer is None:
                raise RuntimeError("Logger not opened")
            self._csv_writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
