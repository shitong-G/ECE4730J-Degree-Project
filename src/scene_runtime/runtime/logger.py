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
    "thermal_state",
    "raw_thermal_state",
    "control_thermal_state",
    "action_mode",
    "decision_reason",
    "thermal_pressure_level",
    "temp_slope_c_per_min",
    "temp_c",
    "freq_mhz_avg",
    "arm_clock_mhz",
    "power_w",
    "throttling_raw",
    "under_voltage",
    "arm_freq_capped",
    "currently_throttled",
    "soft_temp_limit",
    "did_infer",
    "latency_ms",
    "fps",
    "loop_fps",
    "effective_inference_fps",
    "actual_inference_fps",
    "input_resolution",
    "inference_interval",
    "cpu_threads",
    "governor",
    "requested_governor",
    "applied_governor",
    "governor_applied",
    "governor_apply_error",
    "requested_cpu_affinity",
    "applied_cpu_affinity",
    "cpu_affinity_applied",
    "cpu_affinity_apply_error",
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
    thermal_state: str | None
    raw_thermal_state: str | None
    control_thermal_state: str | None
    action_mode: str | None
    decision_reason: str | None
    thermal_pressure_level: int | None
    temp_slope_c_per_min: float
    temp_c: float | None
    freq_mhz_avg: float | None
    arm_clock_mhz: float | None
    power_w: float | None
    throttling_raw: str | None
    under_voltage: bool | None
    arm_freq_capped: bool | None
    currently_throttled: bool | None
    soft_temp_limit: bool | None
    did_infer: bool
    latency_ms: float
    fps: float
    loop_fps: float
    effective_inference_fps: float
    actual_inference_fps: float
    input_resolution: int
    inference_interval: int
    cpu_threads: int
    governor: str | None
    requested_governor: str | None
    applied_governor: str | None
    governor_applied: bool | None
    governor_apply_error: str | None
    requested_cpu_affinity: str | None
    applied_cpu_affinity: str | None
    cpu_affinity_applied: bool | None
    cpu_affinity_apply_error: str | None
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
