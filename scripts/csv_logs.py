"""Load experiment and thermal monitor CSV logs with schema migration."""

from __future__ import annotations

import csv
from pathlib import Path

# Keep in sync with scene_runtime.runtime.logger.LOG_COLUMNS (do not import — pulls cv2/onnx).
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

OLD_LOG_COLUMNS = [c for c in LOG_COLUMNS if c != "arm_clock_mhz"]

THERMAL_FREQ_COLUMNS = [
    "timestamp",
    "temp_c",
    "arm_clock_hz",
    "scaling_cur_freq_hz",
    "throttled_raw",
]


def _row_dict(fields: list[str], columns: list[str]) -> dict[str, str]:
    row = {col: "" for col in columns}
    for idx, col in enumerate(columns):
        if idx < len(fields):
            row[col] = fields[idx]
    return row


def normalize_experiment_row(fields: list[str]) -> dict[str, str]:
    """Map a raw CSV row to the current LOG_COLUMNS schema."""
    if len(fields) == len(LOG_COLUMNS):
        return _row_dict(fields, LOG_COLUMNS)
    if len(fields) == len(OLD_LOG_COLUMNS):
        row = _row_dict(fields, OLD_LOG_COLUMNS)
        row["arm_clock_mhz"] = ""
        return {col: row.get(col, "") for col in LOG_COLUMNS}
    raise ValueError(
        f"Expected {len(OLD_LOG_COLUMNS)} or {len(LOG_COLUMNS)} fields, saw {len(fields)}"
    )


def load_experiment_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return rows
        for line_no, fields in enumerate(reader, start=2):
            if not fields:
                continue
            try:
                rows.append(normalize_experiment_row(fields))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def is_thermal_freq_log(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        first = handle.readline().strip()
    if not first:
        return False
    if first.startswith("timestamp,"):
        return False
    fields = first.split(",")
    if len(fields) != len(THERMAL_FREQ_COLUMNS):
        return False
    try:
        float(fields[0])
        float(fields[1])
        int(fields[2], 0)
        int(fields[3], 0)
    except ValueError:
        return False
    return True


def load_thermal_freq_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_no, fields in enumerate(reader, start=1):
            if not fields:
                continue
            if len(fields) != len(THERMAL_FREQ_COLUMNS):
                raise ValueError(
                    f"{path}:{line_no}: expected {len(THERMAL_FREQ_COLUMNS)} fields, saw {len(fields)}"
                )
            rows.append(_row_dict(fields, THERMAL_FREQ_COLUMNS))
    return rows
