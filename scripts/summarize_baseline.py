#!/usr/bin/env python3
"""Summarize experiment CSV logs into baseline metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from collections import Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scene-runtime baseline CSV logs")
    parser.add_argument("--input", type=Path, required=True, help="Experiment CSV log")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/baseline_native"),
        help="Directory for JSON/CSV summary outputs",
    )
    parser.add_argument("--label", default=None, help="Optional run label")
    return parser.parse_args()


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _series(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _to_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def _to_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _bool_ratio(rows: list[dict[str, str]], column: str) -> float | None:
    values = [_to_bool(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _time_above(rows: list[dict[str, str]], column: str, threshold: float) -> float | None:
    if len(rows) < 2:
        return None
    total = 0.0
    for current, nxt in zip(rows, rows[1:]):
        timestamp = _to_float(current.get("timestamp"))
        next_timestamp = _to_float(nxt.get("timestamp"))
        value = _to_float(current.get(column))
        if timestamp is None or next_timestamp is None or value is None:
            continue
        dt = max(0.0, next_timestamp - timestamp)
        if value >= threshold:
            total += dt
    return total


def _counts_json(rows: list[dict[str, str]], column: str) -> str:
    counts = Counter(row.get(column) or "unknown" for row in rows)
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def _positive_series(rows: list[dict[str, str]], column: str) -> list[float]:
    return [value for value in _series(rows, column) if value > 0.0]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stat(values: list[float], op: str) -> float | None:
    if not values:
        return None
    if op == "mean":
        return mean(values)
    if op == "median":
        return median(values)
    if op == "min":
        return min(values)
    if op == "max":
        return max(values)
    raise ValueError(f"Unknown stat op: {op}")


def summarize(rows: list[dict[str, str]], label: str) -> dict[str, float | int | str | None]:
    timestamps = _series(rows, "timestamp")
    latencies = _positive_series(rows, "latency_ms")
    fps = _series(rows, "fps")
    loop_fps = _series(rows, "loop_fps") or fps
    effective_inference_fps = _series(rows, "effective_inference_fps")
    actual_inference_fps = _series(rows, "actual_inference_fps")
    temps = _series(rows, "temp_c")
    freqs = _series(rows, "freq_mhz_avg")
    arm_clocks = _series(rows, "arm_clock_mhz")
    detections = _series(rows, "detection_count")
    confidences = _series(rows, "confidence_mean")

    return {
        "label": label,
        "strategy": rows[0].get("strategy") if rows else None,
        "wall_time_sec": (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0,
        "total_frames": len(rows),
        "inference_frames": len(latencies),
        "skipped_frames": len(rows) - len(latencies),
        "skip_ratio": ((len(rows) - len(latencies)) / len(rows)) if rows else None,
        "latency_ms_mean": _stat(latencies, "mean"),
        "latency_ms_median": _stat(latencies, "median"),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "latency_ms_p99": _percentile(latencies, 0.99),
        "latency_ms_max": _stat(latencies, "max"),
        "fps_mean": _stat(fps, "mean"),
        "fps_median": _stat(fps, "median"),
        "fps_min": _stat(fps, "min"),
        "fps_max": _stat(fps, "max"),
        "loop_fps_mean": _stat(loop_fps, "mean"),
        "effective_inference_fps_mean": _stat(effective_inference_fps, "mean"),
        "effective_inference_fps_median": _stat(effective_inference_fps, "median"),
        "effective_inference_fps_min": _stat(effective_inference_fps, "min"),
        "effective_inference_fps_max": _stat(effective_inference_fps, "max"),
        "actual_inference_fps_mean": _stat(actual_inference_fps, "mean"),
        "actual_inference_fps_median": _stat(actual_inference_fps, "median"),
        "actual_inference_fps_min": _stat(actual_inference_fps, "min"),
        "actual_inference_fps_max": _stat(actual_inference_fps, "max"),
        "temp_c_start": temps[0] if temps else None,
        "temp_c_end": temps[-1] if temps else None,
        "temp_c_mean": _stat(temps, "mean"),
        "temp_c_max": _stat(temps, "max"),
        "time_above_70c_sec": _time_above(rows, "temp_c", 70.0),
        "time_above_75c_sec": _time_above(rows, "temp_c", 75.0),
        "time_above_80c_sec": _time_above(rows, "temp_c", 80.0),
        "currently_throttled_ratio": _bool_ratio(rows, "currently_throttled"),
        "soft_temp_limit_ratio": _bool_ratio(rows, "soft_temp_limit"),
        "arm_freq_capped_ratio": _bool_ratio(rows, "arm_freq_capped"),
        "under_voltage_ratio": _bool_ratio(rows, "under_voltage"),
        "thermal_state_counts": _counts_json(rows, "thermal_state"),
        "raw_thermal_state_counts": _counts_json(rows, "raw_thermal_state"),
        "control_thermal_state_counts": _counts_json(rows, "control_thermal_state"),
        "action_mode_counts": _counts_json(rows, "action_mode"),
        "freq_mhz_avg_mean": _stat(freqs, "mean"),
        "arm_clock_mhz_mean": _stat(arm_clocks, "mean"),
        "arm_clock_mhz_min": _stat(arm_clocks, "min"),
        "arm_clock_mhz_max": _stat(arm_clocks, "max"),
        "detection_count_mean": _stat(detections, "mean"),
        "confidence_mean": _stat(confidences, "mean"),
    }


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    label = args.label or args.input.stem
    summary = summarize(rows, label)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{label}_summary.json"
    csv_path = args.output_dir / f"{label}_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved JSON summary: {json_path}")
    print(f"Saved CSV summary:  {csv_path}")


if __name__ == "__main__":
    main()
